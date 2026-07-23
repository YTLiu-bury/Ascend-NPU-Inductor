# npu_inductor 在大模型训练/推理上的优劣势分析与创新优化方案

> 基于 `npu_inductor_2.13.0` 全量源码精读（codegen/triton.py 6955 行、npu_triton_heuristics.py 2698 行、npu_patch.py 2566 行、lowering.py 994 行等）+ 真实 Llama3.2-3B 训练数据 + torchbench/UT/单算子基准 + 11 条已知问题清单（problems.csv）的系统性分析。
>
> 所有结论均带 `file:line` 代码证据或实测数据，关键数字来自 `npu_inductor_report.md`、`npu_inductor设计与对标分析.md`、`NPU_Inductor_Linearize_技术报告.md` 及 document/ 下的实测报告。

---

## 〇、一句话结论

`npu_inductor` 是一座**质量过硬但工程承重**的桥：它把上游 Inductor「CUDA 多维 grid + 硬件调度」的假设，用一套与算子无关的**线性化代数变换 + 固定 40 核 group dispatch** 翻译到 Ascend NPU。

- **算子级 kernel 质量是真的好**：torchbench 34 模型 OP 几何平均 **1.30×**（vs 官方 dvm 1.15×，22/32 模型占优），京东 OneRec 真实训练 4 backbone 吞吐全第一且**是唯一跑通的 Triton 方案**。
- **但 E2E 是另一回事**：在 Llama3.2-3B 4 卡 200 步训练上，compile 总时间反而比 eager **慢 3.9×**（730s vs 187s），单步 device 时间虽快（610ms vs 889ms）却被 ~608s 的 host 侧开销吞噬。根因是**小算子密集图的 host launch 税 + autotune 冷启动 + 动态 shape 重编译**，而 torchbench 那份漂亮的 E2E 1.51× 是开了 aclgraph（图捕获）测出来的——**图捕获不开，npu_inductor 的 kernel 优势就无法落到端到端**。

下面的优化方案，核心就是把"算子级快"真正兑现成"E2E 快"，并补齐 reduction/gather/MM 三块结构性短板。

---

## 一、核心机制速览（理解优劣势的前提）

### 1.1 架构：import 即打补丁的 monkey-patch 后端

`import npu_inductor` 时 `__init__.py` 按 12 步固定顺序装配：注册 backend（`__init__.py:148`）→ 与 torch_npu 自带后端"快照-预导入-恢复"博弈（`__init__.py:69-146`）→ `add_npu_patch()`（`npu_patch.py:2442`）→ lowering/decomp 覆盖。**不重写 Inductor 调度框架，只在上游 `TritonKernel`/`TritonScheduling` 的扩展点上挂子类**。净代码 ~7.6k 行，其中 ~2.2k 行本质是补偿 triton-ascend/bishengir 编译器缺陷（应下沉），~5.4k 行是 NPU 必要的（reduction 切分 + autotune + 动态 shape）。

### 1.2 Linearize：把任意多维迭代空间压成 1D，再均衡切到 40 核

Ascend 910B3 只有 **40 个 AI-Vector Core（CU）**，dispatch 是固定宽度一维 grid `grid=(40,1,1)`。Linearize（`TRITON_CODEGEN_LINEARIZE`/`codegen_linearize`，`codegen/triton.py:4787`）做三件事：

1. **展平**：多维迭代空间 → 一维 block 索引 `[0, total_blocks)`，`total_blocks = ∏ ceildiv(numel, real_block)`；
2. **划分**：`group_size/group_base` 把 `total_blocks` 均衡、精确覆盖地切到 40 个 CU（`codegen/triton.py:3463`）；
3. **反线性化**：每个 CU 在 `for i in range(group_size)` 里把 `f=group_base+i` 用 `//`/`%` 拆回每 node 的 tile offset（`npu_patch.py:1161`，odometer div/mod，B1/B2 优化把累积 block 积 hoist 到 pre_loop）。

关键数据结构是 `tree_node_mapping`（`codegen/triton.py:5728` `_apply_linearize`）——多遍 fold 把"置换轴/扁平辅助轴/双 decomp 链"坍缩到基础轴，**防止笛卡尔积爆炸**（不 fold 会慢且算错，softmax-bw+sum+permute 实测不 fold 慢且错 ~2464×，`triton.py:5921`）。

### 1.3 三条 lowering 路径

`lowering.py:153-167` 是路由总闸：**GENERATE_LIST 里的算子走 Triton codegen；有 decomposition 的先分解再 codegen；其余 `make_fallback` 掉回 eager aclnn**。对 LLM 关键算子：

| 算子 | 路径 | 证据 |
|---|---|---|
| pointwise/view/cat/layernorm/rmsnorm/softmax/RoPE | **Triton codegen，可融合** | `lowering.py:34-64`、`npu_patch.py:1537-1609` |
| matmul/linear/bmm | Triton codegen，但 **bias-add→addmm 融合被禁** | `lowering.py:107-111`、`npu_patch.py:1632-1662` |
| attention (sdpa) | **NPU 原生 FlashAttention（aclnn），不与前后 op 融合** | `npu_patch.py:68-143` |
| dropout | NPU 原生（bool mask→uint8 hack） | `npu_patch.py:1563-1609` |
| gather/index/scatter/index_put/argmax/topk/sort/cumsum/embedding_backward | **全部 eager fallback**（A5 gather 路径是死代码 `if False and is_a5()`） | `lowering.py:134` |

---

## 二、优势（带证据）

### 优势 1：算子级 kernel 质量过硬，全面领先官方后端

- torchbench 34 模型（静态）：**OP 几何平均 1.30×，E2E 几何平均 1.51×，OP>1.0 的模型 32/34**（`npu_inductor_report.md §8.3`）。
- 对标官方 dvm：1.30× vs 1.15×，**逐模型占优 22/32**（`设计与对标分析 §0.1`）。
- 单算子级（60 个可比用例）：算子设备耗时几何平均 **dvm/NI=1.27**（NI 更快 38/60）；视图重排融合收益最大——`clip_qkv_bias_grad_sum` **8.83×**、`select_scatter_qkv_bw` **6.24×**、`masked_fill_softmax_bw` **5.90×**、`var_mean_norm` **5.42×**（`设计与对标分析 §0.6`）。
- 京东 OneRec 真实客户训练：4 backbone 吞吐**全部第一**，相对 eager 几何平均 **1.25×**，且**是唯一能跑通的 Triton 方案**（官方 default triton 4 个全崩，`设计与对标分析 §0.5`）。

### 优势 2：固定 40-grid + 核内循环是经过量化验证的正确默认

实测（910B3，profiler 纯 device 口径）固定 grid 相对"每块一 program"的多 grid：

- **轻 tile（elementwise/短归约）是主受益面**：grid=8192 时 add 加速 **8.89×**，行归约 R=64 加速 **5.16×**（`npu_inductor_report.md §三`）。
- 大尺寸（33M+）仍快 **3.2×**；且 `grid_0 = max(1, min(ceil(xnumel/XBLOCK), 40))`（`npu_triton_heuristics.py:461`）让 grid≤40 时与多 grid 等价、零开销。
- **免疫 65535 coreDim 上限和 coreDim=0 空切片崩溃**（`EE1003`），永远合法。

> 本质：调度开销 **O(核数)** 而非 O(tile 数)。这是 npu_inductor 区别于上游的、经过严谨 A/B 的设计决策。

### 优势 3：动态 shape 是结构性强项（一次编译吃整族 shape）

`codegen_linearize` 按 `isinstance(length/divisor, (int, sympy.Integer))` 逐轴决策：静态轴折字面常量、符号轴追加为运行时标量参数（如 `y1divisor`）。**一份 kernel 自适应整族 shape，不重编译**（`npu_inductor_report.md §七`）。京东 OneRec 4 个单轴动态 shape backbone 全第一、全跑通，正是这一能力的实证。对照官方 default 后端是"分桶穷举编译"。

### 优势 4：host launch 开销已压到物理下限

经分层探针定位，单次 `kernel.run` host 开销从 **~23us（慢于 eager 1.5×）压到 ~10us（稳定快于 eager），下降 55%**（`npu_triton_launch_host_overhead.md`）。大头是 `RunOpApiV2` 轻量保序入队（~8us）+ lambda 瘦身（~3us）+ getPointer PyLong 快路径（~1us）+ 收敛后重绑 `run`（省 1.85us）。剩余 ~6us 是"保持与 eager 算子相对顺序"的物理下限。

### 优势 5：精细 autotune + UB 容量感知

- UB 容量感知裁剪（`_NPU_UB_CAPACITY_BYTES=192KiB`，`npu_triton_heuristics.py:1670`）+ OutOfResources/MLIRCompilationError 兜底（`:776`），避免崩溃 config。
- 精细 sweep（30~60 config）相对粗放启发式默认，device 耗时压下 **1.5×–2×**（`npu_inductor_report.md §六`）。
- 分层 primary/fallback 预编译 + 并行编译（`npu_triton_heuristics.py:694,724`）降低冷启动。

### 优势 6：视图重排融合（Linearize 的杀手锏）

`_apply_linearize` 的索引化简 + greedy prefix-fill tiler（`npu_patch.py:794`）让 permute/transpose/select 的非连续访存**尽可能折回连续 burst**——这是单算子基准里 4-8× 收益的来源，也是 910B 上「real-block 按输入步长优先分配」把 T5 position-bias 从 ~290us 降到 ~40us 的机制（`npu_inductor_report.md §一`）。

---

## 三、劣势（带证据）—— LLM 场景的结构性短板

### 劣势 1：离散访存/gather/div-mod 静默退标量 —— 头号性能杀手（5×–250×）

- **NPU 向量单元面向连续访存设计**。一旦 trailing/stride-1/归约轴非连续，访存退化为 gather，性能断崖式下降，**且不报错、只静默变慢**。
- **致命传染性**：**一个连续 vector op 和一个非连续访问共存于同一 kernel，会整段退化为逐元素标量执行**（`npu_inductor_optimization_report.md §一`）。
- 实测形态：T5 相对位置 bias gather（softmax key 轴 stride=H，**~250×**）；T5 attn-output 反向 transpose（~4ms@S=1024，占该片段 95%）；degenerate rank-4 tile（~40× 慢，`triton.py:116` 4969us vs 116us）。
- **当前只能靠上层 realize 物化（多一次 HBM 往返）或 codegen fold（只覆盖"`%`/`//` 同变量"特例）补偿，未根治**。混合形态仍要等编译器侧 `indirect_load/store` SIMT 通路下沉。

### 劣势 2：归约轴默认串行 —— LLM 的最大瓶颈

- `should_use_persistent_reduction` **恒返回 False**（`codegen/triton.py:1501`），所有归约走 looped 多 pass。
- 唯一的跨核归约 `npu_rsplit_outer` 条件**极严**：仅 OUTER、单一非 welford 归约、x≤40·256、r≥2048 且 r≥x（`triton.py:4533`）。
- **LLM 主力归约全是 INNER 且 rsplit 不触发**：
  - attention 的 `QK^T`（K-dim）、softmax（last-dim max/sum）、`P·V`（N-dim）—— 单核串行走整个 seq_len/head_dim，seq_len≥2048 时 R0_BLOCK≤16384 必多 pass；
  - layernorm/rmsnorm 的 var/mean 是 **welford**，rsplit 明确不支持（tuple 语义）；
  - 单算子基准里 dvm 反超的全是 reduction/归一化反向类：`bn_backward_reduce` 0.28×、`softmax_dyn` 0.57×、`rms_norm_bw_view_sum_dyn` 0.84×（`设计与对标分析 §0.6`）。

### 劣势 3：E2E 被 host 侧开销吞噬 —— 608s 谜团

这是最该被读懂的一点。Llama3.2-3B 4 卡训练（`eager_vs_compile_comparison.md`）：

```
compile 单步 device 时间 610ms < eager 889ms  ✓（融合 + 减少 launch 确实更快）
compile 总时间      730s  ≈ 3.9×  eager 187s  ✗（端到端反而慢 3.9×）
未解释开销          730 − 200×610ms ≈ 608s（82%）
```

**根因（用源码 + 基准数据交叉验证）**：
1. **610ms 是编译区域的局部计时**，不是真实 wall-clock/step（真实 ≈ 730/200 = **3.65s/step**）。profiler 自承"未捕获 NPU Device 时间，Host 端占主导"（`npu_inductor_deep_analysis.md §三`）。
2. **小算子密集图 × host launch 税**：device-event 有 **~100us 的 host launch 地板**——单 program marker kernel device 实测仅 **0.96us，但 event 报 102.8us**，这 102us 全是 host 下发开销（`npu_inductor_report.md §三`）。LLM 反向有几十~上百个融合 kernel，每个付 10-23us host 税。
3. **autotune 冷启动**：每个新 (kernel,shape) 编译 30~60 config（`npu_triton_heuristics.py:1819`），首迭代分钟级 stall；动态 batch/seq 持续触发重编译。
4. **图捕获未启用**：torchbench 那份 E2E 1.51× **开了 aclgraph**（设计文档 §0.1 明示"aclgraph 会消除大量 host 侧调度/launch 开销，对 E2E 影响极大"）。llama3 实验没开 → host 开销原形毕露。

> 结论：**npu_inductor 的 kernel 质量没问题，问题是「没有图捕获兜底」时，host 侧开销把算子优势全吃掉了。** 这是当前 NPU 上 compile 模式"看着该赢却输了"的根本原因。

### 劣势 4：MM/attention 不融合，缺少图级融合框架

- **matmul 保持独立**（`extern_kernels.mm`/aclnnMm），bias-add 融合被 `disable_addmm_fusion` 关掉 → LLM 每个 linear 层的 `GEMM + bias + act + residual + dropout` 是一串独立 kernel，多付 launch + HBM。
- **attention 走原生 FlashAttention，与前后 RMSNorm/RoPE/residual 完全不融合**（`npu_patch.py:68-143`）。
- **图级 pattern matching 融合框架是"规划但未实现"**（`trial_period_plan.md` 第 4-6 月）——当前只有 codegen 层单 kernel 融合，没有跨算子 matmul→add→activation、layernorm→dropout→add 的系统性融合。

### 劣势 5：gather/scatter/index/argmax 全部 eager fallback

`lowering.py:134` 是死代码（`if False and is_a5()`），导致 `gather/index/index_put/scatter/scatter_reduce/argmax/topk/sort/cumsum/embedding_backward` **全部掉回 eager aclnn，无法与 Triton 融合**。对 LLM：训练的 gradient scatter、embedding 反向，推理的 top-k/采样，都是融合隔断点。

### 劣势 6：固定 40 核的物理上限

910B3 只有 40 个 vector core（`npu_config.py:40` 兜底返回 40）。无论 tensor 多大，fused pointwise 只能靠加大 XBLOCK 饱和，无法像 GPU 扩到数千 thread。大 hidden（4096×8192）的 elementwise 难以占满。

### 劣势 7：正确性覆盖与维护脆弱性

- **UT 通过率**：静态 ~82%、动态 shape ~71%、整体 ~76%（`npu_inductor_report.md §8.4`）；opinfo 仅 ~57%。
- **torchbench 4 模型精度失败**：hf_Bart / hf_T5_base / hf_T5_large / soft_actor_critic（T5 系是 gather + dual-decomp 折叠最密集处）。
- **problems.csv P0**：`qkv_rmsnorm_rope_half_static` 精度/功能错误（疑 triton-ascend/npuir 层）；`t5_softmax_position_bias_fwd` 性能差（transpose copy 76.8us vs CANN 41.7us）。
- **维护脆弱**：①`__init__.py:69-146` 与 torch_npu 自带后端的"快照-恢复"博弈，注释自述"WORSE: torch_npu reassigns class methods at import time, globally"；②torch 2.13 把 `range(` 改 `tl.range(` 曾静默引发 **3 处 reduction 回归（含一处精度全错）**；③一批补丁改在 site-packages（ACL 头顺序、expand_shape fix），**重装即丢**；④`TORCHINDUCTOR_NPU_BACKEND="mlir"` 是借用的 env 名，仅为 `ACL_CONTINUE_ON_FAILURE` 副作用，否则一个坏 autotune config 会故障整个 stream（507035）；⑤config 已迁移到 typed config **无 env 层，文档与代码漂移**。

---

## 四、创新优化方案（重点）

> 以下方案按 **LLM 收益×创新度** 排序，每条给出：根因、方案、落地位置（file:line）、预期收益、创新点。前 3 个是"立竿见影解开 E2E 困局"的，后几个是"补结构性短板"的。

### 🥇 方案 1：编译区域默认 aclgraph 图捕获 + DDP-aware 切分 —— 直接解开 608s 谜团

**根因**：劣势 3。算子级 OP 1.30× 但 E2E 反慢 3.9×，唯一变量是 torchbench 用了图捕获、llama3 没用。

**方案**：
- 对 compiled fwd/bwd region 默认启用 **aclgraph 捕获**（即 `torch.compile(mode="reduce-overhead")` 的 NPU 对应物），把数千次 triton launch + aclnnMm + dropout op **压成一次图提交**，host 开销从 `N×10us` 降到 `O(1)`。
- **创新点——DDP/HCCL-aware graph 切分**：DDP 的 allreduce 是跨卡同步点，不能进单卡 graph。把每个 compiled region 在 allreduce 边界切成多段图，HCCL 通信用 HCCL-graph 或 graph 外单独提交；用 mspti profiler 自动定位"可捕获的连续 triton 段" vs "必须破图的同步点"。
- 配合 `NPU_RSPLIT_OUTER` 的 workspace allocation 在图捕获前完成（避免捕获期动态分配）。

**落地**：新增 `npu_inductor/graph_capture.py`，hook `NPUWrapperCodeGen` 生成可捕获的 wrapper；wrapper.py 的 stream/workspace 规划（`codegen/wrapper.py`）改造为图友好。

**预期收益**：消除 ~600s 量级的 host 开销，把 E2E 从"慢于 eager 3.9×"拉回到 torchbench 的 **1.5×** 水平——**这是 ROI 最高的一项，且不改任何 kernel**。

---

### 🥇 方案 2：Cube 单元卸载归约（Reduction-as-GEMM）—— 攻克 LLM 头号算子瓶颈

**根因**：劣势 2。归约串行 + welford/INNER 不支持 rsplit。Ascend 有 **Cube（矩阵）核 + Vector 核**，当前归约只用 vector 串行 add。

**方案**（创新）：
- **核心洞察**：`sum(x, dim=-1)` 等价于 `ones(1,R) @ x(R,N)`，`mean`/`sum_of_squares` 同理。把"大轴归约 + 小输出"重写为 **batched GEMV 走 aclnnMm**，用 Cube 核的 MAC 阵列算，吞吐远高于 vector 串行。
- **welford（layernorm/rmsnorm）拆解**：`var = E[x²] - E[x]²` → 用两个 GEMV 算 `S1=sum(x)`、`S2=sum(x²)`，再一个轻量 vector finalize 出 mean/rsqrt。把 layernorm 的三趟串行归约变成 **2 个 Cube GEMV + 1 个 vector 收尾**。
- **INNER 归约的"转置降级"**：attention 的 softmax（last-dim）是 INNER，rsplit 不触发。创新地插入一次**融合到上游 bmm 输出的布局转置**（把 last-dim 变 leading），转成 OUTER 后用 rsplit 跨核，或直接走 Cube。转置成本被 GEMM 的 transposed-output（Cube 原生支持）吃掉。

**落地**：`lowering.py` 新增 reduction-detection pass（识别 reduction_numel 大、输出小的 `sum/mean/var`），在 `_register_npu_inductor_fallbacks` 之前改写为 GEMV pattern；welford 在 `npu_var_mean_helper_`（`lowering.py:193`）里拆解。

**预期收益**：layernorm/rmsnorm/softmax 反向从"单核串行多 pass"变成"Cube 高吞吐并行"，量级上看齐 dvm 在 `bn_backward_reduce` 上的优势（当前 NI 0.28×）。

---

### 🥇 方案 3：学习型 tile 代价模型（Learned Cost Model）替代 runtime autotune sweep

**根因**：劣势 3 的冷启动 + 劣势 5 的 autotune 臃肿（2.7k 行）。每个新 (kernel,shape) 编译 30-60 config + bench，LLM 动态 shape 首迭代分钟级 stall；且 event-timing autotune 对快 kernel失真（19.9us 测成 223us，`heuristics.py:69`）。

**方案**（创新）：
- 训练一个**轻量回归/排序模型**：输入 (op 指纹: pointwise/reduction/permute, 形状 signature, dtype, UB 估算负载, load 数) → 直接预测最优 (XBLOCK, R0_BLOCK, num_warps)，**跳过 runtime bench**。
- **训练数据现成**：用现有 60 个 test_all case + torchbench 30 模型 + 京东 OneRec 4 backbone，离线跑全 sweep 收集 `(输入, 最优config, device耗时)`，几万条样本足够。
- 在线时只编译**预测 top-1（或 top-3 容错）** config，冷启动从"编译 60 个 + bench"降到"编译 1-3 个"。
- 模型与 cubin hash 一起持久化、跨卡共享（4 卡不再各自 autotune）。

**落地**：`npu_triton_heuristics.py` 的 `_partition_configs_by_tier`（`:694`）前插一个 `_predict_config`，命中 cache/model 时短路 sweep；`config.py` 加 `learned_autotune=True`。

**预期收益**：首迭代编译时间 **分钟级 → 秒级**，动态 shape 重编译近乎消除；顺带把 autotune 代码 2.7k → ~900 行（`npu_inductor_report.md §九` 的瘦身目标）。

---

### 方案 4：软件持久化 Mega-Kernel —— 一次 launch 覆盖整个子图

**根因**：劣势 3 + 4。LLM 一个 transformer block 有几十个连续 pointwise（rope→qkv-proj→scale→mask→...），每个都是独立 launch，付 host 税 + HBM 往返。

**方案**（创新）：
- 把一个 block 内**连续的、tile-compatible 的 pointwise 子图**编译成**单个 grid=(40,) 持久 kernel**：每个 CU 用 **UB ping-pong 双缓冲**流式处理该子图的所有 tile，数据全程驻 UB、不落 HBM，**一次 launch 覆盖整个子图**。
- 在 attention/dropout（原生 op）边界切分 mega-kernel，把可融合的 pointwise 尾巴（residual+layernorm）尽量并进前/后 mega-kernel。
- 这是把 **FlashAttention 的 persistent-kernel 哲学推广到整个 LLM block 的 pointwise 段**。

**落地**：`codegen/triton.py` 新增 mega-kernel 编排（基于现有 `_apply_linearize` + `for i in range(group_size)`，把内层循环从"单 kernel 的 tiles"升级为"多 op 的 tile 流水"）；UB 容量预算用现有 `_NPU_UB_CAPACITY_BYTES`（`heuristics.py:1670`）做准入。

**预期收益**：把 N 个连续 pointwise 的 launch 数从 N 降到 1，HBM 往返大幅减少——对 LLM 的 elementwise 密集段收益最大。

---

### 方案 5：vector/scalar 双路径 kernel 分裂（Heterogeneous-Load Fission）—— 治"一粒 gather 毒化整 kernel"

**根因**：劣势 1。一个非连续 load 让整个 fused kernel 退标量（5-250× 静默退化）。

**方案**（创新）：
- 在 codegen 的 `prepare_indexing`（`codegen/triton.py:477`）后插入**访存连续性逃逸分析**：对每个 load 算 effective trailing stride，把 kernel 内的 load 集合分成 `{contiguous}` 与 `{gather}` 两类。
- **当两者共存时，把 kernel 分裂成两个**：连续部分走 vector-fast kernel 写一个中间 buffer；gather 部分单独 kernel 读，中间 buffer 通过 scheduler 的 memory reuse 尽量留在 cache/UB。**不让一个 gather 把整个 vector kernel 拖下水**。
- 对纯 gather kernel，主动 emit triton-ascend 的 `indirect_load/store`（已部分下沉，`31eadfb97`/`6b283b3e2`）走 SIMT 通路，而非退标量。

**落地**：`codegen/triton.py` 新增 `_analyze_load_strides` + fission 决策点（在 `_simplify_compound_indexing` 之后、`codegen_kernel` 之前）。

**预期收益**：把"5-250× 静默退化"变成"vector 段全速 + gather 段隔离"，hf_T5/Bart 这类 T5 系（当前被 skip）有望转正。

---

### 方案 6：GEMM-Epilogue + FlashAttention 融合 lowering

**根因**：劣势 4。MM 独立、attention 独立，缺图级融合。

**方案**：
- **GEMM-epilogue**：lowering 把 `(addmm → activation → +residual)` 识别为 pattern，调用 CANN 的 fused GEMM API（带 epilogue 的 aclnnMm 变体）或自研 triton epilogue-fused GEMM。把当前被 `disable_addmm_fusion` 拆开的 bias-add 融回去，省一次 launch + HBM。
- **Attention prologue/epilogue**：把入 attention 前的 RMSNorm/RoPE 和出 attention 后的 residual+dropout，用 CANN 支持的 `score+dropout` fused 变体接到 FlashAttention 上下（`npu_patch.py:68` 的 `npu_fusion_attention_v3` 已有 dropout 选项，可扩展）。
- **图级 pattern rewrite pass**（实现 trial_plan 规划未做的部分）：MLIR/pass 层 match `(mm→add→act)`、`(rmsnorm→rope)`、`(layernorm→dropout→residual)` → 重写到 fused NPU op。

**落地**：`npu_patch.py` 的 `_disable_addmm_fusion_pass`（`:1632`）改为条件化（仅对纯 GEMM 关，对带 epilogue 的开）；新增 `lowering.py` 的 pattern matcher。

**预期收益**：LLM 每个 transformer 层省数个 launch + HBM 往返，结合方案 4 的 mega-kernel 可显著压低 launch 密度。

---

### 方案 7：A5/910_95 前瞻 + 把"应下沉 2.2k 行"真正下沉

**根因**：劣势 1/2/7 的根多在编译器侧。

**方案**：
- **A5(910_95) 已用"一程序一 tile"dispatch**（`triton.py:3481`），绕开 40 核 group 循环——A5 上劣势 2/3 的归约/核数约束自动缓解。优先把 LLM 负载迁到 A5。
- **把 ~2.2k 行编译器补偿下沉到 triton-ascend/NPUIR**（离散访存 SIMT 通路、grid 折叠已部分合入 `8232cb886`/`31eadfb97`），上层删 workaround，覆盖面更广、其他前端受益。下沉前用 problems.csv 的 repro 在编译器侧验证性能达标再摘补丁。

**预期收益**：上层维护成本骤降（不再因 torch/triton-ascend 升级频繁回归），正确性覆盖收敛。

---

### 优化路线图（按 ROI 排序）

| 阶段 | 方案 | 投入 | LLM 收益 | 风险 |
|---|---|---|---|---|
| **P0（立即）** | 方案1 图捕获默认化 | 中 | ⭐⭐⭐⭐⭐ E2E 从 −3.9× → +1.5× | DDP 切分正确性 |
| **P0** | 方案3 学习型 cost model | 中 | ⭐⭐⭐⭐ 冷启动分钟→秒 | 模型泛化 |
| **P1** | 方案2 Cube 卸载归约 | 大 | ⭐⭐⭐⭐ layernorm/softmax 量级提升 | Cube/Vector 协同 |
| **P1** | 方案4 mega-kernel | 大 | ⭐⭐⭐ launch 密度↓ | UB 容量 |
| **P2** | 方案6 GEMM/Attention epilogue | 中 | ⭐⭐⭐ MM 段融合 | CANN API 依赖 |
| **P2** | 方案5 vector/scalar fission | 中 | ⭐⭐⭐ T5 系转正 | 分裂决策 |
| **P3** | 方案7 A5 迁移 + 下沉 | 大 | ⭐⭐ 长期维护 | 跨编译器协调 |

---

## 五、给"该不该用 npu_inductor 跑 LLM"的明确建议

1. **算子质量过关，可以放心用**——OP 1.30×、动态 shape 强、固定 grid 设计扎实、launch 已优化到物理下限。它确实是目前 NPU 上**质量最高、唯一能在动态 shape 客户场景跑通**的 Triton 编译路径。
2. **但必须配套图捕获（aclgraph/reduce-overhead）**，否则短训练（<10000 步）的 host 开销 + autotune 冷启动会让 E2E 反慢于 eager——这就是 llama3 200 步实验的教训。**长稳态训练（万步级）+ 图捕获 + shape 稳定时，compile 才稳赢。**
3. **当前真正的工程短板不是 kernel，而是**：归约串行（attention/layernorm）、gather 静默退化、MM 不融合、维护脆弱性。这些需要方案 2/4/5/6 + 编译器下沉来补。

> 一句话：**npu_inductor 的 kernel 已经足够好，现在的任务是别让 host 开销和几类结构性算子短板把它拖下水。** 方案 1（图捕获）是性价比最高的第一步，做完就能把这份报告里"算子级 1.30×"真正变成"E2E 1.5×"。

---

*分析依据：npu_inductor_2.13.0 源码（codegen/triton.py、npu_triton_heuristics.py、npu_patch.py、lowering.py、config.py、npu_config.py、codegen/wrapper.py）+ npu_inductor_report.md + npu_inductor设计与对标分析.md + NPU_Inductor_Linearize_技术报告.md + document/{inductor_fusion_graph_analysis, npu_inductor_deep_analysis, llama3_training_performance_analysis, eager_vs_compile_comparison, cache_effect_comparison}.md + problems.csv + docs/*.md（20 篇）。所有 file:line 与性能数字均来自上述源码/文档实测。*
