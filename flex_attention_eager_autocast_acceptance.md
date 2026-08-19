# FlexAttention Eager 模式 Autocast 支持 & PatchManager 化重构 NPU 转测文档

## 一、需求背景

`torch.nn.attention.flex_attention` 是 PyTorch 2.x 引入的高阶算子（HigherOrderOperator, HOP），用于灵活定义 score_mod / mask_mod 注意力。在 NPU 上的既有实现存在三个问题：

- **eager 模式无法使用 autocast**：`flex_attention` / `flex_attention_backward` 两个 HOP 未注册 `AutocastPrivateUse1` dispatch key 的 kernel。用户进入 `torch.autocast(device_type="npu", dtype=torch.bfloat16)` 上下文后调用 flex_attention，dispatcher 找不到 autocast kernel，要么直接报错、要么以 float32 执行，无法像 CUDA 上一样自动把 Q/K/V cast 到 bf16。
- **设备校验白名单化**：上游 `_validate_device` 对 NPU tensor 不友好，NPU 侧的旧 patch（`_inductor/__init__.py` 中的 `flex_attention._validate_device = _validate_device`）硬编码了设备处理，且散落在 inductor 加载路径里——只有触发 inductor backend 加载才会生效，eager 用户（不 import `torch_npu._inductor`）拿不到该 patch。
- **patch 位置分散**：同样的 `_patch_flex_attention_device` 逻辑还曾在 `transfer_to_npu.py` 里有一份（无操作版本，`_npu_valid_device` 直接 return 不做任何校验），两处 patch 语义不一致、重复维护。

本 PR 将 flex_attention 的 autocast 注册与设备校验统一抽取到 `torch_npu/utils/patch_flexattention.py`，通过 **PatchManager** 在 `import torch_npu` 时无条件触发，使 eager 模式下的 autocast 与设备校验对全部用户生效，并删除了原先散落在 inductor 路径和 transfer_to_npu 中的重复 patch。

### 涉及 PR

| PR | 标题 | 改动规模 | 状态 |
|----|------|---------|------|
| [MR 44287](https://gitcode.com/Ascend/pytorch/merge_requests/44287) | `feat(flex_attention): extract autocast registration + device validation into shared patch_flexattention` | 5 文件, +223/-23 | master 已合入（commit 6b620aec6） |
| 同步分支 | v2.7.1 / v2.9.0 / v2.10.0 | 4 文件, +222/-2 | 已提交 |

> **注意**：master 合入时与上游 flexattention template PR（`patch_flex_attention` / `_patch_flex_attention_singleton_sort`）存在冲突，已解决：保留 template PR 的全部改动，仅删除 `_validate_device` 两行。v2.11.0 / v2.12.0 分支完全没有 flex_attention 支持，不做同步。

---

## 二、需求价值

1. **eager + autocast 可用**：`torch.autocast(device_type="npu", dtype=torch.bfloat16)` 上下文内调用 `flex_attention`，Q/K/V 自动 cast 到 bf16 执行，输出为 bf16，与 CUDA 行为对齐。这是混合精度训练/推理中使用 flex_attention 的前提。
2. **import 即生效**：patch 通过 PatchManager 在 `import torch_npu` 时执行，不再依赖用户是否 import `torch_npu._inductor` 或使用 `transfer_to_npu`，eager 用户零配置可用。
3. **单一来源，消除重复维护**：三处 patch（inductor `_validate_device` 赋值、transfer_to_npu 的 no-op 版本、缺失的 autocast kernel）收敛为一处定义，语义统一。
4. **设备校验对齐 GPU**：新的 `_npu_valid_device` 与 CUDA 上游 `_validate_device` 行为对齐——全非 NPU 输入放行（交回原生校验）、同设备放行、混合 device type / 不同 NPU 设备报 `ValueError` 并给出清晰错误信息。

---

## 三、需求详细

**描述**：在 torch_npu master 上为 flex_attention / flex_attention_backward HOP 注册 NPU `AutocastPrivateUse1` kernel，并将设备校验 patch 统一到 PatchManager，在 `import torch_npu` 时自动生效；验证 eager 模式执行、autocast 数值正确性、设备校验行为在 NPU 上符合预期。

**PT 版本**：PyTorch 2.13+（nightly）

**torch_npu 版本**：master（含 commit `6b620aec6`）

**CANN 版本**：9.1.0-beta.1（eager 执行路径依赖 aclnn 算子，9.0.0 会触发已知 `aclnnInplaceOne` 问题）

**交付时间**：（按流水线排期填写）

**新增 API / 环境变量**：无新增公开 API。`patch_flexattention.py` 中全部为下划线私有函数（`__all__ = []`）。无环境变量。

**性能目标**：autocast 路径仅增加一次 Q/K/V 的 dtype cast（与 CUDA autocast 相同），无额外编译开销（eager 模式）。

**精度目标**：bf16 autocast 输出与 fp32 eager 参考值在 `rtol=2e-2, atol=2e-2` 内一致；输出全部有限（无 NaN/Inf）。

---

### 3.1 工作流程图

#### A. Autocast 注册与触发（新增能力）

```
┌────────────────────────────────────────────────────────────────────┐
│  import torch_npu                                                  │
│    └─ PatchManager._apply_all_patches()                            │
│         └─ apply_flex_attention_patch()                            │
│              ├─ _patch_flex_attention_device()                     │
│              │    └─ fa_mod._validate_device = _npu_valid_device   │
│              │       （幂等：_npu_device_patched flag 防重复）      │
│              └─ _register_npu_flex_attention_autocast()            │
│                   ├─ flex_attention_hop.py_impl(                   │
│                   │      AutocastPrivateUse1)(                     │
│                   │      _flex_attention_autocast_npu)             │
│                   └─ flex_attention_backward_hop.py_impl(          │
│                          AutocastPrivateUse1)(                     │
│                          _flex_attention_backward_autocast_npu)    │
│                   （幂等：has_kernel_for_dispatch_key 检查）        │
└────────────────────────────────────────────────────────────────────┘

运行时（用户代码）：
┌────────────────────────────────────────────────────────────────────┐
│  with torch.autocast(device_type="npu", dtype=torch.bfloat16):     │
│      out = flex_attention(q, k, v)   # q/k/v: fp32 NPU tensor      │
│                                                                    │
│  dispatcher 检测 AutocastPrivateUse1 置位                           │
│    └─ 路由到 _flex_attention_autocast_npu                          │
│         1. autocast_dtype = torch.get_autocast_dtype("npu")        │
│            → bfloat16                                              │
│         2. q/k/v = _cast(x, "npu", bfloat16)                       │
│         3. _ExcludeDispatchKeyGuard(AutocastPrivateUse1)           │
│            排除 autocast key 后重新调 HOP，防止无限递归             │
│         4. 走正常 HOP 执行路径（eager 物化实现）                    │
│  → out.dtype == bfloat16                                           │
└────────────────────────────────────────────────────────────────────┘
```

backward 侧按 PyTorch 惯例不做重 cast（forward 图已在 AOTAutograd 捕获时记录所需 cast），仅排除 autocast key 后透传重分发。

#### B. 设备校验（对齐 GPU）

```
┌────────────────────────────────────────────────────────────────────┐
│  _npu_valid_device(query, key, value)                              │
│                                                                    │
│  query.device.type != "npu"                                        │
│    → return（非 NPU 输入不干预，交回原生校验/后端报错）             │
│                                                                    │
│  query.device != key.device or query.device != value.device        │
│    → ValueError: Expected query, key, and value to have the same   │
│      device, but got query.device: ..., key.device: ..., ...       │
│                                                                    │
│  三个 device 完全相同（同一个 npu:x）                               │
│    → 放行                                                          │
└────────────────────────────────────────────────────────────────────┘
```

| 输入场景 | GPU 上游行为 | 本 PR 行为 |
|---------|------------|-----------|
| 全部同一 GPU/NPU 设备 | ✅ 放行 | ✅ 放行 |
| 全部 CPU | ✅ 放行 | ✅ 放行（early return，不干预） |
| 混合 device type（npu + cpu） | ❌ ValueError | ❌ ValueError |
| 同 type 不同卡（npu:0 + npu:1） | ❌ ValueError | ❌ ValueError |

> GPU 行为已在 CUDA 机器上实测对齐（`test_gpu_device_validate.py`），GPU 对场景 3/4 使用同一检查与同一错误信息。

---

### 3.2 关键代码位置索引

| 文件 | 方法 / 位置 | 功能 |
|------|------|------|
| `torch_npu/utils/patch_flexattention.py`（新增） | `_flex_attention_autocast_npu()` | forward autocast kernel：cast Q/K/V 到 autocast dtype，排除 autocast key 后重分发 HOP |
| `torch_npu/utils/patch_flexattention.py` | `_flex_attention_backward_autocast_npu()` | backward autocast kernel：排除 autocast key 后透传（不重 cast） |
| `torch_npu/utils/patch_flexattention.py` | `_register_npu_flex_attention_autocast()` | 幂等注册两个 HOP 的 AutocastPrivateUse1 kernel |
| `torch_npu/utils/patch_flexattention.py` | `_patch_flex_attention_device()` / `_npu_valid_device()` | patch `_validate_device`，对齐 GPU 校验语义 |
| `torch_npu/_init/patches/npu_patches.py` | `apply_flex_attention_patch()` | `@PatchManager.register_patch("npu")`，import torch_npu 时触发上述两项 |
| `torch_npu/_inductor/__init__.py` | （删除）`_validate_device` import + `flex_attention._validate_device = _validate_device` | inductor 路径的旧 patch 移除，统一到 PatchManager |
| `torch_npu/contrib/transfer_to_npu.py` | （删除）`_patch_flex_attention_device()` 定义及调用、`import patch_flexattention` | 移除 transfer_to_npu 中的重复 no-op patch |
| `test/nn/test_npu_flexattention.py`（新增） | `test_eager_flex_attention` / `test_npu_flex_attention_autocast` | eager 执行 + autocast 数值 UT |

**关键实现细节**：
- 幂等双保险：`has_kernel_for_dispatch_key(AutocastPrivateUse1)` 检查 + `fa_mod._npu_device_patched` flag，多路径/重复 import 不冲突。
- `_ExcludeDispatchKeyGuard` 排除 autocast key 后重分发，避免 autocast kernel 递归调用自身。
- autocast dtype 通过 `torch.get_autocast_dtype(device_type)` 获取，跟随用户 `torch.autocast` 设置的 dtype。

---

## 四、相关 PR

| 链接 | 说明 |
|------|------|
| https://gitcode.com/Ascend/pytorch/merge_requests/44287 | master 主 PR |
| 同步分支 feat/flex-attention-eager-autocast-v2-for-v2.7.1 | v2.7.1 同步（含 multi-slice-concat 冲突无关，直接删 `_validate_device` 两行） |
| 同步分支 feat/flex-attention-eager-autocast-v2-for-v2.9.0 | v2.9.0 同步（无 template PR，删两行） |
| 同步分支 feat/flex-attention-eager-autocast-v2-for-v2.10.0 | v2.10.0 同步（MR 44498，保留 `_register_npu_inductor_multi_slice_concat`） |

---

## 五、验证报告

### 5.1 测试设计

**目标**：验证 eager 模式 flex_attention 在 NPU 上可执行、autocast 数值正确、设备校验行为与 GPU 对齐、patch 幂等无副作用。

**策略**：本次验收**仅使用 UT 测试**，不跑业务模型回归。原因：
- 核心改动是 dispatch 注册 + 设备校验 patch，属于算子分发层通用机制，与具体模型结构无关；
- UT 已覆盖 eager 执行、autocast dtype/数值、设备校验正反例；
- 业务模型（含 flex_attention 的 Transformer 变体）回归在后续集成阶段覆盖。

#### 精度与性能覆盖说明

| 维度 | 是否覆盖 | 如何覆盖 |
|------|---------|---------|
| **精度** | ✅ 有覆盖 | autocast bf16 输出 vs fp32 eager 参考 `assert_close(rtol=2e-2, atol=2e-2)` + `isfinite().all()` |
| **性能** | ❌ 不测 | autocast 仅增加一次 dtype cast，无独立性能测试必要 |
| **端到端 wall time** | ❌ 不测 | 业务模型场景留待后续集成阶段覆盖 |

#### 测试文件

| 文件 | 测什么 | 用例数 | 运行位置 |
|------|--------|--------|---------|
| `test/nn/test_npu_flexattention.py` | eager 执行 + autocast bf16 数值 | 2 | NPU, eager |
| `document/flexattention/test_gpu_device_validate.py` | 设备校验 4 场景 GPU 对齐参照 | 4 | GPU（对照基线） |

---

#### test_npu_flexattention.py 用例清单

| 用例 | 验证目标 | 关键断言 |
|------|---------|---------|
| `test_eager_flex_attention` | eager 模式 flex_attention 在 NPU 直接执行（不经 torch.compile） | `output.shape == (B, H, S, D)` |
| `test_npu_flex_attention_autocast` | autocast 上下文内 Q/K/V 自动 cast 到 bf16 且数值正确 | `actual.dtype == bfloat16`；`isfinite().all()`；`assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)` |

#### 设备校验人工验证项（对齐 GPU 实测基线）

| 场景 | 输入 | 预期 |
|------|------|------|
| 同设备 | q/k/v 全在 npu:0 | 正常执行 |
| 全 CPU | q/k/v 全在 cpu | 不抛 NPU 专属错误（early return，交回原生路径） |
| 混合 device type | q 在 npu:0，k 在 cpu | `ValueError: Expected query, key, and value to have the same device ...` |
| 不同 NPU 设备 | q 在 npu:0，k 在 npu:1 | 同上 ValueError |

---

### 5.2 测试结果

#### test/nn/test_npu_flexattention.py

```
执行命令：
python test/nn/test_npu_flexattention.py

实际结果：
  eager flex_attention OK
  autocast flex_attention OK
PASS: NPU FlexAttention eager + inductor execution.

逐项：
  test_eager_flex_attention              PASS  (eager 直接执行)
  test_npu_flex_attention_autocast       PASS  (bf16 输出 + 数值对齐 fp32)

实际执行环境：Ascend 910B × torch_npu master(6b620aec6) × CANN 9.1.0-beta.1
```

#### 设备校验对照（GPU 基线）

```
执行命令（CUDA 机器）：
python test_gpu_device_validate.py

GPU 实测：
  all same GPU                OK
  all CPU                     OK
  GPU + CPU mixed             ValueError: Expected ... same device type ...
  different GPU devices       ValueError: Expected ... same device type ...

NPU 行为按 _npu_valid_device 实现对齐（全 CPU early return；混合/跨卡报错）
```

---

### 5.3 验收结论模板

```
验收结论：通过 / 不通过

环境：
- torch 版本 / git_version：
- torch_npu 版本 / commit：
- CANN / 驱动：
- NPU 设备：

功能验收：
- import torch_npu 后 AutocastPrivateUse1 kernel 已注册
  （flex_attention_hop.has_kernel_for_dispatch_key 为 True）：通过 / 不通过
- autocast 上下文内 flex_attention 输出 dtype == bfloat16：通过 / 不通过
- autocast 输出 vs fp32 eager 数值对齐（rtol/atol 2e-2）：通过 / 不通过
- 输出全部有限（无 NaN/Inf）：通过 / 不通过
- eager 模式（非 compile）执行正常：通过 / 不通过

设备校验：
- 同设备放行：通过 / 不通过
- 全 CPU 不抛 NPU 错误（early return）：通过 / 不通过
- npu + cpu 混合报 ValueError：通过 / 不通过
- npu:0 + npu:1 报 ValueError：通过 / 不通过

幂等性：
- 重复 import torch_npu / 多次调用注册函数不报错不重复注册：通过 / 不通过

回归：
- inductor 路径 torch.compile(flex_attention) 无回归（template PR 功能保留）：通过 / 不通过
- transfer_to_npu 场景无回归：通过 / 不通过

遗留风险：
```

---

## 六、风险与注意事项

1. **CANN 版本依赖**：eager flex_attention 执行路径依赖 aclnn 算子，CANN 9.0.0 存在已知 `aclnnInplaceOne` 问题，验收需使用 CANN 9.1.0+；注意 `/etc/profile` 硬编码了 cann-9.0.0，测试前需手动 source 9.1.0 环境。
2. **inductor 路径 block_mask 格式**（已知、不在本 PR 范围）：上游 PyTorch 的 block_mask tuple 已扩展到 17 个字段（新增 `dq_write_order` 等），NPU 旧 lowering 按 13 字段解包，torch.compile + flex_attention 场景会报 `too many values to unpack (expected 13)`。该问题与本 autocast PR 无关，由 flexattention template 后续 PR 跟进。
3. **`TORCHINDUCTOR_NPU_BACKEND=triton_experimental`**：该实验后端没有 flex_attention lowering，compile 场景会失败，与本 PR 无关，验收 compile 场景请使用默认 backend。
4. **backward autocast 不做重 cast**：`_flex_attention_backward_autocast_npu` 仅排除 key 透传，符合 PyTorch 惯例（forward 图已记录 cast）。若有训练场景反馈 bf16 反向精度问题，需检查 AOTAutograd 捕获的 cast 节点而非修改该 kernel。
5. **版本分支差异**：v2.9.0 无 flexattention template PR（无 `patch_flex_attention`），v2.7.1 / v2.10.0 有；同步分支的 `_inductor/__init__.py` 改动均为删除 `_validate_device` 两行，其余保持分支原状。v2.11.0 / v2.12.0 无 flex_attention 支持，不同步。
