# AscendForge

昇腾 NPU 测试框架，继承自 PyTorch 上游社区的 inductor 测试体系，面向 `torch_npu` + `triton_experimental` 后端进行日常看护与质量保障。

## 1. 背景与动机

### 1.1 上游 inductor 测试体系

PyTorch 社区维护了一套庞大的 inductor 单元测试套件，用于看护 `torch.compile` 在各类后端（CPU / CUDA / XPU / MTIA …）上的编译正确性、数值精度和性能。主要套件包括：

| 套件 | 上游路径 | 规模 | 用途 |
|------|---------|------|------|
| **inductor_tests** | `test/inductor/test_torchinductor.py` | ~800 条 | torch.compile 全功能编译正确性 |
| **dynamic_shapes** | `test/inductor/test_torchinductor_dynamic_shapes.py` | ~300 条 | 动态 shape 场景下的编译正确性 |
| **opinfo** | `test/inductor/test_torchinductor_opinfo.py` | ~2000 条 | 按 PyTorch 算子枚举，细粒度覆盖 |

上游 CI 跑这些套件的方式是：`pytest` 收集 → 多进程并行 → 每种 dtype / device 各跑一遍。对每个后端来说，一套完整的 inductor 看护能覆盖 `torch.compile` → inductor lowering → 后端代码生成的全链路。

### 1.2 为什么要做 AscendForge

`torch_npu` 是昇腾对 PyTorch 的适配层，其中 `triton_experimental` 是面向 NPU 的 inductor 后端（基于 Triton 昇腾分支）。它的质量直接决定了 `torch.compile` 在昇腾 NPU 上的可用性。

直接在上游社区仓库里跑昇腾 NPU 的 CI 有几个实际问题：

- **版本耦合**：`torch_npu` / CANN / `triton_ascend` 版本需严格匹配，上游 CI 镜像不含 NPU 环境
- **预期失败管理**：部分用例因 CANN 算子未支持 / 精度差异 / 编译期行为差异而预期失败，需要一份**昇腾专用黑名单**持续维护
- **多卡并行约束**：NPU 多卡场景下环境变量（`ASCEND_RT_VISIBLE_DEVICES`）必须正确注入，不能简单复用 CPU/CUDA 的多 worker 调度
- **结果可视化**：需要 Web 看板展示历次结果趋势，做跨日期的回归检测

AscendForge 就是为解决这些问题而生的：**把上游社区的 inductor 测试体系搬到昇腾 NPU 上跑，加上适配层、黑名单管理、并行调度和可视化链路**。

### 1.3 与 PyTorch 上游的关系

```
┌─────────────────────────────────────────────────────────────────┐
│                     PyTorch 上游社区                               │
│  test/inductor/                                                  │
│  ├── test_torchinductor.py               ~800 条用例              │
│  ├── test_torchinductor_dynamic_shapes.py  ~300 条用例            │
│  └── test_torchinductor_opinfo.py          ~2000 条用例            │
│                                                                  │
│  运行方式: pytest + run_test.py (多 GPU 并行)                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    适配 & 继承
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       AscendForge                                 │
│  unit_test/                                                      │
│  ├── test_torchinductor.py               ← 继承上游，适配 NPU      │
│  ├── test_torchinductor_dynamic_shapes.py ← 继承上游              │
│  ├── test_torchinductor_opinfo.py        ← 继承上游 (默认关)      │
│  ├── run_inductor_comparison.py          ← 并行驱动器 (自研)       │
│  ├── run_three_cases.sh                  ← 一键总入口             │
│  ├── rerun_failed.py                     ← 只重跑失败用例          │
│  └── blacklist.txt                       ← 昇腾预期失败列表        │
│                                                                  │
│  op_test/                                                        │
│  └── test_all.py                         ← 101 个单算子测试 (自研) │
│                                                                  │
│  适配点:                                                          │
│  • device 改为 npu                                                │
│  • backend 改为 triton_experimental                               │
│  • blacklist 管理预期失败                                         │
│  • ASCEND_RT_VISIBLE_DEVICES 多卡调度                              │
│  • TORCHINDUCTOR_NPU_BACKEND 环境变量注入                          │
└─────────────────────────────────────────────────────────────────┘
```

> **说明**：被测对象 `torch_npu` / `triton_experimental` 已合入 [Ascend/pytorch](https://gitcode.com/Ascend/pytorch) 主仓。本仓 **仅含测试框架本身**，不包含被测代码。

## 2. 目录结构

```
AscendForge/
├── unit_test/                             # UT 测试 —— 继承上游 inductor 用例
│   ├── run_three_cases.sh                 # 总入口，跑 inductor_tests + dynamic_shapes
│   ├── run_inductor_comparison.py         # 驱动器：pytest 收集 + 多 worker 并行 + 报告
│   ├── rerun_failed.py                    # 失败用例重跑（只重跑指定清单，不重复跑全量）
│   ├── test_torchinductor.py              # 用例集 ① inductor_tests（继承上游同名文件）
│   ├── test_torchinductor_dynamic_shapes.py  # 用例集 ② dynamic_shapes
│   ├── test_torchinductor_opinfo.py       # 用例集 ③ opinfo（默认不跑，可选开启）
│   └── blacklist.txt                      # 预期失败/跳过的用例黑名单
│
├── op_test/                               # 单算子测试 —— 自研
│   └── test_all.py                        # 101 个算子 case，支持 perf / profile
│
├── config/
│   ├── env.sh                             # 本地环境变量（CANN / conda / 卡号）
│   └── env.sh.example                     # 环境变量模板
│
├── docs/
│   ├── wiki.md                            # 本文档
│   └── output_format.md                   # test_results 产出格式（下游工具的接口契约）
│
├── requirements.txt                       # PyPI 轻量依赖（pytest / numpy 等）
└── test_results/                          # 测试产出，按日期分目录（见 §5）
```

## 3. 测试套件详解

### 3.1 UT 套件 —— inductor_tests

- **来源**：继承自 PyTorch 上游 `test/inductor/test_torchinductor.py`
- **规模**：~800 条用例
- **内容**：覆盖 `torch.compile` 全功能，包括算子编译、图切分、dtype 转换、inplace 操作、动态 shape 等场景
- **运行方式**：`run_inductor_comparison.py` 通过 pytest 的 `--collect-only` 收集全部用例，然后分配到多 worker 并行执行；每个用例有超时（默认 600s），超时判 timeout
- **黑名单**：`blacklist.txt` 中列出的用例会跳过（已知不支持的算子 / 已知精度差异 / 已知编译期 bug），由开发者持续维护

### 3.2 UT 套件 —— dynamic_shapes

- **来源**：继承自 PyTorch 上游 `test/inductor/test_torchinductor_dynamic_shapes.py`
- **规模**：~300 条用例
- **内容**：专门测试动态 shape 下的编译正确性（`torch._dynamo.mark_dynamic`），shape 在迭代中变化，考验 inductor 的 recompile 和图缓存逻辑
- **重要性**：生产模型中序列长度 / batch size 经常是动态的，这个套件是真实业务场景的前置看护

### 3.3 UT 套件 —— opinfo（可选）

- **来源**：继承自 PyTorch 上游 `test/inductor/test_torchinductor_opinfo.py`
- **规模**：~2000 条用例
- **内容**：按 PyTorch 算子粒度枚举，每种算子 × 多种 dtype × 多种 shape 排列组合
- **默认关闭**：数据量大跑得慢，通过 `RUN_OPINFO=1` 按需开启

### 3.4 OP 套件 —— 单算子测试

- **来源**：自研，不继承上游
- **规模**：101 个 case
- **内容**：每个 case 包含一个独立算子或算子组合（add / layernorm / softmax / rope / matmul / attention / ...），编译后和 eager 模式做精度对比，可选性能 benchmark 和 torch_npu profiler trace
- **设计哲学**：和 UT 的细粒度 pytest 不同，OP 测试是**场景化**的——每个 case 是一个真实的算子组合 pattern（比如 `add_mul_slice_permute_cat`），更接近生产模型中 inductor 实际产生的融合图

## 4. 环境准备

### 4.1 前置条件

| 组件 | 说明 | 安装方式 |
|------|------|---------|
| **NPU 硬件** | Ascend NPU 卡，`npu-smi info` 可见 | 硬件环境 |
| **CANN** | 匹配 NPU 驱动的 CANN 工具包 | 系统管理员安装 |
| **torch** | 与 torch_npu ABI 匹配的 PyTorch（如 2.13.0） | pip / 源码编译 |
| **torch_npu** | 基于对应 CANN 版本编译 | 源码编译 |
| **triton_ascend** | Triton 昇腾分支，提供 `triton_experimental` 后端 | 源码编译 |
| **pytest** | UT 用例收集和执行 | `pip install -r requirements.txt` |

### 4.2 快速配置

```bash
cd AscendForge

# 1. 安装轻量依赖
pip install -r requirements.txt

# 2. 配置本地环境
cp config/env.sh.example config/env.sh
# 编辑 config/env.sh：填入 ASCEND_RT_VISIBLE_DEVICES、CANN 路径等

# 3. 加载环境
source config/env.sh
```

### 4.3 验证环境

```bash
python -c "
import torch, torch_npu
print(f'torch={torch.__version__}')
print(f'torch_npu={torch_npu.__version__}')
print(f'devices={torch_npu.npu.device_count()}')
"
```

## 5. 使用方法

### 5.1 一键跑全部 UT

```bash
source config/env.sh
bash unit_test/run_three_cases.sh
```

环境变量覆盖：

| 变量 | 默认 | 说明 |
|------|------|------|
| `ASCEND_RT_VISIBLE_DEVICES` | （必填） | 用哪些 NPU 卡，如 `0,1,2,3` |
| `WORKERS` | 4 | 并行 worker 数 |
| `TIMEOUT` | 600 | 单用例超时（秒） |
| `BACKEND` | triton | inductor backend |
| `RUN_OPINFO` | 0 | `=1` 时额外跑 opinfo 套件 |

```bash
# 不用 config/env.sh 也能直接一行跑
ASCEND_RT_VISIBLE_DEVICES=2,3,4,5 WORKERS=8 bash unit_test/run_three_cases.sh

# 带上 opinfo
RUN_OPINFO=1 bash unit_test/run_three_cases.sh
```

### 5.2 重跑失败用例

跑完一轮 UT 后，把失败用例清单喂给重跑脚本：

```bash
python unit_test/rerun_failed.py rerun_list.txt
python unit_test/rerun_failed.py rerun_list.txt --backend triton --workers 4
```

文件格式：每行一个 pytest 节点 ID（如 `test_torchinductor.py::NPUTests::test_addmv_npu`）。脚本会自动按文件名分流到对应套件。

### 5.3 运行 OP 测试

```bash
# 全量正确性
python op_test/test_all.py

# 正确性 + 性能 benchmark（compiled vs eager 延迟对比）
python op_test/test_all.py --perf

# 失败不中断
python op_test/test_all.py --perf --continue-on-fail

# 只跑指定算子
python op_test/test_all.py test_add test_layernorm test_softmax

# 性能 + kernel 级 profiling（产出 torch_npu profiler trace）
python op_test/test_all.py --perf --profile --profile-dir ./profiling_data

# 查看所有可用的 case
python op_test/test_all.py --list

# 产出可读的 comparison_report.txt
python op_test/test_all.py --perf --continue-on-fail --report-dir ./my_results
```

关键参数：

| 参数 | 说明 |
|------|------|
| `--perf` | 正确性之后追加延迟 benchmark |
| `--perf-warmup 5` | benchmark warmup 次数（默认 5） |
| `--perf-iters 20` | benchmark 测量次数（默认 20） |
| `--profile` | 开启 torch_npu profiler 采集 kernel 统计 |
| `--profile-dir <dir>` | profiler trace 输出目录 |
| `--continue-on-fail` | 单个 case 失败后继续跑剩余 case |
| `--report-dir <dir>` | 跑完后写一份 `comparison_report.txt` |

## 6. 产出与下游链路

### 6.1 文件产出

```
test_results/
└── 20260807_1200/                         # 每次跑一个时间戳目录
    ├── inductor_tests/
    │   ├── comparison_report.txt          # ← 主产出：每行 "用例名 status elapsed"
    │   ├── inductor_tests.log             # 整个 suite 完整 stdout/stderr
    │   └── failures/                      # 失败用例的 pytest 原始日志
    ├── dynamic_shapes/                    # 结构同上
    ├── op_test/                           # （由 run_op_test.py 或 --report-dir 生成）
    │   └── comparison_report.txt          # OP 测试结果，含 compiled/eager 延迟
    └── opinfo/                            # 仅 RUN_OPINFO=1 时存在
```

### 6.2 comparison_report.txt 格式

```
test_add_npu                                  passed       31.09
test_adaptive_max_pool2d1_npu                 failed       21.16
test__dyn_quant_matmul_4bit_npu               skipped      15.52
```

- 解析正则：`(\S+)\s+(passed|failed|skipped|timeout)\s+([\d.]+)`
- 用例名是全局唯一 key，跨日期对比靠它匹配

### 6.3 下游可视化链路

```
test_results/<ts>/                   BenchBoard
  ├── inductor_tests/
  │   └── comparison_report.txt  →  convert_ut_results.py  →  ut_data/<ts>.json  →  ut.html
  ├── dynamic_shapes/
  │   └── comparison_report.txt  →  (同上)
  └── op_test/
      └── comparison_report.txt  →  run_op_test.py         →  op_data/<ts>.json  →  optest.html

ut_data/<ts1>.json
    vs                        →  diff.html（跨日期对比，检测回归）
ut_data/<ts2>.json
```

> 产出格式是**接口契约**：下游 `convert_ut_results.py` / `ut_error_classify.py` 都按此解析。修改格式需同步下游。详见 `docs/output_format.md`。

## 7. CI 流水线

### 7.1 架构

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ pytorch 主仓  │────▶│  daily_pipeline  │────▶│   BenchBoard      │
│ (master)     │     │  ① 拉代码        │     │  ut.html          │
│              │     │  ② worktree 编包  │     │  diff.html        │
│              │     │  ③ 安装 whl      │     │  optest.html      │
│              │     │  ④ UT 测试       │     │  compare_ut_results│
│              │     │  ⑤ OP 测试       │     │                   │
│              │     │  ⑥ 转换 JSON     │     │                   │
│              │     │  ⑦ 清理          │     │                   │
└──────────────┘     └──────────────────┘     └──────────────────┘
```

### 7.2 关键设计点

- **git worktree 隔离编包**：从主仓拉出独立 worktree 编译 whl，编完即删，不在主仓产生任何编译垃圾
- **CWD import 阴影防护**：所有 Python 调用前确保 cwd 不在 pytorch / worktree 目录，防止 `import torch_npu` 找到未编译的源码
- **7 阶段流水线**：支持 `--skip-build` / `--skip-ut` / `--skip-op` 按需跳过

### 7.3 运行

```bash
# 完整流水线
ASCEND_RT_VISIBLE_DEVICES=2,3,4,5 bash daily_pipeline.sh

# 跳过编包（已装好最新 whl）
bash daily_pipeline.sh --skip-build

# 只跑 UT
bash daily_pipeline.sh --skip-build --skip-op

# 强制运行（即使没有新提交）
bash daily_pipeline.sh --force
```

## 8. 开发指南

### 8.1 上游同步

当 PyTorch 上游 `test/inductor/test_torchinductor.py` 有更新时：

1. 从上游同步用例文件到 `unit_test/`
2. 跑一轮全量 UT，收集新的失败用例
3. 分析失败原因：
   - **CANN 不支持**：加入 `blacklist.txt`
   - **triton_experimental bug**：提 issue 并加入 `blacklist.txt`，修好后移出
   - **测试本身需要适配**：改用例（如 device 改为 npu）
4. 更新 `blacklist.txt`，提交

### 8.2 黑名单维护

`unit_test/blacklist.txt` 每行一个用例名（不用 pytest 节点 ID，只用最后一段方法名）。维护原则：

- **有明确 CANN / 编译器限制**：长久黑名单
- **有 bug 但计划修**：黑名单 + 关联 issue
- **容忍度边界**：偶尔抖动的用例先观察几轮再决定

### 8.3 新增 OP case

在 `op_test/test_all.py` 中新增一个函数，遵循模式：

```python
def test_my_new_op():
    class Model(nn.Module):
        def forward(self, x):
            return torch.some_op(x)

    model = Model().npu()
    model_compiled = torch.compile(model)

    x = torch.randn([4, 1024], device="npu")
    with torch.no_grad():
        data1 = model_compiled(x)
        data2 = model(x)
    _assert_close(data1, data2)

    _maybe_benchmark("my_new_op",
        lambda: model_compiled(x),
        lambda: model(x),
    )
    _maybe_profile("my_new_op",
        lambda: model_compiled(x),
        lambda: model(x),
    )
```

然后在文件末尾的 `CASES` OrderedDict 中注册：`("my_new_op", test_my_new_op)`。

## 9. 常见问题

**Q: `ModuleNotFoundError: No module named 'torch_npu._C'`？**

A: python cwd 在 pytorch 源码目录下，`import torch_npu` 找到了未编译的源码。切到 `/tmp` 或家目录再运行。

**Q: `npu-smi info` 能看到卡但测试报 device 错？**

A: 检查 `ASCEND_RT_VISIBLE_DEVICES` 是否正确设到了空闲卡，以及 CANN 的 `set_env.sh` 是否 source 过。

**Q: UT 跑了一个多小时还没结束？**

A: 可能是某些用例卡死。降低 `TIMEOUT`（默认 600s）或检查卡是否被其他进程占用（`npu-smi info` 看显存）。

**Q: OP 测试的 perf 数据为 `-`？**

A: 确保传了 `--perf` 参数，否则 `_maybe_benchmark` 会跳过，不产生 compiled/eager 延迟数据。

**Q: blacklist 里为什么有这么多用例？**

A: 昇腾 inductor 后端在持续演进中，黑名单会随着 triton_experimental 的能力提升逐步减小。每个版本的 release note 会公布黑名单变化。
