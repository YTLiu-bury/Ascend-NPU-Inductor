# PyTorch硬件解耦任务背景\&目标

一、背景

### **1\.1 核心矛盾**

PyTorch 在架构设计上长期以 **CUDA 为唯一加速器参考**，导致从框架层到测试层都存在对 CUDA 的隐性依赖。当新硬件加速器（如昇腾 NPU、Intel XPU 等）尝试通过 `PrivateUse1` 机制接入时，会遭遇系统性的兼容障碍。

这种障碍在两个层面同时存在：

两者的根因指向同一个事实：**PyTorch 缺乏标准化的第三方后端注册与扩展机制**。

### **1\.2 当前状态与代价**

上述矛盾在 **特性** 和 **用例** 两个维度上同时制造了障碍：

#### **特性维度：新硬件无法通过标准路径接入 PyTorch 核心特性**

PyTorch 的核心特性模块——图捕获与编译（Dynamo/Inductor）、分布式训练与通信（Distributed/FSDP）、算子验证与能力声明（OpInfo）、设备运行时基础设施（Device/Serialization/DLPack）——在设计上均以 CUDA 为默认加速器参考，缺乏面向第三方后端的标准化扩展与注册机制。

具体表现为：Dynamo 的变量追踪仅识别 CUDA 类型，Inductor 的后端设备判定写死了有限后端列表，分布式通信初始化路径仅考虑 NCCL，FSDP 的梯度同步逻辑内嵌 CUDA 流管理假设，算子能力声明以 `dtypesIfCUDA` 等形式硬编码在 OpInfo 中，设备管理/序列化/DLPack 等基础设施未使用 dispatch 机制。

这导致新硬件接入时，需要以 Monkey Patch 方式拦截和替换 PyTorch 内部函数来注入自定义行为。这些内部 API 随上游重构频繁变动，导致兼容性极度脆弱，版本适配成本随迭代线性累积。

#### **用例维度：新硬件无法复用社区测试来验证功能正确性**

与特性层面的 CUDA 耦合相对应，PyTorch 的测试用例同样将 CUDA 作为唯一的加速器验证目标。测试中大量使用的 `@onlyCUDA`、`TEST_CUDA`、`GPU_TYPE`、`HAS_GPU` 等守卫将新硬件排除在外；分布式测试中 `DEVICE_TYPE` 和 `DISTRIBUTED_BACKEND` 写死为 CUDA/NCCL；OpInfo 中每个后端的 dtype 声明需要侵入 PyTorch 源码逐一追加。

此外，测试分类缺乏显式的硬件需求标记——纯 CPU 用例和需要加速器的用例混合在一起，后端开发者无法区分哪些该跑、哪些可以跳过。现有的 skip 机制是方法级"全有或全无"，无法表达"该算子在某个 dtype 通过、另一个跳过"的精确控制。

这导致新硬件无法直接复用社区庞大的测试资产来验证功能正确性，需要自行编写大量等效用例，重复投入且覆盖度远低于社区测试。同时，社区在测试解耦重构过程中，缺乏自动化的测试数量变化感知机制，可能出现因错误标注导致测试静默丢失的风险。

#### **典型样例：Inductor 编译栈的设备判定**

以下以一个具体样例说明同一问题如何在特性和用例两个维度上同时体现。

PyTorch 的 Inductor 编译栈在判断哪些设备需要生成 device guard 时，将支持的设备列表硬编码：

```Python
# PyTorch 源码：torch/_inductor/scheduler.py
def device_need_guard(device_type: str) -> bool:
    return device_type in ["cuda", "xpu"]   # NPU 不在列表中
```

因为 NPU 不在硬编码列表中，Inductor 不会为 NPU 生成 device guard，导致编译优化路径被跳过。`torch_npu` 为了适配，不得不 Monkey Patch 该函数：

```Python
# torch_npu 的适配代码：torch_npu/_inductor/utils.py
import torch._inductor.scheduler as _scheduler

_original_device_need_guard = _scheduler.device_need_guard

def _patched_device_need_guard(device_type: str) -> bool:
    if device_type == "npu":      # 硬编码注入 NPU 分支
        return True
    return _original_device_need_guard(device_type)

_scheduler.device_need_guard = _patched_device_need_guard   # 运行时替换
```

这种对内部函数的替换依赖 PyTorch 私有 API 的函数签名和模块路径，上游一旦重构（如函数改名、模块重组），适配代码立即失效。同类 Patch 在 Inductor 的 Codegen、CodeCache、AOTI 等路径中同样存在。

与此同时，该特性的测试用例通过 GPU 守卫将 NPU 排除在外：

```Python
# PyTorch 测试代码：test/inductor/test_scheduler.py（示例）
from torch.testing._internal.inductor_utils import HAS_GPU

@unittest.skipIf(not HAS_GPU, "requires GPU")
class TestScheduler(TestCase):
    def test_device_guard(self):
            ...
```

上述代码中，同一个功能点（Inductor 设备判定）在**特性侧**需要 Patch 内部函数才能工作，在**用例侧**被 `HAS_GPU` 守卫排除无法验证——两个维度的问题共享同一个根因：设备列表未被设计为可扩展的注册机制。

#### **维护代价总结**

两个维度的根因是统一的——PyTorch 缺乏标准化的第三方后端扩展点。这带来的代价是系统性的：

### **1\.3 社区窗口期**

PyTorch 社区从 2026 年初启动了系统性的**第三方后端友好化**工作，为问题解决提供了方向对齐的窗口：

**关键判断：** 社区方向已明确，具体接口正在设计。此时将 NPU 的实际需求和使用场景贡献进去，可以有效影响设计决策，从"被动适配"转向"共建标准"。

---

## **二、问题分析**

与 1\.2 节对应，问题集中在 **特性** 和 **用例** 两个维度。

### **2\.1 特性维度：核心模块缺乏标准化的第三方后端扩展点**

PyTorch 的多个核心模块在设计上以 CUDA 为唯一加速器参考，第三方后端无法通过标准路径注册自己的实现。

- **图捕获与编译**：Dynamo 的变量追踪仅识别 CUDA 类型，新硬件 tensor 在图捕获阶段被当成未知变量导致图断裂；Inductor 的设备判定、Codegen、AOTI 等路径中设备列表写死，新硬件被排除在编译优化之外。

- **分布式训练与通信**：分布式初始化路径仅考虑 NCCL/GLOO 后端，新硬件通信库无法通过标准接口传递配置；FSDP 的梯度同步与规约逻辑内嵌 CUDA 特定的流管理假设，新硬件需要完全不同的同步语义。

- **算子验证与能力声明**：每个后端的 dtype 支持、精度覆盖、已知失败等信息以 `dtypesIfCUDA` 等形式硬编码在 OpInfo 条目中，新硬件接入需要侵入 PyTorch 源码逐一追加声明，扩展性极差。

- **设备运行时基础设施**：设备管理、DLPack 互操作、序列化等底层能力未使用 dispatch 机制，新硬件无法通过标准注册路径接入自己的实现。

### **2\.2 用例维度：社区测试无法被新硬件复用**

与特性层面的 CUDA 耦合对应，测试用例同样将 CUDA 作为唯一验证目标，新硬件被系统性地排除在测试体系之外。

- **测试与硬件强绑定**：测试用例中大量使用 `@onlyCUDA`、`TEST_CUDA`、`GPU_TYPE`、`HAS_GPU` 等守卫，分布式测试中 `DEVICE_TYPE` 和 `DISTRIBUTED_BACKEND` 写死为 CUDA/NCCL，新硬件完全无法复用。

- **测试分类缺失**：测试类缺乏显式的硬件需求标记——纯 CPU 用例和需要加速器的用例混在一起，后端开发者无法区分哪些该跑、哪些可以跳过。

- **跳过机制粗粒度**：现有 skip 是方法级"全有或全无"，无法表达"该算子在某个 dtype 通过、另一个 dtype 跳过"的精确控制。新硬件接入时只能在 CI 全红和全跳之间二选一。

- **重构缺乏防护**：社区在测试解耦重构过程中，缺乏自动化的测试数量变化感知机制，可能出现因错误标注导致测试静默丢失的风险。

---

---

## **三、任务目标**

将上述问题的解决方案通过 **RFC / PR** 的方式**合入 PyTorch 上游**，从根本上消除 `torch_npu` 中可通过社区贡献解决的适配代码，同时推动社区测试用例向 device\-agnostic 方向演进，使 NPU 能够无缝复用社区测试资产。







