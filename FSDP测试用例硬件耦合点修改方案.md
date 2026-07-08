# FSDP 测试用例硬件耦合解耦改造指南

本文档面向 PyTorch FSDP 测试用例的硬件耦合问题，按代码中出现的耦合类型进行归并，给出每个耦合类型的改造建议、代码示例及社区 PR 参考。

参考文档：https://docs.google.com/spreadsheets/d/1cDNiLW4KvPcGYPlA3KCDm0zV5PLPUWubno1OyCznKBw/edit?pli=1&gid=1201461581#gid=1201461581

pytorch git节点：fad74248e78716152917a729adb2b44ba2bab16e

## 类型一：`@skip_if_lt_x_gpu(N)` 仅识别 cuda/hpu/xpu，新硬件会被误跳过

### 核心问题

- 测试入口的 skip 装饰器或 `world_size` 计算硬编码了 CUDA/HPU/XPU 三类设备。
- 新加速器既无法被 `@skip_if_lt_x_gpu` 识别，也无法通过 `torch.cuda.device_count()` 获得正确的 world size。

### 改造原则

- 所有设备数量判断都通过 `torch.accelerator.current_accelerator()` 获取`device_type`后再通过`device_count()`获取。

### 修改建议

`@skip_if_lt_x_gpu` 在 `torch/testing/_internal/common_distributed.py` 中硬编码了 CUDA/HPU/XPU 三类设备。推荐在 `common_distributed.py` 中新增通用的 `@skip_if_lt_x_devices(x, *, allow_cpu=False)`，供所有分布式测试（包括 FSDP）使用。

### 推荐做法

1. 在 `torch/testing/_internal/common_distributed.py` 中新增 `@skip_if_lt_x_devices(x, *, allow_cpu=False)`。
2. 装饰器内部通过 `torch.accelerator.current_accelerator()` 获取当前加速器类型。
3. 对加速器调用 `device_count()` 判断可用设备数。
4. 显式处理 CPU 路径，保留 `allow_cpu=True` 的退化运行语义。
5. 异常处理逻辑保持与原代码一致：仍使用 `TEST_SKIPS[f"multi-gpu-{x}"]` 和 `_maybe_handle_skip_if_lt_x_gpu`。在 `TEST_SKIPS` 与 `_maybe_handle_skip_if_lt_x_gpu` 处添加注释，说明其命名中虽包含 `gpu`，但实际跳转为硬件无关的通用逻辑，当前阶段暂不修改其命名。

### 原有代码示例

`@skip_if_lt_x_gpu` 当前实现：

```python
# torch/testing/_internal/common_distributed.py
def skip_if_lt_x_gpu(x, *, allow_cpu=False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available() and torch.cuda.device_count() >= x:
                return func(*args, **kwargs)
            if TEST_HPU and torch.hpu.device_count() >= x:
                return func(*args, **kwargs)
            if TEST_XPU and torch.xpu.device_count() >= x:
                return func(*args, **kwargs)
            if allow_cpu and not (torch.cuda.is_available() or TEST_HPU or TEST_XPU):
                return func(*args, **kwargs)
            test_skip = TEST_SKIPS[f"multi-gpu-{x}"]
            if not _maybe_handle_skip_if_lt_x_gpu(args, test_skip.message):
                sys.exit(test_skip.exit_code)
        return wrapper
    return decorator
```

### 修改后代码示例

在 `common_distributed.py` 中新增通用装饰器：

```python
# torch/testing/_internal/common_distributed.py
import torch
from functools import wraps

def skip_if_lt_x_devices(x, *, allow_cpu=False):
    """Skip if fewer than x devices available for the current accelerator.

    Unlike @skip_if_lt_x_gpu, this does not hard-code cuda/hpu/xpu.
    It uses torch.accelerator.current_accelerator() to determine the device type.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            acc = torch.accelerator.current_accelerator()
            if acc is not None:
                device_type = acc.type
                device_module = torch.get_device_module(device_type)
                if device_module.is_available() and device_module.device_count() >= x:
                    return func(*args, **kwargs)

            # CPU path: allow running a degenerate version if explicitly requested
            if allow_cpu and acc is None:
                return func(*args, **kwargs)

            # NOTE: TEST_SKIPS uses "multi-gpu-{x}" naming for historical/
            # backward-compatibility reasons only. The skip logic itself is
            # hardware-agnostic and does not assume CUDA/GPU.
            test_skip = TEST_SKIPS[f"multi-gpu-{x}"]
            # NOTE: _maybe_handle_skip_if_lt_x_gpu retains "gpu" in its name
            # for backward compatibility; it is hardware-agnostic.
            if not _maybe_handle_skip_if_lt_x_gpu(args, test_skip.message):
                sys.exit(test_skip.exit_code)
        return wrapper

    return decorator
```

### 社区 PR 参考

**尚无直接对应 PR**，因为社区改造多集中在 `instantiate_device_type_tests` 框架，而 FSDP 分布式测试使用 `MultiProcessTestCase`。可参考以下思路作为过渡或补充：

- **PR #176717** (`test_unary_ufuncs.py`)：引入 `@onlyAccelerator` 装饰器，提供“跳过 CPU/meta”的通用加速器判断。
  - URL: https://github.com/pytorch/pytorch/pull/176717
  - 参考方法：新增通用装饰器替代硬编码设备判断。


## 类型二：设备特定 API/专用测试解耦

### 核心问题

- 测试代码中硬编码 `device="cuda"`、`torch.cuda.synchronize()`、`DeviceType.CUDA`、`torch.cuda.Stream()`、`torch.device("cuda", rank)` 等设备特定 API 调用。
- NCCL 环境变量设置、NCCL 日志断言、NCCL 特定功能测试与 CUDA/NCCL 强绑定。

### 改造原则

- 通用计算/同步逻辑使用 `torch.get_device_module(device_type)` 或 `torch.accelerator.*` 替代 `torch.cuda.*`。
- 真正依赖 CUDA 特有机制（CUDA Graph、CUDA multicast、CUDA 硬件能力检查等）的测试保留为 CUDA 专用测试，不作修改。

### 小类 1：原硬编码 CUDA 的逻辑可以支持通用 `device_type`

#### 判断标准

把原来写死的 `device="cuda"`、`DeviceType.CUDA`、`torch.cuda.synchronize()` 等换成基于当前 `device_type` 的调用，语义仍然成立，不会引入 CUDA 特有行为。

#### 推荐做法

1. 在测试文件顶部通过 `torch.accelerator.current_accelerator()` 获取当前 `device_type`（若测试文件已有`device_type`获取逻辑，则直接使用`device_type`）。
2. 将硬编码设备 API 替换为基于 `device_type` 的通用调用：
   - `device="cuda"` -> `device=device_type`
   - `torch.device("cuda", rank)` -> `torch.device(device_type, rank)`
   - `torch.cuda.synchronize()` -> `torch.get_device_module(device_type).synchronize()`
   - `torch.cuda.set_device(rank)` -> `torch.get_device_module(device_type).set_device(rank)`（若设备模块提供）
   - `DeviceType.CUDA` -> 由 `device_type` 字符串转换得到的 `DeviceType` 枚举值（如 `getattr(DeviceType, device_type.upper(), None)`）
3. 保留 `@skip_if_lt_x_devices` 等通用设备数量装饰器（已在类型一中覆盖）。

#### 原有代码示例

```python
# test/distributed/_composable/fsdp/test_fully_shard_state_dict.py
class TestFullyShardStateDictMultiProcess(FSDPTest):
    ...
    def _test_cached_state_dict(self, mlp_dim: int, mutate_after_state_dict: bool):
        ...
        torch.manual_seed(42 + self.rank)
        inp = torch.rand(mlp_dim, mlp_dim, device="cuda")
        for _ in range(5):
            optim.zero_grad()
            loss = model(inp).sum()
            loss.backward()
            optim.step()
            ...
```

#### 修改后代码示例

```python
# test/distributed/_composable/fsdp/test_fully_shard_state_dict.py

# 若测试文件顶部已有 device_type 赋值，则无需添加本行
device_type = getattr(torch.accelerator.current_accelerator(), "type", None)

class TestFullyShardStateDictMultiProcess(FSDPTest):
    ...
    def _test_cached_state_dict(self, mlp_dim: int, mutate_after_state_dict: bool):
        ...
        torch.manual_seed(42 + self.rank)
        inp = torch.rand(mlp_dim, mlp_dim, device=device_type)
        for _ in range(5):
            optim.zero_grad()
            loss = model(inp).sum()
            loss.backward()
            optim.step()
            ...
```

#### 说明

- 这是类型二中典型的**设备名字符串硬编码**场景：`torch.rand(..., device="cuda")` 创建输入 tensor 时直接写死 `"cuda"`。
- `torch.rand` 的 `device` 参数既接受字符串也接受 `torch.device`，因此可直接替换为动态推导的 `device_type`，无需引入 CUDA 特有语义。

### 小类 2：原逻辑确实只适用于特定类型设备

#### 判断标准

代码依赖 CUDA 特有 API（如 `torch.cuda.CUDAGraph`、`torch.cuda.Stream()`、CUDA Graph capture/replay、CUDA 特定硬件能力等），无法通过 `torch.get_device_module(device_type)` 泛化。

#### 推荐做法

1. **完全不做改动，与社区 PR 保持一致。** 社区对真正 CUDA 专属的测试采取 `left unchanged` 策略（如 PR #184593 中的 `torch.cuda._sleep()`、`nvtx`、`pin_memory`、`DataParallel` 等）。
2. 保留已有的 CUDA-only 装饰器（如 `@skip_if_lt_x_gpu`、`@unittest.skipIf(not TEST_CUDA_GRAPH, ...)`、`@onlyCUDA`），不再额外抽取类或删除矛盾分支。
3. 若未来 PyTorch 提供统一的 accelerator graph/stream 抽象，再考虑将此类测试泛化。

#### 示例 1：CUDA Graph

##### 原有代码

```python
# test/distributed/_composable/fsdp/test_fully_shard_training.py
class TestFullyShardCudaGraph(FSDPTest):
    @skip_if_lt_x_gpu(2, allow_cpu=True)
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_two_layer_fully_shard_cudagraph(self):
        if device_type.type == "cuda":
            torch.cuda.set_device(self.rank)
        device = torch.device(device_type.type, self.rank)
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(8, 8, bias=False),
            nn.Linear(8, 8, bias=False),
        ).to(device)
        ...

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            ...
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            ...
```

##### 处理方案

**完全不做改动，与社区 PR 保持一致。**

理由：
1. `TestFullyShardCudaGraph` 是 CUDA 专属测试类，属于`device-specific`。社区 PR 对这类 API 的测试也采取 `left unchanged` 策略。
2. 当前 PyTorch 没有跨设备的图抽象，强行参数化需要引入 `if device_type == "cuda": ... elif device_type == "xpu": ...` 等设备名分支，反而增加硬件耦合。
3. 原有代码虽然混有 `if device_type.type == "cuda"` 和 `torch.device(device_type.type, self.rank)` 等通用设备判断，但 `@unittest.skipIf(not TEST_CUDA_GRAPH, ...)` 已经保证该方法只在支持 CUDA Graph 的环境运行，本质上是 CUDA-only 测试。

**若未来 PyTorch 提供统一的 accelerator graph API（如 `torch.accelerator.graph()`），再考虑将此类测试泛化。目前阶段保持原样。**

#### 示例 2：NCCL LOG

##### 原有代码

```python
# test/distributed/_composable/fsdp/test_fully_shard_comm.py
class TestFullyShardAllocFromPG(FSDPTest):
    MEMORY_REGISTER_RE = (
        "NCCL INFO register comm 0x[0-9a-f]+ buffer 0x[0-9a-f]+ size [0-9]+"
    )

    @classmethod
    def _run(cls, *args, **kwargs):
        cls.nccl_log_dir = tempfile.TemporaryDirectory()
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,ENV,REG"
        os.environ["NCCL_DEBUG_FILE"] = cls.nccl_log_dir.name + "/nccl_log"
        super()._run(*args, **kwargs)

    @skip_if_lt_x_gpu(2)
    def test_fully_shard_alloc_from_pg(self):
        ...
        with open(self.nccl_log_dir.name + "/nccl_log") as f:
            self.assertNotRegex(f.read(), self.MEMORY_REGISTER_RE)
        ...
        with open(self.nccl_log_dir.name + "/nccl_log") as f:
            self.assertRegex(f.read(), self.MEMORY_REGISTER_RE)
```

##### 处理方案

**完全不做改动，与社区 PR 保持一致。**

理由：
1. **NCCL 环境变量是 NCCL 专属机制**
   - `NCCL_DEBUG`、`NCCL_DEBUG_SUBSYS`、`NCCL_DEBUG_FILE` 这些变量名本身就是 NCCL 的。
   - 不存在 `torch.distributed.set_collective_debug_env()` 或 `torch.accelerator.set_debug_env()` 这类通用 API。

2. **NCCL 日志格式是 NCCL 专属**
   - 断言里的 `"NCCL INFO ..."` 是 NCCL 打印的日志格式。
   - XCCL、Gloo、UCC 等后端没有等价格式，因此正则表达式无法泛化。

3. **无法通过动态推导把 NCCL 专属调用变成通用调用**
   - `get_default_backend_for_device(device_type)` 或 `torch.accelerator.current_accelerator()` 只能告诉你当前加速器/默认后端是什么。
   - 它们不能把一个 `NCCL_DEBUG` 设置或 `NCCL INFO` 断言自动转换成 XCCL/Gloo 的等价操作。

因此，任何改造都会停留在“判断当前后端是不是 NCCL，然后执行 NCCL 专属逻辑”这一层，无法真正做到后端无关。

### 如何判断归属小类 1 还是小类 2

对于类型二的耦合点，可借助 **torch_npu 仓库的 upstream patch 文件**进行辅助判断：

- torch_npu 仓库 v2.7.1 分支的 `pytorch/test_upstream` 目录下，为每个 PyTorch 测试文件保留了对应的 `.patch` 文件，例如 `test/distributed/_composable/fsdp/test_fully_shard_autograd.py` 对应 `test/distributed/_composable/fsdp/test_fully_shard_autograd.py.patch`。
- 这些 patch 类似于 `git diff`，记录了 NPU 适配时对原文件的修改。
- 由于 patch 基于 PyTorch 历史节点生成，**行号不一定与当前 PyTorch 代码一一对应**，需要通过 diff 中的上下文内容（如变量名、函数名、 surrounding code）定位到当前测试文件的具体位置。

**判定规则**：

- 若某个类型二耦合点在 patch 中能找到对应的修改（如 `device="cuda"` 被改为 `device="npu"`、`.cuda()` 被改为 `.npu()` 等），说明该耦合点可以通过动态推导 `device_type` 泛化到新加速器，归属**小类 1**。
- 若某个类型二耦合点在 patch 中**找不到对应修改**，说明厂商侧也未找到通用替换方案，通常只能保留为 CUDA/NCCL 专属逻辑，归属**小类 2**。

#### 可泛化示例：`test_fully_shard_state_dict.py` 中的 `device="cuda"`

##### 当前测试用例中的耦合点

```python
# test/distributed/_composable/fsdp/test_fully_shard_state_dict.py:151
inp = torch.rand(mlp_dim, mlp_dim, device="cuda")
```

##### torch_npu patch 中的对应修改

```diff
# torch_npu/pytorch/test_upstream/test/distributed/_composable/fsdp/test_fully_shard_state_dict.py.patch
@@ -148,12 +153,12 @@ class TestFullyShardStateDictMultiProcess(FSDPTest):
             model.load_state_dict(sd, assign=True, strict=False)

         # lazy init without error
-        inp = torch.rand((mlp_dim, mlp_dim), device="cuda")
+        inp = torch.rand((mlp_dim, mlp_dim), device="npu")
```

##### 判定结论

该耦合点在 patch 中找到了对应的设备名替换，说明只需把 `"cuda"` 改为动态推导的 `device_type` 即可在新加速器上运行，归属**小类 1**。

#### 不可泛化示例：`test_fully_shard_training.py` 中的 CUDA Graph

##### 当前测试用例中的耦合点

```python
# test/distributed/_composable/fsdp/test_fully_shard_training.py
class TestFullyShardCudaGraph(FSDPTest):
    @skip_if_lt_x_gpu(2, allow_cpu=True)
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_two_layer_fully_shard_cudagraph(self):
        if device_type.type == "cuda":
            torch.cuda.set_device(self.rank)
        device = torch.device(device_type.type, self.rank)
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(8, 8, bias=False),
            nn.Linear(8, 8, bias=False),
        ).to(device)
        ...

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            ...
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            ...
```

##### torch_npu patch 中的对应修改

查阅 `torch_npu/pytorch/test_upstream/test/distributed/_composable/fsdp/test_fully_shard_training.py.patch`，其中没有针对`TestFullyShardCudaGraph` 的修改，仅修改了同文件中 `init_device_mesh("cuda", ...)`、`torch.randn(..., device="cuda")`、`torch.device("cuda")` 等通用设备字符串。

##### 判定结论

该耦合点在 patch 中找不到对应修改，说明 NPU 侧也无法通过简单替换设备名来泛化，必须保留 CUDA 专属 API，归属**小类 2**。

#### 使用说明

- 上述 patch 文件基于 torch_npu v2.7.1 分支，对应 PyTorch 历史节点，因此行号与当前 PyTorch 代码不一定一致，需通过 diff 中的上下文内容定位当前文件位置。
- 该方法仅作为**辅助判定手段**：patch 中未修改的耦合点，通常意味着社区/厂商侧也未找到通用替换方案，应优先归为不可泛化的小类 2。

### 社区 PR 参考

- **PR #184593** (`test_autograd.py`)：将 `torch.cuda.synchronize()`、`torch.cuda.memory_allocated()`、`torch.cuda.set_default_device("cuda")`、`device="cuda"` 等大量替换为 `torch.accelerator.*` 或参数化 `device`；对 CUDA 特有 API（`nvtx`、`_sleep`、`pin_memory`、`DataParallel` 等）保留为 CUDA-only 测试，不强行泛化。
  - URL: https://github.com/pytorch/pytorch/pull/184593
  - 参考方法：通用设备 API 用 `torch.accelerator.*` / `torch.get_device_module(device_type)`；CUDA/NCCL 特有逻辑保留原样。

- **PR #178336** (`[Distributed] Make DDP tests and tensor parallel dependencies backend agnostic`)：将 `@requires_nccl()` 替换为 `@requires_accelerator_dist_backend()`，并用 `torch.accelerator.current_accelerator()` 动态推导 `DEVICE_TYPE` 和 `BACKEND`，减少对 `cuda`/`nccl` 硬编码。
  - URL: https://github.com/pytorch/pytorch/pull/178336
  - 参考方法：分布式测试应尽量通过动态推导获取后端，避免在测试代码中直接写死后端名；对真正依赖 NCCL 行为的断言仍保持 NCCL-only。

- **PR #160158** (`[1/N] Port 6 fsdp distributed test cases to Intel GPU`)：在将 FSDP 测试移植到 Intel GPU 时，把 `backend="cpu:gloo,cuda:nccl"` 改为 `backend="cpu:gloo,xpu:xccl"`， reviewers 建议使用 `get_default_backend_for_device` 进一步去设备名化。
  - URL: https://github.com/pytorch/pytorch/pull/160158
  - 参考方法：后端字符串不应硬编码设备名；若无法完全消除后端名，则按后端隔离的测试保持原样。

- **PR #163063** (`Restore environment after NcclUserBufferRegistrationTest`)：NCCL 专属测试设置 `NCCL_ALGO=NVLS` 后恢复环境，说明 NCCL 环境变量操作只应出现在 NCCL 专属测试范围内。
  - URL: https://github.com/pytorch/pytorch/pull/163063
  - 参考方法：NCCL 环境变量设置属于 NCCL 专属测试逻辑，不强行泛化。

## 类型三：模型/Tensor 显式设备名硬编码解耦

### 核心问题

- 现有测试用例中存在 MLP / Transformer / `nn.Linear` 等在 CPU 默认创建模型，再通过**显式使用设备名字符串**（如 `"cuda"`、`"xpu"`、`"cpu"`）将模型搬移至特定设备上的问题。

### 改造原则

1. 只修改调用点**显式硬编码设备名**的地方，用动态推导的 `device_type` 替换。
2. 对于 `to(device_type)`、默认 CPU 创建后再 `.to(device_type)` 等已参数化路径，**不作修改**。
3. 对于 genuinely CPU-only 的测试（模型始终停留在 CPU，`"cpu"` 是其预期行为），**不作修改**。

### 修改建议

将 `.cuda()` 替换为 `.to(device_type)`；将 `.cpu()` 替换为 `.to("cpu")`（若确实需要 CPU）或 `.to(device_type)`（若只是设备搬移）。

### 原有代码示例

```python
# test/distributed/_composable/fsdp/test_fully_shard_state_dict.py
class TestFullyShardStateDictMultiProcess(FSDPTest):
    @skip_if_lt_x_gpu(2)
    def test_cached_state_dict(self):
        ...
        if not mutate_after_state_dict:
            ...
        else:
            model = model.cpu()
            model = model.cuda()
            ...
```

### 修改后代码示例

```python
# test/distributed/_composable/fsdp/test_fully_shard_state_dict.py
import torch
from torch.testing._internal.common_distributed import skip_if_lt_x_devices

# 若测试文件顶部已有 device_type 赋值，则无需添加本行
device_type = getattr(torch.accelerator.current_accelerator(), "type", None)

class TestFullyShardStateDictMultiProcess(FSDPTest):
    @skip_if_lt_x_devices(2)
    def test_cached_state_dict(self):
        ...
        if not mutate_after_state_dict:
            ...
        else:
            model = model.to("cpu")
            model = model.to(device_type)
            ...
```

### 说明

- `.cpu()` 改为 `.to("cpu")` 只是为了统一写法；但此处测试语义是 CPU offload，属于测试逻辑需要，保留 `"cpu"` 作为目标设备是合理的。
- `.cpu()` 与 `.to("cpu")` 在 `nn.Module` 上是**功能等价**的：前者在 `torch/nn/modules/module.py:1155-1164` 中的实现为 `self._apply(lambda t: t.cpu())`；后者在 `torch/nn/modules/module.py:1340-1383` 中先通过 `torch._C._nn._parse_to("cpu")` 解析出 `device=torch.device("cpu")`，再对每个参数/ buffer 执行 `t.to(device)`。两者都会把所有参数和 buffer 递归移到 CPU 并返回 `self`。因此 `.cpu()` 改写成 `.to("cpu")` 不会引入行为差异，仅是写法上与 `.to(device_type)` 保持一致。

### 社区 PR 参考

- **PR #184593** (`test_autograd.py`)：将大量 `device="cuda"`、`.cuda()`、`model.cuda()` 替换为参数化 `device`。
  - URL: https://github.com/pytorch/pytorch/pull/184593
  - 参考方法：显式设备名统一替换为动态推导的 `device_type`；保留真正需要 CPU 路径的 `"cpu"` 字符串。

- **PR #184261** (`test_serialization.py`)：将 `torch.device("cuda")`、`device="cuda"` 替换为参数化 `device`。
  - URL: https://github.com/pytorch/pytorch/pull/184261
  - 参考方法：tensor/module 创建时的 device 参数统一参数化。

- **PR #184315** (`test_functional.py`)：将 `(x * y).cuda()` 改为 `(x * y).to(device)`。
  - URL: https://github.com/pytorch/pytorch/pull/184315
  - 参考方法：`.cuda()` 替换为 `.to(device)`。

- **PR #183728** (`test_optim.py`)：将 `"cuda" in optim_info.supports_fused_on`、`params_cuda = [p.to(device="cuda")]` 改为基于参数化 device 的判断。
  - URL: https://github.com/pytorch/pytorch/pull/183728
  - 参考方法：避免在字符串层面硬编码 `"cuda"`，统一使用 `_get_device_type(device)` 或 `device_type`。

- **PR #184192** (`test_lazy_modules.py`)：将 `if TEST_CUDA: device = "cuda"` 手动分支删除，改为由框架注入 `device` 参数。
  - URL: https://github.com/pytorch/pytorch/pull/184192
  - 参考方法：删除显式设备名分支，用参数化 device 替代。
