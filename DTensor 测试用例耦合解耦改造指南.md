# DTensor 测试用例耦合解耦改造指南

本文面向 PyTorch Distributed/DTensor 模块相关用例的解耦改造。目标是把测试里不必要的 CUDA 写死逻辑，改成跟随当前测试设备或当前后端能力的写法。

核心原则：先看测试真正想验证什么，再替换硬编码设备；通用逻辑跟随当前设备，后端专属逻辑保留专属入口。

参考判断顺序：

1. 这个测试是否只验证 Python 逻辑、规则是否写对”，不真的依赖 GPU、通信或分布式运行？如果是，可以保持普通 `TestCase`、CPU、fake 或 meta 写法,不需要强行改成 accelerator 测试。
2. 这个测试是否创建真实 DTensor、DeviceMesh等，如果是，优先让它使用当前测试设备,而不是写死"cuda"。
3. 这个测试是不是专门验证某个后端独有能力，比如 XLA、NCCL、CUDA profiler、CUDA Graph、Triton 或某个特定 backend？如果是，保留专属 class 或专属 skip。

常见的 2 个现有基础设施：

1. `self.device_type`

   作用：表示当前测试应该使用的设备类型字符串，例如 `"cuda"`、`"xpu"`、PrivateUse1 后端名或 `"cpu"`。

   适用场景：创建 tensor、module、DeviceMesh 时替换硬编码 `"cuda"`。例如 `torch.randn(shape, device=self.device_type)`。

   源码片段：[torch/testing/_internal/distributed/_tensor/common_dtensor.py](/Users/aurora/.codex/worktrees/85dc/pytorch/torch/testing/_internal/distributed/_tensor/common_dtensor.py:642)

   ```python
    @property
    def device_type(self) -> str:
        if (
            not (TEST_CUDA or TEST_XPU or TEST_HPU or TEST_PRIVATEUSE1)
            or DEVICE_COUNT < self.world_size
        ):
            return "cpu"
        else:
            return DEVICE_TYPE
   ```

   不同情况的结果：当测试环境没有可用 accelerator，或设备数量少于 `self.world_size`，返回 `"cpu"`；否则返回当前 accelerator 类型。具体 accelerator 类型由同文件中 `DEVICE_TYPE = torch.accelerator.current_accelerator().type` 决定，见 [common_dtensor.py](/Users/aurora/.codex/worktrees/85dc/pytorch/torch/testing/_internal/distributed/_tensor/common_dtensor.py:76)。

2. `torch.accelerator`

   作用：PyTorch 提供的统一 accelerator 入口，用来获取当前加速设备、设备数量、设置设备、同步设备等。

   适用场景：把 `torch.cuda.*` 这类 CUDA 专属 runtime API 改成通用写法。例如设备数量用 `torch.accelerator.device_count()`，当前设备类型用 `torch.accelerator.current_accelerator().type`。

   DTensor 公共逻辑里是这样取设备的：

   ```python
   if TEST_CUDA or TEST_XPU or TEST_HPU or TEST_PRIVATEUSE1:
       DEVICE_TYPE = torch.accelerator.current_accelerator().type
       DEVICE_COUNT = torch.accelerator.device_count()
   ```

   NPU 后端注册成 PyTorch PrivateUse1 后端后，也可以被这个统一入口识别，例如：

   ```python
   torch.accelerator.current_accelerator().type
   ```
   返回 `"npu"`。

## 一、耦合类型一：设备创建和 CUDA runtime API 耦合

核心问题：

测试逻辑本身是通用 DTensor 或普通 accelerator 行为，但 tensor、module、`DeviceMesh`、`init_device_mesh`、`torch.cuda.*`等写死 CUDA，导致非 CUDA accelerator 无法复用。

改造建议：

- DTensor 测试中优先使用 `self.device_type` 与 `self.build_device_mesh()`。
- 设备数量、设置设备、同步等 runtime 操作优先使用 `torch.accelerator`。
- 后端模块能力使用 `torch.get_device_module(self.device_type)` 获取；不是所有后端都支持的 API 要按能力 skip。
- CPU-only 测试不要带 `ProfilerActivity.CUDA`。

### 社区已合入 PR 参考与修改示例：

PR #184241 把卷积测试从 CUDA-only 改成 accelerator 通用。

文件：`test/nn/test_convolution.py`

> PR 整体修改点：把  `@onlyCUDA` 测试改成 `@onlyAccelerator`；把 普通测试迁到 device-type class；把 `use_cuda` 布尔参数改成 `device` 参数；把 `.cuda()`、`device="cuda"` 改成使用传入的 `device`；把 `torch.cuda.synchronize()` 改成 `torch.accelerator.synchronize()`；把 `self.device_type == "cuda"` 这类判断改成更通用的 accelerator 判断。

修改前后代码：`test/nn/test_convolution.py @@ -1428,14 +1267,11`

来源：[PR #184241](https://github.com/pytorch/pytorch/pull/184241)

```diff
        groups=1,
-        use_cuda=False,
+        device="cpu",
        use_bias=True,
        dtype=torch.double,
    ):
-        if use_cuda:
-            device = torch.device("cuda")
-        else:
-            device = torch.device("cpu")
+        device = torch.device(device)
```

修改点：

- 不再让 helper 只回答“要不要用 CUDA”，而是由外面直接传入要用的设备。
- 这样同一个 helper 可以服务 CUDA、XPU、HPU、PrivateUse1/NPU 等不同 accelerator。
- 这类 PR 是类型一的典型参考：如果测试逻辑本身是通用的，就把创建 tensor、module、同步设备的地方从 CUDA API 换成通用设备入口。

#### 示例1：DTensor dispatch overhead 写死 CUDA。

文件：[test/distributed/tensor/test_dtensor_dispatch_overhead.py](/Users/aurora/.codex/worktrees/85dc/pytorch/test/distributed/tensor/test_dtensor_dispatch_overhead.py:69)

修改前后代码 ：`test/distributed/tensor/test_dtensor_dispatch_overhead.py @@ test_dtensor_add_op_dispatch_overhead`

```diff
    @skip_if_lt_x_gpu(4)
    @with_comms
    def test_dtensor_add_op_dispatch_overhead(self):
-        if torch.cuda.is_available():
-            device_props = torch.cuda.get_device_name(0)
-            gpu_name = device_props
+        device_module = torch.get_device_module(self.device_type)
+        if hasattr(device_module, "get_device_name"):
+            gpu_name = device_module.get_device_name(0)
            logger.info("running on %s", gpu_name)
            # TODO: adjust `expected_propagate_time` and `expected_dispatch_time` to target different hardware
        else:
-            self.skipTest("CUDA not available")
+            logger.info("running on %s", self.device_type)
        expected_propagate_time = 880  # noqa: F841
        expected_dispatch_time = 90  # noqa: F841
        diff_percent_threshold = 0.20  # noqa: F841
        propagator = DTensor._op_dispatcher.sharding_propagator
-        device_mesh = init_device_mesh("cuda", (self.world_size,))
-        input_data = torch.rand(512, 512, device="cuda")
+        device_mesh = self.build_device_mesh()
+        input_data = torch.rand(512, 512, device=self.device_type)
```

修改点：

- 不再用 `torch.cuda.is_available()` 决定测试能不能跑，改为使用当前测试设备对应的模块。
- 不再写死 `"cuda"` 创建 mesh 和 tensor，改为跟随 `self.device_type`。
- 如果当前后端没有 `get_device_name`，就只记录设备类型，避免因为缺少 CUDA 专属接口而跳过整个测试。

hasattr(obj, "name") 用来判断一个对象里有没有某个属性或方法。  检查 device_module 这个设备模块里有没有 get_device_name 这个方法，如果有返回true。

源码里也有类似判断思路：
```diff
# torch/sparse/_triton_ops_meta.py
if torch.cuda.is_available():
    return torch.cuda.get_device_name()
if torch.xpu.is_available():
    return torch.xpu.get_device_name()
return ""
```
意思是：PyTorch 自己也只对 CUDA/XPU 取设备名，其他设备直接返回空字符串。

## 二、耦合类型二：测试准入、skip 装饰器和 device-type 参数化耦合 CUDA

核心问题：

测试主体可以被多个 accelerator 复用，但测试准入层使用 `@requires_cuda`、`@onlyCUDA`、`skip_if_lt_x_gpu()` 的 CUDA-only 实现、`only_for=("cuda",)` 或整类 skip，导致非 CUDA 后端无法实例化或运行测试。

### 社区已合入 PR 参考与修改示例：

#### 示例：PR #180820 给 device-type tests 增加 class/method 级跳过入口，并在实例化时生效。

文件：`torch/testing/_internal/common_device_type.py`
> PR 整体修改点：给 `DeviceTypeTestBase` 增加 `test_exclusions` 配置；实例化 device-type tests 时先读取该配置；支持 `"*"` 跳过整个 class，也支持只跳过某几个 test method；在 OpenReg 测试里新增用例证明整个类跳过和单个方法跳过都能生效。
> 
> 下面两段代码属于同一个 PR 的一组修改：第一段先提供“跳过名单”，第二段让生成测试时真正执行这份名单。

修改前后代码：`torch/testing/_internal/common_device_type.py @@ DeviceTypeTestBase.test_exclusions`

来源：[PR #180820](https://github.com/pytorch/pytorch/pull/180820)

```diff
+    # An optional skip mechanism built upon instantiate_device_type_tests(),
+    # designed to facilitate skipping either an entire class or specific test cases
+    # within a class.
+    #
+    # Format:
+    #   test_exclusions = {
+    #       "TestClassA": ["test_a", "test_b"],   # Selective: Skips specific
+    #       "TestClassB": "*",                    # Global: Skips the entire class
+    #   }
+    test_exclusions: ClassVar[dict[str, Collection[str]]]
```

修改点：

- 新增 `test_exclusions`，让后端可以按测试类或测试方法精确跳过。
- 这样不用把 `@requires_cuda` 这类粗粒度条件写进通用测试本体。

同一个 PR 里的第二处关键修改：实例化 device-type tests 时读取跳过配置。

修改前后代码：`torch/testing/_internal/common_device_type.py @@ instantiate_device_type_tests`

来源：[PR #180820](https://github.com/pytorch/pytorch/pull/180820)

```diff
+        skipped = base._get_test_exclusions(generic_test_class.__name__)
+        # Skip the entire class
+        if "*" in skipped:
+            continue
```

```diff
+                # Skip the specified methods.
+                if name in skipped:
+                    continue
```

修改点：

- 生成测试用例前先看 `test_exclusions`。
- 如果配置里写了 `"*"`，就跳过整个测试类；如果只写了方法名，就只跳过对应方法。

#### 示例1：DTensor logging 整个 class 被 `@requires_cuda` 包住。

文件：[test/distributed/tensor/test_dtensor_logging.py](/Users/aurora/.codex/worktrees/85dc/pytorch/test/distributed/tensor/test_dtensor_logging.py:11)

修改前后代码 ：`test/distributed/tensor/test_dtensor_logging.py @@ imports and setUp`

```diff
-from torch.testing._internal.common_utils import requires_cuda, run_tests, TestCase
+from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed.fake_pg import FakeStore


-@requires_cuda
class TestDTensorLogging(TestCase):
    """Test DTensor logging."""

    def setUp(self):
        super().setUp()
        _clear_sharding_prop_cache()
        self.world_size = 2
        store = FakeStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=self.world_size, store=store
        )
-        self.device_type = "cuda"
+        accelerator = torch.accelerator.current_accelerator(check_available=True)
+        self.device_type = accelerator.type if accelerator is not None else "cpu"
```

修改点：

- 去掉 `@requires_cuda`，让这个日志测试不再只允许 CUDA 环境运行。
- 不再把 `self.device_type` 固定成 `"cuda"`，改成根据当前可用设备自动决定。
- 这个改动还需要配合修改日志里的 expected string，否则日志内容仍会写死 `'cuda'`。

补充说明：

```python
# torch/testing/_internal/common_utils.py
requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
```

> `@requires_cuda` 的意思很直接：只有 `torch.cuda.is_available()` 为 true 才运行；没有 CUDA 就跳过。因此它适合 CUDA 专属测试，不适合只是检查 DTensor 日志、graph 字符串这类通用逻辑的测试。

## 三、耦合类型三：通信 backend 和多进程初始化绑定 NCCL/Gloo

核心问题：

DTensor 通信测试需要 process group。`backend="nccl"` 会把测试锁到 CUDA/NCCL；
通用 DTensor 测试应跟随当前 device type 的默认 distributed backend。

改造建议：

- 通用 accelerator 通信测试默认使用 `dist.get_default_backend_for_device(self.device_type)`。
  >源码位置：
[torch/distributed/distributed_c10d.py (line 1525)](/Users/aurora/.codex/worktrees/85dc/pytorch/torch/distributed/distributed_c10d.py:1525)

- `DeviceTypeTestBase.distributed_backend()` 是 device-type tests 的默认用来获取通信 backend 的统一入口；新硬件后端可以通过这个入口接入自己的 backend,而不用到处改代码。
  >源码位置：
[torch/testing/_internal/common_device_type.py (line 420)](/Users/aurora/.codex/worktrees/85dc/pytorch/torch/testing/_internal/common_device_type.py:420)
- NCCL-only 测试保留 specific class，不强行泛化。

### 社区已合入 PR 参考与修改示例：

类型三提出的问题是：通信测试不能把 backend 写死成 `"nccl"` 或 `"gloo"`，否则新 accelerator 很难复用。PR #184181 给 device-type tests 增加了统一的 `distributed_backend()` 入口。

示例：PR #184181 给 device-type tests 增加默认 distributed backend hook。并验证自定义后端能接入。

文件：`torch/testing/_internal/common_device_type.py`

PR 整体修改点：在 DeviceTypeTestBase 上新增 distributed_backend()；默认实现调用 dist.get_default_backend_for_device(cls.device_type)；在 OpenReg 测试中新增 TestDistributedBackendHook，验证自定义后端可以返回自己的默认通信 backend。下面两段代码属于同一个 PR 的一组修改：第一段增加统一入口，第二段用 OpenReg 证明新后端可以通过这个入口声明自己的 backend。

修改前后代码：`torch/testing/_internal/common_device_type.py @@ DeviceTypeTestBase.distributed_backend`

来源：[PR #184181](https://github.com/pytorch/pytorch/pull/184181)

```diff
+    @classmethod
+    def distributed_backend(cls) -> str:
+        """
+        Default distributed backend for this device type.
+        """
+        import torch.distributed as dist
+
+        return dist.get_default_backend_for_device(cls.device_type)
```

修改点：

- 新增 `distributed_backend()`，让测试框架能按当前设备自动拿到合适的通信 backend。
- 这样测试不用在代码里手写 `"nccl"`、`"gloo"` 或某个后端私有名称。

同一个 PR 里的验证代码：OpenReg 后端验证自定义 distributed backend。

修改前后代码：`test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_testing.py @@ TestDistributedBackendHook`

来源：[PR #184181](https://github.com/pytorch/pytorch/pull/184181)

```diff
+@unittest.skipIf(not dist.is_available(), "Distributed not available, skipping tests")
+class TestDistributedBackendHook(TestCase):
+    def test_distributed_backend_for_openreg(self, device):
+        self.assertEqual(type(self).distributed_backend(), "occl")
+
+
+instantiate_device_type_tests(
+    TestDistributedBackendHook, globals(), only_for=("openreg",)
+)
```
修改点：

- 新增一个小测试，确认 OpenReg 后端能通过统一 hook 返回自己的 backend。
- 这说明新后端不需要改测试主体，也能接入分布式测试。

## 四、耦合类型四：compile/export/debug 的预期字符串或 backend 写死特定后端

核心问题：

DTensor compile/export/debug 测试常用 `assertExpectedInline()` 校验 graph/log 字符串。如果 golden 字符串写死 `'cuda'`，或 compile backend 使用 CUDA/Triton/Inductor 专属能力，非 CUDA 后端会失败。这里要区分通用 graph 语义与 backend-specific 能力。

### 社区已合入 PR 参考与修改示例：

示例1：PR #180328 把 Dynamo efficient attention 测试从 CUDA-only 改成按能力判断。

文件：`test/dynamo/test_unspec.py`

> PR 整体修改点：把 `test_no_recompilations_with_efficient_attention` 从普通 `UnspecTests` 移到 `UnspecTestsDevice`；删除 `@requires_cuda`；
> 
> 删除函数内部 `device = "cuda"`；新增 `PLATFORM_SUPPORTS_MEM_EFF_ATTENTION` skip；CPU 在运行时明确跳过；
> 
> 删除 `instantiate_device_type_tests(..., only_for=["cuda", "hpu", "xpu"])` 的固定设备列表；顺手把 `torch._dynamo.optimize()` 更新为 `torch.compile()`。

修改前后代码：`test/dynamo/test_unspec.py @@ efficient attention test`

来源：[PR #180328](https://github.com/pytorch/pytorch/pull/180328)

```diff
-    @requires_cuda
-    def test_no_recompilations_with_efficient_attention(self):
+class UnspecTestsDevice(torch._dynamo.test_case.TestCase):
+    @torch._dynamo.config.patch(assume_static_by_default=False)
+    @unittest.skipIf(
+        not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
+        "Platform does not support efficient attention",
+    )
+    def test_no_recompilations_with_efficient_attention(self, device):
+        if self.device_type == "cpu":
+            raise unittest.SkipTest("EFFICIENT_ATTENTION requires a non-CPU device")
```

```diff
-            device = "cuda"
             make_tensor = partial(
                 torch.rand, device=device, dtype=dtype, requires_grad=True
             )
```

```diff
-devices = ["cuda", "hpu", "xpu"]
-instantiate_device_type_tests(
-    UnspecTestsDevice, globals(), only_for=devices, allow_xpu=True
-)
+instantiate_device_type_tests(UnspecTestsDevice, globals(), allow_xpu=True)
```

修改点：

- 旧测试是“只要不是 CUDA 就不进门”，新测试改成“框架传入当前 device，平台支持 efficient attention 才跑”。
- 这保留了硬件能力约束：efficient attention 仍然不是 CPU 通用测试，不支持的平台会明确 skip。
- 它是好的硬件绑定例子：绑定的是“功能能力”，不是“设备名字”。只要别的后端也支持这个能力，就有机会复用同一份测试。

#### 示例1：logging golden 写死 CUDA。

文件：[test/distributed/tensor/test_dtensor_logging.py](/Users/aurora/.codex/worktrees/85dc/pytorch/test/distributed/tensor/test_dtensor_logging.py:61)

修改前后代码：`test/distributed/tensor/test_dtensor_logging.py @@ test_sharding_prop_cache_logging`

```diff
        self.assertExpectedInline(
            log_string(),
-            """\
-sharding_prop MISS (C++ fast path): aten.add.Tensor(Spec(f32[4, 4](S(0))), Spec(f32[4, 4](S(0)))) on DeviceMesh((2,), 'cuda', stride=(1,))) -> Spec(f32[4, 4](S(0)))
-sharding_prop HIT (C++ fast path): aten::add.Tensor(Spec(f32[4, 4](S(0))), Spec(f32[4, 4](S(0))), 4822678189205111) -> Spec(f32[4, 4](S(0)))
-sharding_prop MISS (C++ fast path): aten.add.Tensor(Spec(f32[4, 4](R)), Spec(f32[4, 4](R))) on DeviceMesh((2,), 'cuda', stride=(1,))) -> Spec(f32[4, 4](R))
-sharding_prop MISS (C++ fast path): aten.add.Tensor(Spec(f32[8, 4](S(0))), Spec(f32[8, 4](S(0)))) on DeviceMesh((2,), 'cuda', stride=(1,))) -> Spec(f32[8, 4](S(0)))""",
+            f"""\
+sharding_prop MISS (C++ fast path): aten.add.Tensor(Spec(f32[4, 4](S(0))), Spec(f32[4, 4](S(0)))) on DeviceMesh((2,), '{self.device_type}', stride=(1,))) -> Spec(f32[4, 4](S(0)))
sharding_prop HIT (C++ fast path): aten::add.Tensor(Spec(f32[4, 4](S(0))), Spec(f32[4, 4](S(0))), 4822678189205111) -> Spec(f32[4, 4](S(0)))
+sharding_prop MISS (C++ fast path): aten.add.Tensor(Spec(f32[4, 4](R)), Spec(f32[4, 4](R))) on DeviceMesh((2,), '{self.device_type}', stride=(1,))) -> Spec(f32[4, 4](R))
+sharding_prop MISS (C++ fast path): aten.add.Tensor(Spec(f32[8, 4](S(0))), Spec(f32[8, 4](S(0)))) on DeviceMesh((2,), '{self.device_type}', stride=(1,))) -> Spec(f32[8, 4](S(0)))""",
        )
```

修改点：

- 日志期望值里不再固定写 `'cuda'`，而是使用当前测试设备名。
- 同一个文件里后续 Python cache 日志也要一起改，否则只改一半仍会在非 CUDA 后端失败。

#### 示例2：DTensor compile 中使用 `backend="inductor"`：NPU 支持时无需改 backend。

文件：[test/distributed/tensor/test_dtensor_compile.py](/Users/aurora/.codex/worktrees/85dc/pytorch/test/distributed/tensor/test_dtensor_compile.py:1641)

> 是否适合通用硬件测试：要分情况。当前整项工作的目标是 GPU 和 NPU 能跑即可；如果 NPU 已支持 Inductor，就不需要把 `backend="inductor"` 改成 `backend="aot_eager"`。只有当测试目标只是通用 graph 语义、且npu不支持 Inductor 时，才考虑换成通用写法。

当前代码：`test/distributed/tensor/test_dtensor_compile.py @@ test_dtensor_different_gradient_placement`

```python
        x = torch.randn(4, 2, 4, requires_grad=True, device=self.device_type)
        x_dt = DTensor.from_local(x, mesh, [Shard(1)], run_check=False)

        y = torch.randn(4, requires_grad=True, device=self.device_type)
        y_dt = DTensor.from_local(y, mesh, [Replicate()], run_check=False)

        z = torch.randn(4, requires_grad=True, device=self.device_type)
        z_dt = DTensor.from_local(z, mesh, [Replicate()], run_check=False)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        tmp_dt = opt_fn(x_dt, y_dt, z_dt)
```

- 对 GPU+NPU 目标来说，NPU 已支持 Inductor 的场景不要改，否则会把原本要覆盖的 Inductor 路径测没了。
  
## 五、耦合类型五：OpInfo、operator 支持、skip/xfail 和 allowlist 耦合

核心问题：

DTensor tests 常通过 OpInfo 批量生成。不同 out-of-tree backend 的 operator 支持不同；

改造原则：

- operator 支持差异应通过 `op_overrides`、`op_allowlist`、`test_exclusions` 等统一配置表达。
- 暂时只能 CPU 跑的 OpInfo 文件，不要一次性打开 accelerator；先建立失败列表/allowlist，再逐步放开。

### 社区已合入 PR 参考与修改示例：

#### 示例：PR #181554 给后端测试差异增加统一注册 API，并把临时配置改为走该 API。

文件：`test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_testing.py`

> PR 整体修改点：  
> 
> 给 `DeviceTypeTestBase` 增加 `set_test_configs(...)`；
> 
> 允许后端一次性设置 `op_overrides`、`op_allowlist`、`test_exclusions`；  
> 
> OpenReg 测试里的临时配置上下文从手动 `setattr` 改成调用配置 API；  
> 
> 保留上下文管理器恢复原配置，避免测试之间相互污染。
> 
> 下面两段代码属于同一个 PR 的一组修改：第一段把临时配置改成调用统一 API，第二段给出这个统一 API 的定义。


修改前后代码：`test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_testing.py @@ _temp_test_configs`

来源：[PR #181554](https://github.com/pytorch/pytorch/pull/181554)

```diff
-def _temp_attrs(obj, **attrs):
-    backup = {k: getattr(obj, k) for k in attrs}
-    for k, v in attrs.items():
-        setattr(obj, k, v)
+def _temp_test_configs(obj, **configs):
+    backup = {k: getattr(obj, k, None) for k in configs}
+    obj.set_test_configs(**configs)
     try:
         yield
     finally:
-        for k, v in backup.items():
-            setattr(obj, k, v)
+        obj.set_test_configs(**backup)
```

修改点：

- 临时测试配置不再手动 `setattr` 类属性，而是通过 `set_test_configs(...)` 设置和恢复。
- 改用统一 API 管理这类测试配置，含义更清楚，也更容易维护。

测试配置 API 统一为 `set_test_configs`。

文件：`torch/testing/_internal/common_device_type.py`

修改前后代码：`torch/testing/_internal/common_device_type.py @@ DeviceTypeTestBase.set_test_configs`

来源：[PR #181554](https://github.com/pytorch/pytorch/pull/181554)

```diff
+    @classmethod
+    def set_test_configs(
+        cls,
+        *,
+        op_overrides=None,
+        op_allowlist=None,
+        test_exclusions=None,
+    ):
+        cls.op_overrides = op_overrides
+        cls.op_allowlist = op_allowlist
+        cls.test_exclusions = test_exclusions
```

修改点：

- 把多个配置入口收敛成一个清晰的函数。
- 后端只需要调用 `set_test_configs(...)`，就能一次性设置算子覆盖、算子白名单和测试跳过规则。
- 这一步和上面的临时配置上下文是配套关系：上面负责调用统一入口，这里定义统一入口具体能设置哪些内容。

#### 示例1：DTensor OpInfo 测试当前固定 CPU。

文件：[test/distributed/tensor/test_dtensor_ops.py](/Users/aurora/.codex/worktrees/85dc/pytorch/test/distributed/tensor/test_dtensor_ops.py:412)

对应代码：`test/distributed/tensor/test_dtensor_ops.py @@ OP_DB_WORLD_SIZE`

```python
OP_DB_WORLD_SIZE = 4
# DEVICE_TYPE = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() >= OP_DB_WORLD_SIZE else "cpu"
# TODO: debug cuda illegal memory access issue and re-enable cuda tests
DEVICE_TYPE = "cpu"
```

- 这段代码说明以前尝试过用 CUDA 跑，但遇到过非法内存访问问题。
- 当前固定 CPU 是一种保守选择，不能直接改成所有硬件都跑。
- 正确做法是先列清楚哪些算子在目标后端支持，再用 allowlist、skip、xfail 分批打开。

### 真实参考写法：当前 PyTorch device-type infra 已支持 allowlist 和测试配置。

文件：[torch/testing/_internal/common_device_type.py](/Users/aurora/.codex/worktrees/85dc/pytorch/torch/testing/_internal/common_device_type.py:436)

```python
    @classmethod
    def _apply_op_allowlist(cls, ops):
        """Filters ops.op_list to only include ops declared in op_allowlist.

        If op_allowlist is None (default), no filtering is applied.
        If op_allowlist is set, only ops whose full_name is in the collection
        will generate test variants.

        Args:
            ops: The ops decorator instance whose op_list will be filtered.
        """
        if cls.op_allowlist is None:
            return

        supported_set = set(cls.op_allowlist)
        ops.op_list = [op for op in ops.op_list if op.full_name in supported_set]
```

文件：[torch/testing/_internal/common_device_type.py](/Users/aurora/.codex/worktrees/85dc/pytorch/torch/testing/_internal/common_device_type.py:503)

```python
    @classmethod
    def set_test_configs(
        cls,
        *,
        op_overrides=None,
        op_allowlist=None,
        test_exclusions=None,
    ):
```

- `_apply_op_allowlist` 只生成后端声明支持的算子测试，减少无意义失败。
- `set_test_configs` 把 allowlist、skip、xfail 等配置集中到一个入口，避免测试代码里到处写特殊判断。
- 两段代码配合起来看：set_test_configs 负责登记规则，_apply_op_allowlist 负责在真正生成 OpInfo 测试前执行白名单过滤。
