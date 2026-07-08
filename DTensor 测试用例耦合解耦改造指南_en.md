# DTensor Test Case Coupling Refactoring Guide

This document targets the refactoring of test cases in the PyTorch Distributed/DTensor module. The goal is to replace unnecessary CUDA hard-coded logic in tests with approaches that follow the current test device or the capabilities of the current backend.

Core Principle: First determine what the test actually intends to verify, then replace hard-coded devices; for generic logic, follow the current device; for backend-specific logic, retain dedicated entry points.

Reference judgment order:

1. If a test only verifies Python logic, whether rules are written correctly, and does not actually rely on GPU, communication, or distributed execution, it can remain as a normal `TestCase`, using CPU, fake, or meta approaches — no forced migration to accelerator testing is needed.
2. If a test creates real DTensors, DeviceMesh, etc., prioritize having it use the current test device rather than hard-coding "cuda."
3. If a test specifically verifies a backend-specific capability (e.g., XLA, NCCL, CUDA profiler, CUDA Graph, Triton, or a specific backend), retain the dedicated class or dedicated skip.

Two commonly used existing infrastructure components:

1. `self.device_type`

   Purpose: Represents the device type string that the current test should use, such as `"cuda"`, `"xpu"`, PrivateUse1 backend name, or `"cpu"`.

   Applicable scenarios: Replace hard-coded `"cuda"` when creating tensors, modules, and DeviceMesh. Example: `torch.randn(shape, device=self.device_type)`.

   Source code snippet: [torch/testing/_internal/distributed/_tensor/common_dtensor.py](/Users/aurora/.codex/worktrees/85dc/pytorch/torch/testing/_internal/distributed/_tensor/common_dtensor.py:642)

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

   Results in different cases: Returns `"cpu"` when the test environment has no available accelerator or when the device count is less than `self.world_size`; otherwise returns the current accelerator type. The specific accelerator type is determined by `DEVICE_TYPE = torch.accelerator.current_accelerator().type` in the same file, see [common_dtensor.py](/Users/aurora/.codex/worktrees/85dc/pytorch/torch/testing/_internal/distributed/_tensor/common_dtensor.py:76).

2. `torch.accelerator`

   Purpose: A unified accelerator entry point provided by PyTorch for obtaining the current accelerator device, device count, setting the device, synchronizing the device, etc.

   Applicable scenarios: Replace CUDA-exclusive runtime APIs such as `torch.cuda.*` with generic constructs. For example, use `torch.accelerator.device_count()` for device count and `torch.accelerator.current_accelerator().type` for the current device type.

   In the DTensor common logic, the device is obtained as follows:

   ```python
   if TEST_CUDA or TEST_XPU or TEST_HPU or TEST_PRIVATEUSE1:
       DEVICE_TYPE = torch.accelerator.current_accelerator().type
       DEVICE_COUNT = torch.accelerator.device_count()
   ```

   After the NPU backend is registered as a PyTorch PrivateUse1 backend, it can also be recognized through this unified entry point. For example:

   ```python
   torch.accelerator.current_accelerator().type
   ```
   returns `"npu"`.

## I. Coupling Type 1: Device Creation and CUDA Runtime API Coupling

Core Issue:

The test logic itself is generic DTensor or ordinary accelerator behavior, but tensors, modules, `DeviceMesh`, `init_device_mesh`, `torch.cuda.*`, etc. are hard-coded to CUDA, preventing non-CUDA accelerators from being able to reuse the tests.

Refactoring Recommendations:

- In DTensor tests, prioritize using `self.device_type` and `self.build_device_mesh()`.
- Runtime operations such as device count, device setting, and synchronization should prioritize using `torch.accelerator`.
- Backend module capabilities should be obtained using `torch.get_device_module(self.device_type)`; APIs not supported by all backends should be skipped based on capability.
- CPU-only tests should not carry `ProfilerActivity.CUDA`.

### Community-Merged PR References and Modification Examples:

PR #184241 changed convolution tests from CUDA-only to accelerator-generic.

File: `test/nn/test_convolution.py`

> PR's overall modification points: Changed `@onlyCUDA` tests to `@onlyAccelerator`; migrated regular tests to device-type class; changed `use_cuda` boolean parameter to `device` parameter; changed `.cuda()` and `device="cuda"` to use the incoming `device`; changed `torch.cuda.synchronize()` to `torch.accelerator.synchronize()`; changed `self.device_type == "cuda"` checks to more generic accelerator checks.

Before and after code: `test/nn/test_convolution.py @@ -1428,14 +1267,11`

Source: [PR #184241](https://github.com/pytorch/pytorch/pull/184241)

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

Modification points:
- Instead of having the helper answer only "whether to use CUDA," the device to use is now passed in directly from outside.
- This way, the same helper can serve different accelerators such as CUDA, XPU, HPU, and PrivateUse1/NPU.
- This type of PR is a typical reference for Type 1: if the test logic itself is generic, replace the tensor creation, module creation, and device synchronization points from CUDA APIs to generic device entry points.

#### Example 1: DTensor dispatch overhead test hard-coded CUDA.

File: [test/distributed/tensor/test_dtensor_dispatch_overhead.py](/Users/aurora/.codex/worktrees/85dc/pytorch/test/distributed/tensor/test_dtensor_dispatch_overhead.py:69)

Before and after code: `test/distributed/tensor/test_dtensor_dispatch_overhead.py @@ test_dtensor_add_op_dispatch_overhead`

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

Modification points:
- Instead of using `torch.cuda.is_available()` to decide whether the test can run, the device module corresponding to the current test device is used.
- Instead of hard-coding `"cuda"` to create mesh and tensors, `self.device_type` is followed.
- If the current backend does not have `get_device_name`, only the device type is recorded to avoid skipping the entire test due to the absence of CUDA-exclusive interfaces.

`hasattr(obj, "name")` is used to determine whether an object has a certain property or method. It checks whether the `device_module` has a `get_device_name` method, and returns true if it does.

There are similar judgment patterns in the source code:
```diff
# torch/sparse/_triton_ops_meta.py
if torch.cuda.is_available():
    return torch.cuda.get_device_name()
if torch.xpu.is_available():
    return torch.xpu.get_device_name()
return ""
```
Meaning: PyTorch itself only retrieves device names for CUDA/XPU; for other devices, it directly returns an empty string.

## II. Coupling Type 2: Test Entry, Skip Decorators, and Device-Type Parameterization Coupled to CUDA

Core Issue:

The test body can be reused by multiple accelerators, but the test entry layer uses CUDA-only implementations of `@requires_cuda`, `@onlyCUDA`, `skip_if_lt_x_gpu()`, `only_for=("cuda",)`, or class-level skips, preventing non-CUDA backends from instantiating or running tests.

### Community-Merged PR References and Modification Examples:

#### Example: PR #180820 adds class/method-level skip entry points for device-type tests, taking effect during instantiation.

File: `torch/testing/_internal/common_device_type.py`
> PR's overall modification points: Added `test_exclusions` configuration to `DeviceTypeTestBase`; reads this configuration when instantiating device-type tests; supports `"*"` to skip an entire class and also supports skipping only certain test methods; adds new test cases in OpenReg tests to prove that both class-level and single-method skips work.
> 
> The following two code snippets belong to a set of modifications in the same PR: the first provides the "skip list," and the second makes test generation actually execute this list.

Before and after code: `torch/testing/_internal/common_device_type.py @@ DeviceTypeTestBase.test_exclusions`

Source: [PR #180820](https://github.com/pytorch/pytorch/pull/180820)

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

Modification points:
- Added `test_exclusions`, allowing backends to skip precisely by test class or test method.
- This avoids writing coarse-grained conditions like `@requires_cuda` into the generic test body.

A second key modification in the same PR: reading skip configuration when instantiating device-type tests.

Before and after code: `torch/testing/_internal/common_device_type.py @@ instantiate_device_type_tests`

Source: [PR #180820](https://github.com/pytorch/pytorch/pull/180820)

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

Modification points:
- Check `test_exclusions` before generating test cases.
- If `*` is written in the configuration, skip the entire test class; if only method names are written, skip only the corresponding methods.

#### Example 1: DTensor logging class is wrapped by `@requires_cuda`.

File: [test/distributed/tensor/test_dtensor_logging.py](/Users/aurora/.codex/worktrees/85dc/pytorch/test/distributed/tensor/test_dtensor_logging.py:11)

Before and after code: `test/distributed/tensor/test_dtensor_logging.py @@ imports and setUp`

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

Modification points:
- Removed `@requires_cuda`, allowing this logging test to no longer run only in CUDA environments.
- `self.device_type` is no longer fixed to `"cuda"` but is automatically determined based on the current available device.
- This change also requires modifying the expected strings in the logs; otherwise, the log content will still hard-code `'cuda'`.

Supplementary explanation:

```python
# torch/testing/_internal/common_utils.py
requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
```

> `@requires_cuda` is straightforward: the test runs only when `torch.cuda.is_available()` is true; without CUDA, it is skipped. Therefore, it is suitable for CUDA-exclusive tests but not for tests that only check DTensor logging, graph strings, and other generic logic.

## III. Coupling Type 3: Communication Backend and Multi-Process Initialization Bound to NCCL/Gloo

Core Issue:

DTensor communication tests require a process group. `backend="nccl"` locks the test to CUDA/NCCL;
generic DTensor tests should follow the default distributed backend of the current device type.

Refactoring Recommendations:

- Generic accelerator communication tests should default to `dist.get_default_backend_for_device(self.device_type)`.
  > Source location:
[torch/distributed/distributed_c10d.py (line 1525)](/Users/aurora/.codex/worktrees/85dc/pytorch/torch/distributed/distributed_c10d.py:1525)

- `DeviceTypeTestBase.distributed_backend()` is the unified entry point for device-type tests to obtain the communication backend; new hardware backends can access their own backend through this entry point without modifying code everywhere.
  > Source location:
[torch/testing/_internal/common_device_type.py (line 420)](/Users/aurora/.codex/worktrees/85dc/pytorch/torch/testing/_internal/common_device_type.py:420)
- NCCL-only tests should retain specific classes, without forcing generalization.

### Community-Merged PR References and Modification Examples:

The issue raised in Type 3 is: communication tests should not hard-code the backend as `"nccl"` or `"gloo"`, otherwise new accelerators are very difficult to reuse. PR #184181 added a unified `distributed_backend()` entry point for device-type tests.

Example: PR #184181 adds a default distributed backend hook for device-type tests and verifies that custom backends can be integrated.

File: `torch/testing/_internal/common_device_type.py`

PR's overall modification points: Added `distributed_backend()` on `DeviceTypeTestBase`; default implementation calls `dist.get_default_backend_for_device(cls.device_type)`; added `TestDistributedBackendHook` in OpenReg tests to verify that custom backends can return their own default communication backend. The following two code snippets belong to a set of modifications in the same PR: the first adds the unified entry point, and the second uses OpenReg to prove that new backends can declare their own backend through this entry point.

Before and after code: `torch/testing/_internal/common_device_type.py @@ DeviceTypeTestBase.distributed_backend`

Source: [PR #184181](https://github.com/pytorch/pytorch/pull/184181)

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

Modification points:
- Added `distributed_backend()`, allowing the test framework to automatically retrieve the appropriate communication backend for the current device.
- This way, tests do not need to manually write `"nccl"`, `"gloo"`, or some backend-specific name.

Verification code in the same PR: OpenReg backend verifies custom distributed backend.

Before and after code: `test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_testing.py @@ TestDistributedBackendHook`

Source: [PR #184181](https://github.com/pytorch/pytorch/pull/184181)

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
Modification points:
- Added a small test to confirm that the OpenReg backend can return its own backend through the unified hook.
- This shows that new backends can integrate into distributed tests without modifying the test body.

## IV. Coupling Type 4: Compile/Export/Debug Expected Strings or Backends Hard-Coded for Specific Backends

Core Issue:

DTensor compile/export/debug tests commonly use `assertExpectedInline()` to verify graph/log strings. If the golden string hard-codes `'cuda'`, or the compile backend uses CUDA/Triton/Inductor-specific capabilities, non-CUDA backends will fail. Distinction must be made between generic graph semantics and backend-specific capabilities.

### Community-Merged PR References and Modification Examples:

Example 1: PR #180328 changed Dynamo efficient attention tests from CUDA-only to capability-based judgment.

File: `test/dynamo/test_unspec.py`

> PR's overall modification points: Moved `test_no_recompilations_with_efficient_attention` from regular `UnspecTests` to `UnspecTestsDevice`; removed `@requires_cuda`; 
> 
> removed `device = "cuda"` inside the function; added `PLATFORM_SUPPORTS_MEM_EFF_ATTENTION` skip; CPU explicitly skips at runtime; 
> 
> removed the fixed device list of `instantiate_device_type_tests(..., only_for=["cuda", "hpu", "xpu"])`; updated `torch._dynamo.optimize()` to `torch.compile()`.

Before and after code: `test/dynamo/test_unspec.py @@ efficient attention test`

Source: [PR #180328](https://github.com/pytorch/pytorch/pull/180328)

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

Modification points:
- The old test was "if not CUDA, don't enter"; the new test changes to "the framework passes the current device; run only if the platform supports efficient attention."
- This preserves hardware capability constraints: efficient attention is still not a CPU-generic test; unsupported platforms will explicitly skip.
- It is a good example of hardware binding: the binding is "functional capability," not "device name." As long as other backends support this capability, there is an opportunity to reuse the same test.

#### Example 1: Logging golden hard-codes CUDA.

File: [test/distributed/tensor/test_dtensor_logging.py](/Users/aurora/.codex/worktrees/85dc/pytorch/test/distributed/tensor/test_dtensor_logging.py:61)

Before and after code: `test/distributed/tensor/test_dtensor_logging.py @@ test_sharding_prop_cache_logging`

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

Modification points:
- The log expectation no longer hard-codes `'cuda'` but uses the current test device name.
- Subsequent Python cache logs in the same file also need to be changed together; otherwise, only half of it would have been changed and would still fail on non-CUDA backends.

#### Example 2: Using `backend="inductor"` in DTensor compile: No need to change backend when NPU supports it.

File: [test/distributed/tensor/test_dtensor_compile.py](/Users/aurora/.codex/worktrees/85dc/pytorch/test/distributed/tensor/test_dtensor_compile.py:1641)

> Whether it is suitable for generic hardware tests: It depends. The current overall goal is that GPU and NPU can both run; if NPU already supports Inductor, there is no need to change `backend="inductor"` to `backend="aot_eager"`. Only when the test target is only generic graph semantics and NPU does not support Inductor should a generic approach be considered.

Current code: `test/distributed/tensor/test_dtensor_compile.py @@ test_dtensor_different_gradient_placement`

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

- For GPU+NPU goals, scenarios where NPU already supports Inductor should not be modified; otherwise, the Inductor path coverage that was originally wanted to be achieved would be lost.
  
## V. Coupling Type 5: OpInfo, Operator Support, Skip/xfail, and Allowlist Coupling

Core Issue:

DTensor tests often generate tests in bulk through OpInfo. Different out-of-tree backends have different operator support;

Refactoring Principles:

- Operator support differences should be expressed uniformly through configurations such as `op_overrides`, `op_allowlist`, and `test_exclusions`.
- For OpInfo files that can currently only run on CPU, do not open up accelerators all at once; first establish a failure list/allowlist, then gradually enable them.

### Community-Merged PR References and Modification Examples:

#### Example: PR #181554 adds a unified registration API for backend test differences and changes temporary configurations to use this API.

File: `test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_testing.py`

> PR's overall modification points:  
> 
> Added `set_test_configs(...)` to `DeviceTypeTestBase`;
> 
> Allows backends to set `op_overrides`, `op_allowlist`, and `test_exclusions` at once; 
> 
> Changed temporary configuration contexts in OpenReg tests from manual `setattr` to calling the configuration API; 
> 
> Retains context manager to restore original configurations, avoiding cross-test pollution.
> 
> The following two code snippets belong to a set of modifications in the same PR: the first changes temporary configurations to call a unified API, and the second provides the definition of this unified API.


Before and after code: `test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_testing.py @@ _temp_test_configs`

Source: [PR #181554](https://github.com/pytorch/pytorch/pull/181554)

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
-        for k, v in attrs.items():
-            setattr(obj, k, v)
+        obj.set_test_configs(**backup)
```

Modification points:
- Temporary test configurations no longer manually `setattr` class attributes; instead, they are set and restored through `set_test_configs(...)`.
- Switching to a unified API to manage such test configurations makes the intent clearer and easier to maintain.

Test configuration API unified as `set_test_configs`.

File: `torch/testing/_internal/common_device_type.py`

Before and after code: `torch/testing/_internal/common_device_type.py @@ DeviceTypeTestBase.set_test_configs`

Source: [PR #181554](https://github.com/pytorch/pytorch/pull/181554)

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

Modification points:
- Multiple configuration entry points are consolidated into a clear function.
- Backends only need to call `set_test_configs(...)` to set operator overrides, operator allowlists, and test skip rules all at once.
- This step is complementary to the temporary configuration context above: the former is responsible for calling the unified entry point, while here we define what this unified entry point can specifically set.

#### Example 1: DTensor OpInfo test currently fixed to CPU.

File: [test/distributed/tensor/test_dtensor_ops.py](/Users/aurora/.codex/worktrees/85dc/pytorch/test/distributed/tensor/test_dtensor_ops.py:412)

Corresponding code: `test/distributed/tensor/test_dtensor_ops.py @@ OP_DB_WORLD_SIZE`

```python
OP_DB_WORLD_SIZE = 4
# DEVICE_TYPE = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() >= OP_DB_WORLD_SIZE else "cpu"
# TODO: debug cuda illegal memory access issue and re-enable cuda tests
DEVICE_TYPE = "cpu"
```

- This code indicates that running on CUDA was attempted in the past, but illegal memory access issues were encountered.
- Currently fixing to CPU is a conservative choice; it should not be directly changed to all hardware.
- The correct approach is to first list which operators are supported on the target backend, then enable them in batches using allowlists, skips, and xfails.

### Real Reference Pattern: Current PyTorch device-type infra already supports allowlists and test configurations.

File: [torch/testing/_internal/common_device_type.py](/Users/aurora/.codex/worktrees/85dc/pytorch/torch/testing/_internal/common_device_type.py:436)

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

File: [torch/testing/_internal/common_device_type.py](/Users/aurora/.codex/worktrees/85dc/pytorch/torch/testing/_internal/common_device_type.py:503)

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

- `_apply_op_allowlist` only generates tests for operators declared as supported by the backend, reducing meaningless failures.
- `set_test_configs` consolidates allowlists, skips, xfails, and other configurations into a single entry point, avoiding special-case handling scattered throughout the test code.
- Viewing the two code snippets together: `set_test_configs` is responsible for registering rules, and `_apply_op_allowlist` is responsible for actually executing allowlist filtering before OpInfo tests are generated.
