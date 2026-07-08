# FSDP Test Case Hardware Coupling Refactoring Guide

This document addresses hardware coupling issues in PyTorch FSDP test cases. It categorizes the types of coupling found in the code and provides refactoring recommendations, code examples, and community PR references for each type.

Reference document: https://docs.google.com/spreadsheets/d/1cDNiLW4KvPcGYPlA3KCDm0zV5PLPUWubno1OyCznKBw/edit?pli=1&gid=1201461581#gid=1201461581

PyTorch git node: fad74248e78716152917a729adb2b44ba2bab16e

## Type 1: `@skip_if_lt_x_gpu(N)` Only Recognizes cuda/hpu/xpu — New Hardware Gets Incorrectly Skipped

### Core Issue

- The skip decorator at the test entry point or `world_size` calculation hard-codes three device types: CUDA/HPU/XPU.
- New accelerators are neither recognized by `@skip_if_lt_x_gpu` nor able to obtain the correct world size via `torch.cuda.device_count()`.

### Refactoring Principle

- All device count determinations should use `torch.accelerator.current_accelerator()` to obtain the `device_type`, then call `device_count()` on it.

### Refactoring Recommendation

`@skip_if_lt_x_gpu` in `torch/testing/_internal/common_distributed.py` hard-codes three device types (CUDA/HPU/XPU). It is recommended to add a generic `@skip_if_lt_x_devices(x, *, allow_cpu=False)` in `common_distributed.py` for all distributed tests (including FSDP) to use.

### Recommended Approach

1. Add `@skip_if_lt_x_devices(x, *, allow_cpu=False)` in `torch/testing/_internal/common_distributed.py`.
2. Inside the decorator, use `torch.accelerator.current_accelerator()` to obtain the current accelerator type.
3. Call `device_count()` on the accelerator to determine the number of available devices.
4. Explicitly handle the CPU path, preserving the degenerate execution semantics of `allow_cpu=True`.
5. Exception handling logic should remain consistent with the original code: still use `TEST_SKIPS[f"multi-gpu-{x}"]` and `_maybe_handle_skip_if_lt_x_gpu`. Add comments at `TEST_SKIPS` and `_maybe_handle_skip_if_lt_x_gpu` explaining that although their naming contains `gpu`, the actual skip logic is hardware-agnostic and generic — renaming is deferred for now.

### Original Code Example

Current implementation of `@skip_if_lt_x_gpu`:

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

### Post-Refactoring Code Example

Add a generic decorator in `common_distributed.py`:

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

### Community PR Reference

**No direct corresponding PR exists**, as community refactoring efforts have mostly focused on the `instantiate_device_type_tests` framework, whereas FSDP distributed tests use `MultiProcessTestCase`. The following can serve as transitional or supplementary references:

- **PR #176717** (`test_unary_ufuncs.py`): Introduces the `@onlyAccelerator` decorator, providing a generic accelerator check that "skips CPU/meta."
  - URL: https://github.com/pytorch/pytorch/pull/176717
  - Reference method: Add a generic decorator to replace hard-coded device checks.


## Type 2: Device-Specific API / Dedicated Test Decoupling

### Core Issues

- Test code hard-codes device-specific API calls such as `device="cuda"`, `torch.cuda.synchronize()`, `DeviceType.CUDA`, `torch.cuda.Stream()`, `torch.device("cuda", rank)`.
- NCCL environment variable settings, NCCL log assertions, and NCCL-specific functionality tests are tightly bound to CUDA/NCCL.

### Refactoring Principles

- Generic computation/synchronization logic should use `torch.get_device_module(device_type)` or `torch.accelerator.*` instead of `torch.cuda.*`.
- Tests that genuinely depend on CUDA-specific mechanisms (CUDA Graph, CUDA multicast, CUDA hardware capability checks, etc.) should be retained as CUDA-only tests without modification.

### Subcategory 1: Originally hard-coded CUDA logic that can support a generic `device_type`

#### Determination Criteria

Replace originally hard-coded `device="cuda"`, `DeviceType.CUDA`, `torch.cuda.synchronize()`, etc. with calls based on the current `device_type`, where the semantics still hold and CUDA-specific behavior is not introduced.

#### Recommended Approach

1. At the top of the test file, obtain the current `device_type` via `torch.accelerator.current_accelerator()` (if the test file already has `device_type` retrieval logic, use `device_type` directly).
2. Replace hard-coded device APIs with `device_type`-based generic calls:
   - `device="cuda"` -> `device=device_type`
   - `torch.device("cuda", rank)` -> `torch.device(device_type, rank)`
   - `torch.cuda.synchronize()` -> `torch.get_device_module(device_type).synchronize()`
   - `torch.cuda.set_device(rank)` -> `torch.get_device_module(device_type).set_device(rank)` (if the device module provides it)
   - `DeviceType.CUDA` -> `DeviceType` enum value derived from the `device_type` string (e.g., `getattr(DeviceType, device_type.upper(), None)`)
3. Retain generic device count decorators such as `@skip_if_lt_x_devices` (covered in Type 1).

#### Original Code Example

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

#### Post-Refactoring Code Example

```python
# test/distributed/_composable/fsdp/test_fully_shard_state_dict.py

# If device_type assignment already exists at the top of the test file, this line is unnecessary
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

#### Explanation

- This is a typical **device name string hard-coding** scenario in Type 2: `torch.rand(..., device="cuda")` directly writes `"cuda"` when creating the input tensor.
- The `device` parameter of `torch.rand` accepts both strings and `torch.device`, so it can be directly replaced with the dynamically derived `device_type` without introducing CUDA-specific semantics.

### Subcategory 2: Original logic that genuinely only applies to a specific device type

#### Determination Criteria

The code depends on CUDA-specific APIs (such as `torch.cuda.CUDAGraph`, `torch.cuda.Stream()`, CUDA Graph capture/replay, CUDA-specific hardware capabilities, etc.) that cannot be generalized via `torch.get_device_module(device_type)`.

#### Recommended Approach

1. **Make no changes at all, consistent with community PRs.** The community adopts a `left unchanged` strategy for tests that are genuinely CUDA-exclusive (e.g., `torch.cuda._sleep()`, `nvtx`, `pin_memory`, `DataParallel` in PR #184593).
2. Retain existing CUDA-only decorators (such as `@skip_if_lt_x_gpu`, `@unittest.skipIf(not TEST_CUDA_GRAPH, ...)`, `@onlyCUDA`), without extracting additional classes or removing contradictory branches.
3. If PyTorch provides a unified accelerator graph/stream abstraction in the future, consider generalizing such tests.

#### Example 1: CUDA Graph

##### Original Code

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

##### Handling

**Make no changes at all, consistent with community PRs.**

Rationale:
1. `TestFullyShardCudaGraph` is a CUDA-exclusive test class, classified as `device-specific`. Community PRs also adopt a `left unchanged` strategy for tests of such APIs.
2. PyTorch currently has no cross-device graph abstraction. Forcing parameterization would require introducing device-name branches like `if device_type == "cuda": ... elif device_type == "xpu": ...`, which increases hardware coupling instead.
3. Although the original code mixes generic device checks like `if device_type.type == "cuda"` and `torch.device(device_type.type, self.rank)`, `@unittest.skipIf(not TEST_CUDA_GRAPH, ...)` already ensures the method only runs in environments supporting CUDA Graph — it is fundamentally a CUDA-only test.

**If PyTorch provides a unified accelerator graph API in the future (e.g., `torch.accelerator.graph()`), consider generalizing such tests. For now, keep them as-is.**

#### Example 2: NCCL LOG

##### Original Code

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

##### Handling

**Make no changes at all, consistent with community PRs.**

Rationale:
1. **NCCL environment variables are NCCL-exclusive mechanisms**
   - `NCCL_DEBUG`, `NCCL_DEBUG_SUBSYS`, and `NCCL_DEBUG_FILE` are inherently NCCL variable names.
   - There are no generic APIs such as `torch.distributed.set_collective_debug_env()` or `torch.accelerator.set_debug_env()`.

2. **NCCL log formats are NCCL-exclusive**
   - The `"NCCL INFO ..."` in assertions is NCCL's log format.
   - Backends such as XCCL, Gloo, and UCC have no equivalent format, so the regular expression cannot be generalized.

3. **Dynamic derivation cannot convert NCCL-exclusive calls into generic calls**
   - `get_default_backend_for_device(device_type)` or `torch.accelerator.current_accelerator()` can only tell you what the current accelerator/default backend is.
   - They cannot automatically convert an `NCCL_DEBUG` setting or `NCCL INFO` assertion into an XCCL/Gloo equivalent.

Therefore, any refactoring would remain at the level of "check if the current backend is NCCL, then execute NCCL-exclusive logic," without achieving true backend-agnosticism.

### Determining Whether a Coupling Point Belongs to Subcategory 1 or Subcategory 2

For Type 2 coupling points, the **torch_npu repository's upstream patch files** can serve as an auxiliary reference:

- In the torch_npu repository's v2.7.1 branch, under the `pytorch/test_upstream` directory, each PyTorch test file has a corresponding `.patch` file preserved. For example, `test/distributed/_composable/fsdp/test_fully_shard_autograd.py` corresponds to `test/distributed/_composable/fsdp/test_fully_shard_autograd.py.patch`.
- These patches are similar to `git diff`, recording modifications made to the source file during NPU adaptation.
- Since patches are based on historical PyTorch nodes, **line numbers may not correspond one-to-one with current PyTorch code**. Position the specific location in the current test file using context in the diff (such as variable names, function names, surrounding code).

**Determination Rules**:

- If a Type 2 coupling point has a corresponding modification in the patch (e.g., `device="cuda"` changed to `device="npu"`, `.cuda()` changed to `.npu()`, etc.), it means the coupling point can be generalized to new accelerators by dynamically deriving `device_type`, belonging to **Subcategory 1**.
- If a Type 2 coupling point **has no corresponding modification** in the patch, it means the vendor side has also found no generic replacement. Such points should generally be retained as CUDA/NCCL-exclusive logic, belonging to **Subcategory 2**.

#### Generalizable Example: `device="cuda"` in `test_fully_shard_state_dict.py`

##### Coupling Point in Current Test Case

```python
# test/distributed/_composable/fsdp/test_fully_shard_state_dict.py:151
inp = torch.rand(mlp_dim, mlp_dim, device="cuda")
```

##### Corresponding Modification in torch_npu Patch

```diff
# torch_npu/pytorch/test_upstream/test/distributed/_composable/fsdp/test_fully_shard_state_dict.py.patch
@@ -148,12 +153,12 @@ class TestFullyShardStateDictMultiProcess(FSDPTest):
             model.load_state_dict(sd, assign=True, strict=False)

         # lazy init without error
-        inp = torch.rand((mlp_dim, mlp_dim), device="cuda")
+        inp = torch.rand((mlp_dim, mlp_dim), device="npu")
```

##### Determination Conclusion

This coupling point has a corresponding device name replacement in the patch, indicating that simply changing `"cuda"` to a dynamically derived `device_type` allows it to run on new accelerators. It belongs to **Subcategory 1**.

#### Non-Generalizable Example: CUDA Graph in `test_fully_shard_training.py`

##### Coupling Point in Current Test Case

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

##### Corresponding Modification in torch_npu Patch

Examining `torch_npu/pytorch/test_upstream/test/distributed/_composable/fsdp/test_fully_shard_training.py.patch`, there are **no modifications** targeting `TestFullyShardCudaGraph`. Only generic device strings in the same file such as `init_device_mesh("cuda", ...)`, `torch.randn(..., device="cuda")`, and `torch.device("cuda")` were modified.

##### Determination Conclusion

This coupling point has no corresponding modification in the patch, indicating that the NPU side also cannot achieve generalization through simple device name replacement. CUDA-exclusive APIs must be retained. It belongs to **Subcategory 2**.

#### Usage Notes

- The above patch files are based on the torch_npu v2.7.1 branch, corresponding to a historical PyTorch node, so line numbers may not match current PyTorch code. Position the current file location using context in the diff.
- This method serves only as an **auxiliary determination tool**: coupling points unmodified in patches generally mean the community/vendor side has also found no generic replacement and should be prioritized as non-generalizable Subcategory 2.

### Community PR References

- **PR #184593** (`test_autograd.py`): Replaced a large amount of `torch.cuda.synchronize()`, `torch.cuda.memory_allocated()`, `torch.cuda.set_default_device("cuda")`, `device="cuda"`, etc. with `torch.accelerator.*` or parameterized `device`; retained CUDA-specific APIs (`nvtx`, `_sleep`, `pin_memory`, `DataParallel`, etc.) as CUDA-only tests without forcing generalization.
  - URL: https://github.com/pytorch/pytorch/pull/184593
  - Reference method: Generic device APIs use `torch.accelerator.*` / `torch.get_device_module(device_type)`; CUDA/NCCL-specific logic is left as-is.

- **PR #178336** (`[Distributed] Make DDP tests and tensor parallel dependencies backend agnostic`): Replaced `@requires_nccl()` with `@requires_accelerator_dist_backend()`, dynamically deriving `DEVICE_TYPE` and `BACKEND` via `torch.accelerator.current_accelerator()` to reduce hard-coding of `cuda`/`nccl`.
  - URL: https://github.com/pytorch/pytorch/pull/178336
  - Reference method: Distributed tests should derive backends dynamically wherever possible, avoiding hard-coded backend names in test code; assertions that genuinely depend on NCCL behavior remain NCCL-only.

- **PR #160158** (`[1/N] Port 6 fsdp distributed test cases to Intel GPU`): When porting FSDP tests to Intel GPU, changed `backend="cpu:gloo,cuda:nccl"` to `backend="cpu:gloo,xpu:xccl"`. Reviewers recommended using `get_default_backend_for_device` for further de-device-naming.
  - URL: https://github.com/pytorch/pytorch/pull/160158
  - Reference method: Backend strings should not hard-code device names; if backend names cannot be fully eliminated, keep backend-isolated tests as-is.

- **PR #163063** (`Restore environment after NcclUserBufferRegistrationTest`): After an NCCL-specific test sets `NCCL_ALGO=NVLS`, it restores the environment — illustrating that NCCL environment variable operations should only appear within NCCL-specific test scope.
  - URL: https://github.com/pytorch/pytorch/pull/163063
  - Reference method: NCCL environment variable settings are NCCL-specific test logic and should not be forcibly generalized.

## Type 3: Explicit Device Name Hard-Coding in Models / Tensors

### Core Issue

- Existing test cases have models such as MLP / Transformer / `nn.Linear` created on CPU by default, then moved to a specific device by **explicitly using device name strings** (such as `"cuda"`, `"xpu"`, `"cpu"`).

### Refactoring Principles

1. Only modify call sites where device names are **explicitly hard-coded**, replacing them with a dynamically derived `device_type`.
2. For `to(device_type)`, default CPU creation followed by `.to(device_type)`, and other already-parameterized paths, **make no changes**.
3. For genuinely CPU-only tests (where the model always stays on CPU and `"cpu"` is the expected behavior), **make no changes**.

### Refactoring Recommendation

Replace `.cuda()` with `.to(device_type)`; replace `.cpu()` with `.to("cpu")` (CPU is genuinely needed) or `.to(device_type)` (if it's just a device transfer).

### Original Code Example

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

### Post-Refactoring Code Example

```python
# test/distributed/_composable/fsdp/test_fully_shard_state_dict.py
import torch
from torch.testing._internal.common_distributed import skip_if_lt_x_devices

# If device_type assignment already exists at the top of the test file, this line is unnecessary
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

### Explanation

- Changing `.cpu()` to `.to("cpu")` is just for notational uniformity; however, the test semantics here involve CPU offloading, which is a test logic requirement. Retaining `"cpu"` as the target device is reasonable.
- `.cpu()` and `.to("cpu")` are **functionally equivalent** on `nn.Module`: the former's implementation in `torch/nn/modules/module.py:1155-1164` is `self._apply(lambda t: t.cpu())`; the latter, in `torch/nn/modules/module.py:1340-1383`, first resolves `device=torch.device("cpu")` via `torch._C._nn._parse_to("cpu")`, then executes `t.to(device)` for each parameter/buffer. Both recursively move all parameters and buffers to CPU and return `self`. Therefore, rewriting `.cpu()` as `.to("cpu")` introduces no behavioral difference — it only aligns notationally with `.to(device_type)`.

### Community PR References

- **PR #184593** (`test_autograd.py`): Replaced a large amount of `device="cuda"`, `.cuda()`, and `model.cuda()` with parameterized `device`.
  - URL: https://github.com/pytorch/pytorch/pull/184593
  - Reference method: Explicit device names are uniformly replaced with dynamically derived `device_type`; retain `"cpu"` strings for paths that genuinely require CPU.

- **PR #184261** (`test_serialization.py`): Replaced `torch.device("cuda")` and `device="cuda"` with parameterized `device`.
  - URL: https://github.com/pytorch/pytorch/pull/184261
  - Reference method: The device parameter during tensor/module creation is uniformly parameterized.

- **PR #184315** (`test_functional.py`): Changed `(x * y).cuda()` to `(x * y).to(device)`.
  - URL: https://github.com/pytorch/pytorch/pull/184315
  - Reference method: Replace `.cuda()` with `.to(device)`.

- **PR #183728** (`test_optim.py`): Changed `"cuda" in optim_info.supports_fused_on` and `params_cuda = [p.to(device="cuda")]` to use parameterized device-based checks.
  - URL: https://github.com/pytorch/pytorch/pull/183728
  - Reference method: Avoid hard-coding `"cuda"` at the string level; uniformly use `_get_device_type(device)` or `device_type`.

- **PR #184192** (`test_lazy_modules.py`): Removed manual branching of `if TEST_CUDA: device = "cuda"`, changing to framework-injected `device` parameter.
  - URL: https://github.com/pytorch/pytorch/pull/184192
  - Reference method: Delete explicit device name branches and replace with parameterized device.
