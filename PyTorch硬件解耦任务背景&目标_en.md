# PyTorch Hardware Decoupling: Background & Objectives

## I. Background

### 1.1 Core Contradiction

PyTorch has long adopted **CUDA as its sole accelerator reference** in architectural design, resulting in implicit CUDA dependencies from the framework layer to the test layer. When new hardware accelerators (such as Ascend NPU, Intel XPU, etc.) attempt to integrate via the `PrivateUse1` mechanism, they encounter systemic compatibility barriers.

These barriers exist on simultaneously at two levels:

The root cause of both points to the same fact: **PyTorch lacks a standardized third-party backend registration and extension mechanism.**

### 1.2 Current State and Costs

The above contradiction creates obstacles across both **features** and **test cases**:

#### Feature Dimension: New Hardware Cannot Access PyTorch Core Features Through Standard Paths

PyTorch's core feature modules — Graph Capture and Compilation (Dynamo/Inductor), Distributed Training and Communication (Distributed/FSDP), Operator Verification and Capability Declaration (OpInfo), and Device Runtime Infrastructure (Device/Serialization/DLPack) — are all designed with CUDA as the default accelerator reference, lacking standardized extension and registration mechanisms for third-party backends.

Specifically: Dynamo's variable tracing only recognizes CUDA types; Inductor's backend device determination hard-codes a limited backend list; distributed communication initialization paths only consider NCCL; FSDP's gradient synchronization logic embeds CUDA stream management assumptions; operator capability declarations are hard-coded in OpInfo in forms such as `dtypesIfCUDA`; and infrastructure like device management/serialization/DLPack does not use the dispatch mechanism.

This forces new hardware to intercept and replace PyTorch internal functions via Monkey Patch to inject custom behavior. These internal APIs change frequently with upstream refactoring, resulting in extremely fragile compatibility and version adaptation costs that accumulate linearly with iterations.

#### Test Case Dimension: New Hardware Cannot Reuse Community Tests to Validate Functional Correctness

Corresponding to CUDA coupling at the feature level, PyTorch test cases also treat CUDA as the sole accelerator validation target. Guards such as `@onlyCUDA`, `TEST_CUDA`, `GPU_TYPE`, and `HAS_GPU` widely used in tests exclude new hardware; in distributed tests, `DEVICE_TYPE` and `DISTRIBUTED_BACKEND` are hard-coded as CUDA/NCCL; dtype declarations for each backend in OpInfo require intrusively appending entries one by one in PyTorch source code.

Furthermore, test classification lacks explicit hardware requirement markers — pure CPU test cases and accelerator-requiring test cases are mixed together, making it impossible for backend developers to distinguish which to run and which to skip. The existing skip mechanism is method-level "all-or-nothing," unable to express precise control like "this operator passes for one dtype but skips for another."

This prevents new hardware from directly reusing the community's vast test assets to validate functional correctness, requiring them to write large numbers of equivalent test cases independently — duplicative investment with far lower coverage than community tests. Additionally, during the community's test decoupling refactoring process, there is no automated test count change perception mechanism, creating a risk of tests being silently lost due to mislabeling.

#### Typical Example: Device Determination in the Inductor Compilation Stack

The following concrete example illustrates how the same problem manifests simultaneously at both the feature and test case levels.

PyTorch's Inductor compilation stack hard-codes the supported device list when determining which devices need to generate device guards:

```Python
# PyTorch source: torch/_inductor/scheduler.py
def device_need_guard(device_type: str) -> bool:
    return device_type in ["cuda", "xpu"]   # NPU is not in the list
```

Because NPU is not in the hard-coded list, Inductor will not generate device guards for NPU, causing the compilation optimization path to be skipped. `torch_npu` has had to adapt by monkey-patching this function:

```Python
# torch_npu adaptation code: torch_npu/_inductor/utils.py
import torch._inductor.scheduler as _scheduler

_original_device_need_guard = _scheduler.device_need_guard

def _patched_device_need_guard(device_type: str) -> bool:
    if device_type == "npu":      # Hard-coded injection of NPU branch
        return True
    return _original_device_need_guard(device_type)

_scheduler.device_need_guard = _patched_device_need_guard   # Runtime replacement
```

This kind of internal function replacement depends on the function signatures and module paths of PyTorch's private APIs. Once upstream refactoring occurs (such as function renaming or module reorganization), the adaptation code immediately breaks. Similar patches exist in Inductor's Codegen, CodeCache, AOTI, and other paths.

Meanwhile, the test cases for this feature exclude NPU via GPU guards:

```Python
# PyTorch test code: test/inductor/test_scheduler.py (example)
from torch.testing._internal.inductor_utils import HAS_GPU

@unittest.skipIf(not HAS_GPU, "requires GPU")
class TestScheduler(TestCase):
    def test_device_guard(self):
            ...
```

In the above code, the same functionality point (Inductor device determination) requires **patching internal functions** to work on the feature side and is **excluded by the `HAS_GPU` guard** on the test case side — the problems in both dimensions share the same root cause: the device list was not designed as an extensible registration mechanism.

#### Maintenance Cost Summary

The root cause of both dimensions is unified — PyTorch lacks standardized third-party backend extension points. The costs this imposes are systemic:

### 1.3 Community Window of Opportunity

The PyTorch community initiated systematic **third-party backend-friendliness** work starting in early 2026, providing an aligned window for problem resolution:

**Key Judgment:** The community direction has been clarified, and specific interfaces are under design. Contributing NPU's actual requirements and use cases at this time can effectively influence design decisions, shifting from "passive adaptation" to "co-building standards."

---

## II. Problem Analysis

Corresponding to Section 1.2, problems concentrate in two dimensions: **features** and **test cases**.

### 2.1 Feature Dimension: Core Modules Lack Standardized Third-Party Backend Extension Points

Multiple core modules in PyTorch are designed with CUDA as the sole accelerator reference, preventing third-party backends from registering their implementations through standard paths.

- **Graph Capture and Compilation:** Dynamo's variable tracing only recognizes CUDA types. Tensors from new hardware are treated as unknown variables during graph capture, causing graph breaks. In Inductor's device determination, Codegen, AOTI, and other paths, the device list is hard-coded, excluding new hardware from compilation optimization.

- **Distributed Training and Communication:** Distributed initialization paths only consider NCCL/Gloo backends. New hardware communication libraries cannot pass configurations through standard interfaces. FSDP's gradient synchronization and reduction logic embeds CUDA-specific stream management assumptions. New hardware requires entirely different synchronization semantics.

- **Operator Verification and Capability Declaration:** Information such as dtype support, precision coverage, and known failures for each backend is hard-coded in OpInfo entries in forms like `dtypesIfCUDA`. New hardware integration requires intrusively appending declarations one by one in PyTorch source code, resulting in extremely poor extensibility.

- **Device Runtime Infrastructure:** Underlying capabilities such as device management, DLPack interoperability, and serialization do not use the dispatch mechanism. New hardware cannot access their own implementations through standard registration paths.

### 2.2 Test Case Dimension: Community Tests Cannot Be Reused by New Hardware

Corresponding to CUDA coupling at the feature level, test cases also treat CUDA as the sole validation target, and new hardware is systematically excluded from the test system.

- **Tests Strongly Bound to Hardware:** Test cases extensively use guards such as `@onlyCUDA`, `TEST_CUDA`, `GPU_TYPE`, and `HAS_GPU`. In distributed tests, `DEVICE_TYPE` and `DISTRIBUTED_BACKEND` are hard-coded as CUDA/NCCL, making it completely impossible for new hardware to reuse them.

- **Missing Test Classification:** Test classes lack explicit hardware requirement markers — pure CPU test cases and accelerator-requiring test cases are mixed together, making it impossible for backend developers to distinguish which to run and which to skip.

- **Coarse-Grained Skip Mechanism:** The existing skip is method-level "all-or-nothing," unable to express precise control like "this operator passes for one dtype but skips for another dtype." When integrating new hardware, the only options are between "everything fails in CI" and "everything is skipped."

- **Lack of Safeguards During Refactoring:** During the community's test decoupling refactoring process, there is no automated test count change perception mechanism, creating a risk of tests being silently lost due to mislabeling.

---

---

## III. Task Objectives

Deliver the solutions to the above problems **into PyTorch upstream via RFC / PR**, fundamentally eliminating adaptation code in `torch_npu` that can be resolved through community contributions, while driving the community's test cases toward device-agnostic evolution, enabling NPU to seamlessly reuse community test assets.
