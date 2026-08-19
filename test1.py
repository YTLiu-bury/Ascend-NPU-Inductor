"""Framework-only acceptance test for NPU FlexAttention AMP/HOP dispatch.

The test intentionally uses FakeTensor plus ``torch.compile`` with a recording
AOTAutograd backend.  The backend returns a boxed ``gm.forward`` directly: it
records Dynamo/AOTAutograd graph products but never invokes Inductor lowering,
Triton, CANN, autotune, or a generated NPU kernel.
"""

from __future__ import annotations

from collections.abc import Callable
import operator
from typing import Any, Literal

import torch
import torch_npu  # noqa: F401: initializes the torch_npu framework registrations
from functorch.compile import make_boxed_func
from torch._C import DispatchKey
from torch._dynamo.backends.common import aot_autograd
from torch._higher_order_ops.flex_attention import (
    flex_attention as flex_attention_hop,
    flex_attention_backward as flex_attention_backward_hop,
)
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx import GraphModule, Node
from torch.nn.attention.flex_attention import flex_attention


Stage = Literal["forward", "backward"]


class _GraphRecorder:
    """A no-codegen AOT backend that preserves the captured FX graph products."""

    def __init__(self) -> None:
        self.graphs: dict[Stage, list[GraphModule]] = {"forward": [], "backward": []}
        self.example_inputs: dict[Stage, list[list[Any]]] = {
            "forward": [],
            "backward": [],
        }

    def compiler(self, stage: Stage) -> Callable[[GraphModule, list[Any]], Callable]:
        def _record_and_run(gm: GraphModule, example_inputs: list[Any]) -> Callable:
            self.graphs[stage].append(gm)
            self.example_inputs[stage].append(example_inputs)
            return make_boxed_func(gm.forward)

        return _record_and_run


def _resolve_get_attr(gm: GraphModule, value: Any) -> Any:
    if isinstance(value, Node) and value.op == "get_attr":
        return getattr(gm, value.target)
    return value


def _graph_modules_in_value(gm: GraphModule, value: Any) -> list[GraphModule]:
    """Find traced score/mask/joint GraphModules referenced by a HOP node."""
    value = _resolve_get_attr(gm, value)
    if isinstance(value, GraphModule):
        return [value]
    if isinstance(value, (tuple, list)):
        return [submodule for item in value for submodule in _graph_modules_in_value(gm, item)]
    if isinstance(value, dict):
        return [
            submodule
            for item in value.values()
            for submodule in _graph_modules_in_value(gm, item)
        ]
    return []


def _hop_nodes(gm: GraphModule, hop: Any) -> list[Node]:
    return [
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and node.target == hop
    ]


def _getitem_indices_from_hop(gm: GraphModule, hop_node: Node) -> set[int]:
    """Return tuple fields of a HOP result that are consumed in this FX graph."""
    return {
        node.args[1]
        for node in gm.graph.nodes
        if node.op == "call_function"
        and node.target == operator.getitem
        and node.args[0] is hop_node
        and isinstance(node.args[1], int)
    }


def _assert_npu_fake_tensor(tensor: torch.Tensor, dtype: torch.dtype) -> None:
    assert isinstance(tensor, FakeTensor)
    assert tensor.device.type == "npu"
    assert tensor.dtype == dtype


def test_npu_flex_attention_autocast_fake_tensor_graph_capture() -> None:
    """Validate NPU Device/AMP/HOP graph semantics without backend codegen.

    AOTAutograd is part of the framework contract of this requirement, so the
    test uses ``torch.compile``.  Its custom backend only records GraphModules
    and returns their boxed eager ``forward`` method; compiled NPU code is
    never generated or executed.
    """
    assert flex_attention_hop.has_kernel_for_dispatch_key(
        DispatchKey.AutocastPrivateUse1
    ), "torch_npu must register the forward NPU autocast HOP"
    assert flex_attention_backward_hop.has_kernel_for_dispatch_key(
        DispatchKey.AutocastPrivateUse1
    ), "torch_npu must register the backward NPU autocast HOP"

    recorder = _GraphRecorder()
    no_codegen_backend = aot_autograd(
        fw_compiler=recorder.compiler("forward"),
        bw_compiler=recorder.compiler("backward"),
    )
    compiled_flex_attention = torch.compile(
        flex_attention,
        backend=no_codegen_backend,
        fullgraph=True,
    )

    fake_mode = FakeTensorMode()
    with fake_mode:
        shape = (2, 4, 16, 32)
        query = torch.empty(
            shape, device="npu", dtype=torch.float32, requires_grad=True
        )
        key = torch.empty(
            shape, device="npu", dtype=torch.float32, requires_grad=True
        )
        value = torch.empty(
            shape, device="npu", dtype=torch.float32, requires_grad=True
        )
        for tensor in (query, key, value):
            _assert_npu_fake_tensor(tensor, torch.float32)

        with torch.autocast(device_type="npu", dtype=torch.bfloat16):
            output = compiled_flex_attention(query, key, value)

        _assert_npu_fake_tensor(output, torch.bfloat16)
        assert output.shape == query.shape

        # Run fake backward to make AOTAutograd produce the backward FX graph.
        output.sum().backward()
        for tensor in (query, key, value):
            assert tensor.grad is not None
            _assert_npu_fake_tensor(tensor.grad, torch.float32)
            assert tensor.grad.shape == tensor.shape

    # Intermediate graph artifacts: the graph capture must contain both HOPs,
    # their traced subgraphs, and HOP metadata on NPU FakeTensors.
    assert recorder.graphs["forward"], "AOTAutograd forward graph was not captured"
    assert recorder.graphs["backward"], "AOTAutograd backward graph was not captured"

    forward_hops = [
        node
        for gm in recorder.graphs["forward"]
        for node in _hop_nodes(gm, flex_attention_hop)
    ]
    backward_hops = [
        node
        for gm in recorder.graphs["backward"]
        for node in _hop_nodes(gm, flex_attention_backward_hop)
    ]
    assert forward_hops, "forward flex_attention HOP did not enter the FX graph"
    assert backward_hops, "backward flex_attention HOP did not enter the FX graph"

    for gm in recorder.graphs["forward"]:
        for hop_node in _hop_nodes(gm, flex_attention_hop):
            # The score_mod and mask_mod are materialized as FX GraphModule
            # arguments, rather than being executed outside the captured graph.
            assert len(_graph_modules_in_value(gm, hop_node.args)) >= 2
            # ``out`` and ``logsumexp`` are materialized from the HOP tuple and
            # consumed by the captured forward/autograd graph.
            assert {0, 1}.issubset(_getitem_indices_from_hop(gm, hop_node))

    for gm in recorder.graphs["backward"]:
        for hop_node in _hop_nodes(gm, flex_attention_backward_hop):
            # Backward carries traced forward and joint graphs.  These are the
            # key intermediate products needed before any NPU lowering begins.
            assert len(_graph_modules_in_value(gm, hop_node.args)) >= 2
            # ``dq``, ``dk`` and ``dv`` must all flow out of the backward HOP
            # as graph values before the autocast-to-leaf-gradient conversions.
            assert {0, 1, 2}.issubset(_getitem_indices_from_hop(gm, hop_node))


if __name__ == "__main__":
    test_npu_flex_attention_autocast_fake_tensor_graph_capture()
    print("PASS: NPU FlexAttention FakeTensor AOT graph capture and AMP/HOP metadata.")
