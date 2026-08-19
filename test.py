import torch
from torch.nn.attention.flex_attention import flex_attention

def test_npu_flex_attention_autocast() -> None:
    torch.manual_seed(0)

    batch_size = 2
    num_heads = 4
    sequence_length = 16
    head_dim = 32

    shape = (batch_size, num_heads, sequence_length, head_dim)
    query = torch.randn(shape, device="npu", dtype=torch.float32)
    key = torch.randn(shape, device="npu", dtype=torch.float32)
    value = torch.randn(shape, device="npu", dtype=torch.float32)

    # Keep an FP32 result as a numerical reference.
    expected = flex_attention(query, key, value)
    assert expected.device.type == "npu"
    assert expected.dtype == torch.float32

    with torch.autocast(device_type="npu", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("npu")
        actual = flex_attention(query, key, value)

    assert actual.device.type == "npu"
    assert actual.dtype == torch.bfloat16
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual.float(),
        expected,
        rtol=2e-2,
        atol=2e-2,
    )

if __name__ == "__main__":
    test_npu_flex_attention_autocast()
    print("PASS: npu FlexAttention autocast produced a finite bfloat16 result.")