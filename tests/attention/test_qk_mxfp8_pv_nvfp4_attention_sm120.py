"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import torch
import torch.nn.functional as F

import flashinfer
from flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120 import (
    nvfp4_prefill_convert,
    nvfp4_side_dequant,
)
from flashinfer.utils import is_sm120a_supported, is_sm121a_supported


def _require_sm120():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if not (is_sm120a_supported(device) or is_sm121a_supported(device)):
        pytest.skip("SM120 or SM121 GPU is required")


def _reference_attention(q, k, v, causal, sm_scale):
    num_qo_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    # Match the public preprocessing recipe. K centering leaves the softmax
    # probabilities unchanged but changes the LSE by a row-wise constant.
    k = k - k.mean(dim=-2, keepdim=True)
    if num_qo_heads != num_kv_heads:
        group_size = num_qo_heads // num_kv_heads
        k = k.repeat_interleave(group_size, dim=1)
        v = v.repeat_interleave(group_size, dim=1)

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale
    if causal:
        qo_len, kv_len = q.shape[2], k.shape[2]
        qo_idx = torch.arange(qo_len, device=q.device)[:, None]
        kv_idx = torch.arange(kv_len, device=q.device)[None, :]
        scores.masked_fill_(kv_idx > qo_idx + kv_len - qo_len, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float()), torch.logsumexp(scores, dim=-1)


def _logical_ue8m0_codes(scale, num_tokens, head_dim):
    """Undo the SM120 scale-factor byte layout for MXFP8 tensors."""
    token = torch.arange(num_tokens, device=scale.device)
    token_local = token % 128
    base = (
        (token // 128) * (head_dim * 4) + (token_local % 32) * 16 + (token_local // 32) * 4
    )
    flat = scale.flatten(start_dim=2)
    return torch.stack(
        [flat[:, :, base + (index // 4) * 512 + index % 4] for index in range(head_dim // 32)],
        dim=-1,
    )


def _expected_mxfp8(x):
    blocks = x.float().reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    amax = blocks.abs().amax(dim=-1)
    normalized = amax / 448.0
    codes = torch.where(
        amax == 0,
        torch.zeros_like(amax, dtype=torch.int32),
        torch.ceil(torch.log2(normalized)).to(torch.int32) + 127,
    )
    scales = torch.where(
        codes == 0,
        torch.zeros_like(amax),
        torch.exp2(codes.float() - 127),
    )
    normalized_blocks = torch.where(
        scales[..., None] == 0,
        torch.zeros_like(blocks),
        blocks / scales[..., None],
    )
    quantized = normalized_blocks.to(torch.float8_e4m3fn).float().reshape_as(x.float())
    return quantized, codes.to(torch.uint8)


def _k_physical_to_logical_tokens(num_tokens, device):
    physical = torch.arange(num_tokens, device=device)
    local = physical % 128
    residue = local % 32
    logical_local = (
        (local // 32) * 32 + (residue // 8) * 2 + ((residue % 8) // 2) * 8 + residue % 2
    )
    return (physical // 128) * 128 + logical_local


@pytest.mark.parametrize(
    (
        "num_qo_heads",
        "num_kv_heads",
        "qo_len",
        "kv_len",
        "causal",
        "out_dtype",
    ),
    [
        pytest.param(4, 4, 128, 128, False, torch.bfloat16, id="mha-noncausal"),
        pytest.param(4, 4, 256, 256, True, torch.float16, id="mha-causal-fp16"),
        pytest.param(8, 2, 193, 317, True, torch.bfloat16, id="gqa-causal-tails"),
        pytest.param(8, 1, 257, 193, False, torch.bfloat16, id="mqa-rectangular"),
    ],
)
@torch.inference_mode()
def test_qk_mxfp8_pv_nvfp4_attention_sm120_accuracy(
    num_qo_heads,
    num_kv_heads,
    qo_len,
    kv_len,
    causal,
    out_dtype,
):
    _require_sm120()
    torch.manual_seed(42)
    q = torch.randn((1, num_qo_heads, qo_len, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, num_kv_heads, kv_len, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    sm_scale = 128**-0.5

    quantized = flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(q, k, v)
    out, lse = flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(
        *quantized,
        sm_scale=sm_scale,
        causal=causal,
        out_dtype=out_dtype,
        return_lse=True,
        unpadded_q_len=qo_len,
        unpadded_k_len=kv_len,
    )
    ref, ref_lse = _reference_attention(q, k, v, causal, sm_scale)
    assert out.dtype == out_dtype
    out = out[:, :, :qo_len].float()
    lse = lse[:, :, :qo_len]

    assert lse.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert torch.isfinite(lse).all()
    cosine = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    mean_abs_error = (out - ref).abs().mean().item()
    relative_l2 = torch.linalg.vector_norm(out - ref) / torch.linalg.vector_norm(ref)
    max_abs_error = (out - ref).abs().max().item()
    lse_max_error = (lse - ref_lse).abs().max().item()
    assert cosine >= 0.985
    assert mean_abs_error <= 0.07
    assert relative_l2.item() <= 0.20
    assert max_abs_error <= 0.75
    assert lse_max_error <= 0.15


@pytest.mark.parametrize(
    (
        "num_qo_heads",
        "num_kv_heads",
        "qo_len",
        "kv_len",
        "causal",
    ),
    [
        pytest.param(4, 4, 128, 128, False, id="d256-mha-noncausal"),
        pytest.param(4, 4, 256, 256, True, id="d256-mha-causal"),
        pytest.param(8, 2, 256, 256, False, id="d256-gqa-noncausal"),
        pytest.param(8, 2, 256, 317, True, id="d256-gqa-causal-longkv"),
    ],
)
@torch.inference_mode()
def test_qk_mxfp8_pv_nvfp4_attention_sm120_accuracy_d256(
    num_qo_heads,
    num_kv_heads,
    qo_len,
    kv_len,
    causal,
):
    _require_sm120()
    torch.manual_seed(42)
    head_dim = 256
    q = torch.randn((1, num_qo_heads, qo_len, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, num_kv_heads, kv_len, head_dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    sm_scale = head_dim**-0.5

    quantized = flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(q, k, v)
    out, lse = flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(
        *quantized,
        sm_scale=sm_scale,
        causal=causal,
        out_dtype=torch.bfloat16,
        return_lse=True,
        unpadded_q_len=qo_len,
        unpadded_k_len=kv_len,
    )
    ref, ref_lse = _reference_attention(q, k, v, causal, sm_scale)
    assert out.dtype == torch.bfloat16
    out = out[:, :, :qo_len].float()
    lse = lse[:, :, :qo_len]

    assert lse.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert torch.isfinite(lse).all()
    cosine = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    mean_abs_error = (out - ref).abs().mean().item()
    relative_l2 = torch.linalg.vector_norm(out - ref) / torch.linalg.vector_norm(ref)
    max_abs_error = (out - ref).abs().max().item()
    lse_max_error = (lse - ref_lse).abs().max().item()
    assert cosine >= 0.985
    assert mean_abs_error <= 0.07
    assert relative_l2.item() <= 0.20
    assert max_abs_error <= 0.75
    assert lse_max_error <= 0.15


@pytest.mark.parametrize("causal", [False, True])
@torch.inference_mode()
def test_qk_mxfp8_pv_nvfp4_attention_sm120_lse_specialization(causal):
    _require_sm120()
    torch.manual_seed(7)
    q = torch.randn((1, 8, 256, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 2, 256, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    quantized = flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(q, k, v)

    out_only = flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(
        *quantized, causal=causal
    )
    out_with_lse, lse = flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(
        *quantized, causal=causal, return_lse=True
    )

    assert isinstance(out_only, torch.Tensor)
    assert lse.shape == (1, 8, 256)
    torch.testing.assert_close(out_only, out_with_lse, rtol=0, atol=4e-3)


@torch.inference_mode()
def test_qk_mxfp8_pv_nvfp4_attention_sm120_fully_masked_causal_rows():
    _require_sm120()
    torch.manual_seed(9)
    qo_len, kv_len = 257, 65
    q = torch.randn((1, 4, qo_len, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 2, kv_len, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    quantized = flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(q, k, v)

    call_kwargs = {
        "causal": True,
        "unpadded_q_len": qo_len,
        "unpadded_k_len": kv_len,
    }
    out_only = flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(
        *quantized, **call_kwargs
    )
    out, lse = flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(
        *quantized, return_lse=True, **call_kwargs
    )

    empty_rows = qo_len - kv_len
    torch.testing.assert_close(out_only, out, rtol=0, atol=4e-3)
    assert torch.count_nonzero(out[:, :, :empty_rows]) == 0
    assert torch.isposinf(lse[:, :, :empty_rows]).all()

    ref, ref_lse = _reference_attention(q, k, v, True, 128**-0.5)
    actual_valid = out[:, :, empty_rows:qo_len].float()
    ref_valid = ref[:, :, empty_rows:]
    cosine = F.cosine_similarity(
        actual_valid.flatten(), ref_valid.flatten(), dim=0
    ).item()
    assert cosine >= 0.985
    assert (actual_valid - ref_valid).abs().mean().item() <= 0.07
    assert (
        lse[:, :, empty_rows:qo_len] - ref_lse[:, :, empty_rows:]
    ).abs().max() <= 0.15


@torch.inference_mode()
def test_qk_mxfp8_pv_nvfp4_attention_sm120_caller_buffers_and_scale_alias():
    _require_sm120()
    torch.manual_seed(11)
    q = torch.randn((1, 4, 128, 128), device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    quantized = flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(q, k, v)
    out = torch.empty_like(q)
    lse = torch.empty((1, 4, 128), device="cuda", dtype=torch.float32)

    actual_out, actual_lse = flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(
        *quantized,
        softmax_scale=0.1,
        out=out,
        lse=lse,
        return_lse=True,
    )

    assert actual_out.data_ptr() == out.data_ptr()
    assert actual_lse.data_ptr() == lse.data_ptr()
    assert actual_out.dtype == torch.float16
    assert torch.isfinite(actual_out).all()
    assert torch.isfinite(actual_lse).all()


@torch.inference_mode()
def test_qk_mxfp8_pv_nvfp4_attention_sm120_quantized_shapes():
    _require_sm120()
    q = torch.randn((2, 8, 129, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((2, 2, 257, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    q_fp8, k_fp8, v_fp4_t, q_scale, k_scale, v_scale_t = (
        flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(q, k, v)
    )

    assert q_fp8.shape == (2, 8, 256, 128)
    assert k_fp8.shape == (2, 2, 384, 128)
    assert v_fp4_t.shape == (2, 2, 128, 192)
    assert q_scale.shape == (2, 8, 256, 4)
    assert k_scale.shape == (2, 2, 384, 4)
    assert v_scale_t.shape == (2, 2, 128, 24)
    assert q_fp8.dtype == torch.float8_e4m3fn
    assert k_fp8.dtype == torch.float8_e4m3fn
    assert v_fp4_t.dtype == torch.uint8
    assert q_scale.dtype == torch.uint8
    assert k_scale.dtype == torch.uint8
    assert v_scale_t.dtype == torch.float8_e4m3fn


@torch.inference_mode()
def test_qk_mxfp8_pv_nvfp4_attention_sm120_quantized_shapes_d256():
    _require_sm120()
    q = torch.randn((2, 8, 128, 256), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((2, 2, 128, 256), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    q_fp8, k_fp8, v_fp4_t, q_scale, k_scale, v_scale_t = (
        flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(q, k, v)
    )

    assert q_fp8.shape == (2, 8, 128, 256)
    assert k_fp8.shape == (2, 2, 128, 256)
    assert v_fp4_t.shape == (2, 2, 256, 64)
    assert q_scale.shape == (2, 8, 128, 8)
    assert k_scale.shape == (2, 2, 128, 8)
    assert v_scale_t.shape == (2, 2, 256, 8)
    assert q_fp8.dtype == torch.float8_e4m3fn
    assert k_fp8.dtype == torch.float8_e4m3fn
    assert v_fp4_t.dtype == torch.uint8
    assert q_scale.dtype == torch.uint8
    assert k_scale.dtype == torch.uint8
    assert v_scale_t.dtype == torch.float8_e4m3fn


@pytest.mark.parametrize("tensor_layout", [0, 1], ids=["nhd", "hnd"])
@torch.inference_mode()
def test_qk_mxfp8_pv_nvfp4_attention_sm120_scaled_fp4_quant_padded_output(
    tensor_layout,
):
    _require_sm120()
    batch, num_heads, num_tokens, padded_tokens, head_dim = 1, 2, 129, 256, 128
    if tensor_layout == 0:
        input_shape = (batch, num_tokens, num_heads, head_dim)
        output_shape = (batch, padded_tokens, num_heads, head_dim // 2)
        scale_shape = (batch, padded_tokens, num_heads, head_dim // 16)
        tail = (slice(None), slice(num_tokens, None))
    else:
        input_shape = (batch, num_heads, num_tokens, head_dim)
        output_shape = (batch, num_heads, padded_tokens, head_dim // 2)
        scale_shape = (batch, num_heads, padded_tokens, head_dim // 16)
        tail = (slice(None), slice(None), slice(num_tokens, None))

    x = torch.randn(input_shape, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(output_shape, device="cuda", dtype=torch.uint8)
    output_sf = torch.empty(scale_shape, device="cuda", dtype=torch.float8_e4m3fn)
    module = flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120.get_qk_mxfp8_pv_nvfp4_attention_sm120_module()
    module.scaled_fp4_quant(x, output, output_sf, tensor_layout)

    assert torch.count_nonzero(output[tail]) == 0


@pytest.mark.parametrize(
    ("head_dim", "qo_len", "kv_len"),
    [
        pytest.param(128, 129, 257, id="d128"),
        # D256 quantize runs 64-token blocks (scaled_fp8_quant_impl BS=64), so
        # use lengths whose 128-round-up padding is fully kernel-covered.
        pytest.param(256, 128, 256, id="d256"),
    ],
)
@torch.inference_mode()
def test_qk_mxfp8_pv_nvfp4_attention_sm120_mxfp8_recipe(head_dim, qo_len, kv_len):
    _require_sm120()
    torch.manual_seed(19)
    q = torch.randn((1, 2, qo_len, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 1, kv_len, head_dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    q_fp8, k_fp8, _, q_scale, k_scale, _ = (
        flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(q, k, v)
    )
    q_padded = F.pad(q, (0, 0, 0, (-qo_len) % 128))
    k_centered = k - k.mean(dim=-2, keepdim=True)
    k_padded = F.pad(k_centered, (0, 0, 0, (-kv_len) % 128))
    expected_q, expected_q_codes = _expected_mxfp8(q_padded)
    expected_k, expected_k_codes = _expected_mxfp8(k_padded)

    torch.testing.assert_close(q_fp8.float(), expected_q, rtol=0, atol=0)
    torch.testing.assert_close(
        _logical_ue8m0_codes(q_scale, q_fp8.shape[2], head_dim),
        expected_q_codes,
        rtol=0,
        atol=0,
    )

    physical_to_logical = _k_physical_to_logical_tokens(k_fp8.shape[2], k.device)
    torch.testing.assert_close(
        k_fp8.float(), expected_k[:, :, physical_to_logical], rtol=0, atol=0
    )
    torch.testing.assert_close(
        _logical_ue8m0_codes(k_scale, k_fp8.shape[2], head_dim),
        expected_k_codes[:, :, physical_to_logical],
        rtol=0,
        atol=0,
    )

    assert torch.count_nonzero(q_fp8[:, :, q.shape[2] :]) == 0
    assert torch.count_nonzero(k_fp8[:, :, k.shape[2] :]) == 0


@torch.inference_mode()
def test_qk_mxfp8_pv_nvfp4_attention_sm120_rejects_invalid_lengths():
    _require_sm120()
    q = torch.randn((1, 4, 128, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    quantized = flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(q, k, v)

    with pytest.raises(ValueError, match="unpadded_q_len"):
        flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(*quantized, unpadded_q_len=0)
    with pytest.raises(ValueError, match="unpadded_q_len"):
        flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(*quantized, unpadded_q_len=129)
    with pytest.raises(ValueError, match="unpadded_k_len"):
        flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(*quantized, unpadded_k_len=0)
    with pytest.raises(ValueError, match="Specify only one"):
        flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(
            *quantized, sm_scale=0.1, softmax_scale=0.1
        )

    empty = torch.empty((1, 4, 0, 128), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="sequence lengths must be positive"):
        flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(empty, empty, empty)


@torch.inference_mode()
def test_qk_mxfp8_pv_nvfp4_attention_sm120_empty_prequantized_q():
    _require_sm120()
    device = torch.device("cuda")
    q_fp8 = torch.empty((1, 4, 0, 128), device=device, dtype=torch.float8_e4m3fn)
    k_fp8 = torch.empty((1, 2, 128, 128), device=device, dtype=torch.float8_e4m3fn)
    v_fp4_t = torch.empty((1, 2, 128, 64), device=device, dtype=torch.uint8)
    q_scale = torch.empty((1, 4, 0, 4), device=device, dtype=torch.uint8)
    k_scale = torch.empty((1, 2, 128, 4), device=device, dtype=torch.uint8)
    v_scale_t = torch.empty((1, 2, 128, 8), device=device, dtype=torch.float8_e4m3fn)

    out, lse = flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(
        q_fp8,
        k_fp8,
        v_fp4_t,
        q_scale,
        k_scale,
        v_scale_t,
        return_lse=True,
        unpadded_q_len=0,
        unpadded_k_len=128,
    )

    assert out.shape == (1, 4, 0, 128)
    assert lse.shape == (1, 4, 0)
    assert out.numel() == 0
    assert lse.numel() == 0


@torch.inference_mode()
def test_qk_mxfp8_pv_nvfp4_attention_sm120_rejects_invalid_inputs():
    _require_sm120()
    q = torch.randn((1, 4, 128, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 2, 128, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    with pytest.raises(ValueError, match="q must have dtype"):
        flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(q.float(), k, v)
    with pytest.raises(ValueError, match="q must be contiguous"):
        flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(
            q.transpose(-1, -2), k, v
        )
    with pytest.raises(ValueError, match="num_qo_heads"):
        invalid_k = torch.randn((1, 3, 128, 128), device="cuda", dtype=torch.bfloat16)
        flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(
            q, invalid_k, invalid_k
        )
    with pytest.raises(ValueError, match="head_dim must be 128"):
        q64 = q[..., :64].contiguous()
        k64 = k[..., :64].contiguous()
        v64 = v[..., :64].contiguous()
        flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(q64, k64, v64)

    quantized = flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(q, k, v)
    q_fp8, k_fp8, v_fp4_t, q_scale, k_scale, v_scale_t = quantized
    with pytest.raises(ValueError, match="q_scale shape"):
        flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(
            q_fp8,
            k_fp8,
            v_fp4_t,
            q_scale[..., :-1].contiguous(),
            k_scale,
            v_scale_t,
        )
    with pytest.raises(ValueError, match="q_fp8 and k_fp8"):
        flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(
            q_fp8.view(torch.uint8),
            k_fp8,
            v_fp4_t,
            q_scale,
            k_scale,
            v_scale_t,
        )
    with pytest.raises(ValueError, match="out shape"):
        flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(
            *quantized,
            out=torch.empty((1, 4, 127, 128), device="cuda", dtype=torch.bfloat16),
        )
    with pytest.raises(ValueError, match="return_lse=True"):
        flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(
            *quantized,
            lse=torch.empty((1, 4, 128), device="cuda", dtype=torch.float32),
        )


def test_nvfp4_side_dequant_d256():
    """Dequant of a hand-crafted NVFP4 side matches the documented layout bit-exactly."""
    _require_sm120()
    torch.manual_seed(0)
    H, S, D = 2, 130, 256  # S not a multiple of 128 (stride-bug probe)
    F = D // 2 + D // 16  # 144
    global_scale = 1.7
    # Arbitrary fp4 nibble bytes (all nibbles are valid e2m1) + finite e4m3 scale
    # bytes (0x7F is the e4m3fn NaN code, remap it so the scales stay finite).
    sf = torch.randint(0, 255, (H, S, D // 16), dtype=torch.uint8, device="cuda")
    sf[sf == 0x7F] = 0
    data = torch.randint(0, 256, (H, S, D // 2), dtype=torch.uint8, device="cuda")
    side = torch.cat([data, sf.view(torch.uint8)], dim=-1).contiguous()
    out = nvfp4_side_dequant(side, global_scale)
    assert out.shape == (H, S, D) and out.dtype == torch.bfloat16
    # dim 2b = low nibble of data byte b, dim 2b+1 = high nibble; group g's scale
    # multiplies dims [16g, 16g+16).
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
        device=data.device,
    )
    lo = (data & 0x0F).long()
    hi = ((data >> 4) & 0x0F).long()
    per_dim = torch.stack([lut[lo], lut[hi]], dim=-1).reshape(H, S, D)
    sfv = sf.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    ref = (per_dim * sfv * float(global_scale)).to(torch.bfloat16)
    assert torch.equal(out, ref)


def test_nvfp4_side_dequant_d128():
    """Dequant of a hand-crafted NVFP4 side (D128) matches the documented layout bit-exactly."""
    _require_sm120()
    torch.manual_seed(0)
    H, S, D = 4, 64, 128
    F = D // 2 + D // 16  # 72
    global_scale = 1.7
    # Arbitrary fp4 nibble bytes (all nibbles are valid e2m1) + finite e4m3 scale
    # bytes (0x7F is the e4m3fn NaN code, remap it so the scales stay finite).
    sf = torch.randint(0, 255, (H, S, D // 16), dtype=torch.uint8, device="cuda")
    sf[sf == 0x7F] = 0
    data = torch.randint(0, 256, (H, S, D // 2), dtype=torch.uint8, device="cuda")
    side = torch.cat([data, sf.view(torch.uint8)], dim=-1).contiguous()
    out = nvfp4_side_dequant(side, global_scale)
    assert out.shape == (H, S, D) and out.dtype == torch.bfloat16
    # dim 2b = low nibble of data byte b, dim 2b+1 = high nibble; group g's scale
    # multiplies dims [16g, 16g+16).
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
        device=data.device,
    )
    lo = (data & 0x0F).long()
    hi = ((data >> 4) & 0x0F).long()
    per_dim = torch.stack([lut[lo], lut[hi]], dim=-1).reshape(H, S, D)
    sfv = sf.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    ref = (per_dim * sfv * float(global_scale)).to(torch.bfloat16)
    assert torch.equal(out, ref)


def _quantize_to_fp4_bytes(x, sf_bytes):
    """Round BF16 ``x`` (1, H, S, D) to packed NVFP4 data bytes.

    ``sf_bytes`` is ``(H, S, D // 16)`` uint8 holding one e4m3 group scale per
    16 dims. Returns ``(H, S, D // 2)`` uint8 where dim 2b is the low
    nibble and dim 2b+1 the high nibble of data byte b, on the e2m1 grid.
    """
    H, S, D = x.shape[1], x.shape[2], x.shape[3]
    sf = sf_bytes.view(torch.float8_e4m3fn).float().reshape(H, S, D // 16, 1)
    y = x.float().reshape(1, H, S, D // 16, 16) / sf
    bounds = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        dtype=torch.float32,
        device=x.device,
    )
    idx = torch.bucketize(y.abs(), bounds)
    nibble = idx + ((y < 0).to(idx.dtype) * 8)
    byte = nibble[..., 0::2] + (nibble[..., 1::2] << 4)
    return byte.reshape(H, S, D // 2).to(torch.uint8)


def _torch_dequant_side(side):
    """Dequant an NVFP4 cache side (H, S, F) uint8 to float32 (H, S, D)."""
    H, S, F = side.shape
    D = 2 * (F * 8 // 9)
    data = side[..., : D // 2]
    sf = side[..., D // 2 :]
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
        device=side.device,
    )
    lo = (data & 0x0F).long()
    hi = ((data >> 4) & 0x0F).long()
    per_dim = torch.stack([lut[lo], lut[hi]], dim=-1).reshape(H, S, D)
    sfv = sf.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    return per_dim * sfv


def _ref_attention(q, k, v, num_qo_heads, num_kv_heads, causal, sm_scale):
    """Exact float32 GQA attention over (Hq, M, D) q and (Hk, N, D) k/v."""
    if num_qo_heads != num_kv_heads:
        group = num_qo_heads // num_kv_heads
        k = k.repeat_interleave(group, dim=0)
        v = v.repeat_interleave(group, dim=0)
    scores = q @ k.transpose(-2, -1) * sm_scale
    if causal:
        qo_len, kv_len = q.shape[1], k.shape[1]
        qo_idx = torch.arange(qo_len, device=q.device)[:, None]
        kv_idx = torch.arange(kv_len, device=q.device)[None, :]
        valid = kv_idx <= qo_idx + kv_len - qo_len
        scores = scores.masked_fill(~valid, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = probs @ v
        # Fully-masked rows (the kv_len < qo_len prefix) have no valid keys;
        # the kernel emits zeros there.
        return torch.where(valid.any(dim=-1).unsqueeze(0).unsqueeze(-1), out, 0.0)
    return torch.softmax(scores, dim=-1) @ v


def _cosine(a, b):
    return (a @ b) / (a.norm() * b.norm())


@torch.inference_mode()
def test_nvfp4_prefill_convert_matches_exact_attention():
    """Convert NVFP4 cache sides + BF16 q, run fwd; compare vs exact
    attention computed from the same dequantized BF16 K/V (GQA 12/2, D256,
    causal, variable lengths not 128-aligned). Tolerances match the T4
    precision-gate regime (mxfp8/fp4 requant rounding is the only diff)."""
    _require_sm120()
    torch.manual_seed(1)
    Hq, Hk, D, M, N = 12, 2, 256, 300, 131
    q = torch.randn(1, Hq, M, D, device="cuda", dtype=torch.bfloat16)
    k_bf16 = torch.randn(1, Hk, N, D, device="cuda", dtype=torch.bfloat16)
    v_bf16 = torch.randn(1, Hk, N, D, device="cuda", dtype=torch.bfloat16)
    # Build a valid NVFP4 cache side from the same bf16 (quantize in torch:
    # per-group e4m3 scale + e2m1 rounding), so the reference attention and
    # the kernel see identical pre-quantization values.
    def make_side(x, sf_bytes):
        data = _quantize_to_fp4_bytes(x, sf_bytes)
        return torch.cat([data, sf_bytes], dim=-1).contiguous()

    def group_scales(x):
        # Per-16-dim e4m3 scale bytes: group max / 6.0 keeps every |dim/scale|
        # <= 6, the e2m1 range. 0x7F/0xFF are the e4m3fn NaN bytes.
        amax = x.float().abs().reshape(*x.shape[1:-1], x.shape[-1] // 16, 16)
        amax = amax.amax(dim=-1)
        sf = (amax / 6.0).to(torch.float8_e4m3fn).view(torch.uint8)
        sf[(sf == 0x7F) | (sf == 0xFF)] = 0
        return sf.reshape(x.shape[1:-1] + (x.shape[-1] // 16,))

    k_side = make_side(k_bf16, group_scales(k_bf16))
    v_side = make_side(v_bf16, group_scales(v_bf16))

    q_fp8, k_fp8, v_fp4_t, q_scale, k_scale, v_scale_t = nvfp4_prefill_convert(
        k_side, v_side, q[0], 1.0, 1.0, 1.0
    )
    out = flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(
        q_fp8, k_fp8, v_fp4_t, q_scale, k_scale, v_scale_t,
        sm_scale=D ** -0.5, causal=True,
        unpadded_q_len=M, unpadded_k_len=N,
    )
    # Reference: exact attention on the SAME bf16 (dequant of the crafted side)
    k_ref = _torch_dequant_side(k_side)      # (Hk, N, D) float32
    v_ref = _torch_dequant_side(v_side)
    out_ref = _ref_attention(
        q[0].float(), k_ref, v_ref, Hq, Hk, True, D ** -0.5
    )
    cos = _cosine(out[0, :, :M, :].float().flatten(), out_ref[:, :M, :].flatten())
    rel = (out[0, :, :M, :].float() - out_ref[:, :M, :]).norm() / \
          out_ref[:, :M, :].norm()
    assert cos >= 0.99, f"cos={cos:.4f}"
    assert rel.item() <= 0.2, f"rel={rel.item():.4f}"


@torch.inference_mode()
def test_nvfp4_prefill_convert_kv_longer_scales():
    """Convert NVFP4 cache sides + BF16 q in the chunked-prefill regime (KV
    history N > Q chunk M, causal) with non-unity global scales; compare vs
    exact attention on the SAME effective values the kernel sees — dequant K
    x k_scale, dequant V x v_scale, q x q_scale (nvfp4_side_dequant applies the
    global scale; nvfp4_prefill_convert does not center K, so neither does this
    reference). Tolerances match the sibling convert test."""
    _require_sm120()
    torch.manual_seed(1)
    Hq, Hk, D, M, N = 12, 2, 256, 256, 2048
    q_scale, k_scale, v_scale = 1.0, 2.5, 0.75
    q = torch.randn(1, Hq, M, D, device="cuda", dtype=torch.bfloat16)
    k_bf16 = torch.randn(1, Hk, N, D, device="cuda", dtype=torch.bfloat16)
    v_bf16 = torch.randn(1, Hk, N, D, device="cuda", dtype=torch.bfloat16)
    # Build a valid NVFP4 cache side from the same bf16 (per-group e4m3 scale +
    # e2m1 rounding), so the reference and the kernel share pre-quant values.
    def make_side(x, sf_bytes):
        data = _quantize_to_fp4_bytes(x, sf_bytes)
        return torch.cat([data, sf_bytes], dim=-1).contiguous()

    def group_scales(x):
        # Per-16-dim e4m3 scale bytes: group max / 6.0 keeps every |dim/scale|
        # <= 6, the e2m1 range. 0x7F/0xFF are the e4m3fn NaN bytes.
        amax = x.float().abs().reshape(*x.shape[1:-1], x.shape[-1] // 16, 16)
        amax = amax.amax(dim=-1)
        sf = (amax / 6.0).to(torch.float8_e4m3fn).view(torch.uint8)
        sf[(sf == 0x7F) | (sf == 0xFF)] = 0
        return sf.reshape(x.shape[1:-1] + (x.shape[-1] // 16,))

    k_side = make_side(k_bf16, group_scales(k_bf16))
    v_side = make_side(v_bf16, group_scales(v_bf16))

    q_fp8, k_fp8, v_fp4_t, q_scale_t, k_scale_t, v_scale_t = (
        nvfp4_prefill_convert(
            k_side,
            v_side,
            q[0],
            k_scale_global=k_scale,
            v_scale_global=v_scale,
            q_scale_global=q_scale,
        )
    )
    out = flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(
        q_fp8, k_fp8, v_fp4_t, q_scale_t, k_scale_t, v_scale_t,
        sm_scale=D ** -0.5, causal=True,
        unpadded_q_len=M, unpadded_k_len=N,
    )
    # Reference: exact attention on the SAME effective values the kernel sees.
    # nvfp4_side_dequant multiplies the raw dequant by the global scale, so
    # k_eff = dequant(K) x k_scale, v_eff = dequant(V) x v_scale; q_eff = q x
    # q_scale (1.0, a no-op). _torch_dequant_side is the raw float32 dequant.
    q_ref = q[0].float() * q_scale                      # (Hq, M, D) float32
    k_ref = _torch_dequant_side(k_side) * k_scale       # (Hk, N, D) float32
    v_ref = _torch_dequant_side(v_side) * v_scale       # (Hk, N, D) float32
    out_ref = _ref_attention(q_ref, k_ref, v_ref, Hq, Hk, True, D ** -0.5)
    cos = _cosine(out[0, :, :M, :].float().flatten(), out_ref[:, :M, :].flatten())
    rel = (out[0, :, :M, :].float() - out_ref[:, :M, :]).norm() / \
          out_ref[:, :M, :].norm()
    assert cos >= 0.99, f"cos={cos:.4f}"
    assert rel.item() <= 0.2, f"rel={rel.item():.4f}"
