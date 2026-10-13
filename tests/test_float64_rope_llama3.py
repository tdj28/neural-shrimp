import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import explore_llm_attractors as ela  # noqa: E402


def tiny_llama3(dtype):
    cfg = transformers.LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=131072,
                                   rope_theta=500000.0,
                                   rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                                 "high_freq_factor": 4.0, "original_max_position_embeddings": 8192},
                                   attn_implementation="sdpa")
    torch.manual_seed(0)
    return transformers.LlamaForCausalLM(cfg).to(dtype)


def test_llama3_rope_is_rebuilt_in_float64_and_matches_float32():
    model = tiny_llama3(torch.float64)
    ela.keep_float64_norms(model)
    rot = [m for m in model.modules() if type(m).__name__.endswith("RotaryEmbedding")][0]
    own = rot.inv_freq.detach().to(torch.float64)
    assert rot._inv_freq64.dtype == torch.float64
    assert torch.allclose(rot._inv_freq64, own, rtol=1e-6, atol=0)
    # the scaling actually applies: the lowest frequencies are divided by the factor
    plain = 1.0 / (500000.0 ** (torch.arange(0, 16, 2, dtype=torch.float64) / 16))
    assert torch.allclose(rot._inv_freq64[-1], plain[-1] / 8.0, rtol=1e-12)
    ids = torch.arange(5)[None]
    x = torch.zeros(1, 5, 64, dtype=torch.float64)
    cos64, sin64 = rot(x, ids)
    assert cos64.dtype == torch.float64
    cos32, sin32 = rot._float32_forward(x.float(), ids)
    assert torch.allclose(cos64.float(), cos32, atol=1e-6) and torch.allclose(sin64.float(), sin32, atol=1e-6)
    out = model(torch.tensor([[1, 2, 3]])).logits
    assert out.dtype == torch.float64 and torch.isfinite(out).all()
