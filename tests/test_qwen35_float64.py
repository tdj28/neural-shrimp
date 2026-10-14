"""Tests for tools/qwen35_float64.py and the soft loop's hybrid-cache handling, on a tiny random Qwen3.5 MoE model:
every module computes in float64, the patched float64 forward agrees with the original float32 forward to float32
accuracy, bfloat16 experts used slice by slice equal a float64 copy, and a cached soft-loop step with the restored
prefix state equals a full recomputation and repeats bit for bit."""
import copy
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import qwen35_float64 as q35  # noqa: E402


def tiny_moe(dtype):
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

    cfg = Qwen3_5MoeTextConfig(
        vocab_size=97, hidden_size=64, num_hidden_layers=4, moe_intermediate_size=32,
        shared_expert_intermediate_size=32, num_experts=8, num_experts_per_tok=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"], max_position_embeddings=256,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [1, 1, 0], "mrope_interleaved": True})
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = Qwen3_5MoeForCausalLM(cfg).eval()
    with torch.no_grad():                       # random init leaves the zero-centred norm weights at 0: perturb
        for name, p in model.named_parameters():
            if "norm" in name or "A_log" in name or "dt_bias" in name:
                p.add_(0.1 * torch.randn_like(p))
    for p in model.parameters():                # parameters only, as from_pretrained(dtype=...) leaves buffers alone
        p.data = p.data.to(dtype)
    return model


@pytest.fixture(scope="module")
def models():
    base = tiny_moe(torch.bfloat16)
    m32 = copy.deepcopy(base).to(torch.float32)
    m64 = copy.deepcopy(base)
    info = q35.prepare(m64, keep_experts_bf16=True)
    ref = copy.deepcopy(base).to(torch.float64)
    q35.prepare(ref, keep_experts_bf16=False)
    return base, m32, m64, ref, info


def test_float64_everywhere_and_patched(models):
    _, _, m64, _, info = models
    assert info["experts_bf16"] and "Qwen3_5MoeExperts" in info["patched"]
    ids = torch.tensor([[1, 5, 9, 13, 17, 21, 25, 29, 33, 37]])
    assert q35.non_float64_outputs(m64, input_ids=ids) == []
    experts = [m for m in m64.modules() if type(m).__name__ == "Qwen3_5MoeExperts"]
    assert experts and all(m.gate_up_proj.dtype == torch.bfloat16 for m in experts)


def test_agrees_with_float32_and_with_float64_experts(models):
    _, m32, m64, ref, _ = models
    ids = torch.tensor([[1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45]])
    with torch.no_grad():
        z64 = m64(input_ids=ids).logits
        z32 = m32(input_ids=ids).logits
        zref = ref(input_ids=ids).logits
    assert z64.dtype == torch.float64
    assert torch.allclose(z64, zref, rtol=0, atol=1e-12)            # bf16 experts used exactly
    rel = float((z64 - z32.double()).abs().max() / z64.abs().max())
    assert rel < 1e-4, rel                                           # same function, float32 rounding only


def test_hybrid_cache_restore(models):
    import explore_llm_soft_loop as esl

    _, _, m64, _, _ = models
    loop = object.__new__(esl.SoftLoop)
    loop.torch, loop.model, loop.device = torch, m64, torch.device("cpu")
    loop.dtype, loop.state_dtype = torch.float64, torch.float64
    loop.E = m64.get_input_embeddings().weight.detach()
    loop.sm = types.SimpleNamespace(steer=None)
    loop.prefix_ids = torch.tensor([[1, 5, 9, 13, 17, 21, 25, 29, 33, 37]])
    loop.L0, loop.K, loop.hybrid, loop._saved = loop.prefix_ids.shape[1], 1, True, None
    X0 = loop.prepare(np.zeros((3, 64)))
    rng = np.random.default_rng(0)
    X = X0 + torch.as_tensor(rng.normal(size=X0.shape) * 0.05)
    z1 = loop.logits(X)
    z2 = loop.logits(X)                                              # after the restore: bit for bit
    assert torch.equal(z1, z2)
    emb = m64.get_input_embeddings()(loop.prefix_ids).repeat(3, 1, 1)
    with torch.no_grad():
        full = m64(inputs_embeds=torch.cat([emb, X], dim=1)).logits[:, -1, :]
    assert float((z1 - full).abs().max() / full.abs().max()) < 1e-12
    Y, p = loop.step(X, np.array([0.7, 1.0, 1.5]))
    assert Y.shape == X.shape and torch.allclose(p.sum(-1), torch.ones(3, dtype=torch.float64))


def test_orbit_paths_refuse_hybrid():
    import explore_llm_soft_loop as esl

    loop = types.SimpleNamespace(hybrid=True)
    with pytest.raises(SystemExit):
        esl.SoftOrbit(loop, np.zeros(4), 1.0)


def test_restore_exact_tensors(tmp_path, models):
    from safetensors.torch import save_file

    _, _, m64, _, _ = models
    name = "model.layers.0.linear_attn.A_log"
    p = dict(m64.named_parameters())[name]
    exact = torch.randn(p.shape, dtype=torch.float32)
    save_file({"model.language_model.layers.0.linear_attn.A_log": exact,
               "model.language_model.layers.0.mlp.gate.weight": torch.zeros(2, dtype=torch.bfloat16)},
              str(tmp_path / "model-00001.safetensors"))
    restored = q35.restore_exact_tensors(m64, tmp_path)
    assert restored == [name] and torch.equal(p.data, exact.double())
    save_file({"model.language_model.layers.0.nonexistent": torch.ones(3)}, str(tmp_path / "model-00002.safetensors"))
    with pytest.raises(SystemExit):
        q35.restore_exact_tensors(m64, tmp_path)
