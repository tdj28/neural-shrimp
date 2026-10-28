"""lfm3_float64 on a tiny random LFM3 built from the checkpoint's reviewed code (skipped without the snapshot)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import explore_llm_attractors as ela  # noqa: E402

NAME = "LiquidAI/LFM3-1.7B-Base"
SNAPSHOT = ela.CACHE / ("models--" + NAME.replace("/", "--")) / "snapshots" / ela.REVISIONS[NAME]
pytestmark = pytest.mark.skipif(not SNAPSHOT.exists(), reason="LFM3 snapshot not present")


@pytest.fixture(scope="module")
def tiny():
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    import lfm3_float64

    lfm3_float64.verify_remote_code(SNAPSHOT)
    cfg = AutoConfig.from_pretrained(SNAPSHOT, trust_remote_code=True)
    cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size = 64, 96, 50
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 4, 2, 16
    cfg.layer_types = ["conv", "full_attention", "conv", "sliding_attention"]
    cfg.num_hidden_layers, cfg.sliding_window = 4, 8
    cfg.pad_token_id, cfg.bos_token_id, cfg.eos_token_id = 0, 1, 2
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, attn_implementation="sdpa").eval()
    model = model.to(torch.float64)
    report = lfm3_float64.prepare(model)
    return model, report


def test_prepare_patches_every_float32_path(tiny):
    import torch

    import lfm3_float64

    model, report = tiny
    module = sys.modules[type(model.model.layers[0]).__module__]
    assert report["norms"] == sum(isinstance(m, module.Lfm3RMSNorm) for m in model.modules()) > 0
    assert report["rotary_layer_types"] == ["full_attention", "sliding_attention"]
    assert report["convolutions"] == 2 and report["conv_kernel_sizes"] == [7]
    rot = model.model.rotary_emb
    pos = torch.arange(300)[None]
    for layer_type in rot.layer_types:
        cos, sin = rot(torch.zeros(1, dtype=torch.float64), pos, layer_type)
        freqs = pos[..., None].double() * rot._inv_freq64[layer_type]
        emb = torch.cat((freqs, freqs), dim=-1)
        assert cos.dtype == torch.float64 and torch.equal(cos, emb.cos()) and torch.equal(sin, emb.sin())
    q = torch.randn(1, 4, 3, 16, dtype=torch.float64)
    cos, sin = rot(q, torch.arange(3)[None], "full_attention")
    qe, ke = module.apply_rotary_pos_emb(q, q.clone(), cos, sin)
    ref = q * cos[:, None] + module.rotate_half(q) * sin[:, None]
    assert qe.dtype == torch.float64 and torch.equal(qe, ref) and torch.equal(ke, ref)
    with pytest.raises(SystemExit):
        lfm3_float64.prepare(model.to(torch.float32))
    model.to(torch.float64)


def test_float64_central_differences_converge_like_h_squared(tiny):
    """Successive central differences shrink like h^2 down to h = 1e-4 (float64 rounding takes over only near
    1e-5 at this model's logit scale); a float32 island would put a rounding floor near 1e-7 / h above them."""
    import torch

    model, _ = tiny
    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.normal(size=(1, 6, 64)))
    u = torch.as_tensor(rng.normal(size=(1, 6, 64)))
    u = u / u.norm()
    hs = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
    with torch.no_grad():
        D = [(model(inputs_embeds=x + h * u).logits - model(inputs_embeds=x - h * u).logits) / (2 * h) for h in hs]
    diffs = [float((D[i] - D[i + 1]).norm()) for i in range(len(hs) - 1)]
    ratios = [diffs[i] / diffs[i + 1] for i in range(len(diffs) - 1)]
    expected = [(hs[i] / hs[i + 1]) ** 2 for i in range(len(diffs) - 1)]
    assert all(0.6 * e < r < 1.6 * e for r, e in zip(ratios, expected)), (diffs, ratios)
