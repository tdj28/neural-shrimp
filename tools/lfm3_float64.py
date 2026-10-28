"""Float64 evaluation of LFM3 (LiquidAI/LFM3-1.7B-Base and LFM3-1.7B) for the soft-token map.

LFM3 is not in transformers; its model code (modeling_lfm3.py, configuration_lfm3.py, lfm3_conv.py) comes with each
checkpoint and is loaded with trust_remote_code. It was read in full before first use, and REMOTE_CODE_SHA256 pins
those three files: verify_remote_code() refuses a snapshot whose code differs. The code computes in float32 in four
places even when the model is cast to float64:
- every RMSNorm (operator, feed-forward, sandwich, query-key and final norms) casts its input to float32;
- the rotary tables are built in float32 from float32 inverse frequencies, one set per attention layer type (partial
  rotation is a zero tail of the frequencies);
- apply_rotary_pos_emb casts queries and keys to float32 ("like the reference implementation");
- the eager attention softmax runs in float32.
prepare() makes the first three compute in float64 for float64 inputs (other dtypes keep the original code) and refuses
any attention implementation other than sdpa, which computes in the input dtype. The width-7 short convolution has a
Triton kernel only for 16- and 32-bit CUDA inputs, and the hub causal_conv1d kernels need the absent `kernels`
package; float64 inputs take the PyTorch path, which prepare() checks.
"""
import hashlib
import sys
from pathlib import Path

import torch

REMOTE_CODE_SHA256 = {
    "modeling_lfm3.py": "e8cd5254ea99ce6bd731b57b10908c2e8b5010ce8315341f895a38484885eeab",
    "configuration_lfm3.py": "176cced35657aedfea7d7002221360582d1c05fcaca882c21508bc3a73b282e5",
    "lfm3_conv.py": "80d4a985382a22774212759248a6b4c737bfbf9c2a47b3dd7f797dda748e4bb4",
}


def verify_remote_code(snapshot):
    """Refuse a checkpoint snapshot whose model code is not the reviewed code."""
    snapshot = Path(snapshot)
    for name, digest in REMOTE_CODE_SHA256.items():
        path = snapshot / name
        if not path.exists():
            raise SystemExit(f"LFM3 snapshot {snapshot} lacks {name}")
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != digest:
            raise SystemExit(f"LFM3 {name} differs from the reviewed code: sha256 {got}")


def _norm_forward(self, hidden_states):
    if hidden_states.dtype != torch.float64:
        return self._float32_forward(hidden_states)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    return self.weight * (hidden_states * torch.rsqrt(variance + self.variance_epsilon))


def _rotary_forward(self, x, position_ids, layer_type):
    if x.dtype != torch.float64:
        return self._float32_forward(x, position_ids, layer_type)
    freqs = position_ids[..., None].to(torch.float64) * self._inv_freq64[layer_type].to(x.device)
    emb = torch.cat((freqs, freqs), dim=-1)
    scale = getattr(self, f"{layer_type}_attention_scaling")
    return emb.cos() * scale, emb.sin() * scale


def _inv_freq64(config, layer_type):
    """LFM3's default inverse frequencies with the zero-tail partial rotation (compute_default_rope_parameters), in
    float64."""
    params = config.rope_parameters[layer_type]
    if params["rope_type"] != "default":
        raise SystemExit(f"LFM3 rope type {params['rope_type']!r} ({layer_type}) is not rebuilt in float64")
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    angles = int(params.get("partial_rotary_factor", 1.0) * dim) // 2
    inv = 1.0 / (float(params["rope_theta"]) ** (torch.arange(0, 2 * angles, 2, dtype=torch.float64) / dim))
    return torch.cat((inv, torch.zeros(dim // 2 - angles, dtype=torch.float64)))


def prepare(model):
    """Patch a float64 LFM3 model's float32 code paths. Returns a dict of what was patched."""
    layer0 = model.model.layers[0]
    module = sys.modules[type(layer0).__module__]
    if module.__name__.split(".")[-1] != "modeling_lfm3":
        raise SystemExit(f"not an LFM3 model: {module.__name__}")
    if model.dtype != torch.float64:
        raise SystemExit(f"cast the model to float64 before prepare (it is {model.dtype})")
    impl = getattr(model.config, "_attn_implementation", None)
    if impl != "sdpa":
        raise SystemExit(f"attention implementation {impl!r} may cast to float32; use sdpa for float64 runs")
    report = {"module": module.__name__, "attn_implementation": impl}

    norm_cls = module.Lfm3RMSNorm
    if "_float32_forward" not in norm_cls.__dict__:
        norm_cls._float32_forward = norm_cls.forward
        norm_cls.forward = _norm_forward
    report["norms"] = sum(isinstance(m, norm_cls) for m in model.modules())
    stray = sorted({type(m).__name__ for m in model.modules() if type(m).__name__.endswith("Norm")
                    and not isinstance(m, norm_cls)})
    if stray:
        raise SystemExit(f"unpatched normalization classes: {stray}")

    rotaries = [m for m in model.modules() if isinstance(m, module.Lfm3RotaryEmbedding)]
    if len(rotaries) != 1:
        raise SystemExit(f"expected one rotary embedding, found {len(rotaries)}")
    rot = rotaries[0]
    rot._inv_freq64 = {}
    for layer_type in rot.layer_types:
        inv = _inv_freq64(model.config, layer_type)
        own = getattr(rot, f"{layer_type}_inv_freq").detach().to("cpu", torch.float64)
        if inv.shape != own.shape or not torch.allclose(inv, own, rtol=1e-6, atol=0):
            raise SystemExit(f"rebuilt {layer_type} inverse frequencies do not match the model's")
        if getattr(rot, f"{layer_type}_attention_scaling") != 1.0:
            raise SystemExit(f"{layer_type}: attention scaling other than 1 is not expected for the default rope")
        rot._inv_freq64[layer_type] = inv
    rot_cls = type(rot)
    if "_float32_forward" not in rot_cls.__dict__:
        rot_cls._float32_forward = rot_cls.forward
        rot_cls.forward = _rotary_forward
    report["rotary_layer_types"] = list(rot.layer_types)

    if not hasattr(module, "_float32_apply_rotary_pos_emb"):
        original = module.apply_rotary_pos_emb
        rotate_half = module.rotate_half

        def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
            if q.dtype != torch.float64:
                return original(q, k, cos, sin, unsqueeze_dim)
            if cos.dtype != torch.float64 or sin.dtype != torch.float64 or k.dtype != torch.float64:
                raise RuntimeError("float64 queries with rotary tables or keys in another dtype")
            cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
            return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

        module._float32_apply_rotary_pos_emb = original
        module.apply_rotary_pos_emb = apply_rotary_pos_emb

    convs = [m for m in model.modules() if isinstance(m, module.Lfm3ShortConv)]
    for m in convs:
        w = m.conv.weight
        x = torch.zeros(1, w.shape[0], 2, dtype=torch.float64, device=w.device)
        if w.dtype != torch.float64 or module.can_use_lfm3_conv(x, w):
            raise SystemExit("an LFM3 convolution would run a non-float64 kernel")
    report["convolutions"] = len(convs)
    report["conv_kernel_sizes"] = sorted({m.conv_kernel_size for m in convs})

    # accelerate's device hooks keep the forward a module had at dispatch (see keep_float64_norms): refuse them
    hooked = [n for n, m in model.named_modules() if hasattr(m, "_old_forward")]
    if hooked:
        raise SystemExit(f"LFM3 float64 path supports one device only; hooked modules: {hooked[:3]}")
    return report
