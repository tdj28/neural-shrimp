"""Float64 evaluation of Qwen3.5 language models (dense qwen3_5 and mixture-of-experts qwen3_5_moe) for the
soft-token map.

Hugging Face's Qwen3.5 code computes in float32 in several places even when the model is cast to float64:
the zero-centred RMSNorm (x.float(), 1 + w.float()), the gated RMSNorm of the linear-attention blocks, the rotary
tables (built in float32 from float32 inverse frequencies), the gated delta rule (chunked and recurrent: queries,
keys, values, gates and the recurrent state cast to float32), the decay A = -exp(A_log.float()) and the
mixture-of-experts router softmax (dtype=torch.float). prepare() makes each of these compute in float64 for float64
inputs and leaves float32 and bfloat16 paths as they were. The patched functions are rebuilt from the installed
transformers source with exact text replacements, so an upstream change that removes a replaced line fails loudly
instead of silently reverting to float32.

With keep_experts_bf16=True the experts' stored bfloat16 weights stay bfloat16 on the device and each expert's
slice is converted to float64 when it is used. bfloat16 -> float64 is exact, so this equals a float64 model whose
expert weights are the stored values; it keeps a 35B-parameter model on one 141 GB GPU.

Text-only inputs: the three multimodal rotary sections then share one position, and the rotary map equals the
ordinary one with the configured inverse frequencies (partial rotary factor included), rebuilt here in float64.
"""
import inspect
import sys
import textwrap

import torch
import torch.nn.functional as F

REPLACEMENTS = {
    "torch_chunk_gated_delta_rule": [
        ("x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)",
         "x.transpose(1, 2).to(_work_dtype(initial_dtype), memory_format=torch.contiguous_format)"),
    ],
    "torch_recurrent_gated_delta_rule": [
        ("x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)",
         "x.transpose(1, 2).to(_work_dtype(initial_dtype), memory_format=torch.contiguous_format)"),
    ],
}
DELTANET_G = ("g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)",
              "g = (-self.A_log.exp() * F.softplus(a + self.dt_bias) if a.dtype == torch.float64 else "
              "-self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias))")
ROUTER_SOFTMAX = ("softmax(router_logits, dtype=torch.float, dim=-1)",
                  "softmax(router_logits, dtype=_work_dtype(router_logits.dtype), dim=-1)")


def _work_dtype(dtype):
    return torch.float64 if dtype == torch.float64 else torch.float32


def _rebuild(fn, replacements, namespace):
    """Re-create fn from its source with exact replacements (each must occur), decorators dropped."""
    src = textwrap.dedent(inspect.getsource(fn))
    lines = src.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("def "))
    src = "\n".join(lines[start:]) + "\n"
    for old, new in replacements:
        if old not in src:
            raise SystemExit(f"{fn.__qualname__}: expected source line not found (transformers changed?): {old!r}")
        src = src.replace(old, new)
    ns = dict(namespace)
    ns["_work_dtype"] = _work_dtype
    exec(compile(src, f"<float64 {fn.__qualname__}>", "exec"), ns)
    return ns[fn.__name__]


def _classes(module, suffix):
    return [obj for name, obj in vars(module).items() if isinstance(obj, type) and name.endswith(suffix)
            and obj.__module__ == module.__name__]


def _one(module, suffix):
    found = _classes(module, suffix)
    if len(found) != 1:
        raise SystemExit(f"{module.__name__}: expected one class ending in {suffix!r}, found {[c.__name__ for c in found]}")
    return found[0]


def prepare(model, keep_experts_bf16=False):
    """Cast to float64 (experts optionally kept bfloat16) and patch the float32 code paths. Returns a dict of
    what was patched; refuses models that are not Qwen3.5 text models or use an attention kernel other than sdpa."""
    module = sys.modules[type(model.model.layers[0]).__module__]
    if not module.__name__.split(".")[-1] in ("modeling_qwen3_5", "modeling_qwen3_5_moe"):
        raise SystemExit(f"not a Qwen3.5 model: {module.__name__}")
    impl = getattr(model.config, "_attn_implementation", None)
    if impl not in ("sdpa", None):
        raise SystemExit(f"attention implementation {impl!r} computes its softmax in float32; use sdpa")
    patched = []
    # 1. gated delta rule (module-level functions, looked up by name at call time)
    for name, reps in REPLACEMENTS.items():
        fn = getattr(module, name)
        if not getattr(fn, "_float64_patch", False):                 # idempotent: patch once per process
            new = _rebuild(getattr(fn, "__wrapped__", fn), reps, vars(module))
            new._float64_patch = True
            setattr(module, name, new)
        patched.append(name)
    # 2. linear-attention block: the decay term
    delta = _one(module, "GatedDeltaNet")
    if "_float32_forward" not in delta.__dict__:
        delta._float32_forward = delta.forward
        delta.forward = _rebuild(delta.forward, [DELTANET_G], vars(module))
    patched.append(delta.__name__)
    # 3. zero-centred RMSNorm and gated RMSNorm
    norm = _one(module, "RMSNorm")
    gated = _one(module, "RMSNormGated")

    def norm_forward(self, x):
        if x.dtype != torch.float64:
            return self._float32_forward(x)
        return self._norm(x) * (1.0 + self.weight)

    def gated_forward(self, hidden_states, gate):
        if hidden_states.dtype != torch.float64:
            return self._float32_forward(hidden_states, gate)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = self.weight * (hidden_states * torch.rsqrt(variance + self.variance_epsilon))
        return hidden_states * F.silu(gate)

    for cls, fwd in ((norm, norm_forward), (gated, gated_forward)):
        if "_float32_forward" not in cls.__dict__:
            if cls is gated and getattr(cls, "activation", "silu") != "silu":
                raise SystemExit("gated norm activation is not silu")
            cls._float32_forward = cls.forward
            cls.forward = fwd
        patched.append(cls.__name__)
    for m in model.modules():
        if isinstance(m, gated) and m.activation != "silu":
            raise SystemExit(f"gated norm activation {m.activation!r} is not silu")
    # 4. rotary tables in float64 (text positions)
    rot = _one(module, "TextRotaryEmbedding")

    def rotary_forward(self, x, position_ids):
        if x.dtype != torch.float64:
            return self._float32_forward(x, position_ids)
        position_ids = position_ids.expand(3, -1, -1)
        freqs = position_ids[..., None].to(torch.float64) * self._inv_freq64.to(x.device)
        cos = freqs.cos() * self.attention_scaling
        sin = freqs.sin() * self.attention_scaling
        return self.recomposition_frequencies(cos), self.recomposition_frequencies(sin)

    for m in model.modules():
        if isinstance(m, rot):
            params = m.config.rope_parameters
            if params["rope_type"] != "default":
                raise SystemExit(f"rope type {params['rope_type']!r} is not rebuilt in float64")
            head_dim = getattr(m.config, "head_dim", None) or m.config.hidden_size // m.config.num_attention_heads
            dim = int(head_dim * params.get("partial_rotary_factor", 1.0))
            inv64 = 1.0 / (float(params["rope_theta"]) ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
            own = m.inv_freq.detach().to("cpu", torch.float64)
            if inv64.shape != own.shape or not torch.allclose(inv64, own, rtol=1e-6, atol=0):
                raise SystemExit("rebuilt rotary inverse frequencies do not match the model's")
            if sum(m.mrope_section) != dim // 2:
                raise SystemExit(f"mrope sections {m.mrope_section} do not cover {dim // 2} frequencies")
            m._inv_freq64 = inv64
    if "_float32_forward" not in rot.__dict__:
        rot._float32_forward = rot.forward
        rot.forward = rotary_forward
    patched.append(rot.__name__)
    # 5. mixture of experts: router softmax and (optionally) bfloat16 expert weights used in float64
    experts = None
    if module.__name__.endswith("_moe"):
        router = _one(module, "TopKRouter")
        if "_float32_forward" not in router.__dict__:
            router._float32_forward = router.forward
            router.forward = _rebuild(router.forward, [ROUTER_SOFTMAX], vars(module))
        patched.append(router.__name__)
        experts = _one(module, "Experts")

        def experts_forward(self, hidden_states, top_k_index, top_k_weights):
            final = torch.zeros_like(hidden_states)
            with torch.no_grad():
                mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
                hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
            for e in hit:
                e = e[0]
                if e == self.num_experts:
                    continue
                pos, tok = torch.where(mask[e])
                x = hidden_states[tok]
                gate, up = F.linear(x, self.gate_up_proj[e].to(x.dtype)).chunk(2, dim=-1)
                h = F.linear(self.act_fn(gate) * up, self.down_proj[e].to(x.dtype))
                final.index_add_(0, tok, (h * top_k_weights[tok, pos, None]).to(final.dtype))
            return final

        if "_float32_forward" not in experts.__dict__:
            experts._float32_forward = experts.forward
            experts.forward = experts_forward
        patched.append(experts.__name__)
    # 6. dtypes: everything float64, experts optionally kept bfloat16
    for m in model.modules():
        keep = keep_experts_bf16 and experts is not None and isinstance(m, experts)
        for name, p in list(m.named_parameters(recurse=False)):
            if keep:
                if p.dtype != torch.bfloat16:
                    raise SystemExit(f"expert weights are {p.dtype}, not the stored bfloat16")
                continue
            p.data = p.data.to(torch.float64)
        for name, b in list(m.named_buffers(recurse=False)):
            if b.is_floating_point():
                m._buffers[name] = b.to(torch.float64)
    return {"module": module.__name__, "patched": patched, "experts_bf16": bool(keep_experts_bf16 and experts)}


def restore_exact_tensors(model, checkpoint_dir):
    """Re-read every tensor that the checkpoint stores in a dtype other than bfloat16 (Qwen3.5 keeps A_log and the
    gated-norm weights in float32) and assign it exactly: a bfloat16 load (dtype="auto") rounds them, which changes
    the function by about 1e-3. Every such tensor of the language model must map to one parameter, or this refuses.
    Returns the restored parameter names."""
    import glob
    import json
    import struct
    from pathlib import Path

    from safetensors import safe_open

    params = dict(model.named_parameters())
    restored = []
    for f in sorted(glob.glob(str(Path(checkpoint_dir) / "*.safetensors"))):
        with open(f, "rb") as fh:
            header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        keys = [k for k, v in header.items() if k != "__metadata__" and v["dtype"] != "BF16"]
        if not keys:
            continue
        with safe_open(f, framework="pt") as st:
            for k in keys:
                if k.startswith(("model.visual.", "visual.", "mtp.")):
                    continue                                  # vision tower and multi-token heads are not loaded
                name = k.replace("model.language_model.", "model.")
                if name not in params:
                    raise SystemExit(f"stored {header[k]['dtype']} tensor {k} has no parameter in the model")
                t = st.get_tensor(k)
                p = params[name]
                if tuple(t.shape) != tuple(p.shape):
                    raise SystemExit(f"{k}: stored shape {tuple(t.shape)} != parameter {tuple(p.shape)}")
                p.data = t.to(device=p.device, dtype=torch.float64)
                restored.append(name)
    return restored


def non_float64_outputs(model, **inputs):
    """Names of modules whose floating outputs are not float64 in one forward pass (should be empty)."""
    bad, handles = set(), []

    def hook(name):
        def fn(module, args, output):
            outs = output if isinstance(output, (tuple, list)) else (output,)
            for o in outs:
                if torch.is_tensor(o) and o.is_floating_point() and o.dtype != torch.float64:
                    bad.add(name)
        return fn

    for name, m in model.named_modules():
        handles.append(m.register_forward_hook(hook(name)))
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in handles:
            h.remove()
    return sorted(bad)
