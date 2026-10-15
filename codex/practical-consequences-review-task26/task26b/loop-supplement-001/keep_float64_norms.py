def keep_float64_norms(model):
    """Make a float64 model compute in float64 where Hugging Face code silently drops to float32.

    RMSNorm layers compute in float32 whatever the input dtype (an upcast meant for bf16): their classes are
    patched to compute in the input dtype. Rotary embeddings build their inverse frequencies, cos and sin tables in
    float32 (and CPU and GPU round those tables differently): for float64 inputs the patched classes rebuild them in
    float64 from the configuration, refusing rope types other than the default and frequencies that do not match
    the model's own. float32 inputs keep the original code paths. Attention implementations other than sdpa are
    refused (their kernels may cast to float32)."""
    import torch

    if getattr(model.config, "model_type", "").startswith("qwen3_5"):
        raise SystemExit("Qwen3.5 norms are zero-centred and its linear attention casts to float32: use qwen35_float64")

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return self.weight * (hidden_states * torch.rsqrt(variance + self.variance_epsilon))

    for cls in {type(m) for m in model.modules() if type(m).__name__.endswith("RMSNorm")}:
        cls.forward = forward

    def rotary_forward(self, x, position_ids):
        if x.dtype != torch.float64 or getattr(self, "_inv_freq64", None) is None:
            return self._float32_forward(x, position_ids)
        freqs = position_ids[..., None].to(torch.float64) * self._inv_freq64.to(x.device)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos() * self.attention_scaling, emb.sin() * self.attention_scaling

    for m in model.modules():
        name = type(m).__name__
        if not (name.endswith("RotaryEmbedding") and hasattr(m, "inv_freq")):
            continue
        rope_type = getattr(m, "rope_type", "default")
        if rope_type not in ("default", "llama3"):
            raise SystemExit(f"{name}: rope type {rope_type!r} is not rebuilt in float64")
        cfg = m.config
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        dim = int(head_dim * getattr(cfg, "partial_rotary_factor", 1.0))
        params = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
        base = getattr(cfg, "rope_theta", None) or params["rope_theta"]               # transformers 4 or 5
        inv64 = 1.0 / (float(base) ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
        if rope_type == "llama3":         # Llama 3.1 frequency scaling, as transformers' _compute_llama3_parameters
            factor, low, high = params["factor"], params["low_freq_factor"], params["high_freq_factor"]
            old_len = params["original_max_position_embeddings"]
            wavelen = 2 * math.pi / inv64
            scaled = torch.where(wavelen > old_len / low, inv64 / factor, inv64)
            smooth = (old_len / wavelen - low) / (high - low)
            smoothed = (1 - smooth) * scaled / factor + smooth * scaled
            medium = ~(wavelen < old_len / high) & ~(wavelen > old_len / low)
            inv64 = torch.where(medium, smoothed, scaled)
        own = m.inv_freq.detach().to("cpu", torch.float64)
        if inv64.shape != own.shape or not torch.allclose(inv64, own, rtol=1e-6, atol=0):
            raise SystemExit(f"{name}: rebuilt inverse frequencies do not match the model's")
        m._inv_freq64 = inv64
        cls = type(m)
        if "_float32_forward" not in cls.__dict__:
            cls._float32_forward = cls.forward
            cls.forward = rotary_forward
    # accelerate's device hooks (layers spread over GPUs, CPU offload) keep the forward that a module had when the
    # model was dispatched, before these class patches: a hooked norm or rotary module would still run the float32
    # original (the final norm and the rotary tables of every two-GPU run up to 14 October 2026, measured as a 1e-8
    # wander at fixed points). Rebind each hooked module to its patched class forward, and refuse to continue if any
    # still wraps an original.
    import types

    patched = {cls for cls in {type(m) for m in model.modules()}
               if cls.__name__.endswith("RMSNorm") or "_float32_forward" in cls.__dict__}
    for m in model.modules():
        if type(m) in patched and hasattr(m, "_old_forward"):
            m._old_forward = types.MethodType(type(m).forward, m)
    stale = [n for n, m in model.named_modules() if type(m) in patched and hasattr(m, "_old_forward")
             and getattr(m._old_forward, "__func__", None) is not type(m).forward]
    if stale:
        raise SystemExit(f"float64 patches bypassed by device hooks: {stale[:5]}")
    impl = getattr(model.config, "_attn_implementation", None)
    if impl not in ("sdpa", None):
        raise SystemExit(f"attention implementation {impl!r} may cast to float32; use sdpa for float64 runs")
