class SoftLoop:
    def __init__(self, model, layer, threads, prompt, opener, window, dtype="float64", device="cpu", weights="native"):
        self.sm = elpm.SteeredModel(model, layer, threads, dtype, device, weights)
        self.torch = self.sm.torch
        torch = self.torch
        self.device = self.sm.device                    # the embedding's device when layers are spread ("auto")
        self.model = self.sm.model if device == "auto" else self.sm.model.to(self.device)
        # large models spread over GPUs: the prefix pass of many rows at once needs far more activation memory than
        # the one-token steps, so the prefix caches are built PREFIX_BATCH rows at a time and joined
        self.prefix_batch = PREFIX_BATCH if device == "auto" else None
        self.dtype = self.model.dtype
        self.state_dtype = torch.float64 if dtype == "float64" else torch.float32
        self.E = self.model.get_input_embeddings().weight.detach().to(self.state_dtype)      # (V, d)
        messages = [{"role": "system", "content": ela.PROMPTS[prompt]}, {"role": "user", "content": opener}]
        # reasoning models (Qwen3.5) open a <think> block by default; without thinking the prefix ends at the start
        # of the reply, as for every other model
        extra = {"enable_thinking": False} if str(model).startswith("qwen3.5") else {}
        text = self.sm.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **extra)
        self.prefix_ids = torch.tensor([self.sm.tok(text, add_special_tokens=False)["input_ids"]], device=self.device)
        self.L0 = self.prefix_ids.shape[1]
        self.K = int(window)
        self.cache = None
        self.steer_rows = None
        self.extra = None              # (layer, per-cell rows (n, d)): a second steering block (run TB), or None
        self.extra_rows = None
        # hybrid models (linear-attention layers with recurrent states, e.g. Qwen3.5): the prefix state cannot be
        # cropped back after a step, so it is saved after the prefix pass and restored after every step
        layer_types = getattr(self.model.config.get_text_config(decoder=True), "layer_types", None) or []
        self.hybrid = "linear_attention" in layer_types
        self._saved = None

    def prepare(self, steer_rows):
        """Prefix key-value cache for every row, computed with that row's steering."""
        from transformers import DynamicCache

        torch = self.torch
        B = len(steer_rows)
        self.steer_rows = torch.as_tensor(np.asarray(steer_rows), dtype=self.dtype, device=self.device)[:, None, :]
        self.extra_rows = None
        if self.extra is not None:     # per-cell rows of the second block, repeated like the batch's cells
            rows2 = np.asarray(self.extra[1])
            if B % len(rows2):
                raise ValueError(f"{B} rows are not a multiple of {len(rows2)} second-block cells")
            self.extra_rows = torch.as_tensor(np.concatenate([rows2] * (B // len(rows2))), dtype=self.dtype,
                                              device=self.device)[:, None, :]
        self.cache = DynamicCache(config=self.model.config) if self.hybrid else DynamicCache()
        sub = self.prefix_batch if (self.prefix_batch and not self.hybrid and B > self.prefix_batch) else B
        parts = []
        try:
            for s0 in range(0, B, sub):
                n = min(sub, B - s0)
                cache = self.cache if sub == B else DynamicCache()
                self.sm.steer = self.steer_rows[s0:s0 + n]
                if self.extra_rows is not None:
                    self.sm.steer_extra = {self.extra[0]: self.extra_rows[s0:s0 + n]}
                with torch.no_grad():
                    # logits_to_keep=1: the prefix logits are not needed (all of them cost B x L0 x vocab floats)
                    self.model(input_ids=self.prefix_ids.repeat(n, 1), past_key_values=cache, use_cache=True,
                               logits_to_keep=1)
                parts.append(cache)
        finally:
            self.sm.steer = None
            self.sm.steer_extra = {}
        if sub < B:                       # join the parts layer by layer, freeing each part's tensors as it goes
            for li in range(len(parts[0].layers)):
                k = torch.cat([c.layers[li].keys for c in parts])
                v = torch.cat([c.layers[li].values for c in parts])
                for c in parts:
                    c.layers[li].keys = c.layers[li].values = None
                self.cache.update(k, v, li)
            del parts
        if self.hybrid:
            self._saved = {i: ({k: v.clone() for k, v in layer.conv_states.items() if v is not None},
                               {k: v.clone() for k, v in layer.recurrent_states.items() if v is not None},
                               dict(layer.has_previous_state))
                           for i, layer in enumerate(self.cache.layers) if hasattr(layer, "recurrent_states")}
            if not self._saved:
                raise SystemExit("hybrid model without linear-attention cache layers")
        with torch.no_grad():
            start = self.model.get_input_embeddings()(self.prefix_ids[:, -1:])[0, 0].to(self.state_dtype)
        return start.repeat(B, self.K, 1).clone()                              # initial window (B, K, d)

    def logits(self, X):
        torch = self.torch
        self.sm.steer = self.steer_rows
        if self.extra_rows is not None:
            self.sm.steer_extra = {self.extra[0]: self.extra_rows}
        try:
            with torch.no_grad():
                z = self.model(inputs_embeds=X.to(self.dtype), past_key_values=self.cache, use_cache=True,
                               logits_to_keep=1).logits[:, -1, :].to(self.device, self.state_dtype)
        finally:
            self.sm.steer = None
            self.sm.steer_extra = {}
            self.reset_cache()
        return z

    def reset_cache(self):
        """Put the cache back to the prefix: crop attention layers to the prefix length (argument = length kept);
        for linear-attention layers restore fresh copies of the saved prefix states (their kernels update the
        stored tensors in place)."""
        if not self.hybrid:
            self.cache.crop(self.L0)
            return
        for i, layer in enumerate(self.cache.layers):
            if i in self._saved:
                conv, rec, flags = self._saved[i]
                layer.conv_states = {k: v.clone() for k, v in conv.items()}
                layer.recurrent_states = {k: v.clone() for k, v in rec.items()}
                layer.has_previous_state = dict(flags)
            else:
                layer.crop(self.L0)

    def step(self, X, temps):
        """One soft-decoding step: X (B, K, d) -> shifted window with the new soft embedding appended."""
        torch = self.torch
        z = self.logits(X)
        p = torch.softmax(z / torch.as_tensor(temps, dtype=self.state_dtype, device=self.device)[:, None], dim=-1)
        return torch.cat([X[:, 1:, :], (p @ self.E)[:, None, :]], dim=1), p

    def step_jvp(self, X, temps, V):
        """One step with exact tangents: returns the step of X, p, and the Jacobian-vector products for the tangent
        rows V (B, K, d), by forward-mode AD through the model (math attention kernel). The model runs on a shallow
        copy of the prefix cache, so the stored prefix tensors are never modified (as jvp_rows of soft_family).
        Dual tensors (torch.autograd.forward_ad) rather than torch.func.jvp: accelerate's device hooks need tensors
        with storage when they load offloaded weights, which functorch's wrapped tensors lack."""
        import copy

        import torch.autograd.forward_ad as fwAD
        from torch.nn.attention import SDPBackend, sdpa_kernel

        torch = self.torch
        if self.hybrid:
            raise SystemExit("step_jvp: hybrid (linear-attention) caches are not supported")
        T = torch.as_tensor(temps, dtype=self.state_dtype, device=self.device)[:, None]
        cache = copy.copy(self.cache)
        cache.layers = [copy.copy(layer) for layer in self.cache.layers]

        def new_embedding(inp):
            self.sm.steer = self.steer_rows
            try:
                z = self.model(inputs_embeds=inp.to(self.dtype), past_key_values=cache, use_cache=True,
                               logits_to_keep=1).logits[:, -1, :].to(self.device, self.state_dtype)
            finally:
                self.sm.steer = None
            p = torch.softmax(z / T, dim=-1)
            return p @ self.E, p

        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), fwAD.dual_level():
            y, p = new_embedding(fwAD.make_dual(X, V))
            y, jy = fwAD.unpack_dual(y)
            p = fwAD.unpack_dual(p).primal
        if jy is None:
            raise SystemExit("step_jvp: the tangent was lost in the forward pass")
        return (torch.cat([X[:, 1:, :], y[:, None, :]], dim=1), p,
                torch.cat([V[:, 1:, :], jy[:, None, :]], dim=1))
