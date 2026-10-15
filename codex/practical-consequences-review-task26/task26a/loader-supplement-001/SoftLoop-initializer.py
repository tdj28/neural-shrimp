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
