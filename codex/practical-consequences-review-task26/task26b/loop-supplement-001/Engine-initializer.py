    def __init__(self, model, threads, dtype="float32", device=None, weights="native"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(int(threads))
        self.torch = torch
        self.name = MODELS[model]
        self.weights = weights
        self.tok = AutoTokenizer.from_pretrained(self.name, cache_dir=CACHE)
        if weights not in ("native", "bf16"):
            raise SystemExit(f"weights must be native or bf16, not {weights!r}")
        if (weights == "bf16" or device == "auto") and (dtype != "float64" or model.startswith("qwen3.5")):
            raise SystemExit("bf16 weight storage and device 'auto' are for float64 dense models only")
        if weights == "bf16" or device == "auto":
            # large dense models: decoder linear weights stay in their stored bfloat16 and compute in float64 (exact,
            # see upcast_linear_forward); device "auto" spreads the layers over all GPUs (accelerate)
            upcast_linear_forward()
            kwargs = {}
            if device == "auto":
                # balanced over the GPUs, leaving AUTO_RESERVE_GIB free on each for what grows after placement: the
                # float64 embedding and output layer, the per-row prefix caches and the float64 weight copies
                gib = 2 ** 30
                total = [torch.cuda.get_device_properties(i).total_memory / gib for i in range(torch.cuda.device_count())]
                budget = {i: f"{int(max(t - AUTO_RESERVE_GIB, 0.5 * t))}GiB" for i, t in enumerate(total)}
                kwargs = {"device_map": "balanced", "max_memory": budget}
            self.model = AutoModelForCausalLM.from_pretrained(self.name, dtype=torch.bfloat16, cache_dir=CACHE,
                                                              **kwargs)
            if device not in (None, "auto"):
                self.model = self.model.to(device)
            if weights == "bf16":
                self.bf16_bytes = float64_except_decoder_linears(self.model)
            else:
                self.model = self.model.to(torch.float64)
        elif dtype == "float64" and model.startswith("qwen3.5"):
            # Qwen3.5 (hybrid attention, optionally mixture of experts): load the stored bfloat16 weights, then cast
            # and patch the float32 code paths (tools/qwen35_float64.py); experts stay bfloat16 and are used in
            # float64 slice by slice (exact), so a 35B MoE fits one 141 GB GPU
            import qwen35_float64

            # host first, then the device (device_map would need accelerate); the 35B MoE needs about 72 GB of
            # host RAM for the bfloat16 checkpoint, so its pods ask for at least 128 GB
            self.model = AutoModelForCausalLM.from_pretrained(self.name, dtype="auto", cache_dir=CACHE)
            if device is not None:
                self.model = self.model.to(device)
            self.float64_patch = qwen35_float64.prepare(self.model, keep_experts_bf16="a3b" in model)
            repo = Path(CACHE) / ("models--" + self.name.replace("/", "--"))
            ckpt = repo / "snapshots" / (repo / "refs" / "main").read_text().strip()      # the pinned revision
            self.float64_patch["restored_exact"] = qwen35_float64.restore_exact_tensors(self.model, ckpt)
        elif dtype == "float64" and device is not None and str(device).startswith("cuda"):
            # large models: load in the checkpoint's own dtype, move to the GPU, then cast there (bf16 -> float64 is
            # exact, so the weights equal a direct float64 load; host memory needs only the stored size)
            self.model = AutoModelForCausalLM.from_pretrained(self.name, dtype="auto", cache_dir=CACHE)
            self.model = self.model.to(device).to(torch.float64)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.name, dtype=getattr(torch, dtype), cache_dir=CACHE)
        self.model.eval()
        self.model.requires_grad_(False)                # inference only; forward-mode JVPs need no parameter graph
        if dtype == "float64" and not model.startswith("qwen3.5"):
            keep_float64_norms(self.model)
