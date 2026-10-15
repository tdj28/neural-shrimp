class SteeredModel:
    """Model with an additive per-sequence steering vector on the output of one decoder layer."""

    def __init__(self, model, layer, threads, dtype="float32", device="cpu", weights="native"):
        self.engine = ela.Engine(model, threads, dtype, device, weights)
        self.torch = self.engine.torch
        if device == "auto":                    # layers spread over GPUs: inputs go to the embedding's device
            self.model = self.engine.model
            self.device = self.model.get_input_embeddings().weight.device
        else:
            self.device = self.torch.device(device)
            self.model = self.engine.model.to(self.device)
        self.tok = self.engine.tok
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.layer = int(layer)
        self.steer = None                                  # tensor (batch, 1, d) or None
        self.model.model.layers[self.layer].register_forward_hook(self._hook)
