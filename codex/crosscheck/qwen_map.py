"""Independent float64 CPU Qwen2.5 soft-token map, using tensor primitives only.

No repository tool or Transformers model implementation is imported. The
independent architecture follows the pinned tensor shapes/configuration. All
reference model evaluations belong to separately committed, bounded run cards.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as functional


SNAPSHOT = Path("/data2/neural-shrimp-data/models/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775")
VECTOR_FILE = Path("/data2/neural-shrimp-data/explore/llm-phase-map/vectors-l12/vectors.npz")
PINNED_SHA256 = {
    "model.safetensors": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    "config.json": "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
    "vectors.npz": "b630bb7bef06bde87f8e6a7ca5e8672f83d24bc0585bf19048e9e494a4357f41",
}
PREFIX_IDS = (
    151644, 8948, 198, 2610, 525, 458, 15235, 7404, 448, 2441,
    15235, 13, 1446, 614, 4583, 11290, 13, 31733, 1910, 311,
    22729, 8820, 498, 1366, 13, 151645, 198, 151644, 872, 198,
    9707, 0, 151645, 198, 151644, 77091, 198,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PrefixCache:
    """Immutable container; K/V tensors must never be changed in place."""

    alpha: torch.Tensor
    layers: tuple
    owner: object


class ManualQwenSoftMap:
    """One-token-window, full-vocabulary expected-embedding map.

    States are float64 CPU tensors of shape(d,) or(batch,d). Alpha and log10 T
    may be Python scalars or differentiable scalar float64 CPU tensors.
    A supplied cache is allowed only for fixed-parameter state calculations;
    omit it for parameter derivatives so prefix dependence remains in the graph.
    """

    def __init__(self, config, weights, prefix_ids, steering_vector, steering_layer=12):
        self.config = dict(config)
        self.d = int(config["hidden_size"])
        self.layers = int(config["num_hidden_layers"])
        self.heads = int(config["num_attention_heads"])
        self.kv_heads = int(config["num_key_value_heads"])
        self.intermediate = int(config["intermediate_size"])
        self.vocab = int(config["vocab_size"])
        self.steering_layer = int(steering_layer)
        self.epsilon = float(config["rms_norm_eps"])
        if self.d % self.heads or self.heads % self.kv_heads:
            raise ValueError("head dimensions must divide exactly")
        self.head_dim = self.d // self.heads
        self.groups = self.heads // self.kv_heads
        if self.head_dim % 2 or not 0 <= self.steering_layer < self.layers:
            raise ValueError("even rotary dimension and valid zero-based steering layer required")
        if config.get("use_sliding_window", False) or not config.get("tie_word_embeddings", False):
            raise ValueError("only full attention and tied word embeddings are implemented")
        if config.get("hidden_act") != "silu" or config.get("attention_dropout", 0.) != 0.:
            raise ValueError("only deterministic SwiGLU architecture is implemented")
        if config.get("rope_scaling") is not None or config.get("partial_rotary_factor", 1.) != 1.:
            raise ValueError("only default full-head rotary embedding is implemented")
        base = float(config["rope_theta"])
        if not base > 0 or not self.epsilon > 0:
            raise ValueError("positive rotary base and RMS epsilon required")
        self.inv_freq = 1. / (base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float64) / self.head_dim))
        self.weights = dict(weights)
        shapes = {"model.embed_tokens.weight": (self.vocab, self.d), "model.norm.weight": (self.d,)}
        for layer in range(self.layers):
            name = f"model.layers.{layer}."
            shapes[name + "input_layernorm.weight"] = (self.d,)
            shapes[name + "post_attention_layernorm.weight"] = (self.d,)
            for projection, rows in [("q", self.d), ("k", self.kv_heads * self.head_dim), ("v", self.kv_heads * self.head_dim)]:
                shapes[name + f"self_attn.{projection}_proj.weight"] = (rows, self.d)
                shapes[name + f"self_attn.{projection}_proj.bias"] = (rows,)
            shapes[name + "self_attn.o_proj.weight"] = (self.d, self.d)
            for projection in ["gate", "up"]:
                shapes[name + f"mlp.{projection}_proj.weight"] = (self.intermediate, self.d)
            shapes[name + "mlp.down_proj.weight"] = (self.d, self.intermediate)
        if set(shapes) != set(self.weights):
            raise ValueError("tensor inventory does not match the explicit tied Qwen architecture")
        for name, shape in shapes.items():
            tensor = self.weights[name]
            if tuple(tensor.shape) != shape or tensor.dtype != torch.float64 or tensor.device.type != "cpu":
                raise ValueError(f"wrong shape, dtype or device: {name}")
            self.weights[name] = tensor.detach()
        self.embedding = self.weights["model.embed_tokens.weight"]
        self.prefix_ids = torch.as_tensor(prefix_ids, dtype=torch.long, device="cpu").clone()
        if self.prefix_ids.ndim != 1 or not len(self.prefix_ids) or torch.any(self.prefix_ids < 0) or torch.any(self.prefix_ids >= self.vocab):
            raise ValueError("nonempty in-vocabulary prefix IDs required")
        self.vector = torch.as_tensor(steering_vector, dtype=torch.float64, device="cpu").detach().clone()
        if self.vector.shape != (self.d,) or not torch.isfinite(self.vector).all():
            raise ValueError("finite full-dimensional steering vector required")

    @classmethod
    def from_pinned_files(cls, snapshot=SNAPSHOT, vector_file=VECTOR_FILE):
        """Hash-check original assets, then convert stored BF16 tensors to float64.

        This loads approximately4GB of float64 weights; only call under the
        task's admitted model-run plan. It performs no forward evaluation.
        """
        from safetensors import safe_open
        import numpy as np

        snapshot, vector_file = Path(snapshot), Path(vector_file)
        for path, pin in [(snapshot / "config.json", PINNED_SHA256["config.json"]),
                          (snapshot / "model.safetensors", PINNED_SHA256["model.safetensors"]),
                          (vector_file, PINNED_SHA256["vectors.npz"])]:
            if sha256(path) != pin:
                raise ValueError(f"pinned asset hash mismatch: {path}")
        config = json.loads((snapshot / "config.json").read_text())
        with safe_open(snapshot / "model.safetensors", framework="pt", device="cpu") as source:
            weights = {name: source.get_tensor(name).to(torch.float64) for name in source.keys()}
        with np.load(vector_file, allow_pickle=False) as source:
            vector = source["philosophy"].copy()
        return cls(config, weights, PREFIX_IDS, vector, steering_layer=12)

    @staticmethod
    def _scalar(value):
        if isinstance(value, torch.Tensor):
            if value.ndim != 0 or value.dtype != torch.float64 or value.device.type != "cpu":
                raise ValueError("parameter must be a scalar float64 CPU tensor")
            return value
        return torch.tensor(value, dtype=torch.float64, device="cpu")

    def _state(self, state):
        if not isinstance(state, torch.Tensor) or state.dtype != torch.float64 or state.device.type != "cpu":
            raise ValueError("state must be a float64 CPU tensor")
        if state.ndim not in (1, 2) or state.shape[-1] != self.d:
            raise ValueError("state shape must be(d,) or(batch,d)")
        return state[None, :] if state.ndim == 1 else state

    def _norm(self, hidden, weight):
        variance = hidden.square().mean(-1, keepdim=True)
        return weight * (hidden * torch.rsqrt(variance + self.epsilon))

    def _rotary(self, value, positions):
        phases = positions.to(torch.float64)[:, None] * self.inv_freq[None, :]
        phases = torch.cat((phases, phases), dim=-1)[None, None, :, :]
        half = self.head_dim // 2
        rotated = torch.cat((-value[..., half:], value[..., :half]), dim=-1)
        return value * phases.cos() + rotated * phases.sin()

    def _block(self, hidden, index, alpha, positions, prior=None):
        name = f"model.layers.{index}."
        weight = self.weights
        n = self._norm(hidden, weight[name + "input_layernorm.weight"])
        batch, length, _ = n.shape
        qkv = []
        for projection in ("q", "k", "v"):
            prefix = name + f"self_attn.{projection}_proj."
            value = functional.linear(n, weight[prefix + "weight"], weight[prefix + "bias"])
            qkv.append(value.reshape(batch, length, -1, self.head_dim).transpose(1, 2))
        query, key, value = qkv
        query = self._rotary(query, positions)
        key = self._rotary(key, positions)
        current = (key, value)
        if prior is not None:
            key = torch.cat((prior[0].expand(batch, -1, -1, -1), key), dim=2)
            value = torch.cat((prior[1].expand(batch, -1, -1, -1), value), dim=2)
        key = key.repeat_interleave(self.groups, dim=1)
        value = value.repeat_interleave(self.groups, dim=1)
        scores = torch.matmul(query, key.transpose(-2, -1)) * (self.head_dim ** -.5)
        if prior is None:
            permitted = torch.arange(length)[:, None] >= torch.arange(length)[None, :]
            scores = scores.masked_fill(~permitted[None, None, :, :], -torch.inf)
        elif length != 1:
            raise ValueError("cached forward is restricted to the one-token window")
        attention = torch.softmax(scores, dim=-1, dtype=torch.float64)
        mixed = torch.matmul(attention, value).transpose(1, 2).reshape(batch, length, self.d)
        hidden = hidden + functional.linear(mixed, weight[name + "self_attn.o_proj.weight"])
        n = self._norm(hidden, weight[name + "post_attention_layernorm.weight"])
        gate = functional.linear(n, weight[name + "mlp.gate_proj.weight"])
        up = functional.linear(n, weight[name + "mlp.up_proj.weight"])
        hidden = hidden + functional.linear(functional.silu(gate) * up, weight[name + "mlp.down_proj.weight"])
        if index == self.steering_layer:
            hidden = hidden + alpha * self.vector
        return hidden, current

    def prefix_cache(self, alpha):
        """Compute all24 prefix KV pairs without detaching alpha derivatives."""
        alpha = self._scalar(alpha)
        hidden = self.embedding[self.prefix_ids][None, :, :]
        positions = torch.arange(len(self.prefix_ids))
        layers = []
        for index in range(self.layers):
            hidden, current = self._block(hidden, index, alpha, positions)
            layers.append(current)
        return PrefixCache(alpha, tuple(layers), self)

    def _external_cache(self, cache, alpha):
        if cache.owner is not self or not torch.equal(cache.alpha, alpha):
            raise ValueError("cache belongs to another map or steering parameter")
        if alpha.requires_grad or torch.autograd.forward_ad.unpack_dual(alpha).tangent is not None:
            raise ValueError("omit external cache for parameter derivatives")
        return cache

    def _logits_with_cache(self, state, alpha, cache):
        hidden = self._state(state)[:, None, :]
        positions = torch.tensor([len(self.prefix_ids)])
        for index in range(self.layers):
            hidden, _ = self._block(hidden, index, alpha, positions, cache.layers[index])
        normalized = self._norm(hidden[:, -1, :], self.weights["model.norm.weight"])
        logits = functional.linear(normalized, self.embedding)
        return logits[0] if state.ndim == 1 else logits

    def logits(self, state, alpha, cache=None):
        alpha = self._scalar(alpha)
        cache = self.prefix_cache(alpha) if cache is None else self._external_cache(cache, alpha)
        return self._logits_with_cache(state, alpha, cache)

    def forward(self, state, alpha, log_temperature, cache=None):
        logits = self.logits(state, alpha, cache)
        temperature = 10. ** self._scalar(log_temperature)
        probabilities = torch.softmax(logits / temperature, dim=-1, dtype=torch.float64)
        return probabilities @ self.embedding

    __call__ = forward

    def initial_state(self):
        return self.embedding[self.prefix_ids[-1]].clone()

    def period_map(self, state, alpha, log_temperature, period, cache=None):
        if type(period) is not int or period < 1:
            raise ValueError("period must be a positive integer")
        alpha = self._scalar(alpha)
        cache = self.prefix_cache(alpha) if cache is None else self._external_cache(cache, alpha)
        temperature = 10. ** self._scalar(log_temperature)
        result = state
        for _ in range(period):
            logits = self._logits_with_cache(result, alpha, cache)
            result = torch.softmax(logits / temperature, dim=-1, dtype=torch.float64) @ self.embedding
        return result

    def jvp(self, state, direction, alpha, log_temperature, period=1, cache=None):
        alpha = self._scalar(alpha)
        cache = self.prefix_cache(alpha) if cache is None else self._external_cache(cache, alpha)
        function = lambda value: self.period_map(value, alpha, log_temperature, period, cache)
        return torch.func.jvp(function, (state,), (direction,))[1]

    def jacobian(self, state, alpha, log_temperature, period=1, chunk_size=16, cache=None):
        """Dense column-oriented Jacobian, batched tangent rows without vmap."""
        if state.ndim != 1 or type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("one state and a positive integer chunk size required")
        self._state(state)
        alpha = self._scalar(alpha)
        cache = self.prefix_cache(alpha) if cache is None else self._external_cache(cache, alpha)
        eye = torch.eye(self.d, dtype=torch.float64)
        rows = []
        for start in range(0, self.d, chunk_size):
            directions = eye[start:start + chunk_size]
            points = state[None, :].expand(len(directions), -1).clone()
            rows.append(self.jvp(points, directions, alpha, log_temperature, period, cache))
        return torch.cat(rows, dim=0).T.contiguous()
