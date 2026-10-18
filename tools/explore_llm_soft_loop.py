"""Exploratory: soft-token decoding as a smooth map, scanned for periodic windows and chaos.

Instead of sampling a token, the loop feeds back the probability-weighted input
embedding x_new = E^T softmax(z / T), where z are the next-token logits after a
fixed prefix plus the last K fed-back embeddings (a sliding window at fixed
positions, so the map is autonomous). As T -> 0 the map becomes greedy decoding
with a K-token window; at finite T it is a smooth map of R^(K d), so periodic
orbits, multipliers, Lyapunov exponents and bifurcation curves are defined in
the usual sense. An additive steering vector alpha * v on one decoder layer's
output gives a second continuous parameter.

Subcommands
  scan   Grid in (log10 T, alpha): iterate all cells in one batch, then measure
         the period of the attractor (state recurrence) and the largest Lyapunov
         exponent (twin trajectory, renormalized every step).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import explore_llm_attractors as ela  # noqa: E402
import explore_llm_phase_map as elpm  # noqa: E402

DATA = ela.DATA


PREFIX_BATCH = 32          # rows per prefix pass when a large model is spread over GPUs (device "auto")


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
            if self.extra_rows is not None:          # the second steering block (run TB), as in logits
                self.sm.steer_extra = {self.extra[0]: self.extra_rows}
            try:
                z = self.model(inputs_embeds=inp.to(self.dtype), past_key_values=cache, use_cache=True,
                               logits_to_keep=1).logits[:, -1, :].to(self.device, self.state_dtype)
            finally:
                self.sm.steer = None
                self.sm.steer_extra = {}
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


class MultiPrefixSoftLoop(SoftLoop):
    """SoftLoop with its own fixed prefix for every cell of a batch, so that cells of different prompts (of different
    prefix lengths) are iterated together and a large model's weights are read once per step for all of them.

    set_prefixes gives one token-id list per cell; prepare(steer_rows) then takes B rows, a multiple of the number n of
    cells, and row r uses the prefix of cell r % n (hidden_chaos.run_batch stacks the base trajectories and their twins
    in blocks of n cells, like the rows of the second steering block). The prefixes are left-padded to a common length
    Lp, and L0 = Lp is the length the cache is cropped back to after every step (reset_cache). An attention mask hides
    the padding in the prefix pass and in every step, and position ids give every row its own true positions: its
    prefix at 0 .. L_r - 1 and the fed-back window at L_r .. L_r + K - 1 at every step, the fixed positions of SoftLoop
    with that prefix alone (the rotary tables, float64 in float64 runs, are built from these positions). Each row is
    therefore the map of SoftLoop with its own prefix, up to rounding: the padded attention sums in another order.
    Hybrid (linear-attention) models are refused: their recurrent prefix states would take in the padding."""

    def __init__(self, model, layer, threads, window, dtype="float64", device="cpu", weights="native"):
        super().__init__(model, layer, threads, "open", ela.OPENER, window, dtype, device, weights)
        self._init_prefixes()

    @classmethod
    def from_loop(cls, loop):
        """A multi-prefix loop on the model of an existing SoftLoop: the same weights, tokenizer and steering hooks (no
        second copy of the model). The two share the hooks' steering state, so they must not be stepped concurrently."""
        new = cls.__new__(cls)
        new.__dict__.update(loop.__dict__)
        new.cache, new.steer_rows, new.extra_rows, new._saved = None, None, None, None
        new._init_prefixes()
        return new

    def _init_prefixes(self):
        if self.hybrid:
            raise SystemExit("hybrid (linear-attention) models: padded per-row prefixes are not supported")
        self.prefix_ids, self.L0 = None, None     # the default prefix of SoftLoop is not used
        self.prefixes = None                      # one token-id list per cell (set_prefixes)
        self.prefix_mask = None                   # (B, Lp) after prepare: 1 at prefix tokens, 0 at the padding
        self.row_lengths = None                   # (B,) true prefix length of every row
        self._step_inputs = None                  # attention mask and position ids of the fed-back window

    def set_prefixes(self, prefixes):
        """One prefix per cell (token ids, any lengths); the next prepare builds the cache for them."""
        ids = [[int(t) for t in p] for p in prefixes]
        if not ids or min(len(p) for p in ids) == 0:
            raise ValueError("every cell needs a prefix of at least one token")
        self.prefixes = ids
        self.cache, self.prefix_mask, self.row_lengths, self._step_inputs = None, None, None, None

    def prepare(self, steer_rows):
        """Prefix key-value cache for every row, computed with that row's steering and its own left-padded prefix."""
        from transformers import DynamicCache

        torch = self.torch
        if self.prefixes is None:
            raise ValueError("no prefixes: call set_prefixes first")
        B, n = len(steer_rows), len(self.prefixes)
        if B % n:
            raise ValueError(f"{B} rows are not a multiple of {n} prefix cells")
        rows = [self.prefixes[r % n] for r in range(B)]
        lengths = torch.tensor([len(p) for p in rows])
        Lp = int(lengths.max())
        pad = self.sm.tok.pad_token_id or 0                   # any valid id: the padding is masked
        ids = torch.full((B, Lp), pad, dtype=torch.long)
        mask = torch.zeros((B, Lp), dtype=torch.long)
        for r, p in enumerate(rows):
            ids[r, Lp - len(p):] = torch.tensor(p)
            mask[r, Lp - len(p):] = 1
        # true positions: 0 .. L_r - 1 on the prefix tokens (the padding's positions are masked; 0 there)
        positions = (torch.arange(Lp)[None, :] - (Lp - lengths)[:, None]).clamp(min=0)
        ids, mask, positions = ids.to(self.device), mask.to(self.device), positions.to(self.device)
        self.L0, self.prefix_mask, self.row_lengths = Lp, mask, lengths.to(self.device)
        self._step_inputs = {"attention_mask": torch.cat([mask, torch.ones((B, self.K), dtype=mask.dtype,
                                                                           device=self.device)], dim=1),
                             "position_ids": self.row_lengths[:, None] + torch.arange(self.K, device=self.device)[None]}
        self.steer_rows = torch.as_tensor(np.asarray(steer_rows), dtype=self.dtype, device=self.device)[:, None, :]
        self.extra_rows = None
        if self.extra is not None:     # per-cell rows of the second block, repeated like the batch's cells
            rows2 = np.asarray(self.extra[1])
            if B % len(rows2):
                raise ValueError(f"{B} rows are not a multiple of {len(rows2)} second-block cells")
            self.extra_rows = torch.as_tensor(np.concatenate([rows2] * (B // len(rows2))), dtype=self.dtype,
                                              device=self.device)[:, None, :]
        self.cache = DynamicCache()
        sub = self.prefix_batch if (self.prefix_batch and B > self.prefix_batch) else B
        parts = []
        try:
            for s0 in range(0, B, sub):
                m = min(sub, B - s0)
                cache = self.cache if sub == B else DynamicCache()
                self.sm.steer = self.steer_rows[s0:s0 + m]
                if self.extra_rows is not None:
                    self.sm.steer_extra = {self.extra[0]: self.extra_rows[s0:s0 + m]}
                with torch.no_grad():
                    self.model(input_ids=ids[s0:s0 + m], attention_mask=mask[s0:s0 + m],
                               position_ids=positions[s0:s0 + m], past_key_values=cache, use_cache=True,
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
        # a masked key still enters the attention products (a zero weight times a NaN value is NaN): an attention kernel
        # that left NaN at the fully masked padding queries would poison every later step, so refuse such a cache
        for li, layer in enumerate(self.cache.layers):
            if not bool(torch.isfinite(layer.keys).all() and torch.isfinite(layer.values).all()):
                raise SystemExit(f"non-finite keys or values in the padded prefix cache (layer {li})")
        with torch.no_grad():
            last = torch.tensor([p[-1] for p in self.prefixes], device=self.device)
            start = self.model.get_input_embeddings()(last).to(self.state_dtype)          # (n, d)
        return start[torch.arange(B, device=self.device) % n][:, None, :].repeat(1, self.K, 1).clone()

    def _window_inputs(self, X):
        """Attention mask (padding hidden) and position ids (true positions) for the window X (B, K, d)."""
        if self._step_inputs is None or tuple(X.shape[:2]) != tuple(self._step_inputs["position_ids"].shape):
            raise ValueError(f"window of shape {tuple(X.shape)} does not match the prepared prefix rows")
        return self._step_inputs

    def logits(self, X):
        torch = self.torch
        inputs = self._window_inputs(X)
        self.sm.steer = self.steer_rows
        if self.extra_rows is not None:
            self.sm.steer_extra = {self.extra[0]: self.extra_rows}
        try:
            with torch.no_grad():
                z = self.model(inputs_embeds=X.to(self.dtype), past_key_values=self.cache, use_cache=True,
                               logits_to_keep=1, **inputs).logits[:, -1, :].to(self.device, self.state_dtype)
        finally:
            self.sm.steer = None
            self.sm.steer_extra = {}
            self.reset_cache()
        return z

    def step_jvp(self, X, temps, V):
        """SoftLoop.step_jvp with the padded prefixes: the window's attention mask and true positions."""
        import copy

        import torch.autograd.forward_ad as fwAD
        from torch.nn.attention import SDPBackend, sdpa_kernel

        torch = self.torch
        inputs = self._window_inputs(X)
        T = torch.as_tensor(temps, dtype=self.state_dtype, device=self.device)[:, None]
        cache = copy.copy(self.cache)
        cache.layers = [copy.copy(layer) for layer in self.cache.layers]

        def new_embedding(inp):
            self.sm.steer = self.steer_rows
            if self.extra_rows is not None:          # the second steering block (run TB), as in logits
                self.sm.steer_extra = {self.extra[0]: self.extra_rows}
            try:
                z = self.model(inputs_embeds=inp.to(self.dtype), past_key_values=cache, use_cache=True,
                               logits_to_keep=1, **inputs).logits[:, -1, :].to(self.device, self.state_dtype)
            finally:
                self.sm.steer = None
                self.sm.steer_extra = {}
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


def solve_beta(z, targets, iters=60):
    """Inverse temperatures beta (B,) with entropy H(softmax(beta z)) = targets (B,) nats, for logits z (B, V): Newton
    in u = log beta, safeguarded by bisection (H decreases strictly in beta when the logits are not all equal;
    dH/du = -beta^2 Var_p(z)). A fixed number of iterations, so the solution is one deterministic function of z."""
    torch = __import__("torch")
    H = torch.as_tensor(targets, dtype=z.dtype, device=z.device)
    lo = torch.full_like(H, -12.0)                 # u bracket: beta from 6e-6 to 1.6e5
    hi = torch.full_like(H, 12.0)
    u = torch.zeros_like(H)
    for _ in range(iters):
        b = u.exp()
        p = torch.softmax(b[:, None] * z, dim=-1)
        logp = torch.log_softmax(b[:, None] * z, dim=-1)
        h = -(p * logp).sum(-1)
        m = (p * z).sum(-1)
        var = (p * (z - m[:, None]) ** 2).sum(-1)
        hi = torch.where(h < H, u, hi)              # entropy too low: beta too large
        lo = torch.where(h >= H, u, lo)
        step = (h - H) / torch.clamp(b * b * var, min=1e-300)
        nxt = u + step
        u = torch.where((nxt > lo) & (nxt < hi), nxt, 0.5 * (lo + hi))
    return u.exp()


class SoftOrbit:
    """Periodic orbits of the soft-decoding map at one (temperature, steering) cell: Newton's method on the
    p-fold map with dense central-difference Jacobians (2n rows per Jacobian in chunked batches over a prefix
    cache computed once and repeated), and the orbit's multipliers from the product of those Jacobians."""

    def __init__(self, loop, steer, temperature, h_rel=1e-5, chunk=512, jacobian_method="ad"):
        if getattr(loop, "hybrid", False):
            raise SystemExit("hybrid (linear-attention) models: orbits need prefix-state copies; only scans are supported")
        self.loop, self.torch = loop, loop.torch
        self.T = float(temperature)
        self.start = loop.prepare(np.asarray(steer)[None])        # one-row prefix cache and initial window
        self.cache1, self.steer1 = loop.cache, loop.steer_rows
        self.extra1 = getattr(loop, "extra_rows", None)            # second steering block of this cell (run TB) or None
        self.shape = tuple(self.start.shape[1:])
        self.n = int(np.prod(self.shape))
        self.h_rel, self.chunk = h_rel, int(chunk)
        self.jacobian_method = jacobian_method
        self.caches = {1: self.cache1}                          # prefix caches repeated to each batch size used

    def _use(self, rows):
        import copy

        if rows not in self.caches:
            c = copy.deepcopy(self.cache1)
            c.batch_repeat_interleave(rows)
            self.caches[rows] = c
        self.loop.cache, self.loop.steer_rows = self.caches[rows], self.steer1.repeat(rows, 1, 1)
        self.loop.extra_rows = None if self.extra1 is None else self.extra1.repeat(rows, 1, 1)

    def f(self, X):
        """One soft-decoding step for a batch of windows X (B, K, d) at this cell."""
        self._use(X.shape[0])
        Y, _ = self.loop.step(X, np.full(X.shape[0], self.T))
        return Y

    def probabilities(self, X):
        self._use(1)
        _, p = self.loop.step(X, np.array([self.T]))
        return p[0]

    def jvp_rows(self, X, V):
        """Exact Jacobian-vector products J(X) v for tangent rows V (m, K, d) at X (1, K, d): forward-mode AD
        through the model (math attention kernel), on a shallow copy of the repeated prefix cache so that the
        stored prefix tensors are never modified."""
        import copy

        from torch.nn.attention import SDPBackend, sdpa_kernel

        torch, loop = self.torch, self.loop
        m = V.shape[0]
        self._use(m)
        base = self.caches[m]
        cache = copy.copy(base)
        cache.layers = [copy.copy(layer) for layer in base.layers]
        steer = self.steer1.repeat(m, 1, 1)

        def new_embedding(inp):
            loop.sm.steer = steer
            if self.extra1 is not None:              # the second steering block (run TB), as in the steps
                loop.sm.steer_extra = {loop.extra[0]: self.extra1.repeat(m, 1, 1)}
            try:
                z = loop.model(inputs_embeds=inp.to(loop.dtype), past_key_values=cache, use_cache=True,
                               logits_to_keep=1).logits[:, -1, :].to(loop.state_dtype)
            finally:
                loop.sm.steer = None
                loop.sm.steer_extra = {}
            return torch.softmax(z / self.T, dim=-1) @ loop.E

        with sdpa_kernel(SDPBackend.MATH):
            _, jv = torch.func.jvp(new_embedding, (X.repeat(m, 1, 1),), (V,))
        return torch.cat([V[:, 1:, :], jv[:, None, :]], dim=1)          # the window shifts

    def jacobian(self, X, method=None):
        """Dense Jacobian of one step at X (1, K, d): forward-mode AD (exact) or central differences with
        step h_rel * RMS(X)."""
        torch = self.torch
        method = method or self.jacobian_method
        eye = torch.eye(self.n, dtype=X.dtype, device=X.device).reshape((self.n,) + self.shape)
        h = self.h_rel * float(X.norm()) / np.sqrt(self.n)
        cols = []
        for s in range(0, self.n, self.chunk):
            e = eye[s:s + self.chunk]
            m = e.shape[0]
            if method == "ad":
                cols.append(self.jvp_rows(X, e).reshape(m, -1))
            else:
                Y = self.f(torch.cat([X + h * e, X - h * e]))
                cols.append(((Y[:m] - Y[m:]) / (2 * h)).reshape(m, -1))
        return torch.cat(cols).T.double().cpu().numpy()                 # J[:, i] = d f / d x_i

    def iterate(self, X, steps):
        for _ in range(steps):
            X = self.f(X)
        return X

    def return_distances(self, X, max_period):
        """Relative distances |f^q(X) - X| / |X| for q = 1..max_period."""
        hist = [X]
        for _ in range(max_period):
            hist.append(self.f(hist[-1]))
        return [float((hist[q] - X).norm() / X.norm()) for q in range(1, max_period + 1)]

    def detect_period(self, X, max_period, tol):
        dist = self.return_distances(X, max_period)
        return next((q for q, d in enumerate(dist, 1) if d < tol), 0)

    def newton(self, X, period, tol=1e-11, max_newton=8):
        """Newton for f^p(X) = X; returns (X, relative residual history, multipliers, orbit points)."""
        torch = self.torch
        history = []
        for it in range(max_newton + 1):
            pts = [X]
            for _ in range(period):
                pts.append(self.f(pts[-1]))
            r = (pts[period] - X).reshape(-1).double().cpu().numpy()
            rel = float(np.linalg.norm(r) / float(X.norm()))
            history.append(rel)
            Js = [self.jacobian(x) for x in pts[:period]]
            M = np.eye(self.n)
            for J in Js:
                M = J @ M
            if rel < tol or it == max_newton:
                break
            delta = np.linalg.solve(M - np.eye(self.n), -r)
            X = X + torch.as_tensor(delta.reshape((1,) + self.shape), dtype=X.dtype, device=X.device)
        vals = np.linalg.eigvals(M)
        return X, history, vals[np.argsort(-np.abs(vals))], pts[:period]


    def points(self, X, period):
        pts = [X]
        for _ in range(period):
            pts.append(self.f(pts[-1]))
        return pts

    def subspace_multipliers(self, X, period, k=8, iters=40, seed=0):
        """Leading multipliers of the p-fold map at X by block subspace iteration with batched forward-mode
        JVPs (k tangent rows per model call) and a final Rayleigh-Ritz step. Unlike Arnoldi with a relative
        tolerance, it does not stall when every multiplier is tiny (super-stable orbits). Returns (multipliers,
        converged), converged meaning the moduli of the leading half of the block changed by less than 1e-6
        (relative to the largest) over the last 5 iterations; the trailing Ritz values converge last and are
        returned only for information."""
        torch = self.torch
        pts = self.points(X, period)[:period]

        def apply(V):
            for x in pts:
                V = self.jvp_rows(x, V)
            return V

        rng = np.random.default_rng(seed)
        Q = np.linalg.qr(rng.normal(size=(self.n, k)))[0]
        history = []
        for it in range(iters):
            W = apply(torch.as_tensor(Q.T.reshape((k,) + self.shape), dtype=X.dtype, device=X.device))
            W = W.reshape(k, -1).double().cpu().numpy().T                  # columns M q_i
            H = Q.T @ W
            vals = np.linalg.eigvals(H)
            history.append(np.sort(np.abs(vals))[::-1][: max(2, k // 2)])
            Q = np.linalg.qr(W)[0]
        ok = len(history) > 5 and float(np.abs(history[-1] - history[-6]).max()) <= 1e-6 * max(history[-1][0], 1e-300)
        return vals[np.argsort(-np.abs(vals))], ok

    def phi(self, period):
        """The p-fold soft map as a differentiable function of (window X (1, K, d), log10 T) for continuation in
        temperature (the prefix cache does not depend on T); every model call uses a shallow copy of the
        one-row prefix cache, so forward-mode tangents never reach the stored prefix tensors."""
        import copy

        from torch.nn.attention import SDPBackend, sdpa_kernel

        torch, loop = self.torch, self.loop

        def step(X, temperature):
            cache = copy.copy(self.cache1)
            cache.layers = [copy.copy(layer) for layer in self.cache1.layers]
            loop.sm.steer = self.steer1
            try:
                with sdpa_kernel(SDPBackend.MATH):
                    z = loop.model(inputs_embeds=X.to(loop.dtype), past_key_values=cache, use_cache=True,
                                   logits_to_keep=1).logits[:, -1, :].to(loop.state_dtype)
            finally:
                loop.sm.steer = None
            p = torch.softmax(z / temperature, dim=-1)
            return torch.cat([X[:, 1:, :], (p @ loop.E)[:, None, :]], dim=1)

        def phi(X, log_t):
            temperature = 10.0 ** log_t
            for _ in range(period):
                X = step(X, temperature)
            return X

        return phi


def orbit_report(orb, X, history, vals, pts, k=8, tol=1e-9):
    tok = orb.loop.sm.tok
    seps = [float((p - pts[0]).norm() / pts[0].norm()) for p in pts[1:]]
    tokens = []
    for x in pts:
        p = orb.probabilities(x).double().cpu().numpy()
        top = int(p.argmax())
        tokens.append({"token": tok.decode([top]), "probability": float(p[top]),
                       "entropy": float(-(p * np.log(np.maximum(p, 1e-300))).sum())})
    return {"newton_residuals": history, "converged": history[-1] < tol,
            "spectral_radius": float(np.abs(vals[0])),
            "multipliers": [[float(v.real), float(v.imag), float(abs(v)), float(np.angle(v) / (2 * np.pi))]
                            for v in vals[:k]],
            "min_separation_from_first_point": min(seps) if seps else None, "points": tokens}


def orbit(args):
    loop = SoftLoop(args.model, args.layer, args.threads, args.prompt, args.opener, args.window, args.dtype,
                    args.device)
    v = elpm.load_vectors(args.vectors)[args.concept]
    orb = SoftOrbit(loop, args.alpha * v, 10.0 ** args.log_temperature, args.h_rel, args.chunk, args.jacobian)
    tic = time.perf_counter()
    X = orb.iterate(orb.start, args.transient)
    returns = orb.return_distances(X, args.max_period)
    period = args.period or next((q for q, d in enumerate(returns, 1) if d < args.detect_tol), 0)
    if period == 0:
        raise SystemExit(f"no period up to {args.max_period} at tolerance {args.detect_tol}; nothing to verify "
                         f"(smallest return distance {min(returns):.3g} at q = {int(np.argmin(returns)) + 1})")
    X, history, vals, pts = orb.newton(X, period, args.tol, args.max_newton)
    rec = {"args": vars(args), "period": period, "seconds": round(time.perf_counter() - tic, 1),
           "return_distances_after_transient": returns}
    rec.update(orbit_report(orb, X, history, vals, pts, tol=args.tol))
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "orbit.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    np.save(out / "orbit-state.npy", X.double().cpu().numpy())
    print(json.dumps({k: v for k, v in rec.items() if k != "args"}, ensure_ascii=False))


def tline(args):
    """Continuation in log10 T, at fixed steering, of the periodic orbit reached at (log10 T, alpha): pseudo-
    arclength through folds with multipliers at every point (explore_latent_loop.pseudo_arclength), and the
    classification of every unit-circle crossing found along the branch."""
    import explore_latent_loop as ell

    loop = SoftLoop(args.model, args.layer, args.threads, args.prompt, args.opener, args.window, args.dtype,
                    args.device)
    v = elpm.load_vectors(args.vectors)[args.concept]
    orb = SoftOrbit(loop, args.alpha * v, 10.0 ** args.log_temperature, chunk=args.chunk)
    X = orb.iterate(orb.start, args.transient)
    period = args.period or orb.detect_period(X, args.max_period, args.detect_tol)
    if period == 0:
        raise SystemExit(f"no period up to {args.max_period} at tolerance {args.detect_tol}; nothing to continue")
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    states = []

    def on_point(rec, x):
        states.append(x.double().cpu().numpy()[0])
        print(json.dumps({k: v for k, v in rec.items() if k != "multipliers"} | {"leading": rec["multipliers"][:3]}),
              flush=True)

    def multipliers(x, log_t):
        orb.T = 10.0 ** log_t
        return orb.subspace_multipliers(x, period, k=args.k, iters=args.subspace_iter)

    records, reason = ell.pseudo_arclength(
        orb.phi(period), X, args.log_temperature, direction=args.direction, steps=args.steps, ds=args.ds,
        ds_min=args.ds_min, ds_max=args.ds_max, tol=args.tol, max_newton=args.max_newton, gmres_tol=args.gmres_tol,
        restart=args.restart, gmres_iter=args.gmres_iter, k=args.k, beyond=args.beyond,
        lam_bounds=tuple(args.bounds), on_point=on_point, multipliers_fn=multipliers)
    crossings = []
    for a, b in zip(records, records[1:]):
        if a["spectral_radius"] is None or b["spectral_radius"] is None:
            continue
        if (a["spectral_radius"] - 1) * (b["spectral_radius"] - 1) < 0:
            outer = b if b["spectral_radius"] > 1 else a          # the side where the crossing multiplier is outside
            vals = [complex(re, im) for re, im in outer["multipliers"]]
            lead = max(vals, key=abs)
            kind = ("complex pair (Neimark-Sacker)" if abs(lead.imag) > 1e-9 else
                    "real +1 (fold)" if lead.real > 0 else "real -1 (flip)")
            others = sorted((abs(z) for z in vals if z != lead and z != lead.conjugate()), reverse=True)
            crossings.append({"between_log10_T": [a["lambda"], b["lambda"]], "type": kind,
                              "crossing_multiplier": [lead.real, lead.imag, abs(lead), float(np.angle(lead) / (2 * np.pi))],
                              "next_largest_modulus": others[0] if others else None,
                              "multipliers_converged": bool(a["multipliers_converged"] and b["multipliers_converged"])})
    np.save(out / "states.npy", np.array(states))
    rec = {"args": vars(args), "period": period, "stop": reason, "crossings": crossings, "points": records}
    (out / "tline.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({"period": period, "stop": reason, "points": len(records), "crossings": crossings}))


def precision(args):
    """Effective precision of the float32 map: at cells (log10 T, alpha), iterate the float64 map (norms patched)
    to its attractor, then compare one step and q steps of the float32 map (on --device) with the float64 map from
    the same states. Also reports the float64 attractor's return distances, so a period read at tolerance tol in
    float32 can be checked against the float32 error level."""
    v = elpm.load_vectors(args.vectors)[args.concept]
    hi = SoftLoop(args.model, args.layer, args.threads, args.prompt, args.opener, args.window, "float64",
                  args.device64)
    lo = SoftLoop(args.model, args.layer, args.threads, args.prompt, args.opener, args.window, "float32",
                  args.device)
    torch = hi.torch
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for log_t, alpha in args.cells:
        oh = SoftOrbit(hi, alpha * v, 10.0 ** log_t)
        ol = SoftOrbit(lo, alpha * v, 10.0 ** log_t)
        X = oh.iterate(oh.start, args.transient)
        returns = oh.return_distances(X, args.max_period)
        Xh, Xl, step_err = X, X.to(lo.state_dtype).to(lo.device), []
        for _ in range(args.steps):
            Xh, Xl = oh.f(Xh), ol.f(Xl.to(lo.state_dtype))
            step_err.append(float((Xl.double().to(Xh.device) - Xh).norm() / Xh.norm()))
        one = []
        Y = X
        for _ in range(args.steps):                         # one-step errors from float64 states
            Yh = oh.f(Y)
            Yl = ol.f(Y.to(lo.state_dtype).to(lo.device)).double().to(Yh.device)
            one.append(float((Yl - Yh).norm() / Yh.norm()))
            Y = Yh
        row = {"log10_temperature": log_t, "alpha": alpha, "float64_return_distances": returns,
               "float32_one_step_error": {"median": float(np.median(one)), "max": float(np.max(one))},
               "float32_trajectory_error": step_err, "state_norm": float(X.norm())}
        rows.append(row)
        print(json.dumps({k: (v if k != "float32_trajectory_error" else [v[0], v[-1]]) for k, v in row.items()}),
              flush=True)
    (out / "precision.json").write_text(json.dumps({"args": vars(args), "cells": rows}, indent=1))


def soft_family(loop, v, period, alpha_scale=10.0, chunk=512, reference=None, exact_jvp=False, second=None,
                log_temperature=None):
    """The p-fold soft map as a two-parameter family for bifurcation_curves, theta = (log10 T, alpha * alpha_scale).
    The steering acts on the prefix too, so phi recomputes the prefix cache inside the differentiable function
    (derivatives in alpha include the prefix); the dense monodromy reuses SoftOrbit's batched forward-mode
    Jacobians at fixed theta. With a `reference` SoftLoop (float64, norms patched) the residual Phi(x) - x is
    computed there and the dense derivatives in `loop`'s precision: Newton then converges to the reference
    precision. float32 derivatives of one step are accurate to about 5e-5, but those of the P8 orbit's 8-step map
    only to about 3 % (dense monodromy and JVPs alike), enough for a Newton matrix but not for multipliers or the
    test function g. exact_jvp=True takes the Jacobian-vector products (jvp_x: g by GMRES, refined multipliers)
    from the reference as well.

    Two-block plane (run TB): with second = (layer2, w) and a fixed log_temperature, theta = (alpha1, alpha2) *
    alpha_scale, alpha1 steering along v at loop's layer and alpha2 along w at layer2 (registered on every loop with
    sm.add_steer_layer before this call). Both blocks act on the prefix too."""
    import copy

    import bifurcation_curves as bc
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import DynamicCache

    torch = loop.torch
    v = np.asarray(v)
    if getattr(loop, "hybrid", False):
        raise SystemExit("hybrid (linear-attention) models: continuation is not supported (prefix states are not copied)")
    if second is not None:
        if log_temperature is None:
            raise ValueError("a two-block family needs the fixed log_temperature")
        for lp in (loop, reference):
            if lp is not None and int(second[0]) not in lp.sm._extra_hooks:
                raise ValueError(f"layer {second[0]} has no second steering hook (call sm.add_steer_layer first)")
        w = np.asarray(second[1])

    def make_orbit(lp, key):
        """SoftOrbit of the cell theta = key (a tuple of floats) on SoftLoop lp."""
        if second is None:
            return SoftOrbit(lp, key[1] / alpha_scale * v, 10.0 ** key[0], chunk=chunk)
        lp.extra = (int(second[0]), np.array([key[1] / alpha_scale * w]))
        return SoftOrbit(lp, key[0] / alpha_scale * v, 10.0 ** log_temperature, chunk=chunk)

    def make_phi(lp):
        v_t = torch.as_tensor(v, dtype=lp.dtype, device=lp.device)[None, None, :]
        w_t = None if second is None else torch.as_tensor(w, dtype=lp.dtype, device=lp.device)[None, None, :]

        def phi(X, theta):
            if second is None:
                temperature, a1 = 10.0 ** theta[0], theta[1]
            else:                                           # two-block plane at a fixed temperature
                temperature, a1 = 10.0 ** log_temperature, theta[0]
            cache = DynamicCache()
            lp.sm.steer = (a1 / alpha_scale).to(lp.dtype) * v_t
            if second is not None:
                lp.sm.steer_extra = {int(second[0]): (theta[1] / alpha_scale).to(lp.dtype) * w_t}
            try:
                lp.model(input_ids=lp.prefix_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
                for _ in range(period):
                    c = copy.copy(cache)                    # every step sees the prefix only (fixed positions)
                    c.layers = [copy.copy(layer) for layer in cache.layers]
                    z = lp.model(inputs_embeds=X.to(lp.dtype), past_key_values=c, use_cache=True,
                                 logits_to_keep=1).logits[:, -1, :].to(lp.state_dtype)
                    p = torch.softmax(z / temperature, dim=-1)
                    X = torch.cat([X[:, 1:, :], (p @ lp.E)[:, None, :]], dim=1)
            finally:
                lp.sm.steer = None
                lp.sm.steer_extra = {}
            return X

        return phi

    phi = make_phi(loop)
    if exact_jvp and reference is None:
        raise ValueError("exact_jvp needs a reference SoftLoop")
    exact = bc.Family(make_phi(reference), (1, loop.K, loop.E.shape[1]), reference.state_dtype, reference.device,
                      context=lambda: sdpa_kernel(SDPBackend.MATH)) if exact_jvp else None

    class Family(bc.Family):
        def __init__(self):
            super().__init__(phi, (1, loop.K, loop.E.shape[1]), loop.state_dtype, loop.device,
                             context=lambda: sdpa_kernel(SDPBackend.MATH))
            self.orbit, self.ref_orbit = None, None

        def soft_orbit(self, theta):
            key = tuple(float(t) for t in theta)
            if self.orbit is None or self.orbit[0] != key:
                self.orbit = None                           # release the old repeated caches first
                self.orbit = (key, make_orbit(loop, key))
            return self.orbit[1]

        def residual(self, x, theta):
            if reference is None:
                X = self.soft_orbit(theta).iterate(self.to_x(x), period)
                return self.to_np(X) - np.asarray(x).ravel()
            key = tuple(float(t) for t in theta)
            if self.ref_orbit is None or self.ref_orbit[0] != key:
                self.ref_orbit = (key, make_orbit(reference, key))
            ref = self.ref_orbit[1]
            X = torch.as_tensor(np.asarray(x, dtype=np.float64).reshape(self.shape), dtype=reference.state_dtype,
                                device=reference.device)
            return self.to_np(ref.iterate(X, period)) - np.asarray(x).ravel()

        def monodromy(self, x, theta):
            orb = self.soft_orbit(theta)
            X, M = self.to_x(x), np.eye(self.n)
            for _ in range(period):
                M = orb.jacobian(X) @ M
                X = orb.f(X)
            return M

        def jvp_x(self, x, theta, v):
            return exact.jvp_x(x, theta, v) if exact is not None else super().jvp_x(x, theta, v)

    return Family()


def curve_start(tl, states, kind, index):
    """Starting point for curve: the tline point nearest the first turn (fold) or the first multiplier -1
    crossing (flip). kind 'auto' takes the first boundary found along the tline. Refuses a boundary whose
    crossing was classified otherwise (or not at all)."""
    pts = tl["points"]
    turns = [i for i, r in enumerate(pts) if r.get("turned_since_previous")]
    flips = [c for c in tl["crossings"] if c["type"].startswith("real -1")]
    if kind == "auto":
        lam0 = pts[0]["lambda"]
        cands = []
        if turns:
            cands.append((abs(pts[turns[0]]["lambda"] - lam0), "fold"))
        if flips:
            cands.append((min(abs(v - lam0) for v in flips[0]["between_log10_T"]), "flip"))
        if not cands:
            kinds = sorted({c["type"] for c in tl["crossings"]})
            raise SystemExit(f"no fold turn and no flip crossing along the tline (crossings: {kinds}); nothing to trace")
        kind = min(cands)[1]
    if kind == "fold":
        if not turns:
            raise SystemExit("no turn along the tline: no fold to start from")
        i = turns[0]
        est = pts[i].get("turn_lambda_estimate", pts[i]["lambda"])
        near = [c for c in tl["crossings"] if min(abs(v - est) for v in c["between_log10_T"]) < 0.02]
        if not any(c["type"].startswith("real +1") for c in near):
            raise SystemExit(f"the turn near log10 T {est:.5f} has no real +1 crossing recorded (crossings there: "
                             f"{[c['type'] for c in near]}); refusing to trace it as a fold")
        auto = min((i - 1, i), key=lambda j: abs(pts[j]["lambda"] - est))
    else:
        if not flips:
            raise SystemExit("no real -1 crossing along the tline: no flip to start from")
        a, b = flips[0]["between_log10_T"]
        cand = [j for j, r in enumerate(pts) if r["lambda"] in (a, b) and r["spectral_radius"] is not None]
        auto = min(cand, key=lambda j: abs(pts[j]["spectral_radius"] - 1))
    i = auto if index < 0 else index
    return kind, i, states[i].ravel(), float(pts[i]["lambda"])


def branch(args):
    """Dense continuation in log10 T, at fixed steering, of the periodic orbit reached at (log10 T, alpha):
    bifurcation_curves.trace_branch with all multipliers at every point (eigenvalues of the dense monodromy),
    residuals in the reference precision (float64 by default) and derivatives in --dtype. Writes branch.json in
    the format of tline.json (points, classified crossings), which curve can start from. With --block2 (two-block
    plane, run TB) the temperature stays at --log-temperature and the free parameter is alpha (block --layer) or
    alpha2 (block --block2)."""
    import bifurcation_curves as bc

    loop = SoftLoop(args.model, args.layer, args.threads, args.prompt, args.opener, args.window, args.dtype,
                    args.device)
    ref = (SoftLoop(args.model, args.layer, args.threads, args.prompt, args.opener, args.window,
                    args.reference_dtype, args.device) if args.reference_dtype != "none" else None)
    v = elpm.load_vectors(args.vectors)[args.concept]
    second = None
    if args.block2 is not None:                                 # two-block plane at a fixed temperature
        if None in (args.vectors2, args.concept2, args.alpha2):
            raise SystemExit("branch: a two-block plane needs --vectors2, --concept2 and --alpha2")
        if args.free == "temperature":
            raise SystemExit("branch: in a two-block plane the free parameter is alpha or alpha2 (log10 T is fixed)")
        for lp in (loop, ref):
            if lp is not None:
                lp.sm.add_steer_layer(args.block2)
        second = (args.block2, elpm.load_vectors(args.vectors2)[args.concept2])
    elif args.free == "alpha2":
        raise SystemExit("branch: --free alpha2 needs a second block (--block2)")
    top = ref or loop
    if second is not None:
        top.extra = (args.block2, np.array([args.alpha2 * second[1]]))
    march = None
    if args.march_from is not None:                             # reach (alpha, alpha2) from a fixed-start seed
        if second is None:
            raise SystemExit("branch: --march-from needs a two-block plane")
        import llm_feigenbaum as lfb

        def cell_at(a1, a2):
            top.extra = (args.block2, np.array([a2 * second[1]]))
            return SoftOrbit(top, a1 * v, 10.0 ** args.log_temperature)

        seed = cell_at(*args.march_from)
        X = seed.iterate(seed.start, args.transient)
        returns = seed.return_distances(X, args.max_period)
    else:
        orb = SoftOrbit(top, args.alpha * v, 10.0 ** args.log_temperature)
        X = orb.iterate(orb.start, args.transient)
        returns = orb.return_distances(X, args.max_period)
    period = args.period or next((q for q, d in enumerate(returns, 1) if d < args.detect_tol), 0)
    if period == 0:
        raise SystemExit(f"no period up to {args.max_period} at tolerance {args.detect_tol} (smallest return "
                         f"distance {min(returns):.3g} at q = {int(np.argmin(returns)) + 1}); nothing to continue")
    if args.march_from is not None:
        # Newton continuation of this orbit through the waypoints (a path of cells where the orbit was followed, e.g.
        # the parent chain of tools/follow_attractor.py) to (alpha, alpha2); a waypoint where Newton fails stops the
        # run: the orbit is lost, and nothing is continued from a substitute
        way = [tuple(q) for q in json.loads((DATA / "explore" / args.march_path).read_text())] if args.march_path else []
        way.append((args.alpha, args.alpha2))
        tol = lfb.tol_for(period, 1e-12)
        Q = np.linalg.qr(np.random.default_rng(0).normal(size=(X.numel(), 12)))[0]
        march = []
        for k, (b1, b2) in enumerate([tuple(args.march_from)] + way):
            X, Q, hist = lfb.newton(cell_at(b1, b2), X, period, Q, tol, 12)
            march.append({"alpha": b1, "alpha2": b2, "residual": hist[-1], "newton_steps": len(hist) - 1})
            if not hist[-1] < tol:
                raise SystemExit(f"branch: the march lost the period-{period} orbit at waypoint {k} ({b1}, {b2}): "
                                 f"residual {hist[-1]:.3g}")
        orb = cell_at(args.alpha, args.alpha2)
        returns = orb.return_distances(X, args.max_period)
    fam = soft_family(loop, v, period, args.alpha_scale, args.chunk, reference=ref, exact_jvp=args.exact_jvp,
                      second=second, log_temperature=args.log_temperature)
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)

    progress = out / "progress.jsonl"                           # readable while the run is going
    progress.write_text("")

    def on_point(rec):
        line = json.dumps({k: v for k, v in rec.items() if k != "multipliers"} | {"leading": rec["multipliers"][:3]})
        print(line, flush=True)
        with progress.open("a") as fh:
            fh.write(f"{time.strftime('%H:%M:%S', time.gmtime())} {line}\n")

    tic = time.perf_counter()
    if second is None:
        free = 0 if args.free == "temperature" else 1
        sc = 1.0 if free == 0 else args.alpha_scale            # the free parameter as continued
        theta0 = [args.log_temperature, args.alpha * args.alpha_scale]
        bounds = args.bounds or ([-1.5, 0.5] if free == 0 else [-0.2, 0.2])
    else:                                                      # theta = (alpha, alpha2) * scale
        free = 0 if args.free == "alpha" else 1
        sc = args.alpha_scale
        theta0 = [args.alpha * sc, args.alpha2 * sc]
        bounds = args.bounds or [-0.2, 0.2]
    records, reason, states = bc.trace_branch(
        fam, fam.to_np(X), theta0, free=free,
        direction=args.direction, steps=args.steps, ds=args.ds, ds_min=args.ds_min, ds_max=args.ds_max, tol=args.tol,
        max_newton=args.max_newton, beyond=args.beyond * sc, bounds=(bounds[0] * sc, bounds[1] * sc),
        on_point=on_point, k=args.k, refine_k=args.refine_k)
    for r in records:                                          # 'lambda' in natural units (alpha unscaled)
        r["lambda"] /= sc
        if "turn_lambda_estimate" in r:
            r["turn_lambda_estimate"] /= sc
    crossings = bc.classify_crossings(records)
    np.save(out / "states.npy", np.array(states).reshape((len(states),) + tuple(X.shape[1:])))
    rec = {"args": vars(args), "free": args.free, "period": period, "return_distances_after_transient": returns,
           "stop": reason, "seconds": round(time.perf_counter() - tic, 1), "crossings": crossings,
           "points": records}
    if march is not None:
        rec["march"] = march
    (out / "branch.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({"period": period, "stop": reason, "points": len(records), "crossings": crossings,
                      "seconds": rec["seconds"]}))


def curve(args):
    """Fold or flip curve of a period-p orbit in (log10 T, alpha), traced by two-parameter continuation
    (bifurcation_curves.trace_curve) from the boundary point a tline run found at fixed alpha."""
    import bifurcation_curves as bc

    src = DATA / "explore" / args.start
    tl = json.loads((src / ("branch.json" if (src / "branch.json").exists() else "tline.json")).read_text())
    states = np.load(src / "states.npy")
    kind, index, x0, lam0 = curve_start(tl, states, args.kind, args.start_index)
    ta = tl["args"]
    period = int(tl["period"])
    sc = args.alpha_scale
    two = ta.get("block2") is not None                         # a branch of a two-block plane (run TB)
    if two:                                                    # theta = (alpha, alpha2) * scale at fixed log10 T
        a1, a2 = float(ta["alpha"]), float(ta["alpha2"])
        if tl["free"] == "alpha2":                             # located at fixed alpha
            a2, fix = lam0, 0
        else:                                                  # located at fixed alpha2
            a1, fix = lam0, 1
        theta0 = [a1 * sc, a2 * sc]
        start = {"tline_index": index, "alpha": a1, "alpha2": a2, "fixed_log10_temperature": ta["log_temperature"]}
    elif tl.get("free", "temperature") == "temperature":        # located at fixed alpha
        log_t0, alpha0, fix = lam0, float(ta["alpha"]), 1
    else:                                                      # located at fixed log10 T
        log_t0, alpha0, fix = float(ta["log_temperature"]), lam0, 0
    if not two:
        theta0 = [log_t0, alpha0 * sc]
        start = {"tline_index": index, "log10_temperature": log_t0, "alpha": alpha0}
    loop = SoftLoop(ta["model"], ta["layer"], args.threads, ta["prompt"], ta["opener"], ta["window"], args.dtype,
                    args.device)
    ref = (SoftLoop(ta["model"], ta["layer"], args.threads, ta["prompt"], ta["opener"], ta["window"],
                    args.reference_dtype, args.device) if args.reference_dtype != "none" else None)
    v = elpm.load_vectors(ta["vectors"])[ta["concept"]]
    second = None
    if two:
        for lp in (loop, ref):
            if lp is not None:
                lp.sm.add_steer_layer(ta["block2"])
        second = (ta["block2"], elpm.load_vectors(ta["vectors2"])[ta["concept2"]])
    fam = soft_family(loop, v, period, args.alpha_scale, args.chunk, reference=ref, exact_jvp=args.exact_jvp,
                      second=second, log_temperature=ta["log_temperature"] if two else None)
    if args.gmres_tol is None:                                  # above the JVPs' floor (float32: 4.3e-6 at P8)
        args.gmres_tol = 1e-9 if args.exact_jvp else 1e-5
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    names = ("alpha", "alpha2") if two else ("log10_temperature", "alpha")

    def readable(rec):
        rec = dict(rec)
        if two:
            rec["alpha"], rec["alpha2"] = rec["theta"][0] / sc, rec["theta"][1] / sc
        else:
            rec["log10_temperature"], rec["alpha"] = rec["theta"][0], rec["theta"][1] / sc
        return rec

    progress = out / "progress.jsonl"                           # readable while the run is going
    progress.write_text("")

    def on_point(rec):
        r = readable(rec)
        keys = ("index", "log10_temperature", "alpha", "alpha2", "residual", "g", "g_dense", "newton", "jvps",
                "gmres_unconverged", "rejected", "critical_multiplier", "critical_multiplier_refined",
                "refine_residual", "other_spectral_radius", "parameter_speed", "cusp_candidate")
        line = json.dumps({k: r[k] for k in keys if k in r})
        print(line, flush=True)
        with progress.open("a") as fh:
            fh.write(f"{time.strftime('%H:%M:%S', time.gmtime())} {line}\n")

    tic = time.perf_counter()
    if two:
        bounds = ((args.alpha_bounds[0] * sc, args.alpha_bounds[1] * sc),
                  (args.alpha2_bounds[0] * sc, args.alpha2_bounds[1] * sc))
    else:
        bounds = ((args.log_temperature_bounds[0], args.log_temperature_bounds[1]),
                  (args.alpha_bounds[0] * sc, args.alpha_bounds[1] * sc))
    records, reason, curve_states = bc.trace_curve(
        fam, x0, theta0, kind, fix=fix, direction=args.direction, steps=args.steps, ds=args.ds,
        ds_min=args.ds_min, ds_max=args.ds_max, tol=args.tol, gtol=args.gtol, max_newton=args.max_newton,
        bounds=bounds, on_point=on_point, gmres_tol=args.gmres_tol, gmres_maxiter=args.gmres_maxiter,
        refine_k=args.refine_k)
    rec = {"args": vars(args), "kind": kind, "period": period, "start": start, "parameters": list(names),
           "stop": reason, "seconds": round(time.perf_counter() - tic, 1), "points": [readable(r) for r in records]}
    (out / "curve.json").write_text(json.dumps(rec, indent=1))
    np.save(out / "states.npy", np.array(curve_states))
    cusps = [(r[names[0]], r[names[1]]) for r in rec["points"] if r["cusp_candidate"]]
    print(json.dumps({"kind": kind, "period": period, "points": len(records), "stop": reason,
                      "cusp_candidates": cusps, "seconds": rec["seconds"]}))


def scan(args):
    loop = SoftLoop(args.model, args.layer, args.threads, args.prompt, args.opener, args.window, args.dtype,
                    args.device, args.weights)
    torch = loop.torch
    v = elpm.load_vectors(args.vectors)[args.concept] if args.vectors else None      # --vectors "": unsteered
    if v is not None and len(v) != loop.E.shape[1]:
        raise SystemExit(f"steering vector {args.vectors}/{args.concept} has dimension {len(v)}, the model "
                         f"{loop.E.shape[1]}")
    log_t = np.linspace(*args.log_temperature, args.resolution[0])
    alphas = np.linspace(*args.alpha, args.resolution[1])
    cells = [(lt, a) for lt in log_t for a in alphas]
    B = len(cells)
    d = loop.E.shape[1]
    lyap = np.zeros(B)
    period = np.zeros(B, dtype=int)
    entropy = np.zeros(B)
    family = np.full(B, -1, dtype=int)            # identity of the periodic attractor: its leading-token loop
    families = {}                                 # canonical token loop -> family index
    rng = np.random.default_rng(0)
    proj, dirs = None, None
    if args.projections:                          # bifurcation diagrams: the recorded states on two fixed directions
        r = np.random.default_rng(12345).normal(size=d)
        first = v / np.linalg.norm(v) if v is not None else np.random.default_rng(54321).normal(size=d)
        dirs = np.stack([first / np.linalg.norm(first), r / np.linalg.norm(r)])
        proj = np.full((B, args.record, 2), np.nan)
    init = None
    if args.init_states:                          # warm start: each cell from a previous scan's final state, same grid
        init = np.load(DATA / "explore" / args.init_states / "states.npz")["states"]
        if init.shape[0] != B or init.shape[2] != d:
            raise SystemExit(f"init states {init.shape} do not match {B} cells of dimension {d}")
    final_states = np.zeros((B, loop.K, d), dtype=np.float32) if args.save_states else None
    tic = time.perf_counter()
    chunk = args.chunk or B
    for start in range(0, B, chunk):
        idx = list(range(start, min(start + chunk, B)))
        n = len(idx)
        steer = np.zeros((n, d)) if v is None else np.array([cells[i][1] * v for i in idx])
        temps2 = np.tile(np.array([10.0 ** cells[i][0] for i in idx]), 2)
        XY = loop.prepare(np.concatenate([steer, steer]))     # rows 0..n-1: trajectory, n..2n-1: its twin
        if init is not None:
            X0 = torch.as_tensor(init[idx], dtype=loop.state_dtype, device=loop.device)
            if X0.shape[1] != XY.shape[1]:
                raise SystemExit("init states have another window length")
            XY = torch.cat([X0, X0.clone()])
        progress = DATA / "explore" / args.out / "progress.jsonl"       # timing for long runs (big models)
        progress.parent.mkdir(parents=True, exist_ok=True)
        for k in range(args.transient):
            XY, _ = loop.step(XY, temps2)
            if k % 100 == 99:
                with open(progress, "a") as handle:
                    handle.write(json.dumps({"chunk_start": start, "phase": "transient", "step": k + 1,
                                             "seconds": round(time.perf_counter() - tic, 1)}) + "\n")
        delta = torch.as_tensor(rng.normal(size=tuple(XY[:n].shape)), dtype=loop.state_dtype, device=loop.device)
        delta = delta / delta.flatten(1).norm(dim=1)[:, None, None] * args.eps
        XY = torch.cat([XY[:n], XY[:n] + delta])
        log_growth = np.zeros(n)
        history, tokens = [], []
        for k in range(args.samples):
            XY, p = loop.step(XY, temps2)
            X, Y = XY[:n], XY[n:]
            p = p[:n]
            diff = (Y - X)
            dist = diff.flatten(1).norm(dim=1).double().cpu().numpy()
            # a twin that collapses onto the trajectory (super-stable cycles, exact zero in float32) contributes a
            # floor of log(1e-30) per step; the floor also keeps the renormalization factor finite in float32
            dist = np.maximum(dist, args.eps * 1e-30)
            log_growth += np.log(dist / args.eps)
            Y = X + diff * torch.as_tensor(args.eps / dist, dtype=loop.state_dtype, device=loop.device)[:, None, None]
            XY = torch.cat([X, Y])
            if k >= args.samples - args.record:
                history.append(X[:, -1, :].double().cpu().numpy().copy())
                tokens.append(p.argmax(-1).cpu().numpy().copy())
            if k % 100 == 99:
                with open(progress, "a") as handle:
                    handle.write(json.dumps({"chunk_start": start, "phase": "samples", "step": k + 1,
                                             "seconds": round(time.perf_counter() - tic, 1)}) + "\n")
        lyap[idx] = log_growth / args.samples
        hist = np.array(history)
        if proj is not None:
            proj[idx] = np.einsum("rnd,kd->nrk", hist, dirs)
        scale = np.maximum(1.0, np.linalg.norm(hist[-1], axis=1))
        per = np.zeros(n, dtype=int)
        for q in range(min(args.max_period, len(hist) - 1), 0, -1):
            close = np.linalg.norm(hist[-1 - q] - hist[-1], axis=1) < args.tol * scale
            per = np.where(close, q, per)
        period[idx] = per
        toks = np.array(tokens)                               # (record, n): leading token per step
        for c in range(n):
            q = int(per[c])
            if q > 0:
                loop_ = tuple(int(t) for t in toks[-q:, c])
                key = min(loop_[i:] + loop_[:i] for i in range(q))  # rotation-invariant
                family[idx[c]] = families.setdefault(key, len(families))
        last_p = p.double().cpu().numpy()
        entropy[idx] = -(last_p * np.log(np.maximum(last_p, 1e-300))).sum(1)
        if final_states is not None:
            final_states[idx] = X.float().cpu().numpy()
        del p, XY, X, Y
        print(json.dumps({"done": idx[-1] + 1, "of": B, "seconds": round(time.perf_counter() - tic, 1)}), flush=True)
        part = DATA / "explore" / args.out                    # checkpoint: a timed-out scan still yields its rows
        part.mkdir(parents=True, exist_ok=True)
        extra = {} if proj is None else {"projection": proj, "projection_directions": dirs}
        np.savez_compressed(part / "scan-partial.npz", done=idx[-1] + 1, log10_temperature=log_t, alpha=alphas,
                            lyapunov=lyap, period=period, entropy=entropy, family=family,
                            family_keys=np.array(json.dumps([list(k) for k in families])), **extra)
    seconds = round(time.perf_counter() - tic, 1)
    shape = tuple(args.resolution)
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    extra = {} if proj is None else {"projection": proj.reshape(*shape, args.record, 2),
                                     "projection_directions": dirs}
    np.savez_compressed(out / "scan.npz", log10_temperature=log_t, alpha=alphas, lyapunov=lyap.reshape(shape),
                        period=period.reshape(shape), entropy=entropy.reshape(shape), family=family.reshape(shape),
                        **extra)
    tok = loop.sm.tok
    fam_list = [{"family": i, "period": len(key), "cells": int((family == i).sum()),
                 "tokens": [tok.decode([t]) for t in key]} for key, i in families.items()]
    fam_list.sort(key=lambda f: -f["cells"])
    summary = {"args": vars(args), "seconds": seconds, "cells": B, "families": fam_list,
               "lyapunov_positive": int((lyap > args.chaos_level).sum()),
               "period_counts": {int(q): int((period == q).sum()) for q in np.unique(period)},
               "lyapunov_range": [float(lyap.min()), float(lyap.max())]}
    (out / "scan.json").write_text(json.dumps(summary, indent=1))
    if final_states is not None:                  # float32 copy of each cell's final window, for warm starts
        np.savez(out / "states.npz", states=final_states)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("args", "families")}))
    figure(log_t, alphas, lyap.reshape(shape), period.reshape(shape), entropy.reshape(shape), out / "scan.png", args)


def figure(log_t, alphas, lyap, period, entropy, path, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    lim = max(float(np.nanmax(lyap)), 0.05)                 # clip: super-stable cells reach -700
    im0 = axes[0].pcolormesh(alphas, log_t, np.clip(lyap, -lim, lim), cmap="RdBu_r", vmin=-lim, vmax=lim,
                             shading="auto")
    axes[0].set_title("largest Lyapunov exponent per step (clipped)")
    fig.colorbar(im0, ax=axes[0])
    if (period > 0).any():                                  # one colour and one colour-bar block per period
        import period_colors as pc

        cats = pc.categorize(period, max_blocks=pc.blocks_that_fit(axes[1]))
        im1 = axes[1].pcolormesh(alphas, log_t, cats.index(period), cmap=cats.cmap, norm=cats.norm, shading="auto")
        cats.colorbar(im1, ax=axes[1])
    axes[1].set_title("period of the attractor (white: none up to the maximum)")
    im2 = axes[2].pcolormesh(alphas, log_t, entropy, cmap="viridis", shading="auto")
    axes[2].set_title("next-token entropy at the last step (nats)")
    fig.colorbar(im2, ax=axes[2])
    for ax in axes:
        ax.set_xlabel(f"steering along '{args.concept}'" if args.vectors else "alpha (no steering vector)")
        ax.set_ylabel("log10 temperature")
    fig.suptitle(f"soft-token decoding, {ela.MODELS[args.model]}, window {args.window}, layer {args.layer}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def replot(args):
    """Redraw a finished scan's overview figure from scan.npz and the arguments saved in scan.json."""
    src = DATA / "explore" / args.scan
    z = np.load(src / "scan.npz")
    run = argparse.Namespace(**json.loads((src / "scan.json").read_text())["args"])
    figure(z["log10_temperature"], z["alpha"], z["lyapunov"], z["period"], z["entropy"], args.out, run)


def curve_points(src):
    """(kind, points) of a curve run: curve.json when it is complete and newer than the run's progress file,
    otherwise the points written so far to progress.jsonl (kind from the sign of the critical multiplier)."""
    done, prog = src / "curve.json", src / "progress.jsonl"
    if done.exists() and (not prog.exists() or done.stat().st_mtime >= prog.stat().st_mtime):
        cv = json.loads(done.read_text())
        return cv["kind"], cv["points"]
    pts = [json.loads(line.split(" ", 1)[1]) for line in prog.read_text().splitlines() if line.strip()]
    if not pts:
        raise SystemExit(f"{src}: no curve points yet")
    return ("fold" if pts[0]["critical_multiplier"][0] > 0 else "flip"), pts


def render(args):
    """Period map of a finished scan with chaos marked: one colour and one legend entry per period drawn (periods
    above --max-period, if given, as one 'other' entry), black where the largest Lyapunov exponent exceeds
    --chaos-level, grey where no period was found below it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import period_colors as pc

    src = DATA / "explore" / args.scan
    d = np.load(src / "scan.npz")
    lt, al, per, ly = d["log10_temperature"], d["alpha"], d["period"], d["lyapunov"]
    img = np.full(per.shape + (4,), 0.0)
    family_handles, period_handles = [], []
    none_colour = (0.8, 0.8, 0.8, 1.0) if args.families else (0.75, 0.75, 0.75, 1.0)   # 'no period, not chaotic'
    chaos = (ly > args.chaos_level) & (per <= 0)            # the scans' criterion: positive exponent and no period
    if args.style == "lyapunov":                          # as in the shrimp literature: chaos in colour by the
        # largest exponent, periodic regions in grey by how contracting they are (superstable curves dark)
        from matplotlib.colors import Normalize
        hot_norm = Normalize(0.0, max(float(ly.max()), 1e-9))
        hot = plt.get_cmap("plasma")(hot_norm(np.clip(ly, 0, None)))
        neg = np.clip(-ly, 0, None)
        cap = max(float(np.percentile(neg[~chaos], 99)) if (~chaos).any() else 1.0, 1e-12)
        grey = 1.0 - 0.9 * np.clip(neg / cap, 0, 1) ** 0.5
        img[...] = np.stack([grey, grey, grey, np.ones_like(grey)], axis=-1)
        img[chaos] = hot[chaos]
    elif args.families:                                     # colour by attractor identity (token loop), not period
        if "family" not in d:
            raise SystemExit(f"{args.scan} has no family identities (scan made before they were recorded)")
        fam = d["family"]
        info = {f["family"]: f for f in json.loads((src / "scan.json").read_text())["families"]}
        ids, counts = np.unique(fam[fam >= 0], return_counts=True)
        chosen = ids[np.argsort(-counts)][: args.families]
        # tab20 without its greys, then tab20b (grey marks 'no period' and the other attractors)
        palette = np.concatenate([np.delete(plt.get_cmap("tab20")(np.arange(20)), [14, 15], axis=0),
                                  plt.get_cmap("tab20b")(np.arange(20))])
        if len(chosen) > len(palette):
            raise SystemExit(f"--families {args.families}: only {len(palette)} distinct colours")
        img[(per > 0)] = pc.OTHER_COLOUR                    # other periodic attractors
        for k, f in enumerate(chosen):
            colour = palette[k]
            img[fam == f] = colour
            text = "".join(info[int(f)]["tokens"])[:28] if int(f) in info else "?"   # repr below escapes newlines once
            family_handles.append(matplotlib.patches.Patch(color=colour,
                                  label=f"P{info[int(f)]['period'] if int(f) in info else '?'} {text!r}"))
        if ((per > 0) & ~np.isin(fam, chosen) & ~chaos).any():
            family_handles.append(matplotlib.patches.Patch(color=pc.OTHER_COLOUR, label="other periodic attractors"))
        img[per == 0] = none_colour
    else:
        drawn = (per > 0) & ~chaos                          # chaotic cells are drawn black below
        own = drawn & (per <= args.max_period) if args.max_period else drawn
        if own.any():
            cats = pc.categorize(np.where(own, per, 0))
            img[own] = cats.rgba(per[own])
            period_handles = cats.handles()
        if (drawn & ~own).any():                            # longer than --max-period: one labelled entry
            merged = bool(period_handles) and period_handles[-1].get_label() == "other"   # same colour: one entry
            if merged:
                period_handles.pop()
            img[drawn & ~own] = pc.OTHER_COLOUR
            period_handles.append(matplotlib.patches.Patch(
                color=pc.OTHER_COLOUR, label="other" if merged else f"other (> {args.max_period})"))
        img[per <= 0] = none_colour
    if args.style != "lyapunov":
        img[chaos] = (0.0, 0.0, 0.0, 1.0)
    fig, ax = plt.subplots(figsize=(7.5, 8))
    ax.imshow(img, origin="lower", aspect="auto", interpolation="nearest",
              extent=(al[0], al[-1], lt[0], lt[-1]))
    handles = family_handles or period_handles
    if args.style == "lyapunov":
        # two numeric colour bars instead of a legend: the chaotic cells' exponent (plasma, 0 to the map's largest)
        # and the other cells' exponent on the grey scale of the image (1 - 0.9 sqrt(-exponent / cap), cap the 99th
        # percentile of their contraction), so every shade can be read off
        from matplotlib.colors import FuncNorm, LinearSegmentedColormap
        from matplotlib.cm import ScalarMappable
        handles = []
        bars = [(ScalarMappable(norm=hot_norm, cmap="plasma"), "chaotic cells: largest Lyapunov exponent per step"),
                (ScalarMappable(norm=FuncNorm((lambda x: -np.sqrt(np.clip(-x, 0, None)), lambda y: -np.square(y)),
                                              vmin=-cap, vmax=0.0),
                                cmap=LinearSegmentedColormap.from_list("contraction", [(0.1, 0.1, 0.1), (1, 1, 1)])),
                 "other cells: largest exponent (dark: strongly contracting)")]
        # grey ticks: round values at least an eighth of the bar apart on its square-root scale
        grey_ticks = []
        for x in [-5, -3, -2, -1.5, -1, -0.75, -0.5, -0.3, -0.2, -0.1, -0.05, -0.02, 0.0]:
            pos = 1.0 - np.sqrt(-x / cap) if x >= -cap else None
            if pos is None:
                continue
            if grey_ticks and pos - (1.0 - np.sqrt(-grey_ticks[-1] / cap)) < 0.125:
                if x != 0.0:
                    continue
                grey_ticks.pop()
            grey_ticks.append(x)
        for k, (sm, label) in enumerate(bars):
            cax = ax.inset_axes([0.53 * k, -0.155, 0.47, 0.028])
            cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=grey_ticks if k else None)
            cb.ax.tick_params(labelsize=7)
            cb.set_label(label, fontsize=8)
    if args.style != "lyapunov":
        handles += [matplotlib.patches.Patch(color="black", label=f"chaotic (exponent > {args.chaos_level:g}, "
                                                                  "no period)"),
                    matplotlib.patches.Patch(color=none_colour, label="no period, not chaotic")]
    # fold: cyan, long dashes; flip: magenta, short dashes (white would read as a weakly contracting periodic cell)
    styles = {"fold": dict(color="#00e5ff", lw=1.8, ls=(0, (7, 3))), "flip": dict(color="magenta", lw=1.6, ls=(0, (2, 2)))}
    drawn_kinds = []
    for path in args.curves or []:                         # traced bifurcation curves over the brute-force map
        kind, pts = curve_points(DATA / "explore" / path)
        drawn_kinds += [kind] if kind not in drawn_kinds else []
        ax.plot([r["alpha"] for r in pts], [r["log10_temperature"] for r in pts], **styles[kind])
        cusps = [r for r in pts if r["cusp_candidate"]]
        ax.plot([r["alpha"] for r in cusps], [r["log10_temperature"] for r in cusps], "o", ms=6, mfc="none",
                mec="yellow")
    if args.curves:
        handles += [matplotlib.lines.Line2D([], [], label=f"{k} curve", **st) for k, st in styles.items()
                    if k in drawn_kinds]                    # only the kinds drawn on this map
        ax.set_xlim(al[0], al[-1])
        ax.set_ylim(lt[0], lt[-1])
    if args.box:                                           # the region of a zoom shown elsewhere
        a0, a1, t0, t1 = args.box
        ax.add_patch(matplotlib.patches.Rectangle((a0, t0), a1 - a0, t1 - t0, fill=False, edgecolor="white", lw=1.4))
    if args.window:
        ax.set_xlim(*args.window[:2])
        ax.set_ylim(*args.window[2:])
    if args.style == "lyapunov":                            # only the curves need a legend, below the colour bars
        if handles:
            ax.legend(handles=handles, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.235),
                      frameon=False)
    else:                     # legend below the map, in rows, so that the map keeps the full width of the figure
        ax.legend(handles=handles, fontsize=7, ncol=2 if family_handles else 9, loc="upper center",
                  bbox_to_anchor=(0.5, -0.09), frameon=False,
                  title="attractor (leading-token loop)" if family_handles else "period")
    ax.set_xlabel("steering alpha")
    ax.set_ylabel("log10 temperature")
    ax.set_title(args.title if args.title is not None else f"soft-token decoding: {args.scan}")
    fig.tight_layout()
    fig.savefig(src / (args.name or "render.png"), dpi=170, bbox_inches="tight")
    counts = {int(q): int((per == q).sum()) for q in np.unique(per)}
    print(json.dumps({"cells": int(per.size), "chaotic": int(chaos.sum()), "period_counts": counts}))


def stitch(args):
    """Join scan bands (each a full-width scan over consecutive rows of one grid, given in row order) into one scan:
    rows are concatenated, family identities are merged by their token loops, and a summary is written like
    scan's. Refuses bands whose columns differ or whose rows do not continue the same spacing."""
    bands = [DATA / "explore" / b for b in args.bands]
    data = [np.load(b / "scan.npz") for b in bands]
    infos = [json.loads((b / "scan.json").read_text()) for b in bands]
    alpha = data[0]["alpha"]
    if any(not np.allclose(d["alpha"], alpha, rtol=0, atol=1e-15) for d in data):
        raise SystemExit("bands have different alpha columns")
    lt = np.concatenate([d["log10_temperature"] for d in data])
    steps = np.diff(lt)
    if not np.allclose(steps, steps[0], rtol=1e-6, atol=0):
        raise SystemExit("bands do not continue one evenly spaced temperature grid")
    keys, merged, fam_rows = {}, [], []
    for d, info in zip(data, infos):
        local = {f["family"]: f for f in info.get("families", [])}
        fam = d["family"] if "family" in d else np.full(d["period"].shape, -1)
        remap = np.full(max(local, default=-1) + 2, -1, dtype=int)
        for i, f in local.items():
            key = tuple(f["tokens"])
            if key not in keys:
                keys[key] = len(keys)
                merged.append({"family": keys[key], "period": f["period"], "cells": 0, "tokens": f["tokens"]})
            remap[i] = keys[key]
        fam_rows.append(np.where(fam >= 0, remap[np.clip(fam, 0, len(remap) - 1)], -1))
    family = np.concatenate(fam_rows)
    for f in merged:
        f["cells"] = int((family == f["family"]).sum())
    merged.sort(key=lambda f: -f["cells"])
    period = np.concatenate([d["period"] for d in data])
    lyap = np.concatenate([d["lyapunov"] for d in data])
    entropy = np.concatenate([d["entropy"] for d in data])
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "scan.npz", log10_temperature=lt, alpha=alpha, lyapunov=lyap, period=period,
                        entropy=entropy, family=family)
    summary = {"args": vars(args), "bands": [str(b) for b in args.bands], "band_args": [i["args"] for i in infos],
               "seconds": round(sum(i["seconds"] for i in infos), 1), "cells": int(period.size),
               "families": merged, "lyapunov_positive": int((lyap > infos[0]["args"]["chaos_level"]).sum()),
               "period_counts": {int(q): int((period == q).sum()) for q in np.unique(period)},
               "lyapunov_range": [float(lyap.min()), float(lyap.max())]}
    (out / "scan.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("args", "families", "band_args")}))


def readout(args):
    """Top tokens of every point of a saved periodic orbit: the soft token fed back at each step is a mixture,
    E^T p; list its leading components (token, probability) and entropy, step by step around the orbit. The state
    comes from an orbit run (orbit-state.npy) or a branch/curve run (states.npy at --index), with the cell's
    (log10 T, alpha) taken from that run unless given."""
    src = DATA / "explore" / args.run
    if (src / "orbit.json").exists():
        rec = json.loads((src / "orbit.json").read_text())
        X0 = np.load(src / "orbit-state.npy")
        log_t, alpha, period = rec["args"]["log_temperature"], rec["args"]["alpha"], rec["period"]
    else:
        name = next(n for n in ("branch.json", "tline.json", "curve.json") if (src / n).exists())
        rec = json.loads((src / name).read_text())
        X0 = np.load(src / "states.npy")[args.index]
        pt = rec["points"][args.index]
        if name == "curve.json":
            log_t, alpha = pt["log10_temperature"], pt["alpha"]
        elif rec.get("free", "temperature") == "temperature":
            log_t, alpha = pt["lambda"], rec["args"]["alpha"]
        else:
            log_t, alpha = rec["args"]["log_temperature"], pt["lambda"]
        period = rec["period"]
    log_t = args.log_temperature if args.log_temperature is not None else log_t
    alpha = args.alpha if args.alpha is not None else alpha
    loop = SoftLoop(args.model, args.layer, args.threads, args.prompt, args.opener, args.window, args.dtype,
                    args.device)
    v = elpm.load_vectors(args.vectors)[args.concept]
    orb = SoftOrbit(loop, alpha * v, 10.0 ** log_t)
    tok = loop.sm.tok
    X = loop.torch.as_tensor(np.asarray(X0).reshape((1,) + orb.shape), dtype=loop.state_dtype, device=loop.device)
    steps = []
    for i in range(period):
        p = orb.probabilities(X).double().cpu().numpy()
        top = np.argsort(-p)[: args.top]
        steps.append({"step": i, "entropy": float(-(p * np.log(np.maximum(p, 1e-300))).sum()),
                      "top": [[tok.decode([int(t)]), float(p[t])] for t in top]})
        X = orb.f(X)
    closure = float((X - loop.torch.as_tensor(np.asarray(X0).reshape((1,) + orb.shape), dtype=loop.state_dtype,
                                               device=loop.device)).norm())
    out = {"run": args.run, "index": args.index, "log10_temperature": log_t, "alpha": alpha, "period": period,
           "closure_after_one_period": closure, "steps": steps}
    print(json.dumps(out, ensure_ascii=False))


def walk(args):
    """Long trajectories of the soft map at several cells along alpha (one batch, fixed log10 T): after a
    transient, record every step's probabilities of a fixed token list, the entropy and the leading token.
    Because the map contracts all but one direction strongly, a scalar such as p(' it') two steps apart traces a
    one-dimensional return map; near a fold of a 2-cycle it shows the channel (tangency) and the reinjection.
    Writes walk.npz and walk.json; the figure plots p_(n+2) against p_n of the first token for every cell."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loop = SoftLoop(args.model, args.layer, args.threads, args.prompt, args.opener, args.window, args.dtype,
                    args.device)
    torch, tok = loop.torch, loop.sm.tok
    v = elpm.load_vectors(args.vectors)[args.concept]
    alphas = np.repeat(np.array(args.alphas, dtype=float), args.replicas)
    ids = [tok.encode(t, add_special_tokens=False) for t in args.tokens]
    if any(len(i) != 1 for i in ids):
        raise SystemExit(f"each --tokens entry must be one token: {dict(zip(args.tokens, ids))}")
    ids = [i[0] for i in ids]
    X = loop.prepare(alphas[:, None] * v[None])
    temps = np.full(len(alphas), 10.0 ** args.log_temperature)
    probs, ent, lead = [], [], []
    gen = torch.Generator(device=loop.device).manual_seed(args.seed)
    with torch.no_grad():
        for i in range(args.transient + args.steps):
            if args.sample_m:                         # bridge to sampling: mean of m sampled token embeddings
                z = loop.logits(X)
                p = torch.softmax(z / torch.as_tensor(temps, dtype=loop.state_dtype, device=loop.device)[:, None],
                                  dim=-1)
                draws = torch.multinomial(p, args.sample_m, replacement=True, generator=gen)   # (B, m)
                X = torch.cat([X[:, 1:, :], loop.E[draws].mean(1)[:, None, :]], dim=1)
            else:
                X, p = loop.step(X, temps)
            if i >= args.transient:
                probs.append(p[:, ids].double().cpu().numpy())
                ent.append((-(p * torch.log(p.clamp_min(1e-300))).sum(-1)).double().cpu().numpy())
                lead.append(p.argmax(-1).cpu().numpy())
    probs, ent, lead = np.array(probs), np.array(ent), np.array(lead)          # (steps, B, tokens), (steps, B)
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "walk.npz", alphas=alphas, tokens=np.array(args.tokens), token_ids=np.array(ids),
                        probabilities=probs, entropy=ent, leading=lead)
    summary = []
    for b, a in enumerate(alphas):
        top, counts = np.unique(lead[:, b], return_counts=True)
        order = np.argsort(-counts)[:6]
        summary.append({"alpha": float(a), "mean_entropy": float(ent[:, b].mean()),
                        "leading_tokens": [[tok.decode([int(top[j])]), int(counts[j])] for j in order]})
    (out / "walk.json").write_text(json.dumps({"args": vars(args), "cells": summary}, indent=1, ensure_ascii=False))
    cols = min(len(alphas), 4)
    rows_ = int(np.ceil(len(alphas) / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(3.2 * cols, 3.2 * rows_), squeeze=False)
    for b, a in enumerate(alphas):
        ax = axes[b // cols][b % cols]
        q = probs[:, b, 0]
        ax.plot(q[:-2], q[2:], ".", ms=1)
        ax.plot([0, 1], [0, 1], "k-", lw=0.5)
        ax.set_title(f"alpha {a:.5f}", fontsize=8)
        ax.set_xlabel(f"p_n({args.tokens[0]!r})", fontsize=7)
        ax.set_ylabel(f"p_(n+2)({args.tokens[0]!r})", fontsize=7)
    fig.suptitle(f"soft map at log10 T {args.log_temperature}: two-step return map", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "walk.png", dpi=150)
    for row in summary:
        print(json.dumps(row, ensure_ascii=False))


def normal_form(args):
    """Normal form u' = u + a eps + b u^2 of a period-p orbit's fold, in float64, at a point of a curve or branch
    run (bifurcation_curves.fold_normal_form on the p-fold soft map, eps in units of alpha): a includes the
    steering's effect on the prefix; b comes from central differences of JVPs over a step-size table. Writes
    normal-form.json and vectors.npz (x_c, r, l) for `passage`."""
    import bifurcation_curves as bc

    src = DATA / "explore" / args.run
    name = "curve.json" if (src / "curve.json").exists() else "branch.json"
    rec = json.loads((src / name).read_text())
    index = args.index
    if args.nearest_fold:                     # the branch point whose real multiplier is nearest +1 (the turn)
        gaps = [abs(1.0 - p["real_multiplier_nearest_one"]) if p.get("real_multiplier_nearest_one") is not None
                else np.inf for p in rec["points"]]
        index = int(np.argmin(gaps))
        if not np.isfinite(gaps[index]):
            raise SystemExit("no point with a real multiplier near +1 in the run")
    pt = rec["points"][index]
    x = np.load(src / "states.npy")[index].ravel()
    ra = rec["args"]
    if name == "curve.json":
        start = json.loads((DATA / "explore" / ra["start"] / ("branch.json" if (DATA / "explore" / ra["start"] /
                                                                    "branch.json").exists() else "tline.json"))
                           .read_text())["args"]
        log_t, alpha = pt["log10_temperature"], pt["alpha"]
    else:
        start = ra
        log_t, alpha = ((pt["lambda"], ra["alpha"]) if rec.get("free", "temperature") == "temperature"
                        else (ra["log_temperature"], pt["lambda"]))
    loop = SoftLoop(start["model"], start["layer"], args.threads, start["prompt"], start["opener"], start["window"],
                    "float64", args.device)
    v = elpm.load_vectors(start["vectors"])[start["concept"]]
    fam = soft_family(loop, v, int(rec["period"]), args.alpha_scale, args.chunk)
    theta = np.array([log_t, alpha * args.alpha_scale])
    res = float(np.linalg.norm(fam.residual(x, theta)) / np.linalg.norm(x))
    # eps in units of alpha (default) or of log10 T (--param temperature, for folds approached in temperature)
    param, scale = (0, 1.0) if args.param == "temperature" else (1, args.alpha_scale)
    nf = bc.fold_normal_form(fam, x, theta, param=param, param_scale=scale, steps=tuple(args.steps))
    if abs(nf["multiplier"][1]) > 1e-9 or abs(nf["multiplier"][0] - 1.0) > 0.01:      # classify before trusting
        raise SystemExit(f"multiplier nearest +1 is {nf['multiplier']}: not a fold point (refusing coefficients)")
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "vectors.npz", x_c=x, r=nf.pop("r"), l=nf.pop("l"))
    rec_out = {"args": vars(args), "source": {"run": args.run, "index": index, "log10_temperature": log_t,
               "alpha": alpha, "period": int(rec["period"]), "model": start["model"], "layer": start["layer"],
               "prompt": start["prompt"], "opener": start["opener"], "window": start["window"],
               "vectors": start["vectors"], "concept": start["concept"]},
               "orbit_residual": res, **nf,
               "pi_over_sqrt_ab": float(np.pi / np.sqrt(abs(nf["ab"])))}
    (out / "normal-form.json").write_text(json.dumps(rec_out, indent=1))
    print(json.dumps({k: rec_out[k] for k in ("orbit_residual", "multiplier", "second_modulus", "a", "b", "ab",
                                              "b_steps", "pi_over_sqrt_ab")}))


def passage(args):
    """Passage through the bottleneck of a fold, measured on the soft map and predicted by its normal form: from
    controlled entrances x_c + u_in r (and transverse controls x_c + u_in r + |u_in| w with l^T w = 0) at
    alpha = alpha_c + eps, iterate the p-fold map G and count the G-iterates until u = l^T (x - x_c) exceeds
    u_out; compare with the discrete quadratic map and the arctangent formula. All rows run in one batch."""
    import bifurcation_curves as bc

    src = DATA / "explore" / args.normal_form
    nf = json.loads((src / "normal-form.json").read_text())
    vec = np.load(src / "vectors.npz")
    xc, r, l = vec["x_c"], vec["r"], vec["l"]
    so = nf["source"]
    a, b = nf["a"], nf["b"]
    sign = 1.0 if a * b > 0 and a > 0 else -1.0                   # orient u so that a, b > 0 (ab > 0 needed)
    if a * b <= 0:
        raise SystemExit(f"ab = {a * b:.4g} is not positive: no bottleneck for eps > 0 in this orientation")
    a, b, r, l = abs(a), abs(b), sign * r, sign * l
    U = args.u if args.u is not None else 0.1 / b
    rng = np.random.default_rng(args.seed)
    w = rng.normal(size=xc.size)
    w -= (l @ w) / (l @ r) * r                                  # transverse: l^T w = 0
    w /= np.linalg.norm(w)
    rows = []
    for eps in args.epsilons:
        if args.natural:                                        # free trajectories from the decoding start
            for k in range(args.natural):
                rows.append({"eps": eps, "entrance": f"natural-{k}", "u_in": None, "x0": None})
            continue
        for kind, u_in in (("full", -U), ("midpoint", 0.0)):
            rows.append({"eps": eps, "entrance": kind, "u_in": u_in, "x0": xc + u_in * r})
        rows.append({"eps": eps, "entrance": "full+transverse", "u_in": -U, "x0": xc - U * r + U * w})
    loop = SoftLoop(so["model"], so["layer"], args.threads, so["prompt"], so["opener"], so["window"], "float64",
                    args.device)
    torch = loop.torch
    v = elpm.load_vectors(so["vectors"])[so["concept"]]
    in_temperature = nf.get("args", {}).get("param") == "temperature"     # a fold approached in log10 T (unsteered)
    alphas = np.array([so["alpha"] + (0.0 if in_temperature else row["eps"]) for row in rows])
    start = loop.prepare(alphas[:, None] * v[None]).double().cpu().numpy()
    shape = (len(rows), loop.K, loop.E.shape[1])
    for i, row in enumerate(rows):
        if row["x0"] is None:                                   # independent starts: tiny random displacements
            row["x0"] = start[i].ravel() + 1e-6 * np.linalg.norm(start[i]) * rng.normal(size=start[i].size)
    X = torch.as_tensor(np.array([row["x0"] for row in rows]).reshape(shape), dtype=loop.state_dtype,
                        device=loop.device)
    temps = (10.0 ** np.array([so["log10_temperature"] + row["eps"] for row in rows]) if in_temperature
             else np.full(len(rows), 10.0 ** so["log10_temperature"]))
    p = so["period"]
    lt = torch.as_tensor(l, dtype=loop.state_dtype, device=loop.device)
    xct = torch.as_tensor(xc, dtype=loop.state_dtype, device=loop.device)
    rt = torch.as_tensor(r, dtype=loop.state_dtype, device=loop.device)
    us, trans, count = [], [], [None] * len(rows)
    tic = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.transient * p if args.natural else 0):
            X, _ = loop.step(X, temps)
        for g in range(1, args.max_g + 1):
            for _ in range(p):
                X, _ = loop.step(X, temps)
            d = X.reshape(len(rows), -1) - xct
            u_t = d @ lt
            u = u_t.double().cpu().numpy()
            us.append(u)
            trans.append((d - u_t[:, None] * rt[None]).norm(dim=1).double().cpu().numpy())
            for i, ui in enumerate(u):
                if count[i] is None and rows[i]["u_in"] is not None and ui > args.u_out_factor * U:
                    count[i] = g
            if not args.natural and all(c is not None for c in count):
                break
    us, trans = np.array(us), np.array(trans)                       # (G, rows): critical and transverse parts
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "u.npy", us)
    np.save(out / "transverse.npy", trans)
    table = []
    for i, row in enumerate(rows):
        u_out = args.u_out_factor * U
        if row["u_in"] is None:                                 # natural rows: analysed offline from u.npy
            table.append({"eps": row["eps"], "entrance": row["entrance"], "held_out": row["eps"] not in
                          args.development})
            continue
        pred = bc.passage_count(row["u_in"], a, b, row["eps"], u_out)
        table.append({"eps": row["eps"], "entrance": row["entrance"], "u_in": row["u_in"], "u_out": u_out,
                      "measured_G": count[i], "predicted_discrete_G": pred,
                      "predicted_arctan_G": bc.passage_arctan(row["u_in"], u_out, a, b, row["eps"]),
                      "relative_error": (None if count[i] is None or pred is None else (count[i] - pred) / pred),
                      "held_out": row["eps"] not in args.development})
    rec = {"args": vars(args), "normal_form": {k: nf[k] for k in ("a", "b", "ab", "multiplier")}, "U": U,
           "seconds": round(time.perf_counter() - tic, 1), "rows": table}
    (out / "passage.json").write_text(json.dumps(rec, indent=1))
    for t in table:
        print(json.dumps(t))


def branch_plot(args):
    """Leading multipliers along branch or tline runs against log10 T: modulus of the largest (log scale) and the
    real parts of the real multipliers, with turns and classified crossings marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for path in args.runs:
        src = DATA / "explore" / path
        rec = json.loads((src / ("branch.json" if (src / "branch.json").exists() else "tline.json")).read_text())
        pts = rec["points"]
        lam = np.array([r["lambda"] for r in pts])
        rho = np.array([r["spectral_radius"] if r["spectral_radius"] is not None else np.nan for r in pts])
        lead = np.array([r["multipliers"][0][0] if r["multipliers"] else np.nan for r in pts])
        line, = ax1.semilogy(lam, rho, ".-", ms=3, label=f"{path} (period {rec['period']})")
        ax2.plot(lam, lead, ".-", ms=3, color=line.get_color())
        for r in pts:
            if "turn_lambda_estimate" in r:
                for ax in (ax1, ax2):
                    ax.axvline(r["turn_lambda_estimate"], color=line.get_color(), ls=":", lw=1)
        for c in rec.get("crossings", []):
            ax2.annotate(c["type"].split(" (")[0], (np.mean(c["between_log10_T"]), c["crossing_multiplier"][0]),
                         fontsize=7, color=line.get_color())
    ax1.axhline(1.0, color="k", lw=0.8)
    ax2.axhline(1.0, color="k", lw=0.8)
    ax2.axhline(-1.0, color="k", lw=0.8)
    ax2.axhline(0.0, color="grey", lw=0.5)
    ax1.set_ylabel("spectral radius of the monodromy")
    ax2.set_ylabel("largest-modulus multiplier (real part)")
    ax2.set_xlabel("log10 temperature")
    ax1.legend(fontsize=7)
    fig.tight_layout()
    out = DATA / "explore" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(json.dumps({"figure": str(out)}))


def windows(args):
    """Candidate windows in a finished scan: connected regions (4-neighbour) of one period p >= 2, ranked by
    how much of their border touches chaotic cells (periodic islands embedded in chaos)."""
    from scipy import ndimage

    src = DATA / "explore" / args.scan
    d = np.load(src / "scan.npz")
    lt, al, per, ly = d["log10_temperature"], d["alpha"], d["period"], d["lyapunov"]
    chaos = ly > args.chaos_level
    found = []
    for q in np.unique(per):
        if q < args.min_period:
            continue
        lab, n = ndimage.label(per == q)
        for k in range(1, n + 1):
            region = lab == k
            area = int(region.sum())
            if area < args.min_area:
                continue
            border = ndimage.binary_dilation(region) & ~region
            touching = float(chaos[border].mean()) if border.any() else 0.0
            rows, cols = np.nonzero(region)
            i, j = int(np.round(rows.mean())), int(np.round(cols.mean()))
            if not region[i, j]:                                  # centroid outside a curved region
                i, j = int(rows[len(rows) // 2]), int(cols[len(cols) // 2])
            found.append({"period": int(q), "cells": area, "border_cells": int(border.sum()),
                          "border_chaotic_fraction": round(touching, 3),
                          "inside_point": {"log10_temperature": float(lt[i]), "alpha": float(al[j])},
                          "log10_temperature_range": [float(lt[rows.min()]), float(lt[rows.max()])],
                          "alpha_range": [float(al[cols.min()]), float(al[cols.max()])]})
    found.sort(key=lambda w: (-w["border_chaotic_fraction"], -w["cells"]))
    (src / "windows.json").write_text(json.dumps(found, indent=1))
    for w in found[: args.show]:
        print(json.dumps(w))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sc = sub.add_parser("scan")
    sc.add_argument("--model", choices=sorted(ela.MODELS), default="qwen0.5b")
    sc.add_argument("--layer", type=int, default=12)
    sc.add_argument("--threads", type=int, default=8)
    sc.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    sc.add_argument("--device", default="cpu", help="cpu, cuda, cuda:N, or auto (layers spread over all GPUs)")
    sc.add_argument("--weights", default="native", choices=("native", "bf16"),
                    help="bf16: decoder linear weights stay in their stored bfloat16 and compute in float64 (exact)")
    sc.add_argument("--chunk", type=int, default=0, help="cells per batch (0: all at once)")
    sc.add_argument("--prompt", choices=sorted(ela.PROMPTS), default="open")
    sc.add_argument("--opener", default=ela.OPENER)
    sc.add_argument("--window", type=int, default=1)
    sc.add_argument("--vectors", default="llm-phase-map/vectors-l12", help='explore-relative vector set; "" for none')
    sc.add_argument("--concept", default="philosophy", help="a vector name in --vectors (e.g. hostility-vs-kindness)")
    sc.add_argument("--log-temperature", type=float, nargs=2, default=[-1.0, 0.3])
    sc.add_argument("--alpha", type=float, nargs=2, default=[-0.12, 0.12])
    sc.add_argument("--resolution", type=int, nargs=2, default=[16, 16], help="rows (temperature) and columns (alpha)")
    sc.add_argument("--transient", type=int, default=300)
    sc.add_argument("--samples", type=int, default=300)
    sc.add_argument("--record", type=int, default=80)
    sc.add_argument("--max-period", type=int, default=32)
    sc.add_argument("--tol", type=float, default=1e-6)
    sc.add_argument("--eps", type=float, default=1e-6)
    sc.add_argument("--chaos-level", type=float, default=1e-3)
    sc.add_argument("--init-states", default=None, help="explore-relative scan folder whose states.npz (same grid) "
                    "gives each cell's start (warm start, e.g. the previous frame of a morph)")
    sc.add_argument("--save-states", action="store_true", help="write each cell's final window to states.npz")
    sc.add_argument("--projections", action="store_true",
                    help="save the recorded states projected on the unit steering vector and a fixed random direction "
                         "(bifurcation diagrams)")
    sc.add_argument("--out", required=True)
    ob = sub.add_parser("orbit")
    ob.add_argument("--model", choices=sorted(ela.MODELS), default="qwen0.5b")
    ob.add_argument("--layer", type=int, default=12)
    ob.add_argument("--threads", type=int, default=8)
    ob.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    ob.add_argument("--device", default="cpu")
    ob.add_argument("--prompt", choices=sorted(ela.PROMPTS), default="open")
    ob.add_argument("--opener", default=ela.OPENER)
    ob.add_argument("--window", type=int, default=1)
    ob.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    ob.add_argument("--concept", default="philosophy", help="a vector name in --vectors")
    ob.add_argument("--log-temperature", type=float, required=True)
    ob.add_argument("--alpha", type=float, required=True)
    ob.add_argument("--period", type=int, default=0, help="0: detect from the attractor reached")
    ob.add_argument("--transient", type=int, default=600)
    ob.add_argument("--max-period", type=int, default=32)
    ob.add_argument("--detect-tol", type=float, default=1e-6)
    ob.add_argument("--tol", type=float, default=1e-11)
    ob.add_argument("--max-newton", type=int, default=8)
    ob.add_argument("--jacobian", default="ad", choices=("ad", "fd"), help="forward-mode AD or central differences")
    ob.add_argument("--h-rel", type=float, default=1e-5)
    ob.add_argument("--chunk", type=int, default=512)
    ob.add_argument("--out", required=True)
    tl = sub.add_parser("tline")
    tl.add_argument("--model", choices=sorted(ela.MODELS), default="qwen0.5b")
    tl.add_argument("--layer", type=int, default=12)
    tl.add_argument("--threads", type=int, default=8)
    tl.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    tl.add_argument("--device", default="cpu")
    tl.add_argument("--prompt", choices=sorted(ela.PROMPTS), default="open")
    tl.add_argument("--opener", default=ela.OPENER)
    tl.add_argument("--window", type=int, default=1)
    tl.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    tl.add_argument("--concept", default="philosophy", help="a vector name in --vectors")
    tl.add_argument("--log-temperature", type=float, required=True)
    tl.add_argument("--alpha", type=float, required=True)
    tl.add_argument("--period", type=int, default=0, help="0: detect from the attractor reached")
    tl.add_argument("--transient", type=int, default=600)
    tl.add_argument("--max-period", type=int, default=32)
    tl.add_argument("--detect-tol", type=float, default=1e-6)
    tl.add_argument("--direction", type=int, default=1, choices=(-1, 1))
    tl.add_argument("--steps", type=int, default=40)
    tl.add_argument("--ds", type=float, default=0.01)
    tl.add_argument("--ds-min", type=float, default=1e-6)
    tl.add_argument("--ds-max", type=float, default=0.05)
    tl.add_argument("--beyond", type=float, default=0.02, help="stop this far in log10 T past the first turn")
    tl.add_argument("--bounds", type=float, nargs=2, default=[-1.5, 0.5], help="log10 T range")
    tl.add_argument("--tol", type=float, default=1e-10)
    tl.add_argument("--max-newton", type=int, default=8)
    tl.add_argument("--gmres-tol", type=float, default=1e-9)
    tl.add_argument("--restart", type=int, default=60)
    tl.add_argument("--gmres-iter", type=int, default=3)
    tl.add_argument("--k", type=int, default=8, help="block size of the subspace iteration for multipliers")
    tl.add_argument("--subspace-iter", type=int, default=40)
    tl.add_argument("--chunk", type=int, default=512)
    tl.add_argument("--out", required=True)
    rd = sub.add_parser("render")
    rd.add_argument("--scan", required=True, help="explore-relative directory of a finished scan")
    rd.add_argument("--max-period", type=int, default=None,
                    help="periods above this share one 'other' legend entry (default: every period its own colour)")
    rd.add_argument("--chaos-level", type=float, default=1e-3,
                    help="a cell is drawn chaotic when its exponent exceeds this and no period was found (the "
                         "paper's criterion; renders before 15 Oct used 0.005 without the period condition)")
    rd.add_argument("--curves", nargs="*", help="explore-relative directories of curve runs to draw on the map")
    rd.add_argument("--name", default="", help="output file name inside the scan directory (default render.png)")
    rd.add_argument("--title", default=None, help="figure title (default: the scan directory)")
    rd.add_argument("--window", type=float, nargs=4, default=None, metavar=("ALPHA0", "ALPHA1", "LOGT0", "LOGT1"),
                    help="zoom to this part of the map")
    rd.add_argument("--style", default="period", choices=("period", "lyapunov"),
                    help="lyapunov: chaos coloured by the largest exponent, periodic cells in grey by contraction "
                         "(superstable curves dark), as in the shrimp literature")
    rd.add_argument("--box", type=float, nargs=4, default=None, metavar=("ALPHA0", "ALPHA1", "LOGT0", "LOGT1"),
                    help="outline this region on the map (e.g. the region of a zoom shown elsewhere)")
    rd.add_argument("--families", type=int, default=0,
                    help="colour the N most common attractor identities (token loops) instead of periods")
    rp = sub.add_parser("replot", help="redraw a finished scan's overview figure (scan.png) from its saved files")
    rp.add_argument("--scan", required=True, help="explore-relative directory of a finished scan")
    rp.add_argument("--out", required=True, help="output image path")
    ro = sub.add_parser("readout")
    ro.add_argument("--run", required=True, help="explore-relative orbit, branch, tline or curve run directory")
    ro.add_argument("--index", type=int, default=0, help="point of a branch/tline/curve run")
    ro.add_argument("--log-temperature", type=float, default=None)
    ro.add_argument("--alpha", type=float, default=None)
    ro.add_argument("--top", type=int, default=5)
    ro.add_argument("--model", choices=sorted(ela.MODELS), default="qwen0.5b")
    ro.add_argument("--layer", type=int, default=12)
    ro.add_argument("--threads", type=int, default=8)
    ro.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    ro.add_argument("--device", default="cpu")
    ro.add_argument("--prompt", choices=sorted(ela.PROMPTS), default="open")
    ro.add_argument("--opener", default=ela.OPENER)
    ro.add_argument("--window", type=int, default=1)
    ro.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    ro.add_argument("--concept", default="philosophy", help="a vector name in --vectors")
    bp = sub.add_parser("branch-plot")
    bp.add_argument("--runs", nargs="+", required=True, help="explore-relative branch or tline run directories")
    bp.add_argument("--out", required=True, help="explore-relative output png")
    wi = sub.add_parser("windows")
    wi.add_argument("--scan", required=True, help="explore-relative directory of a finished scan")
    wi.add_argument("--chaos-level", type=float, default=0.005)
    wi.add_argument("--min-period", type=int, default=2)
    wi.add_argument("--min-area", type=int, default=4)
    wi.add_argument("--show", type=int, default=15)
    pr = sub.add_parser("precision")
    pr.add_argument("--model", choices=sorted(ela.MODELS), default="qwen0.5b")
    pr.add_argument("--layer", type=int, default=12)
    pr.add_argument("--threads", type=int, default=8)
    pr.add_argument("--device", default="cpu", help="device of the float32 map")
    pr.add_argument("--device64", default="cpu", help="device of the float64 reference map")
    pr.add_argument("--prompt", choices=sorted(ela.PROMPTS), default="open")
    pr.add_argument("--opener", default=ela.OPENER)
    pr.add_argument("--window", type=int, default=1)
    pr.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    pr.add_argument("--concept", default="philosophy", help="a vector name in --vectors")
    pr.add_argument("--cells", type=float, nargs=2, action="append", required=True, metavar=("LOG10_T", "ALPHA"))
    pr.add_argument("--transient", type=int, default=600)
    pr.add_argument("--max-period", type=int, default=32)
    pr.add_argument("--steps", type=int, default=32)
    pr.add_argument("--out", required=True)
    bb = sub.add_parser("branch")
    bb.add_argument("--model", choices=sorted(ela.MODELS), default="qwen0.5b")
    bb.add_argument("--layer", type=int, default=12)
    bb.add_argument("--threads", type=int, default=8)
    bb.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    bb.add_argument("--reference-dtype", default="float64", choices=("none", "float64"))
    bb.add_argument("--device", default="cpu")
    bb.add_argument("--prompt", choices=sorted(ela.PROMPTS), default="open")
    bb.add_argument("--opener", default=ela.OPENER)
    bb.add_argument("--window", type=int, default=1)
    bb.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    bb.add_argument("--concept", default="philosophy", help="a vector name in --vectors")
    bb.add_argument("--log-temperature", type=float, required=True)
    bb.add_argument("--alpha", type=float, required=True)
    bb.add_argument("--period", type=int, default=0, help="0: detect from the attractor reached")
    bb.add_argument("--transient", type=int, default=600)
    bb.add_argument("--max-period", type=int, default=32)
    bb.add_argument("--detect-tol", type=float, default=1e-9)
    bb.add_argument("--alpha-scale", type=float, default=10.0)
    bb.add_argument("--direction", type=int, default=1, choices=(-1, 1))
    bb.add_argument("--steps", type=int, default=60)
    bb.add_argument("--ds", type=float, default=0.004)
    bb.add_argument("--ds-min", type=float, default=1e-7)
    bb.add_argument("--ds-max", type=float, default=0.01)
    bb.add_argument("--beyond", type=float, default=0.01, help="stop this far past the first turn (free parameter)")
    bb.add_argument("--free", default="temperature", choices=("temperature", "alpha", "alpha2"),
                    help="the continued parameter (the other stays fixed; alpha2 with --block2)")
    bb.add_argument("--block2", type=int, default=None,
                    help="second steering block (two-block plane, run TB; log10 T stays at --log-temperature)")
    bb.add_argument("--vectors2", default=None)
    bb.add_argument("--concept2", default=None)
    bb.add_argument("--alpha2", type=float, default=None, help="steering strength of the second block")
    bb.add_argument("--march-from", type=float, nargs=2, default=None, metavar=("A1", "A2"),
                    help="two-block plane: detect the period from the fixed start at this cell, then continue the orbit "
                         "by Newton through --march-path to (--alpha, --alpha2) before the branch starts")
    bb.add_argument("--march-path", default=None, help="explore-relative JSON list of [alpha, alpha2] waypoints")
    bb.add_argument("--bounds", type=float, nargs=2, default=None,
                    help="range of the free parameter (default log10 T -1.5..0.5, alpha -0.2..0.2)")
    bb.add_argument("--tol", type=float, default=1e-10)
    bb.add_argument("--max-newton", type=int, default=10)
    bb.add_argument("--k", type=int, default=12, help="multipliers recorded per point")
    bb.add_argument("--exact-jvp", action=argparse.BooleanOptionalAction, default=True,
                    help="refined multipliers from float64 JVPs of the reference")
    bb.add_argument("--refine-k", type=int, default=4,
                    help="Arnoldi steps for refined leading multipliers (0: dense float32 multipliers only)")
    bb.add_argument("--chunk", type=int, default=512)
    bb.add_argument("--out", required=True)
    st = sub.add_parser("stitch")
    st.add_argument("--bands", nargs="+", required=True, help="explore-relative band scans in row order")
    st.add_argument("--out", required=True)
    nfp = sub.add_parser("normal-form")
    nfp.add_argument("--run", required=True, help="explore-relative curve or branch run holding the fold point")
    nfp.add_argument("--index", type=int, default=0)
    nfp.add_argument("--nearest-fold", action="store_true", help="use the point whose real multiplier is nearest +1")
    nfp.add_argument("--param", choices=("alpha", "temperature"), default="alpha",
                     help="unfolding parameter of the normal form (eps in alpha, or in log10 T)")
    nfp.add_argument("--threads", type=int, default=8)
    nfp.add_argument("--device", default="cpu")
    nfp.add_argument("--alpha-scale", type=float, default=10.0)
    nfp.add_argument("--chunk", type=int, default=512)
    nfp.add_argument("--steps", type=float, nargs="+", default=[1e-2, 3e-3, 1e-3, 3e-4, 1e-4],
                     help="relative finite-difference steps for b")
    nfp.add_argument("--out", required=True)
    ps = sub.add_parser("passage")
    ps.add_argument("--normal-form", required=True, help="explore-relative normal-form run")
    ps.add_argument("--epsilons", type=float, nargs="+", required=True)
    ps.add_argument("--development", type=float, nargs="*", default=[],
                    help="epsilons used to set up the measurement; the others are held out")
    ps.add_argument("--u", type=float, default=None, help="entrance distance U (default 0.1 / b)")
    ps.add_argument("--u-out-factor", type=float, default=1.0, help="exit at u = factor * U")
    ps.add_argument("--max-g", type=int, default=512)
    ps.add_argument("--natural", type=int, default=0,
                    help="instead of controlled entrances: this many free trajectories per eps from the decoding "
                         "start, recorded for --max-g iterates after --transient")
    ps.add_argument("--transient", type=int, default=100, help="G-iterates discarded before recording (natural)")
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument("--threads", type=int, default=8)
    ps.add_argument("--device", default="cpu")
    ps.add_argument("--out", required=True)
    wk = sub.add_parser("walk")
    wk.add_argument("--model", choices=sorted(ela.MODELS), default="qwen0.5b")
    wk.add_argument("--layer", type=int, default=12)
    wk.add_argument("--threads", type=int, default=8)
    wk.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    wk.add_argument("--device", default="cpu")
    wk.add_argument("--prompt", choices=sorted(ela.PROMPTS), default="open")
    wk.add_argument("--opener", default=ela.OPENER)
    wk.add_argument("--window", type=int, default=1)
    wk.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    wk.add_argument("--concept", default="philosophy", help="a vector name in --vectors")
    wk.add_argument("--log-temperature", type=float, required=True)
    wk.add_argument("--alphas", type=float, nargs="+", required=True)
    wk.add_argument("--transient", type=int, default=300)
    wk.add_argument("--steps", type=int, default=4000)
    wk.add_argument("--tokens", nargs="+", default=[" it", " a", " there", " is", ".", " This"])
    wk.add_argument("--replicas", type=int, default=1, help="trajectories per alpha (independent with --sample-m)")
    wk.add_argument("--sample-m", type=int, default=0,
                    help="feed back the mean of m sampled token embeddings instead of the expected embedding "
                         "(m = 1: sampled decoding of the same fixed-prefix, one-token-window system)")
    wk.add_argument("--seed", type=int, default=0)
    wk.add_argument("--out", required=True)
    cv = sub.add_parser("curve")
    cv.add_argument("--start", required=True, help="explore-relative directory of a finished branch or tline run")
    cv.add_argument("--reference-dtype", default="float64", choices=("none", "float64"))
    cv.add_argument("--kind", default="auto", choices=("auto", "fold", "flip"))
    cv.add_argument("--start-index", type=int, default=-1, help="tline point to start from (-1: nearest the boundary)")
    cv.add_argument("--threads", type=int, default=8)
    cv.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    cv.add_argument("--device", default="cpu")
    cv.add_argument("--alpha-scale", type=float, default=10.0, help="alpha is continued as alpha * scale")
    cv.add_argument("--direction", type=int, default=1, choices=(-1, 1),
                    help="sign of the initial change of the parameter that varied along the start run")
    cv.add_argument("--steps", type=int, default=60)
    cv.add_argument("--ds", type=float, default=0.005)
    cv.add_argument("--ds-min", type=float, default=1e-6)
    cv.add_argument("--ds-max", type=float, default=0.02)
    cv.add_argument("--tol", type=float, default=1e-10)
    cv.add_argument("--gtol", type=float, default=1e-4, help="float32 monodromies make g accurate to about 1e-5")
    cv.add_argument("--max-newton", type=int, default=8)
    cv.add_argument("--exact-jvp", action=argparse.BooleanOptionalAction, default=True,
                    help="JVPs (g by GMRES, refined multipliers) from the float64 reference; float32 derivatives of "
                         "the P8 map are off by 3 %%")
    cv.add_argument("--refine-k", type=int, default=4, help="Arnoldi steps for the refined critical multiplier")
    cv.add_argument("--gmres-tol", type=float, default=None,
                    help="relative GMRES tolerance for g (default 1e-9 with exact JVPs, else 1e-5: float32 JVPs "
                         "reach about 4e-6 at the P8 fold)")
    cv.add_argument("--gmres-maxiter", type=int, default=1, help="GMRES restart cycles (30 products each) per g")
    cv.add_argument("--log-temperature-bounds", type=float, nargs=2, default=[-1.5, 0.5])
    cv.add_argument("--alpha-bounds", type=float, nargs=2, default=[-0.2, 0.2])
    cv.add_argument("--alpha2-bounds", type=float, nargs=2, default=[-0.2, 0.2],
                    help="second-block range (start runs of a two-block plane)")
    cv.add_argument("--chunk", type=int, default=512)
    cv.add_argument("--out", required=True)
    args = parser.parse_args()
    {"scan": scan, "orbit": orbit, "tline": tline, "branch": branch, "curve": curve, "precision": precision,
     "walk": walk, "normal-form": normal_form, "passage": passage, "stitch": stitch,
     "render": render, "windows": windows, "branch-plot": branch_plot, "readout": readout,
     "replot": replot}[args.command](args)


if __name__ == "__main__":
    main()
