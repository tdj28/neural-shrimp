"""Routing probe for mixture-of-experts models (OLMoE, Qwen3.5-MoE): margins, switches and frozen routing.

Hugging Face's top-k routers (classes named ...TopKRouter) return (router_logits, top_k_weights, top_k_indices) for
the flattened tokens of a batch. In the soft-token loop every forward pass has one token per row, so the router's
rows are the batch rows. The probe wraps every router instance and can do three things.
- **Margin:** with capture on, it records for every router and row the gap l_(k) - l_(k+1) between the k-th and
  (k+1)-th largest router logits (k = top_k), in float64. A perturbation must move the logits by about this gap to
  switch the expert set, so a small gap means the row is close to a routing boundary, where the map is not smooth.
  Logit gaps, not probability gaps, are the natural scale: with many experts the probabilities near the cut are all
  small, whatever the distance to the boundary.
- **Switches:** for given (row, reference row) pairs it records whether their natural top-k expert sets differ.
- **Frozen routing:** for given (row, reference row) pairs it replaces the row's expert set by the reference row's.
  The weights are the row's own routing probabilities on that set, renormalized when the router renormalizes. The
  frozen row then follows the smooth branch of the map on the reference row's routing region, so comparing a free twin
  with a frozen twin separates smooth stretching from stretching by routing switches.
Routers that renormalize their top-k weights (Qwen3.5-MoE always; OLMoE when config.norm_topk_prob) are detected from
their attribute norm_topk_prob, defaulting to True.
"""
import torch


class RoutingProbe:
    def __init__(self, model):
        self.routers = [m for m in model.modules() if type(m).__name__.endswith("TopKRouter")]
        if not self.routers:
            raise SystemExit("no ...TopKRouter modules: not a supported mixture-of-experts model")
        self.capture = False
        self.freeze = None            # (rows, reference rows), python lists
        self.switch_pairs = None      # (rows, reference rows), python lists
        self.margins = []             # per router call (capture on): float64 tensor (rows,)
        self.switches = []            # per router call (capture on, pairs set): bool tensor (pairs,)
        for r in self.routers:
            r._probe_orig_forward = r.forward
            r.forward = self._wrap(r)

    def _wrap(self, router):
        probe = self

        def forward(hidden_states):
            logits, weights, idx = router._probe_orig_forward(hidden_states)
            k = idx.shape[-1]
            if probe.capture or probe.freeze is not None or probe.switch_pairs is not None:
                probs = torch.softmax(logits.to(torch.float64), dim=-1)
            if probe.capture:
                top = torch.topk(logits.to(torch.float64), k + 1, dim=-1).values
                probe.margins.append((top[:, k - 1] - top[:, k]).detach())
                if probe.switch_pairs is not None:
                    rows, refs = probe.switch_pairs
                    a = torch.sort(idx[rows], dim=-1).values
                    b = torch.sort(idx[refs], dim=-1).values
                    probe.switches.append((a != b).any(dim=-1).detach())
            if probe.freeze is not None:
                rows, refs = probe.freeze
                idx = idx.clone()
                idx[rows] = idx[refs]
                w = probs[rows].gather(1, idx[rows])
                if getattr(router, "norm_topk_prob", True):
                    w = w / w.sum(dim=-1, keepdim=True)
                weights = weights.clone()
                weights[rows] = w.to(weights.dtype)
            return logits, weights, idx
        return forward

    def reset(self):
        self.margins, self.switches = [], []

    def summary(self, rows):
        """Per row of `rows`: the smallest margin over the routers of the captured calls; and per switch pair, the
        number of routers (calls) whose expert sets differed."""
        m = torch.stack(self.margins)[:, rows].min(dim=0).values if self.margins else None
        s = torch.stack(self.switches).sum(dim=0) if self.switches else None
        return m, s
