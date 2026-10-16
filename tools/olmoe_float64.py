"""Float64 evaluation of OLMoE (allenai/OLMoE-1B-7B) for the soft-token map.

Hugging Face's OLMoE code computes the router softmax in float32 (dtype=torch.float) even for a float64 model.
prepare() rebuilds the router's forward with that line computing in the input dtype (float64 for float64 inputs) and
fails loudly if the line is not found. It also replaces the experts' forward so that the stored bfloat16 expert
weights (stacked tensors gate_up_proj and down_proj) stay bfloat16 and each expert's slice is converted to float64
when it is used. bfloat16 -> float64 is exact, so this equals a float64 model whose expert weights are the stored
values, in a fraction of the memory. Every other parameter and buffer is cast to float64. The RMS normalizations and
rotary tables are left to explore_llm_attractors.keep_float64_norms, which the loader applies afterwards as for the
dense models.
"""
import sys

import torch
import torch.nn.functional as F

import qwen35_float64 as q35


def prepare(model, keep_experts_bf16=True):
    module = sys.modules[type(model.model.layers[0]).__module__]
    if module.__name__.split(".")[-1] != "modeling_olmoe":
        raise SystemExit(f"olmoe_float64: not an OLMoE model ({module.__name__})")
    patched = []
    router = q35._one(module, "TopKRouter")
    if "_float32_forward" not in router.__dict__:
        router._float32_forward = router.forward
        router.forward = q35._rebuild(router.forward, [q35.ROUTER_SOFTMAX], vars(module))
    patched.append(router.__name__)
    experts = q35._one(module, "Experts")

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
    kept = 0
    for m in model.modules():
        keep = keep_experts_bf16 and isinstance(m, experts)
        for name, p in list(m.named_parameters(recurse=False)):
            if keep:
                if p.dtype != torch.bfloat16:
                    raise SystemExit(f"expert weights are {p.dtype}, not the stored bfloat16")
                kept += p.numel()
                continue
            p.data = p.data.to(torch.float64)
        for name, b in list(m.named_buffers(recurse=False)):
            if b.is_floating_point():
                m._buffers[name] = b.to(torch.float64)
    return {"module": module.__name__, "patched": patched, "experts_bf16_parameters": int(kept)}
