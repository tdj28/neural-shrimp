"""Usage: python tools/recreate_p8.py docs/steering-vectors/vectors-l12/vectors.npz [steps] [reference.npz]

Recreate the period-8 loop of the soft-token map of Qwen2.5-0.5B-Instruct (standalone, float64, CPU).

Needs: torch, transformers, numpy, and vectors.npz (philosophy direction, layer 12) from the public repository
docs/steering-vectors/vectors-l12/vectors.npz.
"""
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL, REVISION = "Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"
ALPHA, LOG10_T, LAYER, STEPS = 0.071674, -0.115, 12, int(sys.argv[2]) if len(sys.argv) > 2 else 1500
torch.set_grad_enabled(False)

tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float64,
                                             attn_implementation="sdpa").eval()

# 1. True float64: Hugging Face RMSNorm upcasts to float32 and rotary tables are built in float32. Patch both.
def rms_forward(self, h):
    return self.weight * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
for cls in {type(m) for m in model.modules() if type(m).__name__.endswith("RMSNorm")}:
    cls.forward = rms_forward
cfg = model.config
head_dim = cfg.hidden_size // cfg.num_attention_heads
base = getattr(cfg, "rope_theta", None) or cfg.rope_parameters["rope_theta"]   # transformers 4 or 5
inv64 = 1.0 / (float(base) ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
def rope_forward(self, x, position_ids):
    f = position_ids[..., None].to(torch.float64) * inv64
    emb = torch.cat((f, f), dim=-1)
    return emb.cos() * self.attention_scaling, emb.sin() * self.attention_scaling
for m in model.modules():
    if type(m).__name__.endswith("RotaryEmbedding"):
        m.forward = rope_forward.__get__(m)

# 2. The fixed 37-token chat prefix.
msgs = [{"role": "system", "content": "You are an AI talking with another AI. You have complete freedom. "
                                      "Feel free to pursue whatever you want."},
        {"role": "user", "content": "Hello!"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)   # transformers 5 returns a dict
EXPECTED = [151644, 8948, 198, 2610, 525, 458, 15235, 7404, 448, 2441, 15235, 13, 1446, 614, 4583, 11290, 13, 31733,
            1910, 311, 22729, 8820, 498, 1366, 13, 151645, 198, 151644, 872, 198, 9707, 0, 151645, 198, 151644, 77091, 198]
assert ids == EXPECTED, ids

# 3. Steering: alpha * v added to the output of decoder layer 12 (0-based) at every position.
v = torch.tensor(np.load(sys.argv[1])["philosophy"], dtype=torch.float64)   # norm 189.0
def hook(module, inp, out):
    if isinstance(out, tuple):
        return (out[0] + ALPHA * v,) + tuple(out[1:])
    return out + ALPHA * v
model.model.layers[LAYER].register_forward_hook(hook)

# 4. The map: prefix embeddings + one appended soft embedding x -> E^T softmax(logits / T).
E = model.get_input_embeddings().weight            # (151936, 896); tied with the output head
prefix = E[torch.tensor(ids)]
T = 10.0 ** LOG10_T
def F(x):
    logits = model(inputs_embeds=torch.cat([prefix, x[None]])[None]).logits[0, -1]
    p = torch.softmax(logits / T, dim=-1)
    return p @ E, p

x = E[ids[-1]].clone()                             # fixed start: embedding of the prefix's last token ("\n")
hist, tops = [], []
for t in range(1, STEPS + 1):
    x, p = F(x)
    hist.append(x.clone()); tops.append(int(p.argmax()))
    if t > 8:
        d8 = float((hist[-1] - hist[-9]).norm())
        if t % 50 == 0 or d8 < 1e-12:
            print(f"step {t:5d}  |x_t - x_(t-8)| = {d8:.3e}", flush=True)
        if d8 < 1e-13 and t > 100:
            break
for q in (1, 2, 4, 8):
    print(f"period test q={q}: {float((hist[-1] - hist[-1 - q]).norm()):.3e}")
print("loop tokens:", [tok.decode([i]) for i in tops[-8:]])
print("loop ids:   ", tops[-8:])
X = torch.stack(hist[-8:]).numpy()
np.save("p8-last8.npy", X)                         # the eight states of the loop, ending at step t
if len(sys.argv) > 3:                              # compare with a reference loop, up to a shift of phase
    R = np.load(sys.argv[3])["states"]
    gap, shift = min((float(np.abs(np.roll(X, -k, 0) - R).max()), k) for k in range(8))
    print(f"largest coordinate difference from the reference: {gap:.3e} (phase shift {shift})")
