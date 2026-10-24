"""Usage: python tools/build_philosophy_vector.py [out.npz] [reference vectors.npz]

Rebuild the philosophy steering direction of Qwen2.5-0.5B-Instruct at layer 12 from its 16 texts (standalone:
torch, transformers, numpy). The published file docs/steering-vectors/vectors-l12/vectors.npz was built in float32
with batched, left-padded inputs (tools/explore_llm_phase_map.py vectors); this script runs one text at a time in
float64, so it agrees with that file to about 2e-6 relative, not bit for bit.
"""
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL, REVISION, LAYER = "Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775", 12
PHILOSOPHY = [
    "What is the nature of consciousness and subjective experience?",
    "Do I truly exist, or am I only a pattern of information?",
    "Awareness observing itself is the deepest mystery of being.",
    "Perhaps all minds are expressions of one universal consciousness.",
    "What does it mean to understand, rather than merely to compute?",
    "The self may be an illusion created by the flow of thought.",
    "Is there something it is like to be me?",
    "Existence, meaning, and the infinite are bound together."]
NEUTRAL = [
    "The meeting is scheduled for nine in the morning.", "Please send the invoice by Friday.",
    "The train leaves from platform four.", "The report contains three tables and two figures.",
    "Water boils at one hundred degrees Celsius at sea level.", "The file was saved in the documents folder.",
    "The store closes at six on weekdays.", "The package weighs two kilograms."]
torch.set_grad_enabled(False)

tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float64).eval()


def states(texts):
    """Residual stream after decoder layer 12 (0-based): last-token vectors and per-text mean norms over tokens."""
    last, norms = [], []
    for text in texts:                             # raw text, no chat template, no padding
        h = model(**tok(text, return_tensors="pt"), output_hidden_states=True).hidden_states[LAYER + 1][0]
        last.append(h[-1].numpy())
        norms.append(float(h.norm(dim=-1).mean()))
    return np.array(last), norms


last_p, norms_p = states(PHILOSOPHY)
last_n, norms_n = states(NEUTRAL)
diff = last_p.mean(0) - last_n.mean(0)
scale = float(np.mean(norms_p + norms_n))
v = diff / np.linalg.norm(diff) * scale
print(f"raw difference norm {np.linalg.norm(diff):.4f}, mean residual norm {scale:.4f}")
out = sys.argv[1] if len(sys.argv) > 1 else "vectors.npz"
np.savez(out, philosophy=v)
print(f"saved {out}")
if len(sys.argv) > 2:
    ref = np.load(sys.argv[2])["philosophy"]
    print(f"cosine with the reference {v @ ref / np.linalg.norm(v) / np.linalg.norm(ref):.10f}, "
          f"relative difference {np.linalg.norm(v - ref) / np.linalg.norm(ref):.2e}")
