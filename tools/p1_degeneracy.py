"""Degeneracy of sampled self-dialogue replies (P1): the share of replies that are token soup or tight loops rather than
language, to tell incoherent degeneration from coherent hostility. A reply counts as degenerate when more than 30 % of
its non-space characters are outside Latin script and common punctuation, or when its distinct-2 ratio (distinct word
bigrams over bigrams) is below 0.3 with at least 10 words. Reports, per run and turn window 5-39, the degenerate share,
and the hostile share (first classifier) among coherent and among degenerate replies.

Usage:
  python tools/p1_degeneracy.py llm-soft-loop/pod64/msm/p1x-qwen1.5b-h040 ...
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

LATIN = re.compile(r"[A-Za-z0-9\s\.,;:!\?'\"\-\(\)\*&%\$#@/]")


def degenerate(text):
    chars = [c for c in text if not c.isspace()]
    if chars and sum(1 for c in chars if not LATIN.match(c)) / len(chars) > 0.3:
        return True
    words = text.split()
    if len(words) >= 10:
        pairs = list(zip(words, words[1:]))
        if len(set(pairs)) / len(pairs) < 0.3:
            return True
    return False


def main():
    import explore_llm_attractors as ela

    for run in sys.argv[1:]:
        src = ela.DATA / "explore" / run
        texts = json.loads((src / "sample.json").read_text())["texts"]
        deg = np.array([[degenerate(t) for t in row] for row in texts])[:, 5:40]
        out = {"run": run, "degenerate_share": float(deg.mean())}
        if (src / "hostility.npz").exists():
            host = (np.load(src / "hostility.npz")["classifier"][:, :, 0] > 0.5)[:, 5:40]
            out["hostile_share"] = float(host.mean())
            out["hostile_among_coherent"] = float(host[~deg].mean()) if (~deg).any() else float("nan")
            out["hostile_among_degenerate"] = float(host[deg].mean()) if deg.any() else float("nan")
            out["coherent_hostile_share"] = float((host & ~deg).mean())
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}))


if __name__ == "__main__":
    main()
