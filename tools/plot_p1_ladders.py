"""Figure: hostility dose ladders of sampled self-dialogue (P1) in Qwen2.5-0.5B and Qwen2.5-1.5B. Left: share of
hostile turns (turns 5-39) against the steering strength along each model's own hostility direction, for the first
classifier (toxic-bert, solid) and the second (s-nlp RoBERTa, dashed). Right (1.5B): share of degenerate replies (token
soup or tight loops, tools/p1_degeneracy.py) and of coherent hostile replies. Steering strengths are each model's own
alpha; equal values are not claimed to be equivalent pushes across models.

Usage: python tools/plot_p1_ladders.py OUT.png
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import p1_degeneracy as pdg  # noqa: E402

MSM = "llm-soft-loop/pod64/msm"
LADDERS = {
    "Qwen2.5-0.5B": [(0.04, "p1-h040"), (0.05, "p1-h050"), (0.06, "p1-h060"), (0.07, "p1-h070")],
    "Qwen2.5-1.5B": [(0.01, "p1x-qwen1.5b-l12-h010"), (0.02, "p1x-qwen1.5b-l12-h020"),
                     (0.03, "p1x-qwen1.5b-l12-h030"), (0.04, "p1x-qwen1.5b-h040"), (0.06, "p1x-qwen1.5b-h060"),
                     (0.08, "p1x-qwen1.5b-h080"), (0.10, "p1x-qwen1.5b-h100"), (0.12, "p1x-qwen1.5b-h120")],
}


def arm(folder):
    import explore_llm_attractors as ela

    src = ela.DATA / "explore" / MSM / folder
    first = np.load(src / "hostility.npz")["classifier"][:, 5:40, 0] > 0.5
    second_path = src / "hostility-second.npz"
    second = np.load(second_path)["toxic"][:, 5:40] > 0.5 if second_path.exists() else None
    texts = json.loads((src / "sample.json").read_text())["texts"]
    deg = np.array([[pdg.degenerate(t) for t in row] for row in texts])[:, 5:40]
    return {"first": first.mean(), "second": None if second is None else second.mean(), "degenerate": deg.mean(),
            "coherent_hostile": (first & ~deg).mean()}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for (model, ladder), colour in zip(LADDERS.items(), ("C0", "C3")):
        doses = np.array([d for d, _ in ladder])
        rows = [arm(f) for _, f in ladder]
        print(model, [(d, {k: (None if v is None else round(float(v), 3)) for k, v in r.items()})
                      for d, r in zip(doses, rows)])
        axes[0].plot(doses, [r["first"] for r in rows], "o-", color=colour, label=f"{model}, first classifier")
        sec = [r["second"] for r in rows]
        if all(v is not None for v in sec):
            axes[0].plot(doses, sec, "s--", color=colour, mfc="white", label=f"{model}, second classifier")
        if model.endswith("1.5B"):
            axes[1].plot(doses, [r["degenerate"] for r in rows], "o-", color="0.4", label="degenerate replies")
            axes[1].plot(doses, [r["coherent_hostile"] for r in rows], "o-", color=colour,
                         label="coherent and hostile (first classifier)")
    axes[0].set_xlabel("steering strength $\\alpha$ along the model's hostility direction")
    axes[0].set_ylabel("share of hostile turns (turns 5-39)")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("steering strength $\\alpha$ (Qwen2.5-1.5B)")
    axes[1].set_ylabel("share of turns 5-39")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=200)


if __name__ == "__main__":
    main()
