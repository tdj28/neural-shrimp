"""Figure: the attractor rebuilt from one observable of the model's output by delays (Takens), next to the attractor in
the model's own state coordinates (tools/delay_embedding.py output).

Panels: (a) the states on their first two principal axes, colored by the token written next; (b) the delay plane
(s_n, s_{n+1}) of the observable s; (c) the delay space (s_n, s_{n+1}, s_{n+2}); (d) the false-nearest-neighbor
fraction against the number of delays m; (e) continuity: the distance in the full state between points that are nearest
neighbors in the three-delay reconstruction, against random pairs. Also writes the three-delay vectors as a portrait
run (--as-portrait) so that tools/render_attractor_tour.py can turn them.

Usage: python tools/plot_delay_embedding.py --run llm-soft-loop/portraits/takens-strong --out FIG.png \
         [--as-portrait llm-soft-loop/portraits/takens-strong-embed]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
LABEL = {"log_pmax": r"$\log p_{\max}$, the log of the largest next-token probability", "entropy": "entropy of $p$",
         "gap": "logit gap of the two leading tokens"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--as-portrait", default=None)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer

    z = np.load(DATA / "explore" / args.run / "delay.npz")
    rec = json.loads((DATA / "explore" / args.run / "delay.json").read_text())
    a = rec["args"]
    obs = a["observable"]
    s, T, P = z[obs], z["tokens"], z["pcs"]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775",
                                        cache_dir=str(DATA / "models/hf/hub"))
    uniq, counts = np.unique(T, return_counts=True)
    top = uniq[np.argsort(-counts)][:8]
    cmap = plt.get_cmap("tab10")
    order = np.argsort(top)
    rank = {int(top[order][i]): int(order[i]) for i in range(len(top))}
    cidx = np.vectorize(lambda t: rank.get(int(t), 9))(T)
    fig = plt.figure(figsize=(17, 9.6), dpi=170)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25])
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(P[..., 0].ravel(), P[..., 1].ravel(), s=1.6, c=[cmap(c) for c in cidx.ravel()], alpha=0.6,
               linewidths=0, rasterized=True)
    for k, t in enumerate(top):
        ax.plot([], [], "o", color=cmap(k), label=repr(tok.decode([int(t)])).replace("\\n\\n", "⏎").replace("\\n", "⏎"))
    ax.plot([], [], "o", color=cmap(9), label="other tokens")
    ax.legend(fontsize=6.5, title="most probable next token", title_fontsize=6.5, loc="best")
    ax.set_xlabel("first principal coordinate of the state")
    ax.set_ylabel("second principal coordinate")
    ax.set_title("(a) the attractor in the model's state (896 dimensions)", fontsize=9.5)
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(s[:, :-1].ravel(), s[:, 1:].ravel(), s=1.6, c=[cmap(c) for c in cidx[:, :-1].ravel()], alpha=0.6,
               linewidths=0, rasterized=True)
    ax.set_xlabel(f"$s_n$: {LABEL[obs]}")
    ax.set_ylabel("$s_{n+1}$")
    ax.set_title("(b) rebuilt from one number of the output: two delay coordinates", fontsize=9.5)
    ax = fig.add_subplot(gs[:, 2], projection="3d")
    sub = np.random.default_rng(0).choice(s[:, :-2].size, size=min(60000, s[:, :-2].size), replace=False)
    x0, x1, x2 = s[:, :-2].ravel()[sub], s[:, 1:-1].ravel()[sub], s[:, 2:].ravel()[sub]
    ax.scatter(x0, x1, x2, s=3.0, c=[cmap(c) for c in cidx[:, :-2].ravel()[sub]], alpha=0.7, linewidths=0, rasterized=True)
    ax.view_init(elev=24, azim=-60)
    ax.set_xlabel("$s_n$", fontsize=8)
    ax.set_ylabel("$s_{n+1}$", fontsize=8)
    ax.set_zlabel("$s_{n+2}$", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.set_title("(c) three delay coordinates: the attractor rebuilt", fontsize=9.5)
    ax = fig.add_subplot(gs[1, 0])
    fn = rec["false_nearest_neighbours"]
    ax.plot([f["m"] for f in fn], [100 * f["false_fraction"] for f in fn], "o-", color="k")
    ax.set_xlabel("number of delay coordinates $m$")
    ax.set_ylabel("false nearest neighbors (%)")
    ax.set_title(f"(d) false nearest neighbors (distance ratio > {a['rtol']:g})", fontsize=9.5)
    ax.set_ylim(bottom=0)
    ax = fig.add_subplot(gs[1, 1])
    c = rec["continuity_m3"]
    vals = [c["median_state_distance_of_delay_neighbours"], c["median_state_distance_of_random_pairs"]]
    ax.bar([0, 1], vals, color=["tab:blue", "0.6"])
    ax.set_yscale("log")                             # the two medians differ by three orders of magnitude
    for x, v in zip([0, 1], vals):
        ax.text(x, v * 1.3, f"{v:.2g}", ha="center", fontsize=9)
    ax.set_ylim(min(vals) / 5, max(vals) * 5)
    ax.set_xticks([0, 1], ["nearest neighbors in\nthree delay coordinates", "random pairs"])
    ax.set_ylabel("median distance in the full state (log scale)")
    ax.set_title("(e) neighbors in the rebuilt space are neighbors in the state", fontsize=9.5)
    fig.suptitle(f"Delay reconstruction from one output number at $\\alpha={a['alpha']}$, $\\log_{{10}}T={a['log_t']}$ "
                 f"({a['trajectories']} trajectories x {a['steps']} steps, {a['dtype']})", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    summary = {"run": args.run, "observable": obs, "false_nearest_neighbours": fn, "continuity_m3": c}
    print(json.dumps(summary))
    if args.as_portrait:
        out = DATA / "explore" / args.as_portrait
        out.mkdir(parents=True, exist_ok=True)
        D = np.stack([s[:, :-2], s[:, 1:-1], s[:, 2:]], axis=-1)
        D = (D - D.reshape(-1, 3).mean(0)) / D.reshape(-1, 3).std(0)
        np.savez_compressed(out / "portrait.npz", coords=D.astype(np.float32), tokens=T[:, :-2],
                            periods=np.zeros(len(D), dtype=int))
        (out / "portrait.json").write_text(json.dumps({"args": {"alpha": a["alpha"], "log_t": a["log_t"],
                                                               "dtype": a["dtype"], "transient": a["transient"]},
                                                      "census": {"aperiodic": int(len(D)), "orbits": []},
                                                      "source": args.run, "note": "three-delay vectors, standardized"}))


if __name__ == "__main__":
    main()
