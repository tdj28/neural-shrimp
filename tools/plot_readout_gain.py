"""Figure fig-readout-gain: the readout gain g(T) = ||Sigma_p(T)|| / T at fixed logits (paper, Section "Why the loop can
be chaotic", eq. gain).

Qwen2.5-0.5B-Instruct ties its output matrix to the input embedding E, so the soft-token map has
DF(x) = (1/T) Sigma_p J_h(x), with Sigma_p = Cov_{t~p}(e_t) the covariance of the token embeddings under
p = softmax(z(x)/T). Here the logits z = z_alpha(x) of a state are held fixed and only the temperature in the softmax
varies, so g(T) = ||Sigma_p(T)||_2 / T (spectral norm) is the readout factor alone. The network factor J_h is not
included: ||DF|| <= g ||J_h||, and g by itself does not say whether DF expands.

States, exactly: the period-8 orbit of the philosophy direction (Table tab:orbits; vectors llm-phase-map/vectors-l12,
layer 12, alpha 0.071674, log10 T -0.115). Run G1b of tools/softmax_gain.py (orbit mode, 1000 transient steps from
the fixed start) saved its states x_0, ..., x_8 = F^8(x_0) in
explore/llm-soft-loop/softmax-gain/p8b/orbit-jacobians.npz; x_0 is also the state of the identity check of eq. gain
(run G1, same cell and transient). Drawn: x_0, and, of x_0..x_7, the states with the largest and the smallest gap
Delta = z_(1) - z_(2) between the two largest logits. With the stored states these are x_6 (Delta 4.457, next token
' is') and x_1 (Delta 0.0831, next token ' a'); x_0 has Delta 1.395 (next token ' is').

Reference curves, not fitted:
  T -> 0:   ||e_t1 - e_t2||^2 exp(-Delta/T) / T, the leading term (t1, t2 the tokens with the two largest logits);
  T -> oo:  ||Sigma_u|| / T, with Sigma_u the covariance of all V embeddings under equal weights (the limit of
            Sigma_p).
The ratio of g to each reference is also evaluated outside the drawn range (at T = Delta/200 and Delta/600, and at
log10 T = 2, 3 and 4) and recorded. In the figure each low-temperature reference is drawn below its state's peak
temperature and the high-temperature reference above the largest peak temperature, where they apply.

Numerics, float64. Sigma_p = sum_t p_t d_t d_t^T - delta delta^T, with d_t = e_t - e_t1 and delta = sum_t p_t d_t. This
is exact and keeps relative precision when p is nearly one-hot, where E^T diag(p) E - m m^T cancels catastrophically.
Tokens whose summed weight p_t ||d_t||^2 is below 1e-18 of the total are left out of the first sum. A peak is the
largest grid value, refined by a bounded scalar search over the two neighbouring grid intervals and checked on an
independent grid of 17 points at spacing 0.0005 around it (the run stops if that grid finds a larger value or a
maximum more than one spacing away); a maximum at an end of the grid is reported as such, not as a peak. Interior
local maxima of the grid are listed as well.

Gates (the run stops if one fails): the output matrix is the input embedding; at the orbit's own temperature the
recomputed p reproduces G1b's entropy, top-3 probabilities and four leading eigenvalues of Sigma_p at every state, and
E^T p reproduces the next state of the orbit.

Usage: uv run python tools/plot_readout_gain.py [--threads 8] [--recompute] [--out PNG]
Data: ../neural-shrimp-data/explore/llm-soft-loop/readout-gain/p8b/readout-gain.{npz,json}. They are computed once
(CPU, about ten minutes on 8 threads); afterwards only the figure is redrawn, unless --recompute is given.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
RUN = DATA / "explore" / "llm-soft-loop" / "softmax-gain" / "p8b"             # run G1b of tools/softmax_gain.py
OUT = DATA / "explore" / "llm-soft-loop" / "readout-gain" / "p8b"
FIG = ROOT / "paper" / "llm" / "figures" / "fig-readout-gain.png"
GRID = np.round(np.linspace(-1.5, 1.0, 251), 10)        # log10 T, step 0.01
HIGH_CHECK = (2.0, 3.0, 4.0)                             # log10 T of the high-temperature ratio check (not drawn)
LOW_CHECK = (200.0, 600.0)                               # Delta / T of the low-temperature ratio check (not drawn)
FINE = np.linspace(-0.004, 0.004, 17)                    # independent check of each peak: offsets in log10 T
DROP = 1e-18                                             # relative weight of the tokens left out of Sigma_p
GATE = {"entropy": 1e-9, "top3": 1e-10, "eigenvalues_rel": 1e-8, "next_state_rel": 1e-10}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Readout:
    """Sigma_p(T) for the fixed logits z of one state, with the embeddings centred at the leading token."""

    def __init__(self, torch, E, z):
        self.torch, self.z = torch, z
        top = torch.topk(z, 3)
        self.t1, self.t2, self.t3 = (int(i) for i in top.indices)
        self.gap = float(top.values[0] - top.values[1])                  # Delta
        self.gap3 = float(top.values[0] - top.values[2])
        self.D = E - E[self.t1]                                          # d_t = e_t - e_t1, (V, d)
        self.dn2 = (self.D * self.D).sum(1)                              # ||d_t||^2
        self.c12 = float(self.dn2[self.t2])                              # ||e_t1 - e_t2||^2

    def sigma(self, T):
        torch = self.torch
        p = torch.softmax(self.z / T, dim=0)
        w = p * self.dn2
        order = torch.argsort(w)                                         # ascending weight
        tail = torch.cumsum(w[order], 0)
        keep = order[tail > DROP * tail[-1]]
        n = len(keep)
        Dk, pk = (self.D, p) if n > len(p) // 2 else (self.D[keep], p[keep])
        delta = p @ self.D
        return (Dk * pk[:, None]).T @ Dk - torch.outer(delta, delta), p, n

    def norm(self, T):
        S, p, n = self.sigma(T)
        scale = float(S.abs().max())
        lam = 0.0 if scale == 0.0 else float(self.torch.linalg.eigvalsh(S / scale)[-1]) * scale
        return lam, n

    def gain(self, log10_t):
        T = 10.0 ** log10_t
        lam, n = self.norm(T)
        return lam / T, n

    def low_reference(self, T):
        return self.c12 * np.exp(-self.gap / T) / T


def gates(loop, states, Z, run):
    """Recompute G1b's per-step quantities at the orbit's temperature; stop if any differs beyond GATE."""
    torch = loop.torch
    T0 = run["temperature"]
    rows = []
    for k, z in enumerate(Z):
        r = Readout(torch, loop.E, z)
        S, p, _ = r.sigma(T0)
        ev = torch.linalg.eigvalsh(S).flip(0)[:4].cpu().numpy()
        ent = float(-(p * torch.log(torch.clamp(p, min=1e-300))).sum())
        top3 = torch.topk(p, 3).values.cpu().numpy()
        nxt = (p @ loop.E).cpu().numpy()
        stored = run["steps"][k]
        row = {"state": k, "token": loop.sm.tok.decode([r.t1]),
               "entropy": abs(ent - stored["entropy_nats"]),
               "top3": float(np.max(np.abs(top3 - np.array(stored["top3_p"])))),
               "eigenvalues_rel": float(np.max(np.abs(ev - np.array(stored["sigma_p_eigenvalues"]))
                                               / np.array(stored["sigma_p_eigenvalues"]))),
               "next_state_rel": float(np.linalg.norm(nxt - states[k + 1]) / np.linalg.norm(states[k + 1])),
               "g_at_orbit_temperature": float(ev[0]) / T0}
        row["token_matches"] = row["token"] == stored["token"]
        row["pass"] = bool(row["token_matches"] and all(row[key] < tol for key, tol in GATE.items()))
        rows.append(row)
        print(json.dumps(row), flush=True)
    if not all(row["pass"] for row in rows):
        raise SystemExit("gate failed: the recomputed logits do not reproduce run G1b (see the rows above)")
    return rows


def peak(r, g):
    """Largest grid value of g, refined between its neighbours; a maximum at an end of the grid is not a peak."""
    from scipy.optimize import minimize_scalar

    i = int(np.argmax(g))
    interior = [int(j) for j in range(1, len(g) - 1) if g[j] > g[j - 1] and g[j] > g[j + 1]]
    out = {"grid_index": i, "grid_log10_T": float(GRID[i]), "grid_g": float(g[i]),
           "interior_local_maxima_log10_T": [float(GRID[j]) for j in interior],
           "interior_local_maxima_g": [float(g[j]) for j in interior]}
    if i in (0, len(g) - 1):
        out["at_grid_end"] = True
        return out
    res = minimize_scalar(lambda lt: -r.gain(lt)[0], bounds=(float(GRID[i - 1]), float(GRID[i + 1])),
                          method="bounded", options={"xatol": 1e-6})
    x, gx = float(res.x), float(-res.fun)
    fine = np.array([r.gain(x + o)[0] for o in FINE])
    j = int(np.argmax(fine))
    check = {"fine_grid_max_log10_T": float(x + FINE[j]), "fine_grid_max_g": float(fine[j]),
             "fine_grid_spacing": float(FINE[1] - FINE[0])}
    check["pass"] = bool(fine[j] <= gx * (1 + 1e-12) and abs(FINE[j]) <= (FINE[1] - FINE[0]) * 1.01)
    out.update({"at_grid_end": False, "log10_T": x, "g": gx, "evaluations": int(res.nfev), "check": check})
    if not check["pass"]:
        raise SystemExit(f"peak check failed: {out}")
    return out


def compute(args):
    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    script_hash = sha256(Path(__file__))                 # the code that runs, hashed before it runs
    src_json, src_npz = RUN / "softmax-gain-orbit.json", RUN / "orbit-jacobians.npz"
    run = json.loads(src_json.read_text())
    a = run["args"]
    if a["dtype"] != "float64" or a["period"] != 8:
        raise SystemExit("expected run G1b: the float64 period-8 orbit")
    states = np.load(src_npz)["states"]
    loop = esl.SoftLoop(a["model"], a["layer"], args.threads, "open", ela.OPENER, 1, "float64", "cpu")
    torch = loop.torch
    torch.set_num_threads(args.threads)
    model = loop.model
    if not getattr(model.config, "tie_word_embeddings", False) or \
            model.lm_head.weight.data_ptr() != model.get_input_embeddings().weight.data_ptr():
        raise SystemExit("Sigma_p is the readout factor only with tied embeddings (output matrix = input embedding)")
    if getattr(model.lm_head, "bias", None) is not None:
        raise SystemExit("output layer has a bias; the factorisation assumes z = E h")
    v = elpm.load_vectors(a["vectors"])[a["concept"]]
    loop.prepare((a["alpha"] * v)[None])
    Z = [loop.logits(torch.as_tensor(states[k], dtype=torch.float64)[None, None, :])[0] for k in range(8)]
    gate_rows = gates(loop, states, Z, run)

    gaps = [float(-torch.diff(torch.topk(z, 2).values)) for z in Z]                # Delta of x_0..x_7
    chosen = [0, int(np.argmax(gaps)), int(np.argmin(gaps))]
    if len(set(chosen)) < 3:
        raise SystemExit(f"x_0 is itself an extreme of the gap ({gaps}); choose the states again")
    Ec = loop.E - loop.E.mean(0)
    sigma_u = float(torch.linalg.eigvalsh(Ec.T @ Ec / Ec.shape[0])[-1])        # ||Sigma_u||
    curves, rows = [], []
    for k in chosen:
        r = Readout(torch, loop.E, Z[k])
        tic = time.perf_counter()
        g, kept = np.empty(len(GRID)), np.empty(len(GRID), dtype=np.int64)
        for i, lt in enumerate(GRID):
            g[i], kept[i] = r.gain(float(lt))
        pk = peak(r, g)
        low_ratio = {str(c): float(r.gain(np.log10(r.gap / c))[0] / r.low_reference(r.gap / c)) for c in LOW_CHECK}
        high_ratio = {str(lt): r.gain(lt)[0] / (sigma_u / 10.0 ** lt) for lt in HIGH_CHECK}
        row = {"state": k, "next_token": loop.sm.tok.decode([r.t1]), "second_token": loop.sm.tok.decode([r.t2]),
               "third_token": loop.sm.tok.decode([r.t3]), "gap": r.gap, "gap_to_third": r.gap3,
               "e1_minus_e2_sq": r.c12, "peak": pk,
               "g_at_orbit_temperature": r.gain(float(a["log_temperature"]))[0],
               "ratio_to_low_reference": {"at_delta_over_T": low_ratio,
                                          "at_grid_start": float(g[0] / r.low_reference(10.0 ** GRID[0]))},
               "ratio_to_high_reference": {**high_ratio, "at_grid_end": float(g[-1] / (sigma_u / 10.0 ** GRID[-1]))},
               "tokens_kept_min_max": [int(kept.min()), int(kept.max())], "seconds": time.perf_counter() - tic}
        print(json.dumps(row), flush=True)
        rows.append(row)
        curves.append(g)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "readout-gain.npz", log10_T=GRID, gain=np.array(curves), states=np.array(chosen),
                        gaps=np.array([row["gap"] for row in rows]),
                        e1_minus_e2_sq=np.array([row["e1_minus_e2_sq"] for row in rows]), sigma_u=sigma_u,
                        peak_log10_T=np.array([row["peak"].get("log10_T", np.nan) for row in rows]),
                        orbit_log10_T=float(a["log_temperature"]))
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    receipt = {"source_run": str(RUN.relative_to(DATA)), "source_json_sha256": sha256(src_json),
               "source_npz_sha256": sha256(src_npz), "script_sha256": script_hash, "commit": commit,
               "cell": {k: a[k] for k in ("model", "layer", "vectors", "concept", "alpha", "log_temperature")},
               "all_gaps": gaps, "chosen_states": chosen, "sigma_u_norm": sigma_u, "drop": DROP,
               "max_g_at_orbit_temperature_all_states": max(row["g_at_orbit_temperature"] for row in gate_rows),
               "gates": gate_rows,
               "gate_tolerances": GATE, "states": rows}
    (OUT / "readout-gain.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps({"sigma_u_norm": sigma_u, "all_gaps": gaps, "chosen": chosen}), flush=True)


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(OUT / "readout-gain.npz")
    meta = json.loads((OUT / "readout-gain.json").read_text())
    lt, G, sigma_u, peaks = d["log10_T"], d["gain"], float(d["sigma_u"]), d["peak_log10_T"]
    if not np.all(np.isfinite(peaks)):
        raise SystemExit("a state has no interior peak; the reference curves cannot be split at it")
    T = 10.0 ** lt
    colours = ("#2a78d6", "#eb6834", "#1baf7a")         # categorical slots 1-3 (validated all-pairs on white)
    ink, muted, hair = "#0b0b0b", "#52514e", "#c3c2b7"
    dash = (0, (4, 2))
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "legend.fontsize": 7.2, "axes.edgecolor": muted, "xtick.color": muted,
                         "ytick.color": muted, "axes.labelcolor": ink, "axes.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(3.9, 2.7), dpi=300)
    ymin, ymax = 1e-8, 3.0
    orbit = float(d["orbit_log10_T"])
    ax.axvline(orbit, color=hair, lw=0.7, zorder=0)
    ax.text(orbit + 0.03, 1.6, "orbit's $T$", color=muted, fontsize=7, va="center")
    high = lt >= peaks.max()                            # 1/T reference above the largest peak temperature
    ax.plot(lt[high], sigma_u / T[high], color=muted, lw=0.8, ls=dash, zorder=2)
    ax.text(0.62, 0.42 * sigma_u / 10 ** 0.62, r"$\|\Sigma_u\|/T$", color=muted, fontsize=7, ha="center",
            va="top")
    spots = {0: (-1.06, 1.2e-5, "right"), 6: (-0.36, 3e-6, "left"), 1: (-1.27, 0.2, "center")}   # direct labels
    for row, g, c, pk in zip(meta["states"], G, colours, peaks):
        k, gap = row["state"], row["gap"]
        ax.plot(lt, g, color=c, lw=1.5, solid_capstyle="round", zorder=3, label=rf"$x_{k}$, $\Delta={gap:.3g}$")
        low = row["e1_minus_e2_sq"] * np.exp(-gap / T) / T
        keep = (lt <= pk) & (low > ymin / 10)           # low-temperature reference below the state's peak
        ax.plot(lt[keep], low[keep], color=c, lw=0.8, ls=dash, zorder=2)
        x0, y0, ha = spots.get(k, (lt[np.argmax(g)], 1.5 * g.max(), "center"))
        ax.text(x0, y0, rf"$x_{k}$", color=ink, fontsize=7.5, ha=ha, va="center")
    ax.plot([], [], color=muted, lw=0.8, ls=dash, label="asymptotes")
    ax.set_yscale("log")
    ax.set_xlim(lt[0], lt[-1])
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"$\log_{10} T$")
    ax.set_ylabel(r"$g(T)=\|\Sigma_p\|/T$")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", frameon=False, handlelength=2.2, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    print(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--out", default=str(FIG))
    args = ap.parse_args()
    if args.recompute or not (OUT / "readout-gain.npz").exists():
        compute(args)
    plot(args.out)


if __name__ == "__main__":
    main()
