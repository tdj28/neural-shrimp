"""Two runs of the soft-token loop that differ only in the batch grouping (explainer clip N4), from the saved arrays of
the article's reproducibility test (Sec. IV "Reproducibility"): Qwen2.5-0.5B-Instruct, philosophy direction at layer
12, float64, 448 labeled cells of two scans (192 chaotic, 192 periodic, 64 other; docs/h2-horizon-cells.json), run
twice from the same start in lockstep for 1500 steps (tools/repro_noise.py, runs REPRO1 and FS).

Inputs (relative to the data root ../neural-shrimp-data/explore/llm-soft-loop/pod64/repro):
  h2cells/repro.npz  batch64_dist, batch64_agree: per step and cell, the relative distance |x_A - x_B| / |x_A| of the
                     two runs' soft tokens and whether their most probable next tokens agree (row i = step i + 1)
  fs/repro.npz       repeat_dist, fixedperm_dist: the fixed-shape control (same rows twice; rows permuted)
plus docs/h2-horizon-cells.json (label, map_exponent) and docs/repro-summary.json (its numbers are cross-checked).

Gates, recomputed from the arrays (the tool refuses to render if any differs, or if the article or the summary
says otherwise): 190 of the 192 chaotic cells pass 1e-2 (strictly), median first step 102; 182 of the 192 periodic
cells stay at or below 1e-2 for all 1500 steps; repeat_dist and fixedperm_dist are exactly 0 in all 448 cells and
1500 steps. Cells shown (rules of the explainer plan, section 3.4): the chaotic cell with 0.25 <= lambda <= 0.40 whose
crossing step lies in 70-140 and whose ratio of measured to predicted step, ln(1e-2/d_1)/lambda with d_1 its smallest
nonzero distance, is closest to the median ratio (0.91); the periodic cell, among the 182 that stay below 1e-2, whose
largest distance is closest to their median largest distance (2.30e-12).

Timeline (20 s): 0-8 s the two cells drawn step by step over steps 0-300; 8-14 s the axis widens to 1500 steps and all
chaotic and periodic cells fade in with the two counts; 14-20 s the fixed-shape control (exactly 0) in a strip below
the logarithmic axis. Frames go to --frames-dir (on disk), then ffmpeg (H.264, yuv420p, CRF 16, 30 fps); the clip's
sources, gates, chosen cells and every number shown are upserted into --manifest.

Usage: python tools/render_batch_divergence.py --out ../neural-shrimp-data/videos/explainer/parts/C7-02-N4-batch-divergence.mp4
       (add --still 12 to write one PNG at that time instead of the clip)
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import multiprocessing as mp
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "neural-shrimp-data/explore/llm-soft-loop/pod64/repro"
H2, FS = DATA / "h2cells/repro.npz", DATA / "fs/repro.npz"
CELLS, SUMMARY = ROOT / "docs/h2-horizon-cells.json", ROOT / "docs/repro-summary.json"
ARTICLE = ROOT.parent / "chaos-journal-neural-shrimp" / "main.tex"
W, H, DPI, FPS, SECONDS = 1920, 1080, 72, 30, 20.0
BG, WHITE, BODY, MUTED, GRID = "#0B0B0F", "#FFFFFF", "#C8C8D0", "#9A9AA6", "#23232B"
CHAOS, PERIODIC, RED = "#FF8C1A", "#4DB8FF", "#FF6B4D"
SANS = "DejaVu Sans"
EPS = 1e-2
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def fail(msg):
    raise SystemExit(f"render_batch_divergence: gate failed, nothing rendered: {msg}")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sci(x, digits=1):
    m, e = f"{x:.{digits}e}".split("e")
    return f"{m} × 10{str(int(e)).translate(SUP)}"


def first_above(col):
    over = np.nonzero(col > EPS)[0]
    return int(over[0]) + 1 if len(over) else None          # steps are 1-based: row i is step i + 1


def prepare():
    for f in (H2, FS, CELLS, SUMMARY, ARTICLE):
        if not f.exists():
            fail(f"missing input {f}")
    h2, fs = np.load(H2), np.load(FS)
    spec = json.loads(CELLS.read_text())
    lab, lam = np.array(spec["label"]), np.array(spec["map_exponent"], dtype=float)
    run = json.loads((H2.parent / "repro.json").read_text())
    if list(run["labels"]) != list(lab) or not np.allclose(run["cells"], spec["cells"]):
        fail("the arrays' cells and labels do not match docs/h2-horizon-cells.json")
    D, A = h2["batch64_dist"].astype(np.float64), h2["batch64_agree"]
    rep, perm = fs["repeat_dist"], fs["fixedperm_dist"]
    if D.shape != (1500, 448) or rep.shape != D.shape or perm.shape != D.shape:
        fail(f"array shapes {D.shape}, {rep.shape}, {perm.shape}")
    ch, pe = np.nonzero(lab == "chaotic")[0], np.nonzero(lab == "periodic")[0]
    cross = {int(c): first_above(D[:, c]) for c in ch}
    reach = [v for v in cross.values() if v is not None]
    stay = [int(c) for c in pe if D[:, c].max() <= EPS]
    gates = {"chaotic_cells": len(ch), "chaotic_pass_1e-2": len(reach), "median_first_step": float(np.median(reach)),
             "periodic_cells": len(pe), "periodic_stay_below_1e-2": len(stay),
             "repeat_max": float(rep.max()), "fixedperm_max": float(perm.max()), "cells_all": int(D.shape[1]),
             "steps": int(D.shape[0])}
    want = {"chaotic_cells": 192, "chaotic_pass_1e-2": 190, "median_first_step": 102.0, "periodic_cells": 192,
            "periodic_stay_below_1e-2": 182, "repeat_max": 0.0, "fixedperm_max": 0.0, "cells_all": 448, "steps": 1500}
    if gates != want:
        fail(f"recomputed {gates}, expected {want}")
    obs, pred, idx, d1s = [], [], [], {}
    for c in ch:
        c = int(c)
        if cross[c] is not None and lam[c] > 0:
            d1s[c] = float(D[:, c][D[:, c] > 0].min())
            obs.append(cross[c])
            pred.append(math.log(EPS / d1s[c]) / lam[c])
            idx.append(c)
    ratio = np.array(obs) / np.array(pred)
    med_ratio = float(np.median(ratio))
    gates["median_ratio"] = med_ratio
    gates["loglog_r"] = float(np.corrcoef(np.log(obs), np.log(pred))[0, 1])
    gates["chaotic_token_differs_last_500"] = int(sum(np.any(~A[-500:, c]) for c in ch))
    if round(med_ratio, 2) != 0.91 or gates["chaotic_token_differs_last_500"] != 173:
        fail(f"median ratio {med_ratio} or late token differences {gates['chaotic_token_differs_last_500']}")
    s = json.loads(SUMMARY.read_text())["trajectories"]["qwen0.5b steered (REPRO1)"]["batch64"]
    if (s["chaotic"]["reach_1e-2"], s["chaotic"]["median_step"], s["periodic"]["cells"] - s["periodic"]["reach_1e-2"],
            round(s["lyapunov_prediction"]["median_ratio"], 4)) != (190, 102.0, 182, round(med_ratio, 4)):
        fail("docs/repro-summary.json disagrees with the arrays")
    art = re.sub(r"\s+", " ", ARTICLE.read_text())
    for pat in (r"At 190 of the 192 chaotic cells", r"after a median of 102 steps", r"at 182 of the 192 periodic cells",
                r"bitwise identical in all 448 cells", r"median ratio of \$0\.91\$", r"at 173 of them"):
        if not re.search(pat, art):
            fail(f"the article no longer says: {pat}")
    # the cells shown
    cand = [c for c, r in zip(idx, ratio) if 0.25 <= lam[c] <= 0.40 and 70 <= cross[c] <= 140]
    if not cand:
        fail("no chaotic cell satisfies the selection rule")
    rmap = dict(zip(idx, ratio))
    c_ch = min(cand, key=lambda c: abs(rmap[c] - med_ratio))
    mx = {c: float(D[:, c].max()) for c in stay}
    med_mx = float(np.median(list(mx.values())))
    c_pe = min(stay, key=lambda c: abs(mx[c] - med_mx))
    dis = np.nonzero(~A[:, c_ch])[0]
    if not len(dis):
        fail("the chosen chaotic cell never changes its most probable token")
    chosen = {"chaotic": {"index": c_ch, "log10T_alpha": spec["cells"][c_ch], "lambda": float(lam[c_ch]),
                          "d1": d1s[c_ch], "crossing_step": cross[c_ch],
                          "predicted_step": math.log(EPS / d1s[c_ch]) / lam[c_ch], "ratio": float(rmap[c_ch]),
                          "first_token_difference_step": int(dis[0]) + 1},
              "periodic": {"index": c_pe, "log10T_alpha": spec["cells"][c_pe], "period": int(spec["map_period"][c_pe]),
                           "lambda": float(lam[c_pe]), "largest_distance": mx[c_pe],
                           "median_largest_distance_182": med_mx,
                           "token_differences": int(np.sum(~A[:, c_pe]))}}
    if round(med_mx, 14) != 2.30e-12:
        fail(f"median largest periodic distance {med_mx} is not 2.30e-12")
    steps = np.arange(1, D.shape[0] + 1, dtype=float)
    segs = lambda cols: [np.column_stack([steps, D[:, c]]) for c in cols]          # noqa: E731
    return dict(D=D, steps=steps, chosen=chosen, gates=gates, seg_ch=segs(ch), seg_pe=segs(pe),
                zero=np.maximum(rep.max(axis=1), perm.max(axis=1)))


# ----------------------------------------------------------------------------------------------- timeline
def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def ramp(t, start, dur=0.4):
    return min(1.0, max(0.0, (t - start) / dur))


def drawn_steps(t):
    """Steps of the two highlighted runs drawn at time t."""
    if t < 0.5:
        return 0.0
    if t < 4.5:
        return 120 * (t - 0.5) / 4.0
    if t < 8.0:
        return 120 + 180 * min(1.0, (t - 4.5) / 3.0)
    return xmax(t)


def xmax(t):
    return 300 + 1200 * smooth((t - 8.0) / 1.5)


def time_of_step(k):
    """Inverse of drawn_steps over the first 300 steps: when the highlighted runs reach step k."""
    return 0.5 + k / 30.0 if k <= 120 else 4.5 + (k - 120) / 60.0


# ----------------------------------------------------------------------------------------------- drawing
def text(ax, x, y, s, size, color=BODY, weight="normal", alpha=1.0, ha="left"):
    if alpha > 0:
        ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, family=SANS, alpha=alpha, ha=ha,
                va="baseline")


def key_line(ax, x, y, color, ls="-", lw=3, alpha=1.0):
    ax.plot([x, x + 44], [y - 9, y - 9], color=color, lw=lw, ls=ls, alpha=alpha, solid_capstyle="butt")


def key_dot(ax, x, y, face, edge, alpha=1.0, ms=12):
    ax.plot([x + 22], [y - 9], "o", ms=ms, color=face, mec=edge, mew=2.5, alpha=alpha)


def frame(S, t):
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    C, P = S["chosen"]["chaotic"], S["chosen"]["periodic"]
    g = S["gates"]
    text(ax, 80, 100, "Only the batch grouping differs", 52, WHITE, "bold")
    text(ax, 80, 152, "448 cells run twice from the same start; only their grouping into batches differs", 30, BODY)
    text(ax, 80, 1030, "Qwen2.5-0.5B-Instruct · philosophy steering, layer 12 · float64 · cells: 192 chaotic, "
                       "192 periodic, 64 other · article Sec. IV", 26, MUTED)
    # main logarithmic axis
    x0, x1, y0, y1 = 190, 1250, 238, 786
    m = fig.add_axes([x0 / W, 1 - y1 / H, (x1 - x0) / W, (y1 - y0) / H], facecolor=BG)
    xm = 300.0 if t < 8.0 else xmax(t)
    m.set_xlim(0, xm)
    m.set_yscale("log")
    m.set_ylim(1e-16, 10)
    for sp in m.spines.values():
        sp.set_visible(False)
    yt = [1e-16, 1e-12, 1e-8, 1e-4, 1e-2, 1]
    for v in yt:
        if v != EPS:                                          # 1e-2 carries the dashed threshold instead
            m.axhline(v, color=GRID, lw=1, zorder=0)
    m.set_yticks(yt)
    m.set_yticklabels(["10⁻¹⁶", "10⁻¹²", "10⁻⁸", "10⁻⁴", "10⁻²", "1"])
    m.minorticks_off()
    step = 100 if xm <= 450 else (250 if xm <= 1000 else 500)
    m.set_xticks(np.arange(0, xm + 1, step))
    m.tick_params(colors=MUTED, labelsize=24, length=0, pad=10)
    m.axhline(1e-16, color=MUTED, lw=1)
    m.text(0, 30, "relative distance between the two runs", fontsize=26, color=BODY, va="bottom")
    m.axhline(EPS, color=MUTED, lw=1.6, ls=(0, (6, 4)), zorder=2)
    # every chaotic and periodic cell, faint
    fade = ramp(t, 8.3, 1.5)
    if fade > 0:
        for segs, col in ((S["seg_pe"], PERIODIC), (S["seg_ch"], CHAOS)):
            m.add_collection(LineCollection(segs, colors=[col], linewidths=1.2, alpha=0.08 * fade, zorder=1))
    # the two highlighted cells
    n = int(math.floor(drawn_steps(t)))
    st, D = S["steps"], S["D"]
    if n >= 1:
        hl = 1.0 - 0.25 * fade                                # a little softer once every cell is shown
        m.plot(st[:n], D[:n, P["index"]], color=PERIODIC, lw=2.5, zorder=4, solid_joinstyle="round", alpha=hl)
        m.plot(st[:n], D[:n, C["index"]], color=CHAOS, lw=2.5, zorder=5, solid_joinstyle="round", alpha=hl)
        if t < 8.0:
            for c in (P["index"], C["index"]):
                m.plot([st[n - 1]], [D[n - 1, c]], "o", ms=8, color=WHITE, mec=BG, mew=2, zorder=7)
    pk = C["predicted_step"]
    if n >= 1:
        k = min(n, pk)
        m.plot([0, k], [C["d1"], C["d1"] * math.exp(C["lambda"] * k)], color=BODY, lw=2, ls=(0, (1.2, 2.2)),
               zorder=3)
    marks = 1.0 - ramp(t, 8.0, 0.4)
    if n >= C["crossing_step"] and marks > 0:
        ks = C["crossing_step"]
        m.plot([ks], [D[ks - 1, C["index"]]], "o", ms=13, color=BG, mec=CHAOS, mew=2.5, zorder=6, alpha=marks)
    if n >= C["first_token_difference_step"] and marks > 0:
        kf = C["first_token_difference_step"]
        m.plot([kf], [D[kf - 1, C["index"]]], "o", ms=13, color=RED, mec=BG, mew=2, zorder=6, alpha=marks)
    ax.text((x0 + x1) / 2, 846, "step", fontsize=24, color=MUTED, ha="center", va="baseline", family=SANS)
    # strip for exact zero, below the logarithmic axis
    z = fig.add_axes([x0 / W, 1 - 936 / H, (x1 - x0) / W, 56 / H], facecolor=BG)
    z.set_xlim(0, xm)
    z.set_ylim(-1, 1)
    for sp in z.spines.values():
        sp.set_visible(False)
    z.set_xticks([])
    z.set_yticks([0])
    z.set_yticklabels(["0"])
    z.tick_params(colors=MUTED, labelsize=24, length=0, pad=10)
    z.axhline(0, color=GRID, lw=1)
    zf = ramp(t, 14.0, 0.6)
    if zf > 0:
        k = int(min(1500, 1500 * smooth((t - 14.0) / 1.2)))
        z.plot(st[:max(k, 2)], S["zero"][:max(k, 2)], color=WHITE, lw=3, alpha=zf, zorder=3)
        z.text(xm * 0.01, 0.35, "same matrix shapes: distance exactly 0", fontsize=24, color=WHITE, va="bottom",
               alpha=zf)
    ax.plot([x0 - 30, x0 - 14], [842, 860], color=MUTED, lw=1.5)            # axis-break marks
    ax.plot([x0 - 30, x0 - 14], [850, 868], color=MUTED, lw=1.5)
    # right column: legend, then what the frame shows
    rx = 1310
    key_line(ax, rx, 262, CHAOS)
    text(ax, rx + 62, 262, f"a chaotic loop, λ = {C['lambda']:.2f} per step", 28, BODY)
    key_line(ax, rx, 306, PERIODIC)
    text(ax, rx + 62, 306, f"a periodic loop, period {P['period']}", 28, BODY)
    a1 = ramp(t, 0.5) * (1.0 - ramp(t, 8.0, 0.4))
    if a1 > 0:
        ac = ramp(t, time_of_step(C["crossing_step"]), 0.3) * a1
        key_dot(ax, rx, 392, BG, CHAOS, ac)
        text(ax, rx + 62, 392, f"passes 10⁻² at step {C['crossing_step']}", 28, BODY, alpha=ac)
        key_line(ax, rx, 436, BODY, (0, (1.2, 2.2)), 2, ac)
        text(ax, rx + 62, 436, f"predicted from λ: step {pk:.0f}", 28, BODY, alpha=ac)
        af = ramp(t, time_of_step(C["first_token_difference_step"]), 0.3) * a1
        key_dot(ax, rx, 480, RED, BG, af)
        text(ax, rx + 62, 480, f"step {C['first_token_difference_step']}: the most probable", 28, BODY, alpha=af)
        text(ax, rx + 62, 516, "tokens first differ", 28, BODY, alpha=af)
        ap = ramp(t, 4.0, 0.5) * a1
        text(ax, rx + 62, 584, "periodic loop: largest distance", 28, BODY, alpha=ap)
        text(ax, rx + 62, 620, sci(P["largest_distance"]), 28, BODY, alpha=ap)
    a2 = ramp(t, 9.6)
    if a2 > 0:
        text(ax, rx, 420, f"{g['chaotic_pass_1e-2']} of {g['chaotic_cells']}", 46, WHITE, "bold", alpha=a2)
        text(ax, rx, 462, "chaotic loops pass 10⁻²", 30, BODY, alpha=a2)
        text(ax, rx, 500, f"(median step {g['median_first_step']:.0f})", 30, BODY, alpha=a2)
    a3 = ramp(t, 11.2)
    if a3 > 0:
        text(ax, rx, 600, f"{g['periodic_stay_below_1e-2']} of {g['periodic_cells']}", 46, WHITE, "bold", alpha=a3)
        text(ax, rx, 642, "periodic loops stay below", 30, BODY, alpha=a3)
    a4 = ramp(t, 14.2, 0.6)
    if a4 > 0:
        text(ax, rx, 870, "same matrix shapes (rows repeated", 28, WHITE, alpha=a4)
        text(ax, rx, 906, f"or permuted): exactly 0 in all {g['cells_all']}", 28, WHITE, alpha=a4)
        text(ax, rx, 942, f"cells, for all {g['steps']} steps", 28, WHITE, alpha=a4)
    return fig


def _worker(job):
    S, frames, outdir = job
    for i in frames:
        fig = frame(S, i / FPS)
        fig.savefig(outdir / f"{i:05d}.png", dpi=DPI, facecolor=BG)
        plt.close(fig)
    return len(frames)


# ----------------------------------------------------------------------------------------------- output
def upsert(manifest, entry):
    """Replace this clip's entry in the manifest, holding a lock on the manifest file itself."""
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        raw = f.read()
        data = json.loads(raw) if raw.strip() else {"clips": []}
        data["clips"] = [c for c in data["clips"] if c["file"] != entry["file"]] + [entry]
        data["clips"].sort(key=lambda c: c["file"])
        f.seek(0)
        f.truncate()
        f.write(json.dumps(data, indent=1, ensure_ascii=False) + "\n")


def entry(S, out, seconds):
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--",
                                 "tools/render_batch_divergence.py"], capture_output=True, text=True).stdout.strip())
    g, C, P = S["gates"], S["chosen"]["chaotic"], S["chosen"]["periodic"]
    return {"file": out.name, "clip": "N4", "chapter": "C7", "shot": "7.2", "seconds": seconds,
            "format": "1920x1080, 30 fps, H.264 High, yuv420p, CRF 16, no audio",
            "tool": "tools/render_batch_divergence.py", "commit": head, "tool_uncommitted": dirty, "sha256": sha256(out),
            "data_sources": [{"path": str(Path(p).resolve()), "sha256": sha256(p)}
                             for p in (H2, FS, CELLS, SUMMARY, H2.parent / "repro.json")]
                            + [{"path": str(ARTICLE.resolve()), "sha256": sha256(ARTICLE), "use": "statements checked"}],
            "fields": "h2cells: batch64_dist, batch64_agree; fs: repeat_dist, fixedperm_dist; cells: label, "
                      "map_exponent, map_period, cells",
            "numbers_shown": {"chaotic_pass_1e-2": [g["chaotic_pass_1e-2"], g["chaotic_cells"]],
                              "median_first_step": g["median_first_step"],
                              "periodic_stay_below": [g["periodic_stay_below_1e-2"], g["periodic_cells"]],
                              "fixed_shape": {"max_distance": max(g["repeat_max"], g["fixedperm_max"]),
                                              "cells": g["cells_all"], "steps": g["steps"]},
                              "chaotic_cell": {"lambda_shown": round(C["lambda"], 2),
                                               "crossing_step": C["crossing_step"],
                                               "predicted_step_shown": round(C["predicted_step"]),
                                               "first_token_difference_step": C["first_token_difference_step"]},
                              "periodic_cell": {"period": P["period"], "largest_distance_shown": sci(P["largest_distance"])},
                              "cells_in_footer": {"chaotic": 192, "periodic": 192, "other": 64}},
            "gates": g, "chosen_cells": S["chosen"],
            "article": "Sec. IV 'Reproducibility' (190 of 192, median 102 steps, 182 of 192, median ratio 0.91, "
                       "bitwise identical in all 448 cells)",
            "rendered_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames-dir", type=Path, default=ROOT.parent / "neural-shrimp-data/render-tmp/explainer-data")
    ap.add_argument("--manifest", type=Path, default=None, help="default: MANIFEST-data-clips.json beside --out")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--still", type=float, default=None, help="write one PNG at this time (s) and stop")
    a = ap.parse_args()
    S = prepare()
    C, P = S["chosen"]["chaotic"], S["chosen"]["periodic"]
    print(json.dumps({"gates": S["gates"], "chaotic": C, "periodic": P}, indent=1))
    if a.still is not None:
        out = a.out.with_name(a.out.stem + f"-still-{a.still:05.2f}.png")
        fig = frame(S, a.still)
        fig.savefig(out, dpi=DPI, facecolor=BG)
        print(f"wrote {out}")
        return
    frames = a.frames_dir / a.out.stem
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True)
    n = int(round(SECONDS * FPS))
    chunks = [list(range(i, n, a.workers)) for i in range(a.workers)]
    with mp.get_context("fork").Pool(a.workers) as pool:
        done = sum(pool.map(_worker, [(S, c, frames) for c in chunks]))
    if done != n:
        fail(f"rendered {done} of {n} frames")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames / "%05d.png"),
                    "-vf", "setsar=1,format=yuv420p", "-c:v", "libx264", "-profile:v", "high", "-preset", "slow",
                    "-crf", "16", "-pix_fmt", "yuv420p", "-r", str(FPS), "-g", "60", "-movflags", "+faststart", "-an",
                    str(a.out)], check=True)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(a.out)], capture_output=True, text=True, check=True).stdout)
    upsert(a.manifest or a.out.parent / "MANIFEST-data-clips.json", entry(S, a.out, round(dur, 3)))
    shutil.rmtree(frames)
    print(f"wrote {a.out} ({dur:.2f} s, {n} frames)")


if __name__ == "__main__":
    sys.exit(main())
