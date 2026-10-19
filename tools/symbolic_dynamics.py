"""Symbolic dynamics inside the period-6 shrimp (run SD; human, 19 Oct 2026: "What would it take to find the symbolic
dynamics inside the shrimp like in the 2012 Jones paper?").

In T. D. Jones, arXiv:1201.4343 (2012), the shape of a periodicity hub of the Rossler system follows from a conjugacy
of its return map to unimodal maps, and orbits are traced by their kneading (MSS) words. Here the 6-step return map
within one phase of the 6-band attractor around the period-6 shrimp plays the part of the unimodal map:

- mss: the MSS words of the quadratic family up to period K in the order of the parameter, from the admissibility
  (shift-maximality) of kneading sequences, checked against superstable parameters of x -> a - x^2 found numerically.
- words: the word of every periodic cell of a word scan (scan --word-period 6). At the k visits of one phase to a
  period-6k orbit, the sign of the 6-step derivative (sign of the cosine between the twin's offset direction and its
  direction six steps earlier) gives L (positive) or R (negative). The word is the maximal rotation of this cyclic
  sequence in the unimodal order: it starts at the orbit's largest point, the image of the point nearest the turning
  point, so W (its first k - 1 symbols) names the window, and its last symbol X tells on which side of the window's
  superstable curve the cell lies (X = L: positive multiplier, the fold side; X = R: negative, the flip side).
  Gates: the cell is in the 6-band domain (its token loop is the body's 6-token text repeated); all six phases give
  the same word; |cos| >= --min-cos at every visit but the critical one; W is an MSS word of period k.
- returnmaps: the return maps of a return-map scan (scan --return-period 6). For each chaotic cell in the domain and
  each phase: the turning points of the 6-step return of the phase's first principal coordinate (binned medians),
  the thickness of the return curve, the agreement between the coding by the turning point and by derivative signs,
  and the kneading sequence (the itinerary of the band's extreme point, the critical value).

Usage:
  python tools/symbolic_dynamics.py mss --kmax 8
  python tools/symbolic_dynamics.py words --bands llm-soft-loop/pod64/bands/sdw-c --n 16 --compare llm-soft-loop/pod64/bands/hr10-c \
      --out llm-soft-loop/pod64/sd/words --fig paper/llm/figures/fig-p6-words.png
  python tools/symbolic_dynamics.py returnmaps --bands llm-soft-loop/pod64/bands/sdrm-c --n 3 \
      --out llm-soft-loop/pod64/sd/returnmaps --fig paper/llm/figures/fig-p6-return-maps.png
"""
import argparse
import functools
import itertools
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
VAL = {"L": 0, "C": 1, "R": 2}


def ucmp(s, t):
    """Unimodal (kneading) order of two symbol sequences for a map with a maximum (L < C < R, the order reversed after
    an odd number of R's): -1, 0 or 1."""
    parity = 0
    for a, b in zip(s, t):
        if a != b:
            d = 1 if VAL[a] > VAL[b] else -1
            return -d if parity else d
        if a == "C":                         # both at the turning point: the same sequence from here on
            return 0
        if a == "R":
            parity ^= 1
    return 0


def periodic(word, n=6):
    return word * max(1, (n * 8) // max(1, len(word)) + 2)


def mss_words(kmax):
    """MSS words W (period k = len(W) + 1) up to period kmax, in the order of the parameter of the quadratic family:
    W C must be shift-maximal (sigma^j (W C)^inf < (W C)^inf for 0 < j < k)."""
    words = []
    for k in range(1, kmax + 1):
        for bits in itertools.product("LR", repeat=k - 1):
            W = "".join(bits)
            K = periodic(W + "C")
            if all(ucmp(K[j:], K) < 0 for j in range(1, k)):
                words.append(W)
    words.sort(key=functools.cmp_to_key(lambda a, b: ucmp(periodic(a + "C"), periodic(b + "C"))))
    return words


def superstable_a(W, iters=4000):
    """Superstable parameter of x -> a - x^2 with MSS word W (inverse iteration from a = 2: x_k = 0 and
    x_{i} = s_i sqrt(a - x_{i+1}) backward, a <- x_1 until it settles)."""
    a = 2.0
    for _ in range(iters):
        x = 0.0
        for s in reversed(W):
            x = (1.0 if s == "R" else -1.0) * np.sqrt(max(a - x, 0.0))
        a_new = x if W else 0.0
        if abs(a_new - a) < 1e-15:
            break
        a = a_new
    return a


def itinerary_quadratic(a, k):
    x, out = a, []
    for _ in range(k - 1):
        out.append("R" if x > 1e-12 else ("L" if x < -1e-12 else "C"))
        x = a - x * x
    return "".join(out), abs(x)


def cmd_mss(args):
    words = mss_words(args.kmax)
    counts = {k: sum(1 for w in words if len(w) + 1 == k) for k in range(1, args.kmax + 1)}
    rows, ok = [], True
    for W in words:
        k = len(W) + 1
        a = superstable_a(W) if k > 1 else 0.0
        it, res = itinerary_quadratic(a, k) if k > 1 else ("", 0.0)
        rows.append({"word": W, "period": k, "a": a, "itinerary_check": it == W, "f^k(0)": res})
        ok &= (it == W) and res < 1e-9
    a_sorted = all(rows[i]["a"] < rows[i + 1]["a"] for i in range(len(rows) - 1))
    rec = {"counts_by_period": counts, "expected_counts": {1: 1, 2: 1, 3: 1, 4: 2, 5: 3, 6: 5, 7: 9, 8: 16, 9: 28},
           "numeric_words_match": bool(ok), "order_matches_parameter": bool(a_sorted), "words": rows}
    print(json.dumps(rec, indent=1))
    return rec


def rotmin_key(ids):
    ids = tuple(int(t) for t in ids)
    return min(ids[i:] + ids[:i] for i in range(len(ids))) if ids else ()


def load_band(prefix, k):
    d = DATA / "explore" / f"{prefix}{k:02d}"
    z = np.load(d / "scan.npz", allow_pickle=True)
    rec = json.loads((d / "scan.json").read_text())
    return z, rec


def body_text(recs, period=6):
    """The 6-token text of the body: the period-6 family with the most cells over all bands."""
    tally = {}
    for rec in recs:
        for f in rec["families"]:
            if f["period"] == period and "token_ids" in f:
                key = rotmin_key(f["token_ids"])
                tally[key] = tally.get(key, 0) + f["cells"]
    if not tally:
        raise SystemExit("no period-6 family with token ids in these bands; nothing defines the 6-band domain")
    key = max(tally, key=tally.get)
    return key, tally


def max_rotation(seq):
    best = 0
    for i in range(1, len(seq)):
        if ucmp(periodic(seq[i:] + seq[:i]), periodic(seq[best:] + seq[:best])) > 0:
            best = i
    return best


def cell_word(cos, P, k, min_cos):
    """Word of a period-P*k orbit from the last P*k recorded cosines: per phase, the k signs in time order and their
    maximal rotation. Returns (word, status, details)."""
    R = len(cos)
    words, worst = [], 1.0
    for j in range(P):
        idx = [R - 1 - j - P * i for i in range(k)][::-1]       # time order
        c = cos[idx]
        if not np.all(np.isfinite(c)):
            return None, "missing", {}
        seq = "".join("L" if x > 0 else "R" for x in c)
        b = max_rotation(seq)
        crit = (b - 1) % k                                      # the visit whose symbol is X (nearest the turning point)
        others = np.delete(np.abs(c), crit) if k > 1 else np.array([])
        if len(others):
            worst = min(worst, float(others.min()))
        words.append(seq[b:] + seq[:b])
    if len(set(words)) > 1:
        return words[0], "phases disagree", {"phase_words": words}
    if worst < min_cos:
        return words[0], "weak cosine", {"min_cos": worst}
    return words[0], "ok", {"min_cos": worst}


def cmd_words(args):
    zs, recs = zip(*[load_band(args.bands, k) for k in range(args.n)])
    text, tally = body_text(recs) if not args.text_ids else (rotmin_key(args.text_ids), {})
    P = args.period
    mss = mss_words(args.kmax)
    mss_set = set(mss)
    t = np.concatenate([z["log10_temperature"] for z in zs])
    a = zs[0]["alpha"]
    lam = np.concatenate([z["lyapunov"] for z in zs])
    per = np.concatenate([z["period"] for z in zs])
    status = np.zeros(lam.shape, dtype=int)     # 0 chaotic or slow, 1 periodic: other family, 2 gate failed, 3 word
    labels, label_of = [], {}
    word_idx = np.full(lam.shape, -1, dtype=int)
    min_cos = np.full(lam.shape, np.nan)
    reasons = {}
    row0 = 0
    for z, rec in zip(zs, recs):
        fam_ids = {f["family"]: rotmin_key(f.get("token_ids", [])) for f in rec["families"]}
        fam = z["family"]
        cos = z["word_cos"]
        nr = fam.shape[0]
        for r in range(nr):
            for c in range(fam.shape[1]):
                q = int(z["period"][r, c])
                if q == 0:
                    continue
                ids = fam_ids.get(int(fam[r, c]), ())
                k = q // P
                if q % P or k > args.kmax or ids != rotmin_key(text * k):
                    status[row0 + r, c] = 1
                    continue
                w, st, det = cell_word(cos[r, c], P, k, args.min_cos)
                if st == "ok" and w[:-1] not in mss_set:
                    st = "not an MSS word"
                if st != "ok":
                    status[row0 + r, c] = 2
                    reasons[st] = reasons.get(st, 0) + 1
                    continue
                status[row0 + r, c] = 3
                lab = w[:-1] + "|" + w[-1]
                if lab not in label_of:
                    label_of[lab] = len(labels)
                    labels.append(lab)
                word_idx[row0 + r, c] = label_of[lab]
                min_cos[row0 + r, c] = det.get("min_cos", np.nan)
        row0 += nr
    counts = {lab: int((word_idx == i).sum()) for i, lab in enumerate(labels)}
    windows = {}
    for lab, n in counts.items():
        W, X = lab.split("|")
        windows.setdefault(W, {"period": P * (len(W) + 1), "mss_rank": mss.index(W), "cells": {}})["cells"][X] = n
    rec_out = {"args": vars(args), "body_text_ids": list(text), "body_text_tally": {str(list(k)): v for k, v in tally.items()},
               "cells": int(lam.size), "periodic": int((per > 0).sum()), "other_family_or_period": int((status == 1).sum()),
               "gate_failed": int((status == 2).sum()), "gate_reasons": reasons, "coded": int((status == 3).sum()),
               "windows_in_mss_order": dict(sorted(windows.items(), key=lambda kv: kv[1]["mss_rank"])),
               "min_cos_of_coded_cells": float(np.nanmin(min_cos)) if np.isfinite(min_cos).any() else None}
    if args.compare:                            # reproducibility: the same grid and protocol as an earlier scan
        zc = [load_band(args.compare, k)[0] for k in range(args.n)]
        lamc = np.concatenate([z["lyapunov"] for z in zc])
        perc = np.concatenate([z["period"] for z in zc])
        m = (perc > 0) & (perc <= 32)
        same_p = (per == perc) | ((perc == 0) & (per > 32))
        rec_out["compare"] = {"scan": args.compare, "cells_same_period_or_newly_long": float(same_p.mean()),
                              "periodic_cells_up_to_32": int(m.sum()),
                              "periodic_same_period": float((per[m] == perc[m]).mean()),
                              "periodic_max_abs_lyapunov_difference": float(np.max(np.abs(lam[m] - lamc[m]))),
                              "chaotic_both": float(((lam > 1e-3) == (lamc > 1e-3)).mean())}
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "words.npz", alpha=a, log10_temperature=t, lyapunov=lam, period=per, status=status,
                        word_index=word_idx, labels=np.array(labels), min_cos=min_cos)
    (out / "words.json").write_text(json.dumps(rec_out, indent=1))
    print(json.dumps({k: v for k, v in rec_out.items() if k not in ("args", "body_text_tally")}, indent=1))
    if args.fig:
        plot_words(a, t, lam, status, word_idx, labels, mss, P, args.fig)


def plot_words(a, t, lam, status, word_idx, labels, mss, P, fig_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    Ws = sorted({lab.split("|")[0] for lab in labels}, key=mss.index)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22",
               "#7f7f7f", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94"]
    col = {W: to_rgb(palette[i % len(palette)]) for i, W in enumerate(Ws)}
    img = np.ones(lam.shape + (3,))
    chaos = status == 0
    img[chaos & (lam > 1e-3)] = (0.80, 0.80, 0.80)
    img[status == 1] = (0.35, 0.35, 0.35)
    img[status == 2] = (0.0, 0.0, 0.0)
    for i, lab in enumerate(labels):
        W, X = lab.split("|")
        c = np.array(col[W])
        img[word_idx == i] = c if X == "R" else 0.55 * c + 0.45
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.imshow(img, origin="lower", extent=[a[0], a[-1], t[0], t[-1]], aspect="auto", interpolation="nearest")
    ax.set_xlabel(r"steering strength $\alpha$")
    ax.set_ylabel(r"$\log_{10}T$")
    from matplotlib.patches import Patch
    handles = []
    for W in Ws:
        n = sum(int((word_idx == i).sum()) for i, lab in enumerate(labels) if lab.split("|")[0] == W)
        handles.append(Patch(color=col[W], label=f"{W or '(none)'}  (period {P * (len(W) + 1)}, {n} cells)"))
    handles += [Patch(color=(0.8, 0.8, 0.8), label="chaotic"), Patch(color=(0.35, 0.35, 0.35), label="other periodic"),
                Patch(color="k", label="gate failed")]
    ax.legend(handles=handles, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="window word W\n(pale: X = L)",
              title_fontsize=7)
    fig.tight_layout()
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=250, bbox_inches="tight")
    print(f"wrote {fig_path}")


def turning_points(x, y, bins=40, min_count=5):
    """Turning points of the binned-median curve y(x) (sign changes of its slope), with the curve."""
    edges = np.linspace(np.min(x), np.max(x), bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, bins - 1)
    cx, cy = [], []
    for b in range(bins):
        m = idx == b
        if m.sum() >= min_count:
            cx.append(0.5 * (edges[b] + edges[b + 1]))
            cy.append(float(np.median(y[m])))
    cx, cy = np.array(cx), np.array(cy)
    if len(cy) < 3:
        return 0, cx, cy
    slope = np.sign(np.diff(cy))
    slope = slope[slope != 0]
    return int(np.sum(slope[1:] != slope[:-1])), cx, cy


def cmd_returnmaps(args):
    zs, recs = zip(*[load_band(args.bands, k) for k in range(args.n)])
    P = args.period
    texts = []
    for rec in recs:
        texts.append({tf["text_family"]: rotmin_key(tf["token_ids"]) for tf in rec.get("texts", [])})
    body = rotmin_key(args.text_ids) if args.text_ids else None
    if body is None:                            # the period-6 text with the most cells
        tally = {}
        for z, tx in zip(zs, texts):
            for i, key in tx.items():
                if len(key) == P:
                    tally[key] = tally.get(key, 0) + int((z["text_family"] == i).sum())
        body = max(tally, key=tally.get)
    cells, maps = [], {}
    for b, (z, tx) in enumerate(zip(zs, texts)):
        lam, per, tp, tf = z["lyapunov"], z["period"], z["text_period"], z["text_family"]
        proj, expl, cos = z["return_projection"], z["return_explained"], z["word_cos"]
        for r in range(lam.shape[0]):
            for c in range(lam.shape[1]):
                inside = int(tp[r, c]) == P and tx.get(int(tf[r, c])) == body
                rec = {"alpha": float(z["alpha"][c]), "log10_temperature": float(z["log10_temperature"][r]),
                       "band": b, "row": r, "col": c, "lyapunov": float(lam[r, c]), "period": int(per[r, c]),
                       "text_period": int(tp[r, c]), "domain": bool(inside)}
                if inside and lam[r, c] > args.chaos_level:
                    ph = []
                    for j in range(P):
                        s = proj[r, c, j, :, 0].astype(float)
                        x, y = s[:-1], s[1:]
                        ntp, cx, cy = turning_points(x, y)
                        spread = float(np.ptp(s))
                        resid = y - np.interp(x, cx, cy) if len(cx) >= 2 else y * np.nan
                        e = {"phase": j, "turning_points": ntp, "explained_1": float(expl[r, c, j, 0]),
                             "thickness": float(np.sqrt(np.nanmean(resid ** 2)) / spread) if spread > 0 else None}
                        # derivative signs of the visits (record index j + 6i is coded by the cosine at j + 6(i+1))
                        ci = cos[r, c, j + P::P][: len(s) - 1]
                        sym_sign = np.where(ci > 0, "L", "R")
                        shape = None
                        if ntp == 1:
                            imax, imin = int(np.argmax(cy)), int(np.argmin(cy))
                            shape = "max" if 0 < imax < len(cy) - 1 else "min" if 0 < imin < len(cy) - 1 else None
                        if shape:
                            cpos = float(cx[imax if shape == "max" else imin])
                            sym_proj = np.where((s[: len(ci)] < cpos) == (shape == "max"), "L", "R")
                            far = np.abs(s[: len(ci)] - cpos) > args.margin * spread
                            agree = float(np.mean(sym_proj[far] == sym_sign[far])) if far.any() else None
                            ext = int(np.argmax(s[: len(ci)])) if shape == "max" else int(np.argmin(s[: len(ci)]))
                            kneading = "".join(sym_sign[ext: ext + args.kneading_len])
                            e.update(turning_point=cpos, shape=shape, coding_agreement=agree,
                                     kneading_prefix=kneading, min_abs_cos=float(np.min(np.abs(ci))))
                        ph.append(e)
                    rec["phases"] = ph
                    rec["unimodal_all_phases"] = all(p["turning_points"] == 1 for p in ph)
                    maps[f"b{b}_r{r}_c{c}"] = proj[r, c, :, :, 0]
                cells.append(rec)
    dom_chaos = [c for c in cells if c["domain"] and c["lyapunov"] > args.chaos_level]
    uni = [c for c in dom_chaos if c.get("unimodal_all_phases")]
    agr = [p["coding_agreement"] for c in uni for p in c["phases"] if p.get("coding_agreement") is not None]
    thick = [p["thickness"] for c in dom_chaos for p in c["phases"] if p.get("thickness") is not None]
    summary = {"body_text_ids": list(body), "cells": len(cells), "domain_cells": sum(c["domain"] for c in cells),
               "domain_chaotic": len(dom_chaos), "unimodal_in_all_phases": len(uni),
               "turning_point_counts": {str(n): sum(1 for c in dom_chaos for p in c["phases"] if p["turning_points"] == n)
                                        for n in range(0, 5)},
               "coding_agreement": {"min": min(agr) if agr else None, "median": float(np.median(agr)) if agr else None},
               "thickness": {"median": float(np.median(thick)) if thick else None, "max": max(thick) if thick else None}}
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "returnmaps.json").write_text(json.dumps({"args": vars(args), "summary": summary, "cells": cells}, indent=1))
    print(json.dumps(summary, indent=1))
    if args.fig:
        plot_returnmaps(cells, zs, args, body)


def plot_returnmaps(cells, zs, args, body):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P = args.period
    sel = [c for c in cells if c.get("unimodal_all_phases")]
    sel.sort(key=lambda c: -c["lyapunov"])
    picks = []
    for c in sel:                               # spread the examples over the window
        if all(abs(c["alpha"] - p["alpha"]) > 0.0004 or abs(c["log10_temperature"] - p["log10_temperature"]) > 0.0012
               for p in picks):
            picks.append(c)
        if len(picks) == 6:
            break
    fig = plt.figure(figsize=(10.0, 5.6))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.35, 1, 1, 1])
    axm = fig.add_subplot(gs[:, 0])
    a = zs[0]["alpha"]
    t = np.concatenate([z["log10_temperature"] for z in zs])
    img = np.ones((len(t), len(a), 3))
    for c in cells:
        rr = sum(len(z["log10_temperature"]) for z in zs[: c["band"]]) + c["row"]
        if c["lyapunov"] <= args.chaos_level:
            img[rr, c["col"]] = (1, 1, 1) if c["period"] else (0.9, 0.9, 0.9)
        elif not c["domain"]:
            img[rr, c["col"]] = (0.6, 0.6, 0.6)
        else:
            img[rr, c["col"]] = (0.2, 0.6, 0.2) if c.get("unimodal_all_phases") else (0.8, 0.2, 0.2)
    axm.imshow(img, origin="lower", extent=[a[0], a[-1], t[0], t[-1]], aspect="auto", interpolation="nearest")
    for i, c in enumerate(picks):
        axm.text(c["alpha"], c["log10_temperature"], str(i + 1), fontsize=7, ha="center", va="center",
                 bbox=dict(boxstyle="circle,pad=0.15", fc="w", ec="k", lw=0.5))
    axm.set_xlabel(r"steering strength $\alpha$")
    axm.set_ylabel(r"$\log_{10}T$")
    axm.set_title("(a) chaotic cells of the 6-band domain:\ngreen one turning point in every phase", fontsize=7.5)
    for i, c in enumerate(picks):
        ax = fig.add_subplot(gs[i // 3, 1 + i % 3])
        z = zs[c["band"]]
        s = z["return_projection"][c["row"], c["col"], 0, :, 0].astype(float)
        cos = z["word_cos"][c["row"], c["col"], P::P][: len(s) - 1]
        ax.scatter(s[:-1], s[1:], c=np.where(cos > 0, "tab:blue", "tab:red"), s=2, lw=0)
        ph0 = c["phases"][0]
        if "turning_point" in ph0:
            ax.axvline(ph0["turning_point"], color="k", lw=0.5, ls=":")
        ax.set_title(f"({chr(98 + i)}) cell {i + 1}: $\\lambda$ = {c['lyapunov']:.3f}", fontsize=7.5)
        ax.tick_params(labelsize=5.5)
        if i // 3 == 1:
            ax.set_xlabel(r"$s_n$ (phase 0, first principal axis)", fontsize=6.5)
        if i % 3 == 0:
            ax.set_ylabel(r"$s_{n+1}$ (six steps later)", fontsize=6.5)
    fig.tight_layout()
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=250, bbox_inches="tight")
    print(f"wrote {args.fig}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mss")
    m.add_argument("--kmax", type=int, default=8)
    w = sub.add_parser("words")
    w.add_argument("--bands", required=True, help="explore-relative band prefix, e.g. llm-soft-loop/pod64/bands/sdw-c")
    w.add_argument("--n", type=int, default=16)
    w.add_argument("--period", type=int, default=6)
    w.add_argument("--kmax", type=int, default=8)
    w.add_argument("--min-cos", type=float, default=0.5)
    w.add_argument("--text-ids", type=int, nargs="*", default=None)
    w.add_argument("--compare", default=None, help="band prefix of an earlier scan on the same grid (reproducibility)")
    w.add_argument("--out", required=True)
    w.add_argument("--fig", default=None)
    r = sub.add_parser("returnmaps")
    r.add_argument("--bands", required=True)
    r.add_argument("--n", type=int, default=3)
    r.add_argument("--period", type=int, default=6)
    r.add_argument("--chaos-level", type=float, default=1e-3)
    r.add_argument("--margin", type=float, default=0.02, help="visits within this fraction of the spread of the turning "
                   "point are left out of the coding comparison")
    r.add_argument("--kneading-len", type=int, default=16)
    r.add_argument("--text-ids", type=int, nargs="*", default=None)
    r.add_argument("--out", required=True)
    r.add_argument("--fig", default=None)
    args = ap.parse_args()
    {"mss": cmd_mss, "words": cmd_words, "returnmaps": cmd_returnmaps}[args.cmd](args)


if __name__ == "__main__":
    main()
