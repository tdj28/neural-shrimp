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


def cell_word(cos, P, k, min_cos, min_phases=3):
    """Word of a period-P*k orbit from the last P*k recorded cosines. Per phase: the k signs in time order, their maximal
    rotation, and the smallest |cos| at the visits other than the critical one. A phase is usable when that smallest
    |cos| reaches min_cos (in the phase holding the fold's tip the band is a hairpin and the sign is ill defined). The
    usable phases must agree; if they agree on W but not on the critical symbol, the cell lies on the window's
    superstable curve (X = C). Returns (word, status, details)."""
    R = len(cos)
    phases = []
    for j in range(P):
        idx = [R - 1 - j - P * i for i in range(k)][::-1]       # time order
        c = cos[idx]
        if not np.all(np.isfinite(c)):
            return None, "missing", {}
        seq = "".join("L" if x > 0 else "R" for x in c)
        b = max_rotation(seq)
        crit = (b - 1) % k                                      # the visit whose symbol is X (nearest the turning point)
        others = np.delete(np.abs(c), crit)
        phases.append((seq[b:] + seq[:b], float(others.min()) if len(others) else 1.0, float(abs(c[crit]))))
    usable = [w for w, m, _ in phases if m >= min_cos]
    det = {"usable_phases": len(usable), "phase_words": [w for w, _, _ in phases]}
    if not usable:
        return phases[0][0], "too few usable phases", det
    from collections import Counter
    wv = Counter(w[:-1] for w in usable)               # votes for the window word W
    W, nW = wv.most_common(1)[0]
    det["votes_for_W"] = nW
    if nW < min_phases or (len(wv) > 1 and wv.most_common(2)[1][1] == nW):
        return usable[0], "phases disagree", det
    xv = Counter(w[-1] for w in usable if w[:-1] == W)  # votes for the side X among the phases that give W
    X, nX = xv.most_common(1)[0]
    det["votes_for_X"] = nX
    return W + (X if nX >= 2 * nW / 3 else "C"), "ok", det   # split sides: on the window's superstable curve


def load_dir(rel):
    d = DATA / "explore" / rel
    return np.load(d / "scan.npz", allow_pickle=True), json.loads((d / "scan.json").read_text())


def cmd_words(args):
    if args.list:
        zs, recs = zip(*[load_dir(d) for d in args.list])
    else:
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
                w, st, det = cell_word(cos[r, c], P, k, args.min_cos, args.min_phases)
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
                min_cos[row0 + r, c] = det.get("usable_phases", np.nan)
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
               "usable_phases_of_coded_cells": {str(n): int(np.sum(min_cos == n)) for n in range(1, 7)}}
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
                        word_index=word_idx, labels=np.array(labels), usable_phases=min_cos)
    (out / "words.json").write_text(json.dumps(rec_out, indent=1))
    print(json.dumps({k: v for k, v in rec_out.items() if k not in ("args", "body_text_tally")}, indent=1))
    if args.fig:
        plot_words(a, t, lam, status, word_idx, labels, mss, P, args.fig, zoom=args.zoom, unconfirmed=set(args.unconfirmed))


WORD_COLORS = {"": "#9ecae1", "R": "#4292c6", "RLR": "#08519c", "RLRRRLR": "#08306b", "RL": "#ff7f0e", "RLL": "#d62728",
               "RLRR": "#2ca02c", "RLLR": "#9467bd", "RLRRR": "#8c564b", "RLLRL": "#e377c2", "RLLL": "#bcbd22",
               "RLLLL": "#17becf", "RLLRR": "#7f3b08", "RLLLR": "#b15928"}
OTHER_COLOR = "#e6ab02"                                      # every other MSS word (periods up to 48)


def word_image(lam, status, word_idx, labels):
    from matplotlib.colors import to_rgb
    img = np.ones(lam.shape + (3,))
    img[(status == 0) & (lam > 1e-3)] = (0.88, 0.88, 0.88)          # chaos
    img[status == 1] = (0.55, 0.55, 0.55)                            # periodic, other family or period
    img[status == 2] = (0.30, 0.30, 0.30)                            # gate failed
    for i, lab in enumerate(labels):
        W, X = lab.split("|")
        c = np.array(to_rgb(WORD_COLORS.get(W, OTHER_COLOR)))
        positive = ((W + X).count("R") % 2) == 0                    # sign of the leading multiplier
        img[word_idx == i] = (1.0, 0.0, 1.0) if X == "C" else (0.45 * c + 0.55 if positive else c)
    return img


def plot_words(a, t, lam, status, word_idx, labels, mss, P, fig_path, zoom=None, unconfirmed=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    img = word_image(lam, status, word_idx, labels)
    present = {}
    for i, lab in enumerate(labels):
        W = lab.split("|")[0]
        present[W] = present.get(W, 0) + int((word_idx == i).sum())
    fig = plt.figure(figsize=(7.2, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.42], width_ratios=[1.15, 1.0], hspace=0.32, wspace=0.28)
    axa, axb, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    ext = [a[0], a[-1], t[0], t[-1]]
    axa.imshow(img, origin="lower", extent=ext, aspect="auto", interpolation="nearest")
    axa.set_title("(a) the window of the period-6 shrimp", fontsize=8)
    if zoom:
        a0, a1, t0, t1 = zoom
        axa.add_patch(Rectangle((a0, t0), a1 - a0, t1 - t0, fill=False, ec="k", lw=0.8))
        ci = np.nonzero((a >= a0) & (a <= a1))[0]
        ri = np.nonzero((t >= t0) & (t <= t1))[0]
        axb.imshow(img[np.ix_(ri, ci)], origin="lower", extent=[a[ci[0]], a[ci[-1]], t[ri[0]], t[ri[-1]]], aspect="auto",
                   interpolation="nearest")
        axb.set_title("(b) the box in (a): small windows below the body", fontsize=8)
    for ax in (axa, axb):
        ax.set_xlabel(r"steering strength $\alpha$", fontsize=8)
        ax.set_ylabel(r"$\log_{10}T$", fontsize=8)
        ax.tick_params(labelsize=6.5)
        ax.ticklabel_format(useOffset=False)
    # (c) the words at their superstable parameters in the quadratic family x -> a - x^2
    A = np.linspace(0.6, 2.0, 1400)
    xs = []
    for av in A:
        x = 0.0
        for _ in range(400):
            x = av - x * x
        orb = []
        for _ in range(80):
            x = av - x * x
            orb.append(x)
        xs.append(orb)
    xs = np.array(xs)
    axc.plot(np.repeat(A, xs.shape[1]), xs.ravel(), ",", color="0.55", alpha=0.6)
    shown = [W for W in mss if W in present]
    lab_k = 0
    for W in shown:
        aw = superstable_a(W) if W else 0.0
        if aw < A[0]:
            continue
        col = WORD_COLORS.get(W, OTHER_COLOR)
        axc.plot([aw, aw], [2.05, 2.45], color=col, lw=1.4, solid_capstyle="butt")
        if present[W] >= 15:                               # label the larger windows; the rest are ticks only
            axc.text(aw, 2.55 + 0.33 * (lab_k % 2), f"{W} ({P * (len(W) + 1)})", fontsize=5.8, ha="center", va="bottom",
                     color=col)
            lab_k += 1
    axc.set_xlim(A[0], A[-1])
    axc.set_ylim(-2.1, 3.3)
    axc.set_xlabel(r"parameter $a$ of the quadratic map $x\mapsto a-x^2$", fontsize=8)
    axc.set_ylabel(r"$x$", fontsize=8)
    axc.tick_params(labelsize=6.5)
    axc.set_title("(c) the same words at their superstable parameters in the quadratic family (period of the soft-token orbit)",
                  fontsize=8)
    listed = [W for W in shown if W in WORD_COLORS and present[W] >= 3]
    others = [W for W in shown if W not in listed]
    handles = [Patch(color=WORD_COLORS[W], label=f"{W or '(empty)'}: period {P * (len(W) + 1)}, {present[W]} cells"
                     + (" (not confirmed)" if W in unconfirmed else "")) for W in listed]
    if others:
        n_unc = sum(1 for W in others if W in unconfirmed)
        note = (f"{n_unc} not confirmed, {len(others) - n_unc} not checked" if n_unc else "not checked")
        handles.append(Patch(color=OTHER_COLOR, label=f"{len(others)} other MSS words, periods {min(P * (len(W) + 1) for W in others)}"
                             f" to {max(P * (len(W) + 1) for W in others)}: {sum(present[W] for W in others)} cells ({note})"))
    handles += [Patch(color=(1, 0, 1), label="on a superstable curve (X = C)"), Patch(color=(0.88, 0.88, 0.88), label="chaotic"),
                Patch(color=(0.55, 0.55, 0.55), label="other periodic"), Patch(color=(0.30, 0.30, 0.30), label="no word (gate failed)")]
    fig.legend(handles=handles, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=False,
               title="window word W (pale: leading multiplier positive; full: negative)", title_fontsize=6.5)
    fig.subplots_adjust(top=0.86)
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
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


def arc_coordinate(xy, k=8):
    """Arc length along a thin curve of points xy (n, 2): geodesic distance on a k-nearest-neighbor graph from one end
    of the curve. Returns None if the graph is not connected (a banded attractor; use the principal coordinate)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra
    from scipy.spatial import cKDTree

    n = len(xy)
    d, j = cKDTree(xy).query(xy, k=min(k + 1, n))
    rows = np.repeat(np.arange(n), d.shape[1] - 1)
    g = csr_matrix((d[:, 1:].ravel(), (rows, j[:, 1:].ravel())), shape=(n, n))
    g = g.maximum(g.T)
    if connected_components(g, directed=False)[0] > 1:
        return None
    far = int(np.argmax(dijkstra(g, indices=0)))           # one end of the curve
    return dijkstra(g, indices=far)


def bend(xy, k=40):
    """How far the band of points xy (n, 2) in its principal plane turns away from the first principal axis: the 95th
    percentile, in degrees, of the angle between the axis and the band's local direction (the principal direction of
    each point's k nearest neighbors, a scale above the band's thickness). The derivative-sign coding needs every pair
    of tangents within 90 degrees, so a band whose local directions all lie within 45 degrees of the axis qualifies."""
    from scipy.spatial import cKDTree

    n = len(xy)
    if n < 10:
        return None
    k = min(k, n)
    _, j = cKDTree(xy).query(xy, k=k)
    nb = xy[j] - xy[j].mean(axis=1, keepdims=True)                 # (n, k, 2), centered per neighborhood
    cxx, cyy, cxy = (nb[..., 0] ** 2).mean(1), (nb[..., 1] ** 2).mean(1), (nb[..., 0] * nb[..., 1]).mean(1)
    theta = 0.5 * np.arctan2(2 * cxy, cxx - cyy)                     # leading local direction, radians
    ang = np.degrees(np.abs(theta))                                  # 0..90 from the first principal axis
    return float(np.percentile(ang, 95))


def robust_turning_points(x, y, bins=16, min_count=4, frac=0.03):
    """Turning points of the binned-median return curve y(x), ignoring slope changes smaller than frac of y's spread
    (small steps are merged before the direction is read). Returns (count, cx, cy)."""
    edges = np.linspace(x.min(), x.max(), bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, bins - 1)
    keep = [b for b in range(bins) if (idx == b).sum() >= min_count]
    cx = np.array([0.5 * (edges[b] + edges[b + 1]) for b in keep])
    cy = np.array([np.median(y[idx == b]) for b in keep])
    if len(cy) < 3:
        return 0, cx, cy
    thr, dirs, acc = frac * np.ptp(y), [], 0.0
    for v in np.diff(cy):
        acc += v
        if abs(acc) > thr:
            dirs.append(np.sign(acc))
            acc = 0.0
    return sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i - 1]), cx, cy


def separability(s, sym, weak):
    """Best single threshold on the coordinate s separating the L visits from the R visits (the coding of a unimodal map
    has one), leaving out the weak (near-critical) visits. Returns (agreement, threshold, L_below, n_used)."""
    keep = ~weak
    x, y = s[keep], sym[keep] == "L"
    n = len(x)
    if n < 10 or y.all() or (~y).all():
        return (1.0 if n else None), None, None, n
    o = np.argsort(x)
    x, y = x[o], y[o]
    # agreement for "L below threshold" at every cut k (first k points below): L among the first k + R among the rest
    L_below = np.concatenate([[0], np.cumsum(y)]) + (np.sum(~y) - np.concatenate([[0], np.cumsum(~y)]))
    R_below = n - L_below
    kb, ka = int(np.argmax(L_below)), int(np.argmax(R_below))
    if L_below[kb] >= R_below[ka]:
        k, best, lb = kb, L_below[kb], True
    else:
        k, best, lb = ka, R_below[ka], False
    thr = float(x[k - 1] + x[k]) / 2 if 0 < k < n else (float(x[0]) if k == 0 else float(x[-1]))
    return float(best / n), thr, lb, n


def cmd_returnmaps(args):
    if args.list:                                   # explicit band folders in row order (e.g. a rerun of half a band)
        zs, recs = zip(*[load_dir(d) for d in args.list])
    else:
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
                        sv = s[: len(ci)]                 # the principal axis (usable phases are straight)
                        sym = np.where(ci > 0, "L", "R")
                        weak = np.abs(ci) < args.min_cos
                        agree, thr, lb, nused = separability(sv, sym, weak)
                        nL = int(np.sum((sym == "L") & ~weak))
                        minority = min(nL, nused - nL)
                        errors = int(round((1 - agree) * nused)) if agree is not None else None
                        bd = bend(proj[r, c, j, : len(ci), :].astype(float))
                        e["turning_points_robust"] = robust_turning_points(s[:-1], s[1:])[0]
                        e.update(separability=agree, threshold=thr, weak_fraction=float(weak.mean()), visits=int(len(ci)),
                                 minority=minority, errors=errors, bend_deg=bd,
                                 usable=bool(weak.mean() <= args.max_weak and minority >= 5 and bd is not None
                                             and bd < args.max_bend))
                        if thr is not None:
                            so = sv if lb else -sv                # oriented so that L lies below the turning point
                            ext = int(np.argmax(so))              # the critical value f(c): the top of the attractor
                            e["kneading_prefix"] = "".join(sym[ext: ext + args.kneading_len])
                        ph.append(e)
                    rec["phases"] = ph
                    use = [p for p in ph if p["usable"]]
                    rec["usable_phases"] = len(use)
                    rec["errors_over_minority"] = (sum(p["errors"] for p in use) / sum(p["minority"] for p in use)) if use else None
                    rec["unimodal_all_phases"] = bool(use) and all(p["separability"] >= args.min_separability for p in use)
                    rec["min_separability"] = min((p["separability"] for p in ph if p.get("separability") is not None), default=None)
                    maps[f"b{b}_r{r}_c{c}"] = proj[r, c, :, :, 0]
                cells.append(rec)
    dom_chaos = [c for c in cells if c["domain"] and c["lyapunov"] > args.chaos_level]
    uni = [c for c in dom_chaos if c.get("unimodal_all_phases")]
    agr = [c["min_separability"] for c in dom_chaos if c.get("min_separability") is not None]
    thick = [p["thickness"] for c in dom_chaos for p in c["phases"] if p.get("thickness") is not None]
    summary = {"body_text_ids": list(body), "cells": len(cells), "domain_cells": sum(c["domain"] for c in cells),
               "domain_chaotic": len(dom_chaos), "unimodal_in_all_phases": len(uni),
               "turning_point_counts": {str(n): sum(1 for c in dom_chaos for p in c["phases"] if p["turning_points"] == n)
                                        for n in range(0, 5)},
               "min_separability_per_cell": {"min": min(agr) if agr else None, "median": float(np.median(agr)) if agr else None,
                                             "cells_at_least_0.99": sum(1 for x in agr if x >= 0.99), "cells": len(agr)},
               "weak_visit_fraction_median": float(np.median([p["weak_fraction"] for c in dom_chaos for p in c["phases"]]))
               if dom_chaos else None,
               "usable_phases_per_cell": {str(n): sum(1 for c in dom_chaos if c.get("usable_phases") == n) for n in range(0, 7)},
               "errors_over_minority_in_usable_phases": {
                   "max": max((c["errors_over_minority"] for c in dom_chaos if c.get("errors_over_minority") is not None), default=None),
                   "median": float(np.median([c["errors_over_minority"] for c in dom_chaos if c.get("errors_over_minority") is not None]))
                   if any(c.get("errors_over_minority") is not None for c in dom_chaos) else None},
               "thickness": {"median": float(np.median(thick)) if thick else None, "max": max(thick) if thick else None},
               "exceptions_with_two_turning_points": sum(
                   1 for c in dom_chaos if (c.get("errors_over_minority") or 0) > 0.05 and
                   min(([p for p in c["phases"] if p.get("usable")] or c["phases"]),
                       key=lambda p: (p.get("bend_deg") or 99)).get("turning_points_robust") == 2)}
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
    bim = []
    for c in cells:                             # cells that fail the test and whose straightest phase has two turning points
        if c.get("domain") and c["lyapunov"] > args.chaos_level and (c.get("errors_over_minority") or 0) > 0.05:
            use = [p for p in c["phases"] if p.get("usable")] or c["phases"]
            best = min(use, key=lambda p: (p.get("bend_deg") or 99))
            if best.get("turning_points_robust") == 2:
                bim.append((c, best["phase"]))
    bim.sort(key=lambda cb: (cb[0]["alpha"] > 0.0728, cb[0]["log10_temperature"]))
    show_b = bim[:4]
    nrow = 3 if show_b else 2
    fig = plt.figure(figsize=(10.0, 5.6 + (2.9 if show_b else 0)))
    gs = fig.add_gridspec(nrow, 4, width_ratios=[1.35, 1, 1, 1])
    axm = fig.add_subplot(gs[:2, 0])
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
            ok = (c.get("errors_over_minority") is not None and c["errors_over_minority"] <= args.max_errors)
            img[rr, c["col"]] = (0.2, 0.6, 0.2) if ok else (0.8, 0.2, 0.2)   # the text's criterion
    axm.imshow(img, origin="lower", extent=[a[0], a[-1], t[0], t[-1]], aspect="auto", interpolation="nearest")
    for i, c in enumerate(picks):
        axm.text(c["alpha"], c["log10_temperature"], str(i + 1), fontsize=7, ha="center", va="center",
                 bbox=dict(boxstyle="circle,pad=0.15", fc="w", ec="k", lw=0.5))
    axm.set_xlabel(r"steering strength $\alpha$")
    axm.set_ylabel(r"$\log_{10}T$")
    axm.set_title("(a) the cells of the 48 x 48 map", fontsize=7.5)
    from matplotlib.patches import Patch
    axm.legend(handles=[Patch(color=(0.2, 0.6, 0.2), label="6-band chaos, unimodal coding"),
                        Patch(color=(0.8, 0.2, 0.2), label="6-band chaos, not unimodal"),
                        Patch(color=(0.6, 0.6, 0.6), label="other chaos"),
                        Patch(facecolor="w", edgecolor="k", lw=0.5, label="periodic")],
               fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False)
    for i, c in enumerate(picks):
        ax = fig.add_subplot(gs[i // 3, 1 + i % 3])
        z = zs[c["band"]]
        use = [p for p in c["phases"] if p.get("usable")] or c["phases"]
        best = max(use, key=lambda p: (p.get("separability") or 0, -p["weak_fraction"]))
        jb = best["phase"]
        s = z["return_projection"][c["row"], c["col"], jb, :, 0].astype(float)
        cos = z["word_cos"][c["row"], c["col"], jb + P::P][: len(s) - 1]
        ax.scatter(s[:-1], s[1:], c=np.where(cos > 0, "tab:blue", "tab:red"), s=2, lw=0)
        ax.set_title(f"({chr(98 + i)}) cell {i + 1}, phase {jb}: $\\lambda$ = {c['lyapunov']:.3f}", fontsize=7.5)
        ax.tick_params(labelsize=5.5)
        if i // 3 == 1:
            ax.set_xlabel(r"$s_n$ (first principal axis of the phase)", fontsize=6.5)
        if i % 3 == 0:
            ax.set_ylabel(r"$s_{n+1}$ (six steps later)", fontsize=6.5)
    for i, (c, jb) in enumerate(show_b):          # (h)-(k): return maps with two turning points
        ax = fig.add_subplot(gs[2, i])
        z = zs[c["band"]]
        sv = z["return_projection"][c["row"], c["col"], jb, :, 0].astype(float)
        cs = z["word_cos"][c["row"], c["col"], jb + P::P][: len(sv) - 1]
        ax.scatter(sv[:-1], sv[1:], c=np.where(cs > 0, "tab:blue", "tab:red"), s=2, lw=0)
        ax.set_title(f"({chr(104 + i)}) ({c['alpha']:.5f}, {c['log10_temperature']:.5f}), two turning points", fontsize=6.8)
        ax.tick_params(labelsize=5.5)
        ax.set_xlabel(r"$s_n$", fontsize=6.5)
        if i == 0:
            ax.set_ylabel(r"$s_{n+1}$", fontsize=6.5)
        axm.text(c["alpha"], c["log10_temperature"], chr(104 + i), fontsize=6.5, ha="center", va="center", color="w",
                 fontweight="bold")
    fig.tight_layout()
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=250, bbox_inches="tight")
    print(f"wrote {args.fig}")


def cmd_merge(args):
    """One band from the complete rows of a timed-out scan's checkpoint (scan-partial.npz) and a rerun of its missing
    rows (same columns; the rerun's rows must continue the checkpoint's). Families and texts are reconciled by token
    loop. The rerun's twins start from other random directions, so its exponents are not bit-identical to an
    uninterrupted run's."""
    D = DATA / "explore"
    zp = np.load(D / args.partial / "scan-partial.npz", allow_pickle=True)
    zr = np.load(D / args.rerun / "scan.npz", allow_pickle=True)
    rr = json.loads((D / args.rerun / "scan.json").read_text())
    nc = len(zp["alpha"])
    r0 = int(zp["done"]) // nc
    tp = zp["log10_temperature"][:r0]
    tr = zr["log10_temperature"]
    if not np.array_equal(zp["alpha"], zr["alpha"]):
        raise SystemExit("columns differ")
    step = np.diff(zp["log10_temperature"]).mean()
    if r0 and abs(tr[0] - (tp[-1] + step)) > 1e-9 * max(1.0, abs(step)):
        raise SystemExit("the rerun's rows do not continue the checkpoint's complete rows")
    keys_p = {"family": [tuple(k) for k in json.loads(str(zp["family_keys"]))], "text_family": [tuple(k) for k in json.loads(str(zp["text_keys"]))]}
    keys_r = {"family": [tuple(f["token_ids"]) for f in sorted(rr["families"], key=lambda f: f["family"])],
              "text_family": [tuple(f["token_ids"]) for f in sorted(rr.get("texts", []), key=lambda f: f["text_family"])]}
    out, merged = {"alpha": zp["alpha"], "log10_temperature": np.concatenate([tp, tr])}, {}
    for kind in ("family", "text_family"):
        keys = list(dict.fromkeys(keys_p[kind] + keys_r[kind]))
        index = {k: i for i, k in enumerate(keys)}
        mp = np.array([index[k] for k in keys_p[kind]] + [-1], dtype=int)
        mr = np.array([index[k] for k in keys_r[kind]] + [-1], dtype=int)
        a_p = zp[kind][: r0 * nc].reshape(r0, nc)
        a_r = zr[kind]
        out[kind] = np.concatenate([np.where(a_p >= 0, mp[np.clip(a_p, 0, len(mp) - 1)], -1),
                                    np.where(a_r >= 0, mr[np.clip(a_r, 0, len(mr) - 1)], -1)])
        merged[kind] = keys
    for k in ("lyapunov", "period", "entropy", "text_period"):
        out[k] = np.concatenate([zp[k][: r0 * nc].reshape(r0, nc), zr[k]])
    for k in ("word_cos", "word_log_growth"):
        if k in zp.files and k in zr.files:
            out[k] = np.concatenate([zp[k][: r0 * nc].reshape(r0, nc, -1), zr[k]])
    o = D / args.out
    o.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(o / "scan.npz", **out)
    rec = {"args": vars(args), "rows_from_checkpoint": r0, "rows_from_rerun": int(len(tr)),
           "families": [{"family": i, "period": len(k), "token_ids": list(k), "cells": int((out["family"] == i).sum())}
                        for i, k in enumerate(merged["family"])],
           "texts": [{"text_family": i, "period": len(k), "token_ids": list(k), "cells": int((out["text_family"] == i).sum())}
                     for i, k in enumerate(merged["text_family"])]}
    (o / "scan.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({"out": args.out, "rows_from_checkpoint": r0, "rows_from_rerun": int(len(tr)), "shape": list(out["lyapunov"].shape)}))


def cmd_verify_summary(args):
    """Score the float64 Newton checks (sd_verify_words.py) with the scan's rule and compare with the scan's words. A
    phase is clean when its L and R points are separable along the phase's principal axis and the smallest |cos| of
    its visits reaches --min-cos; the word is that of the clean phases (they must agree). A complex leading pair
    marks a cell on a superstable curve (X = C). The scan's label is that of the word map at the nearest cell."""
    from collections import Counter
    rows = []
    for v in args.verify:
        rows += json.loads((DATA / "explore" / v / "verify.json").read_text())["cells"]
    W = np.load(DATA / "explore" / args.words / "words.npz", allow_pickle=True)
    labels = list(W["labels"])
    out = []
    for c in rows:
        r = int(np.argmin(np.abs(W["log10_temperature"] - c["log10_temperature"])))
        k = int(np.argmin(np.abs(W["alpha"] - c["alpha"])))
        wi = int(W["word_index"][r, k])
        scan = labels[wi] if wi >= 0 else {0: "chaotic", 1: "other", 2: "no word"}.get(int(W["status"][r, k]), "?")
        rec = {"alpha": c["alpha"], "log10_temperature": c["log10_temperature"], "period": c["period"], "scan_label": scan,
               "newton_residual": c["newton_residuals"][-1], "leading_multiplier": c["leading_multiplier"],
               "second_modulus": c.get("second_modulus")}
        if c.get("status") == "leading multiplier complex or not simple":
            rec.update(newton_label="C (complex leading pair)", clean_phases=0,
                       agrees=scan.endswith("|C"))
        elif "phases" in c:
            clean = [p["word"] for p in c["phases"] if p["separable"] in (True, None) and p["min_abs_cos"] >= args.min_cos]
            cnt = Counter(clean)
            if clean and len(cnt) == 1:
                w = clean[0]
                rec.update(newton_label=w[:-1] + "|" + w[-1], clean_phases=len(clean), agrees=(w[:-1] + "|" + w[-1]) == scan,
                           mss=w[:-1] in set(mss_words(9)), parity=c.get("parity_matches_multiplier"))
            else:
                rec.update(newton_label="no clean phase" if not clean else "clean phases disagree", clean_phases=len(clean),
                           agrees=None)
        else:
            rec.update(newton_label=c.get("status"), agrees=None)
        out.append(rec)
    summ = {"cells": len(out), "confirmed": sum(1 for r in out if r["agrees"] is True),
            "contradicted": sum(1 for r in out if r["agrees"] is False),
            "unscorable": sum(1 for r in out if r["agrees"] is None), "rows": out}
    o = DATA / "explore" / args.out
    o.mkdir(parents=True, exist_ok=True)
    (o / "verify-summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: summ[k] for k in ("cells", "confirmed", "contradicted", "unscorable")}))
    for r in sorted(out, key=lambda r: (r["period"], r["scan_label"])):
        print(f"q={r['period']:2d} ({r['alpha']:.7f}, {r['log10_temperature']:.7f}) scan {r['scan_label']:>12}  newton {r['newton_label']:>24}"
              f"  clean {r.get('clean_phases')}  agrees {r['agrees']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mss")
    m.add_argument("--kmax", type=int, default=8)
    w = sub.add_parser("words")
    w.add_argument("--bands", default=None, help="explore-relative band prefix, e.g. llm-soft-loop/pod64/bands/sdw-c")
    w.add_argument("--list", nargs="*", default=None, help="explore-relative band folders in row order, instead of --bands")
    w.add_argument("--n", type=int, default=16)
    w.add_argument("--period", type=int, default=6)
    w.add_argument("--kmax", type=int, default=8)
    w.add_argument("--min-cos", type=float, default=0.5)
    w.add_argument("--min-phases", type=int, default=3, help="usable phases (|cos| >= --min-cos off the critical visit) "
                   "that must agree on W (majority vote; a phase holding the fold's tip can flip signs)")
    w.add_argument("--text-ids", type=int, nargs="*", default=None)
    w.add_argument("--compare", default=None, help="band prefix of an earlier scan on the same grid (reproducibility)")
    w.add_argument("--out", required=True)
    w.add_argument("--fig", default=None)
    w.add_argument("--zoom", type=float, nargs=4, default=None, metavar=("A0", "A1", "T0", "T1"), help="box for panel (b)")
    w.add_argument("--unconfirmed", nargs="*", default=[], help="window words the float64 check could not confirm (legend note)")
    r = sub.add_parser("returnmaps")
    r.add_argument("--bands", default=None)
    r.add_argument("--n", type=int, default=3)
    r.add_argument("--list", nargs="*", default=None, help="explore-relative band folders in row order, instead of --bands")
    r.add_argument("--period", type=int, default=6)
    r.add_argument("--chaos-level", type=float, default=1e-3)
    r.add_argument("--margin", type=float, default=0.02, help="visits within this fraction of the spread of the turning "
                   "point are left out of the coding comparison")
    r.add_argument("--kneading-len", type=int, default=16)
    r.add_argument("--min-cos", type=float, default=0.5, help="visits with |cos| below this are near-critical (left out)")
    r.add_argument("--min-separability", type=float, default=0.99, help="unimodal: one threshold separates L from R "
                   "visits at this fraction in every usable phase")
    r.add_argument("--max-errors", type=float, default=0.05, help="coded unimodally: L/R misplacements at most this "
                   "fraction of the minority symbol over the usable phases (the figure's green)")
    r.add_argument("--max-weak", type=float, default=0.1, help="a phase is usable when at most this fraction of its visits "
                   "is near-critical")
    r.add_argument("--max-bend", type=float, default=45.0, help="a phase is usable when its band turns less than this "
                   "(degrees, 95th percentile) from its principal axis")
    r.add_argument("--text-ids", type=int, nargs="*", default=None)
    r.add_argument("--out", required=True)
    r.add_argument("--fig", default=None)
    vs = sub.add_parser("verify-summary")
    vs.add_argument("--verify", nargs="+", required=True, help="explore-relative folders with verify.json")
    vs.add_argument("--words", required=True, help="explore-relative folder with the word map's words.npz")
    vs.add_argument("--min-cos", type=float, default=0.5)
    vs.add_argument("--out", required=True)
    g = sub.add_parser("merge")
    g.add_argument("--partial", required=True, help="explore-relative band folder with scan-partial.npz")
    g.add_argument("--rerun", required=True, help="explore-relative band folder of the rerun of its missing rows")
    g.add_argument("--out", required=True)
    args = ap.parse_args()
    {"mss": cmd_mss, "words": cmd_words, "returnmaps": cmd_returnmaps, "merge": cmd_merge,
     "verify-summary": cmd_verify_summary}[args.cmd](args)


if __name__ == "__main__":
    main()
