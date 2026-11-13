"""Walkthrough figures and numbers for the blog's step-by-step readings of two article figures (run FW).

branch:  the S-shaped branch of the kite figure, drawn on three vertical axes that share alpha: the article's first
         principal coordinate and two probabilities the model actually assigns. The numbers change; the folds do not.
         Writes fig-branch-three-readings.png and branch.json.
passage: how far one lap moves the state at each point of the channel, measured on the network against the normal
         form. Writes fig-passage-steps.png and passage.json.
free:    a free trajectory beyond the fold with the words it writes (needs run FW1). Writes fig-free-orbit-words.png
         and free.json.

Usage: python tools/figure_walkthroughs.py branch|passage|free
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from fold_panel_data import load_panel, draw_passages

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
LOOP = DATA / "explore" / "llm-soft-loop"
OUT = ROOT / "docs" / "figure-walkthroughs"
BLUE, GRAY, GREEN = "tab:blue", "0.45", "tab:green"


def style(plt):
    plt.rcParams.update({"font.size": 8, "axes.spines.top": True, "axes.spines.right": True})


def branch():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style(plt)
    src = LOOP / "pod64/kite/kite-la-branch"
    br = json.loads((src / "branch.json").read_text())
    pts = br["points"]
    w = np.load(LOOP / "branch-words/kite-la-branch/words.npz")
    words = json.loads((LOOP / "branch-words/kite-la-branch/words.json").read_text())
    vocab = words["vocab"]
    alpha, y = w["alpha"], w["y"]
    ids, p = w["ids"], w["p"]
    rho = np.array([q["spectral_radius"] for q in pts])
    res = np.array([q["residual"] for q in pts])
    turns = [i for i, q in enumerate(pts) if q.get("turned_since_previous")]
    assert len(turns) == 2 and len(pts) == len(alpha)
    text = {" ".join(vocab[str(int(j))].replace("\n", "\\n") for j in ids[i, :, 0]) for i in range(len(alpha))}
    assert len(text) == 1, text
    names = [vocab[str(int(j))] for j in w["tokens"]]
    col = {t: names.index(t) for t in ("\n", "\n\n", " is")}
    prob = w["token_p"]
    is_p, nl_p, bl_p = (100 * prob[:, col[t]] for t in (" is", "\n", "\n\n"))
    stable0 = np.nonzero(rho[: turns[0]] <= 1 + 1e-6)[0]
    lo, hi = float(alpha[stable0[0]]), float(alpha[turns[0]])
    fig, axes = plt.subplots(3, 1, figsize=(4.9, 6.3), sharex=True, gridspec_kw={"hspace": 0.2})
    panels = ((y, "first principal coordinate\n(the article's axis)", "(a) one saved state, projected on a direction"),
              (is_p, 'probability of " is" (%)', '(b) the same orbits: how likely " is" is at that step'),
              (nl_p, "probability of a single line break (%)", "(c) the same orbits: how likely a single line break is at that step"))
    good = (rho < 1)[:-1] & (rho < 1)[1:]
    for ax, (v, lab, title) in zip(axes, panels):
        k = 0
        while k < len(pts) - 1:
            j = k
            while j < len(pts) - 1 and good[j] == good[k]:
                j += 1
            ax.plot(alpha[k:j + 1], v[k:j + 1], "-" if good[k] else "--", color=BLUE if good[k] else GRAY,
                    lw=1.7 if good[k] else 1.0)
            k = j
        for i in turns:
            ax.plot(alpha[i], v[i], "o", ms=5, mfc="w", mec="k", zorder=5)
            ax.axvline(alpha[i], color="0.55", lw=0.6, ls=":", zorder=0)
        ax.axvspan(lo, hi, color=GREEN, alpha=0.15, lw=0)
        ax.set_ylabel(lab, fontsize=7.5)
        ax.set_title(title, fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel(r"steering strength $\alpha$ (at $\log_{10}T=%.7f$)" % float(br["args"]["log_temperature"]))
    axes[-1].ticklabel_format(axis="x", useOffset=False, style="plain")
    axes[0].plot([], [], "-", color=BLUE, lw=1.7, label="attracting")
    axes[0].plot([], [], "--", color=GRAY, lw=1.0, label="not attracting")
    axes[0].plot([], [], "o", ms=5, mfc="w", mec="k", label="fold")
    axes[0].fill_between([], [], color=GREEN, alpha=0.15, label="two attracting orbits")
    axes[0].legend(fontsize=6.5, loc="lower right", frameon=False, ncol=1)
    fig.savefig(OUT / "fig-branch-three-readings.png", dpi=300, bbox_inches="tight", pad_inches=0.03)

    S = np.load(src / "states.npy").reshape(len(pts), -1)
    point = lambda i: {"index": int(i), "alpha": float(alpha[i]), "y": float(y[i]), "p_is_percent": float(is_p[i]),  # noqa: E731
                       "p_linebreak_percent": float(nl_p[i]), "p_blankline_percent": float(bl_p[i]),
                       "spectral_radius": float(rho[i])}
    target = 0.07295
    sheets = (("lower, attracting", 0, turns[0]), ("middle, saddle", turns[0], turns[1]),
              ("upper, attracting", turns[1], len(alpha) - 1))
    three = []
    for name, a, b in sheets:
        idx = np.arange(a, b + 1)
        k = int(idx[np.argmin(np.abs(alpha[idx] - target))])
        steps = []
        for s in range(6):
            steps.append([{"token": vocab[str(int(j))], "percent": 100 * float(q)} for j, q in zip(ids[k, s, :3], p[k, s, :3])])
        three.append({"sheet": name, **point(k), "steps": steps})
    share = w["token_share"]
    worked = 97
    parts = {t: float(share[worked, col[t]]) for t in col}
    pbar = prob.mean(0)
    crossing = [c for c in br["crossings"]]
    flips = [c for c in crossing if "flip" in c["type"]]
    i_up = next(i for i in range(turns[1], len(rho) - 1) if rho[i] < 1 <= rho[i + 1])
    rec = {"source": {"branch_sha256": words["branch_sha256"], "states_sha256": words["states_sha256"]},
           "points": len(pts), "text": sorted(text)[0], "largest_residual": float(res.max()),
           "largest_return_gap_cpu": words["largest_return_gap"], "largest_average_gap_cpu": words["largest_average_gap"],
           "pc_variance_share_percent": [100 * x for x in words["variance_share_pc"][:3]],
           "y_range": [float(y.min()), float(y.max())], "y_span": float(y.max() - y.min()),
           "state_norm_mean": float(np.linalg.norm(S, axis=1).mean()),
           "y_span_percent_of_state_norm": float(100 * (y.max() - y.min()) / np.linalg.norm(S, axis=1).mean()),
           "y_strictly_increasing_along_branch": bool((np.diff(y) > 0).all()),
           "start": point(0), "fold_1": point(turns[0]), "fold_2": point(turns[1]), "end": point(len(pts) - 1),
           "upper_flip_alpha_between": [float(alpha[i_up]), float(alpha[i_up + 1])], "upper_flip_y": float(y[i_up]),
           "upper_flip_p_is_percent": float(is_p[i_up]),
           "upper_flip_multiplier": flips[-1]["crossing_multiplier"][0],
           "fold_multipliers": [c["crossing_multiplier"][0] for c in crossing if "fold" in c["type"]],
           "coexistence_alpha": [lo, hi],
           "p_is_range_percent": [float(is_p.min()), float(is_p.max())],
           "p_linebreak_range_percent": [float(nl_p.min()), float(nl_p.max())],
           "p_blankline_range_percent": [float(bl_p.min()), float(bl_p.max())],
           "corr_y_p_is": float(np.corrcoef(y, is_p)[0, 1]), "corr_y_p_linebreak": float(np.corrcoef(y, nl_p)[0, 1]),
           "token_loads": {t: float(w["token_load"][col[t]]) for t in col},
           "branch_mean_percent": {t: float(100 * pbar[col[t]]) for t in col},
           "worked_point": {**point(worked), "shares": parts, "rest": float(w["y"][worked] - sum(parts.values()))},
           "largest_rest_after_three_tokens": float(np.abs(y - sum(share[:, col[t]] for t in col)).max()),
           "middle_sheet_largest_multiplier": float(rho[turns[0]:turns[1]].max()),
           "three_orbits": three}
    c0 = float(w["mean"] @ w["axis"])                           # measure the weights from the average state: same y
    load_c = w["token_load"] - c0
    share_c = (prob - pbar) * load_c
    main_tokens = [col[t] for t in col]
    worked_c = {t: float(share_c[worked, col[t]]) for t in col}
    total = nl_p + bl_p
    assert (np.diff(is_p) > 0).all() and (np.diff(nl_p) < 0).all()
    k0, k2 = three[0]["index"], three[2]["index"]
    gaps = []
    for st in range(6):
        for j0, q0 in zip(ids[k0, st], p[k0, st]):
            hit = np.nonzero(ids[k2, st] == j0)[0]
            if len(hit):
                gaps.append(100 * abs(float(q0) - float(p[k2, st, hit[0]])))
    peak = int(np.argmax(bl_p))
    rec.update({"weights_from_average": {t: float(load_c[col[t]]) for t in col}, "origin_dot_axis": c0,
                "worked_terms_from_average": worked_c, "worked_rest_from_average": float(y[worked] - sum(worked_c.values())),
                "largest_rest_from_average": float(np.abs(y - share_c[:, main_tokens].sum(1)).max()),
                "line_break_total_percent": {"start": float(total[0]), "first_fold": float(total[turns[0]]), "end": float(total[-1])},
                "blank_line_peak": {"percent": float(bl_p[peak]), "index": peak, "y": float(y[peak]),
                                    "share_of_axis_percent": float(100 * (y[peak] - y.min()) / (y.max() - y.min()))},
                "blank_line_start_percent": float(bl_p[0]), "blank_line_end_percent": float(bl_p[-1]),
                "attracting_orbits_largest_gap_points": float(max(gaps)), "dashed_points": int((rho >= 1).sum())})
    (OUT / "branch.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "three_orbits"}, indent=1, ensure_ascii=False))


def passage():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style(plt)
    pas = json.loads((LOOP / "passage-p2-t-0.115/passage.json").read_text())
    V = np.load(LOOP / "passage-p2-t-0.115/u.npy")
    U, nf = float(pas["U"]), pas["normal_form"]
    a, b = abs(float(nf["a"])), abs(float(nf["b"]))
    colors = {3e-6: "#1b9e77", 1e-5: "#d95f02", 3e-5: "#7570b3", 1e-4: "#e7298a"}
    markers = {3e-6: "o", 1e-5: "s", 3e-5: "^", 1e-4: "D"}
    fig, ax = plt.subplots(figsize=(4.9, 3.4))
    grid = np.linspace(-1.1, 1.1, 400)
    rows = {}
    for i, row in enumerate(pas["rows"]):
        e = float(row["eps"])
        if row["entrance"] != "full" or e not in colors:
            continue
        v = np.concatenate([[row["u_in"]], V[:, i]])
        n = int(np.argmax(v > U))
        v = v[:n + 1]
        dv = np.diff(v)
        pred = a * e + b * v[:-1] ** 2
        err = (dv - pred) / pred
        ax.semilogy(grid, (a * e + b * (grid * U) ** 2) / U, "-", color=colors[e], lw=0.9)
        ax.semilogy(v[:-1] / U, dv / U, markers[e], color=colors[e], ms=2.6, mew=0, ls="none",
                    label=rf"$\varepsilon={e:.0e}$: {n} laps".replace("e-0", r"\times10^{-").replace("$:", "}$:"))
        k = np.sqrt(a * b * e)
        half = int(np.argmax(v > 0))
        vv, m = [float(row["u_in"])], 0
        while vv[-1] <= 0:
            vv.append(vv[-1] + a * e + b * vv[-1] ** 2)
        rows[f"{e:.0e}"] = {"laps": n, "laps_to_center": half, "normal_form_laps_to_center": len(vv) - 1,
                            "center_step": float(a * e), "center_step_over_U": float(a * e / U),
                            "edge_step_over_U": float((a * e + b * U ** 2) / U),
                            "edge_over_center": float((a * e + b * U ** 2) / (a * e)),
                            "laps_in_middle_half": int((np.abs(v) < U / 2).sum()),
                            "laps_in_middle_quarter": int((np.abs(v) < U / 4).sum()),
                            "kappa": float(k), "arctan_laps": float(2 * np.arctan(b * U / k) / k), "limit_laps": float(np.pi / k),
                            "largest_step_error_percent": float(100 * np.abs(err).max()),
                            "step_error_first_half_mean_percent": float(100 * err[v[:-1] < 0].mean()),
                            "step_error_second_half_mean_percent": float(100 * err[v[:-1] >= 0].mean()),
                            "signed_error_first_lap_percent": float(100 * err[0]), "signed_error_last_lap_percent": float(100 * err[-1]),
                            "positions_over_U": {str(m): float(v[m] / U) for m in (0, 1, 2, 10, 20, 36, 50, 60, 69, 70, 71) if m < len(v)},
                            "steps_over_U": {str(m): float(dv[m] / U) for m in (0, 10, 20, 36, 50, 60, 70) if m < len(dv)}}
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(2.5e-3, 0.8)
    ax.set_xlabel(r"where the lap starts, $v_n/U$ (entrance at $-1$, exit at $+1$)")
    ax.set_ylabel(r"how far one lap moves it, $(v_{n+1}-v_n)/U$")
    ax.set_title("Each lap's move, measured (markers) and from the formula (lines)", fontsize=8, loc="left")
    ax.legend(fontsize=6.5, loc="upper center", frameon=False, ncol=2, markerscale=1.8)
    fig.savefig(OUT / "fig-passage-steps.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    # after the fact: the per-lap residual fitted to the two lowest dropped terms that change sign across the channel
    Xfit, Yfit, Mfit, tag = [], [], [], []
    for i, row in enumerate(pas["rows"]):
        e = float(row["eps"])
        if row["entrance"] != "full" or e > 1e-4:
            continue
        v = np.concatenate([[row["u_in"]], V[:, i]])
        v = v[:int(np.argmax(v > U)) + 1]
        vv, dv = v[:-1], np.diff(v)
        Xfit.append(np.column_stack([vv ** 3, e * vv])); Yfit.append(dv - (a * e + b * vv ** 2)); tag += [e] * len(vv)
    Xall, Yall = np.concatenate(Xfit), np.concatenate(Yfit)
    (c3, c1), *_ = np.linalg.lstsq(Xall, Yall, rcond=None)
    fit = {"c3": float(c3), "c1": float(c1), "r2": float(1 - ((Yall - Xall @ [c3, c1]) ** 2).sum() / (Yall ** 2).sum()),
           "distances": sorted(set(tag))}
    for e, name in ((3e-6, "3e-06"), (1e-4, "1e-04")):
        vn, mv = 0.9 * U, a * e + b * (0.9 * U) ** 2
        fit[f"cubic_percent_of_move_at_0.9U_{name}"] = float(100 * c3 * vn ** 3 / mv)
        fit[f"eps_u_percent_of_move_at_0.9U_{name}"] = float(100 * c1 * e * vn / mv)
    totals = {f'{float(r["eps"]):.0e}': {"measured": r["measured_G"], "predicted": r["predicted_discrete_G"]}
              for r in pas["rows"] if r["entrance"] == "full"}
    rec = {"source_passage_sha256": __import__("hashlib").sha256((LOOP / "passage-p2-t-0.115/passage.json").read_bytes()).hexdigest(),
           "a": a, "b": b, "U": U, "ab": a * b, "alpha_c": 0.06279827217667086, "rows": rows, "residual_fit": fit, "totals": totals}
    (OUT / "passage.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=0)[:2600])


def runs_of(mask):
    out, n, start = [], 0, None
    for i, m in enumerate(mask):
        if m:
            if n == 0:
                start = i
            n += 1
        elif n:
            out.append((start, i - 1))
            n = 0
    if n:
        out.append((start, len(mask) - 1))
    return out


def free():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style(plt)
    run = LOOP / "passage-natural-words-p2-t-0.115"
    old = np.load(LOOP / "passage-natural-p2-t-0.115/u.npy")
    pa = json.loads((run / "passage.json").read_text())
    U, pick = float(pa["U"]), 7
    assert pa["rows"][pick]["entrance"] == "natural-3" and float(pa["rows"][pick]["eps"]) == 1e-5
    z = np.load(run / "top.npz")
    vocab = json.loads((run / "vocab.json").read_text())
    new = np.load(run / "u.npy")
    gap = float(np.abs(new - old[:len(new)]).max())
    chk = np.load(run / "phase-check.npz")
    v, v1 = chk["u_same_phase"], chk["u_one_step_later"]
    assert np.array_equal(v, new[:, pick])
    ids, p = z["ids"][:, pick], z["p"][:, pick]
    word = lambda j: vocab[str(int(j))]  # noqa: E731
    top = [word(j) for j in ids[:, 0]]

    def percent(tok):
        """Each iterate's probability of tok, at the sampled pair's step that writes " it" (or, if neither does, the
        step with the larger probability of " it"); 0 when tok is outside the recorded top 8."""
        out = np.zeros(len(v))
        for n in range(len(v)):
            steps = (2 * n, 2 * n + 1)
            it = [float(p[t][[word(j) == " it" for j in ids[t]]].sum()) for t in steps]
            t = steps[int(np.argmax(it))]
            out[n] = 100 * float(p[t][[word(j) == tok for j in ids[t]]].sum())
        return out

    p_it, p_a, p_there = percent(" it"), percent(" a"), percent(" there")
    N = 345
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 6.6), sharex=True, gridspec_kw={"hspace": 0.2, "height_ratios": [1.0, 1.0, 0.9]})
    panel = load_panel(LOOP)
    ax = axes[0]
    draw_passages(ax, panel, horizon=N, fontsize=8)
    ax.set_xlabel("")
    ax.set_title("(a) Signed position: slow passages cross zero smoothly; blue follows the shifted phase", fontsize=8, loc="left")
    ax = axes[1]
    for phase, color, label in [(0, 'black', 'every two updates'), (1, '#0072b2', 'one update later')]:
        ax.semilogy(np.arange(N), panel['transverse'][phase, :N], color=color, lw=.8, label=label)
    ax.axhline(.05, color='#a6430a', ls='--', lw=.8, label='transverse cutoff 0.05')
    ax.set_ylabel(r"Transverse remainder $d_\perp$")
    ax.set_title("(b) A second check: the remaining state components must also be close to the channel", fontsize=8, loc="left")
    ax.legend(fontsize=6.5, loc='lower right', frameon=True, ncol=3)
    ax = axes[2]
    ax.plot(np.arange(N), p_it[:N], "-", color="black", lw=0.9, label='" it"')
    ax.plot(np.arange(N), p_there[:N], "-", color="0.55", lw=0.9, label='" there"')
    ax.plot(np.arange(N), p_a[:N], "-", color="#d95f02", lw=1.1, label='" a"')
    ax.set_ylim(0, 60)
    ax.set_ylabel("probability (%)")
    ax.set_xlabel(r"laps $n$ (two steps of the loop each)")
    ax.set_title('(c) what the crawl is: at the step that writes " it", three tokens compete', fontsize=8, loc="left")
    ax.legend(fontsize=6.5, loc="upper right", frameon=True, framealpha=0.9, edgecolor="none", ncol=3)
    axes[-1].set_xlim(0, N)
    fig.savefig(OUT / "fig-free-orbit-words.png", dpi=300, bbox_inches="tight", pad_inches=0.03)

    same = runs_of(np.abs(v) < U)
    other = runs_of(np.abs(v1) < U)
    allrows = {}
    for i in (4, 5, 6, 7):
        w = [vocab[str(int(j))] for j in z["ids"][:, i, 0]]
        allrows[pa["rows"][i]["entrance"]] = {"first_words": "".join(w[:6]), "last_words": "".join(w[-6:]),
                                              "fraction_in_band_first_600": float((np.abs(new[:, i]) < U).mean())}
    full = old[:, pick]
    flats = [r for r in runs_of((np.abs(full) > 0.3) & (np.abs(full) < 0.6) & (np.abs(np.diff(full, prepend=full[0])) < 0.01))
             if r[1] - r[0] >= 4]
    laps = {}
    for n in (0, 47, 54, 56, 57, 58, 59, 60, 61, 62):
        laps[str(n)] = {"v": float(v[n]), "words": [top[2 * n], top[2 * n + 1]],
                        "first_step": [{"token": word(j), "percent": 100 * float(q)} for j, q in zip(ids[2 * n, :3], p[2 * n, :3])],
                        "second_step": [{"token": word(j), "percent": 100 * float(q)} for j, q in zip(ids[2 * n + 1, :3], p[2 * n + 1, :3])]}
    nf = pa["normal_form"]
    A, B = abs(float(nf["a"])), abs(float(nf["b"]))
    kappa = float(np.sqrt(A * B * 1e-5))
    bursts = []
    for k in range(len(same) - 1):
        a, b = same[k][1] + 1, same[k + 1][0]
        text = [top[t] for t in range(2 * a, 2 * b)]
        while text and text[0] in (" it", " is"):
            text = text[1:]
        while text and text[-1] in (" it", " is"):
            text = text[:-1]
        bursts.append({"from_lap": a, "to_lap": b - 1, "text": "".join(text)[:80], "next_entry_v": float(v[b]),
                       "next_entry_over_U": float(v[b] / U), "next_crawl_laps": same[k + 1][1] - same[k + 1][0] + 1,
                       "next_crawl_predicted": float((np.arctan(B * U / kappa) - np.arctan(B * v[b] / kappa)) / kappa)})
    rec = {"source_passage_sha256": __import__("hashlib").sha256((run / "passage.json").read_bytes()).hexdigest(),
           "rerun_largest_gap_from_published": gap, "rows_compared": int(new.shape[1]), "laps_compared": int(new.shape[0]),
           "U": U, "eps": 1e-5, "transient_laps": 100,
           "same_phase_runs": same, "other_phase_runs": other,
           "same_phase_lengths": [b - a + 1 for a, b in same], "other_phase_lengths": [b - a + 1 for a, b in other],
           "flat_first": {"laps": [181, 233], "u_range": [float(v[181:234].min()), float(v[181:234].max())],
                          "one_step_later_range": [float(v1[181:234].min()), float(v1[181:234].max())]},
           "laminar_one_step_later_range": [float(v1[241:311].min()), float(v1[241:311].max())],
           "fraction_first_600_same_phase": float((np.abs(v) < U).mean()),
           "fraction_first_600_either_phase": float(((np.abs(v) < U) | (np.abs(v1) < U)).mean()),
           "fraction_1500_same_phase": float((np.abs(full) < U).mean()),
           "flat_stretches_1500": [[a, b, b - a + 1] for a, b in flats],
           "fraction_1500_flat": float(sum(b - a + 1 for a, b in flats) / len(full)),
           "laps": laps, "bursts": bursts, "other_trajectories_at_this_eps": allrows,
           "p_a_crawl_range": [float(p_a[0:48].min()), float(p_a[0:48].max())],
           "p_it_crawl_range": [float(p_it[0:48].min()), float(p_it[0:48].max())],
           "peak_abs_v_first_345": float(np.abs(v[:N]).max()),
           "state_norm_crawl": float(np.linalg.norm(z["states"][:48, pick], axis=1).mean()), "kappa": kappa,
           "worked_ratio": 0.1 / kappa}
    rec["channel_percent_of_state_norm"] = 100 * 2 * U / rec["state_norm_crawl"]
    t_is = [i for i, x in enumerate(top) if x == " is"]
    gaps = [(t_is[i], t_is[i + 1] - t_is[i] - 1) for i in range(len(t_is) - 1) if t_is[i + 1] - t_is[i] != 2]
    rec["beat"] = {"gaps_other_than_one": [[a0, g] for a0, g in gaps], "beat_changes": sum(g % 2 == 0 for _, g in gaps)}
    rec["v_over_U_after_flat"] = {str(n): float(v[n] / U) for n in range(236, 244)}
    a0, b0 = other[0]
    pred = lambda w0: float((np.arctan(B * U / kappa) - np.arctan(B * w0 / kappa)) / kappa)  # noqa: E731
    rec["hidden_crawl"] = {"first_lap": a0, "last_lap": b0, "laps": b0 - a0 + 1, "entry_over_U": float(v1[a0] / U),
                           "predicted": pred(v1[a0]), "exit_lap_reading_over_U": float(v1[233] / U)}
    crawls, misses = {}, []
    for i in (5, 6, 7):
        u = old[:, i]
        row = []
        for a1, b1 in runs_of(np.abs(u) < U):
            if a1 == 0 or b1 == len(u) - 1:
                continue
            row.append({"first": a1, "last": b1, "laps": b1 - a1 + 1, "entry_over_U": float(u[a1] / U),
                        "next_over_U": float(u[a1 + 1] / U), "predicted": pred(u[a1])})
        crawls[pa["rows"][i]["entrance"]] = {"complete": len(row), "within_one_lap": sum(abs(r["laps"] - r["predicted"]) <= 1 for r in row)}
        misses += [dict(r, row=pa["rows"][i]["entrance"]) for r in row if abs(r["laps"] - r["predicted"]) > 1]
    rec["complete_crawls"] = crawls
    rec["complete_crawls_total"] = sum(c["complete"] for c in crawls.values())
    rec["complete_crawls_within_one_lap"] = sum(c["within_one_lap"] for c in crawls.values())
    rec["crawl_misses"] = misses
    same_ending = []
    for i in (5, 6, 7):
        u = new[:, i]
        tw = [vocab[str(int(j))] for j in z["ids"][:, i, 0]]
        rr = runs_of(np.abs(u) < U)
        for k in range(len(rr) - 1):
            a1, b1 = rr[k][1] + 1, rr[k + 1][0]
            text = tw[2 * a1:2 * b1]
            while text and text[0] in (" it", " is"):
                text = text[1:]
            while text and text[-1] in (" it", " is"):
                text = text[:-1]
            if "".join(text) == " a. This":
                same_ending.append({"row": pa["rows"][i]["entrance"], "entry_over_U": float(u[b1] / U)})
    rec["endings_a_stop_this"] = same_ending
    rec["one_step_reading_on_crawls"] = [float(min(v1[a1:b1 + 1].min() for a1, b1 in same)), float(max(v1[a1:b1 + 1].max() for a1, b1 in same))]
    rec["flat_reading_range_1500"] = [float(min(np.abs(full[a1:b1 + 1]).min() for a1, b1 in flats)), float(max(np.abs(full[a1:b1 + 1]).max() for a1, b1 in flats))]
    rec["flat_count_1500"] = len(flats)
    rec["corr_v_p_a_laps_241_310"] = float(np.corrcoef(v[241:311], p_a[241:311])[0, 1])
    rec["corr_v_p_a_laps_62_101"] = float(np.corrcoef(v[62:102], p_a[62:102])[0, 1])
    rec["indeed_first_crawl_percent"] = [100 * float(p[2 * n + 1][[word(j) == " indeed" for j in ids[2 * n + 1]]].sum()) for n in (0, 47)]
    rec["a_peaks_after_exits"] = [float(p_a[b1 + 1:b1 + 23].max()) for a1, b1 in same if b1 < N]
    vec = np.load(LOOP / "normal-form-p2-t-0.115/vectors.npz")
    states59, xc = z["states"][59, pick], vec["x_c"]
    rec["ruler"] = {"length": float(np.linalg.norm(vec["l"])), "cosine_with_direction": float(abs(vec["l"] @ vec["r"]) / (np.linalg.norm(vec["l"]) * np.linalg.norm(vec["r"]))),
                    "lap_59_reading": float(abs(v[59])), "lap_59_true_distance": float(np.linalg.norm(states59 - xc)),
                    "flat_true_distance": float(np.linalg.norm(z["states"][200, pick] - xc))}
    for key in ("fraction_first_600_same_phase", "fraction_first_600_either_phase", "fraction_1500_same_phase", "fraction_1500_flat"):
        rec[key.replace("_phase", "") + "_percent"] = 100 * rec[key]
    (OUT / "free.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: rec[k] for k in ("rerun_largest_gap_from_published", "same_phase_lengths", "other_phase_lengths", "flat_first",
                                          "fraction_first_600_same_phase", "fraction_first_600_either_phase", "fraction_1500_same_phase",
                                          "flat_stretches_1500", "fraction_1500_flat", "other_trajectories_at_this_eps", "p_a_crawl_range",
                                          "p_it_crawl_range", "peak_abs_v_first_345")}, indent=0, ensure_ascii=False))
    for x in bursts:
        print(x)


if __name__ == "__main__":
    {"branch": branch, "passage": passage, "free": free}[sys.argv[1]]()
