"""Checks, receipt and figure of a batch_fate_long.py run (run BFL, exploratory, 11 to 12 Nov 2026).

Reads the arrays of the long census (long.npz, long.json, run.log, launch.txt) and of run BF (fate.npz and its
receipt), re-derives every number of the census from the arrays, and refuses to write a receipt when a gate fails:

- the start states are those of run BF, bit for bit;
- long.json agrees with the arrays (capture steps, periods, counts), the run reached its horizon, and the run's own
  list of captured rows that left their cycle is empty;
- the tool modules of the run are identical at the launch commit and at the commit recorded when the run ended;
- every captured row stays on its cycle: each state recorded after the capture, and the final state, lies within the
  run's --stay-tol of a saved cycle state (recomputed here);
- every saved cycle is primitive (it is not a shorter cycle traversed several times).

Reported, without a gate: which cycle each captured row is on (compared, up to a shift, with the two cycles of run
BF), how close the uncaptured rows come to those cycles and to their own earlier recorded states, the census by
horizon and by fifth of the run, the capture rate with an exact Poisson interval, two tests of a constant rate, the
comparison with the numbers the run card fixed in advance, and the reading under the run card's criteria.

"Captured" means the recurrence criterion of tools/batch_fate.py with sampled persistence. No cycle here was
refined by Newton's method or tested for stability.

Usage: python tools/batch_fate_long_receipt.py [--run llm-soft-loop/batch-fate/bfl1] [--bf llm-soft-loop/batch-fate/bf1]
           [--out docs/batch-fate/bfl1-receipt.json] [--figure paper/llm/figures/fig-batch-fate-long.png]
"""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
MODULES = ["tools/batch_fate_long.py", "tools/batch_fate.py", "tools/explore_llm_soft_loop.py",
           "tools/explore_llm_attractors.py", "tools/explore_llm_phase_map.py"]
SAME_ORBIT = 1e-6        # largest relative distance (after the best shift) at which two saved cycles count as one orbit
INK, MUTED, HAIR, BLUE, LIGHT = "#0b0b0b", "#52514e", "#d9d8d3", "#1c5cab", "#b7d3f6"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise SystemExit(f"gate failed: {message}")


def cyclic_distance(A, B):
    """Largest relative distance between the states of two cycles after the best cyclic shift (inf if the periods
    differ)."""
    if A.shape != B.shape:
        return math.inf
    scale = np.linalg.norm(B, axis=1).min()
    return min(float(np.linalg.norm(np.roll(A, s, axis=0) - B, axis=1).max() / scale) for s in range(len(A)))


def primitive_gap(C):
    """Smallest relative distance between two different states of a saved cycle (zero for a repeated shorter cycle)."""
    q = len(C)
    if q == 1:
        return math.inf
    return min(float(np.linalg.norm(C[i] - C[j]) / np.linalg.norm(C[i])) for i in range(q) for j in range(i))


def distance_to_cycle(x, C):
    """Relative distance from state x to the nearest state of cycle C."""
    return float(np.linalg.norm(C - x[None], axis=1).min() / np.linalg.norm(x))


def nearest_pair(X):
    """Indices i < j of the two closest rows of X and their relative distance, computed without a Gram matrix."""
    best = (0, 1, math.inf)
    for j in range(1, len(X)):
        d = np.linalg.norm(X[:j] - X[j][None], axis=1)
        i = int(d.argmin())
        if d[i] < best[2]:
            best = (i, j, float(d[i]))
    return best[0], best[1], best[2] / float(np.linalg.norm(X[best[1]]))


def median_pair(X):
    """Median relative distance between two rows of X."""
    d = np.concatenate([np.linalg.norm(X[:j] - X[j][None], axis=1) / np.linalg.norm(X[j]) for j in range(1, len(X))])
    return float(np.median(d))


def poisson_interval(events, level=0.95):
    """Exact (Garwood) interval for the mean of a Poisson count."""
    from scipy.stats import chi2
    a = 1.0 - level
    low = 0.0 if events == 0 else 0.5 * chi2.ppf(a / 2, 2 * events)
    return float(low), float(0.5 * chi2.ppf(1 - a / 2, 2 * events + 2))


def exposure(cap, horizon, start=0):
    """Steps that the rows spent uncaptured between `start` and `horizon` (cap < 0: never captured)."""
    end = np.where(cap >= 0, np.minimum(cap, horizon), horizon)
    return int(np.clip(end - start, 0, None).sum())


def rate_record(events, steps_at_risk):
    low, high = poisson_interval(events)
    return {"captures": int(events), "steps_at_risk": int(steps_at_risk), "per_step": events / steps_at_risk,
            "interval95_per_step": [low / steps_at_risk, high / steps_at_risk]}


def expected_captured(rate, horizon, rows):
    return rows * (1.0 - math.exp(-rate * horizon))


def constant_rate_tests(cap, horizon):
    """Two tests of a constant capture rate over [0, horizon].

    halves: given the total count, the captures of the first half are binomial with the first half's share of the
    steps at risk (exact two-sided test). uniform: for a constant rate r the values (1 - exp(-r t)) / (1 - exp(-r H))
    of the captured rows are uniform on [0, 1]; Kolmogorov-Smirnov with r fitted, which makes the p-value too large
    rather than too small."""
    from scipy.stats import binomtest, kstest
    caught = cap >= 0
    first = int((caught & (cap <= horizon // 2)).sum())
    total = int(caught.sum())
    e1, e2 = exposure(cap, horizon // 2), exposure(cap, horizon, horizon // 2)
    rate = total / (e1 + e2)
    u = (1 - np.exp(-rate * cap[caught])) / (1 - math.exp(-rate * horizon))
    ks = kstest(u, "uniform")
    return {"halves": {"captures": [first, total - first], "steps_at_risk": [e1, e2],
                       "per_step": [first / e1, (total - first) / e2],
                       "binomial_p": float(binomtest(first, total, e1 / (e1 + e2)).pvalue)},
            "uniform": {"ks_statistic": float(ks.statistic), "ks_p": float(ks.pvalue)}}


def densest_window(cap, horizon, width):
    """The window of `width` steps (starting at a capture) whose number of captures is least probable for a constant
    rate, given the steps at risk inside it: (Poisson tail probability, captures, expected, first step)."""
    from scipy.stats import poisson
    caught = cap >= 0
    rate = caught.sum() / exposure(cap, horizon)
    times = np.sort(cap[caught])
    best = (1.0, 0, 0.0, None)
    for a in times:
        count = int(((times >= a) & (times < a + width)).sum())
        expected = rate * exposure(cap, min(a + width, horizon), a)
        tail = float(poisson.sf(count - 1, expected))
        if tail < best[0]:
            best = (tail, count, float(expected), int(a))
    return best


def cluster_test(cap, horizon, width=5000, draws=2000, seed=0):
    """How unusual the least probable window of captures is for a constant rate: the share of simulated censuses
    (the same number of rows, exponential capture times at the fitted rate, cut at the horizon) whose least probable
    window is at least as improbable."""
    tail, count, expected, first = densest_window(cap, horizon, width)
    rate = (cap >= 0).sum() / exposure(cap, horizon)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(draws):
        t = np.ceil(rng.exponential(1 / rate, size=len(cap))).astype(np.int64)
        hits += densest_window(np.where(t <= horizon, t, -1), horizon, width)[0] <= tail
    return {"window_steps": width, "captures": count, "window_starts_at_step": first,
            "expected_at_constant_rate": expected, "poisson_tail_of_that_window": tail,
            "simulated_share_with_a_window_as_improbable": hits / draws, "draws": draws, "seed": seed}


def reading(cap, horizon):
    """The run card's criteria, fixed before the run."""
    total = int((cap >= 0).sum())
    second = int((cap > horizon // 2).sum())
    if total >= len(cap) // 2 and second > 0:
        return "long transient"
    if total < len(cap) // 4 and second == 0:
        return "attractor at this horizon"
    return "unresolved"


def module_blobs(commit):
    return {m: subprocess.check_output(["git", "rev-parse", f"{commit}:{m}"], cwd=ROOT, text=True).strip()
            for m in MODULES}


def analyze(run, bf):
    z, rec_json = np.load(run / "long.npz"), json.loads((run / "long.json").read_text())
    f, bf_receipt = np.load(bf / "fate.npz"), json.loads((ROOT / "docs/batch-fate/bf1-receipt.json").read_text())
    args, cap, per, rec, rec_steps = rec_json["args"], z["cap"], z["per"], z["rec"], z["rec_steps"]
    H, n = args["steps"], len(cap)
    caught = cap >= 0

    # gates: starts, agreement of the record with the arrays, completion
    starts_sha = hashlib.sha256(np.ascontiguousarray(z["starts"]).tobytes()).hexdigest()
    require(starts_sha == rec_json["starts_sha256"] == bf_receipt["starts_sha256"], "start states differ from run BF")
    require(np.array_equal(z["starts"], f["starts"]), "start states differ from fate.npz of run BF")
    require(sha256(bf / "fate.npz") == bf_receipt["fate_npz_sha256"], "fate.npz differs from its receipt")
    require(rec_json["steps_run"] == H and rec_steps[-1] == H, "the run did not reach its horizon")
    require(np.array_equal(rec_steps, args["record_every"] * np.arange(len(rec_steps))), "recorded steps")
    require(rec_json["left"] == [], f"captured rows left their cycle: {rec_json['left']}")
    require(rec_json["capture_steps"] == {str(r): int(cap[r]) for r in np.nonzero(caught)[0]}, "capture steps")
    require(rec_json["captured"] == int(caught.sum()) and rec_json["not_captured"] == int((~caught).sum()), "counts")
    require(np.all(per[~caught] == 0) and np.all(per[caught] >= 1) and np.all(cap[caught] <= H), "periods")
    require(np.all(np.isfinite(rec)) and np.all(np.isfinite(z["final"])), "non-finite state")

    # gate: the code of the run is the same at launch and at the end
    launch = re.search(r"at commit ([0-9a-f]{7,40})", (run / "launch.txt").read_text()).group(1)
    end = rec_json["source"]["commit"]
    require(not rec_json["source"]["tracked_files_modified"], "tracked files were modified when the run ended")
    blobs = module_blobs(launch)
    require(blobs == module_blobs(end), "a tool module changed between launch and the end of the run")

    # reference cycles of run BF (its receipt's orbit 0, period 4, and orbit 1, period 8)
    refs = {}
    for key in f.files:
        m = re.fullmatch(r"(b\d+)_cyc_state_(\d+)", key)
        if m and len(f[key]) not in refs:
            refs[len(f[key])] = (key, f[key])
    require(set(refs) == {4, 8}, "run BF should hold a period-4 and a period-8 cycle")
    tokens = json.loads(str(z["cyc_tok"][0]))

    rows, orbit_count, persistence, other = {}, {}, 0.0, []
    for r in np.nonzero(caught)[0]:
        C = z[f"cyc_state_{r}"]
        require(len(C) == per[r], f"row {r}: saved cycle length")
        gap = primitive_gap(C)
        require(gap > 1e3 * args["tol"], f"row {r}: the saved cycle repeats a shorter one (gap {gap:.1e})")
        after = [distance_to_cycle(rec[k, r], C) for k in np.nonzero(rec_steps > cap[r])[0]]
        after.append(distance_to_cycle(z["final"][r], C))
        require(max(after) < args["stay_tol"], f"row {r}: leaves its cycle (distance {max(after):.1e})")
        persistence = max(persistence, max(after))
        d = {q: cyclic_distance(C, ref) for q, (_k, ref) in refs.items()}
        q = min(d, key=d.get)
        name = f"period {q} of run BF" if d[q] < SAME_ORBIT else "other"
        if name == "other":
            other.append(int(r))
        orbit_count[name] = orbit_count.get(name, 0) + 1
        rows[str(int(r))] = {"capture_step": int(cap[r]), "period": int(per[r]), "orbit": name,
                             "distance_to_reference": d[q], "smallest_gap_within_cycle": gap,
                             "largest_distance_after_capture": max(after), "tokens": tokens[str(int(r))]}

    # uncaptured rows: nearest approach of a recorded state to the two cycles and to an earlier recorded state
    free = {}
    for r in np.nonzero(~caught)[0]:
        X = rec[1:, r]                                    # without the start state
        i, j, d = nearest_pair(X)
        free[str(int(r))] = {
            "nearest_recorded_state_to_period_4": min(distance_to_cycle(x, refs[4][1]) for x in X),
            "nearest_recorded_state_to_period_8": min(distance_to_cycle(x, refs[8][1]) for x in X),
            "final_state_to_period_4": distance_to_cycle(z["final"][r], refs[4][1]),
            "nearest_pair_of_recorded_states": {"steps": [int(rec_steps[1 + i]), int(rec_steps[1 + j])],
                                                "distance": d},
            "median_distance_between_recorded_states": median_pair(X)}

    # census and rates
    horizons = list(range(H // 10, H + 1, H // 10))
    fifths = []
    for a in range(0, H, H // 5):
        b = a + H // 5
        fifths.append({"steps": [a, b], **rate_record(int(((cap > a) & (cap <= b)).sum()), exposure(cap, b, a))})
    overall = rate_record(int(caught.sum()), exposure(cap, H))
    bf_events = bf_exposure = 0
    bf_caps = {}
    for label in bf_receipt["labels"]:
        c = f[f"{label}_cap"]
        bf_events += int((c >= 0).sum())
        bf_exposure += exposure(c, bf_receipt["args"]["steps"])
        bf_caps[label] = {str(int(r)): int(c[r]) for r in np.nonzero(c >= 0)[0]}
    earlier = rate_record(bf_events, bf_exposure)
    predicted = {str(h): {"at_rate": expected_captured(earlier["per_step"], h, n),
                          "interval": [expected_captured(x, h, n) for x in earlier["interval95_per_step"]],
                          "observed": int((caught & (cap <= h)).sum())} for h in (20000, 50000, 100000) if h <= H}
    lifetime = {"mean_steps": 1 / overall["per_step"],
                "interval95_steps": [1 / x for x in reversed(overall["interval95_per_step"])],
                "half_of_the_starts_captured_by_step": int(np.sort(np.where(caught, cap, H + 1))[n // 2 - 1])
                if caught.sum() >= n // 2 else None}
    both = sorted(set(map(int, np.nonzero(caught)[0]))
                  & {int(r) for d in bf_caps.values() for r in d})
    return {
        "run": str(run.relative_to(DATA / "explore")), "args": args,
        "labels": "captured = recurrence candidate (relative distance below tol to a state up to max_period steps "
                  "back) with sampled persistence; no Newton refinement, no stability test",
        "source": {"launch_commit": launch, "commit_recorded_at_end": end, "module_blobs_identical": True,
                   "module_blobs": blobs},
        "files": {name: {"sha256": sha256(run / name), "bytes": (run / name).stat().st_size}
                  for name in ("long.npz", "long.json", "run.log", "launch.txt")},
        "starts_sha256": starts_sha, "starts_equal_run_bf": True,
        "steps": H, "starts": n, "captured": int(caught.sum()), "not_captured": int((~caught).sum()),
        "captured_in_second_half": int((cap > H // 2).sum()),
        "first_and_last_capture_step": [int(cap[caught].min()), int(cap[caught].max())],
        "periods": {str(q): int((per == q).sum()) for q in sorted(set(per[caught].tolist()))},
        "orbits": orbit_count, "rows_on_another_orbit": other,
        "reference_cycles_of_run_bf": {str(q): k for q, (k, _a) in refs.items()},
        "same_orbit_tolerance": SAME_ORBIT,
        "largest_distance_from_cycle_after_capture": persistence,
        "captured_by": {str(h): int((caught & (cap <= h)).sum()) for h in [1000, 2000, 5000] + horizons},
        "fifths": fifths, "rate": overall, "lifetime": lifetime,
        "rate_of_run_bf": earlier, "predicted_from_run_bf": predicted,
        "constant_rate_tests": constant_rate_tests(cap, H), "densest_window": cluster_test(cap, H),
        "reading_under_the_run_card": reading(cap, H),
        "starts_captured_here_and_in_run_bf": both, "captures_of_run_bf": bf_caps,
        "captured_rows": rows, "uncaptured_rows": free,
        "iteration_seconds": rec_json["iteration_seconds"], "software": rec_json["software"],
        "limits": "one cell, one model, 64 starts, one realization on the CPU; states recorded every "
                  f"{args['record_every']} steps; a cycle longer than {args['max_period']} steps would not be detected",
    }


def figure(receipt, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    H, n = receipt["steps"], receipt["starts"]
    caps = np.sort([v["capture_step"] for v in receipt["captured_rows"].values()])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw={"width_ratios": [1.0, 1.25]})

    # (a) starts still uncaptured, with the prediction made from run BF before this run
    t = np.linspace(0, H, 400)
    low, high = receipt["rate_of_run_bf"]["interval95_per_step"]
    ax.fill_between(t, n * np.exp(-high * t), n * np.exp(-low * t), color=LIGHT, lw=0,
                    label="95 % interval of that rate")
    ax.plot(t, n * np.exp(-receipt["rate_of_run_bf"]["per_step"] * t), color=BLUE, lw=1, ls="--",
            label="at the rate of the 5000-step runs")
    ax.step(np.concatenate([[0], caps, [H]]), np.concatenate([[n], n - np.arange(1, len(caps) + 1), [n - len(caps)]]),
            where="post", color=INK, lw=1.6, label="this run")
    ax.set_yscale("log")
    ax.set_yticks([64, 32, 16, 8])
    ax.set_yticks([], minor=True)
    ax.set_yticklabels(["64", "32", "16", "8"])
    ax.set_ylim(7, 70)
    ax.set_xlim(0, H)
    ax.set_xticks(range(0, H + 1, H // 5))
    ax.set_xticklabels([f"{x // 1000}" for x in range(0, H + 1, H // 5)])
    ax.set_xlabel("step (thousands)")
    ax.set_ylabel("starts not yet captured (of 64)")
    ax.grid(axis="y", color=HAIR, lw=0.6)
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    ax.set_title("(a) trajectories still on the chaotic set", loc="left", fontsize=9.5, color=INK)

    # (b) capture step of every start, here and in the three layouts of run BF
    rows = {int(r): v for r, v in receipt["captured_rows"].items()}
    for r in range(n):
        bx.plot([1e2, rows[r]["capture_step"] if r in rows else H * 1.6], [r, r], color=HAIR, lw=0.6, zorder=1)
    four = [(v["capture_step"], r) for r, v in rows.items() if v["period"] == 4]
    rest = [(v["capture_step"], r) for r, v in rows.items() if v["period"] != 4]
    bx.scatter(*zip(*four), s=18, color=INK, zorder=3, label="this run: period 4")
    if rest:
        bx.scatter(*zip(*rest), s=30, marker="D", facecolor="white", edgecolor=INK, lw=1.1, zorder=3,
                   label="this run: period 8")
    free = [r for r in range(n) if r not in rows]
    bx.scatter([H * 1.6] * len(free), free, s=22, marker=">", color=MUTED, zorder=3,
               label=f"this run: not captured in {H:,} steps".replace(",", " "))
    marks = {"b32": ("s", "two batches of 32"), "b64": ("o", "one batch of 64"), "b1": ("^", "one start at a time")}
    for label, d in receipt["captures_of_run_bf"].items():
        bx.scatter(list(d.values()), [int(r) for r in d], s=30, marker=marks[label][0], facecolor="white",
                   edgecolor=BLUE, lw=1.1, zorder=2, label=f"5000-step runs: {marks[label][1]}")
    bx.axvline(5000, color=BLUE, lw=0.7, ls=":")
    bx.set_xscale("log")
    bx.set_xlim(5e2, H * 2.2)
    bx.set_ylim(-1.5, n + 0.5)
    bx.set_xlabel("step of capture")
    bx.set_ylabel("start (index)")
    bx.legend(frameon=False, fontsize=7.2, loc="center left", bbox_to_anchor=(1.0, 0.5))
    bx.set_title("(b) when each start was captured", loc="left", fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="llm-soft-loop/batch-fate/bfl1")
    ap.add_argument("--bf", default="llm-soft-loop/batch-fate/bf1")
    ap.add_argument("--out", default="docs/batch-fate/bfl1-receipt.json")
    ap.add_argument("--figure", default="paper/llm/figures/fig-batch-fate-long.png")
    a = ap.parse_args()
    receipt = analyze(DATA / "explore" / a.run, DATA / "explore" / a.bf)
    figure(receipt, ROOT / a.figure)
    receipt["figure"] = {"path": a.figure, "sha256": sha256(ROOT / a.figure)}
    (ROOT / a.out).write_text(json.dumps(receipt, indent=1) + "\n")
    short = {k: receipt[k] for k in ("captured", "not_captured", "captured_in_second_half", "periods", "orbits",
                                     "captured_by", "rate", "lifetime", "predicted_from_run_bf",
                                     "constant_rate_tests", "densest_window", "reading_under_the_run_card")}
    print(json.dumps(short, indent=1))


if __name__ == "__main__":
    main()
