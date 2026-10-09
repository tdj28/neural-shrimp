"""Derive every number quoted in the gd-route manuscript from saved evidence.

Writes five files next to the manuscript:

  generated-numbers.tex   one \\newcommand per quoted number (the text uses only these)
  generated-design-table.tex  the design table, built from the frozen plan and summary
  generated-consequence-table.tex  the consequence table, built from the pilot's receipts
  data/numbers.json       the manifest: each macro's value, evidence label, source
                          file(s) with SHA-256 and the computation that yields it
  provenance.json         index of the manifest and every figure receipt, by hash

With --verify nothing is written: every number is recomputed, and the run
fails if a value, a source hash, the generated TeX or the index differs.
"""

import json
import sys
from fractions import Fraction
from decimal import Decimal
from math import gcd
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER / "figures"))

import common as c  # noqa: E402

TEX = PAPER / "generated-numbers.tex"
TABLE = PAPER / "generated-design-table.tex"
MANIFEST = PAPER / "data/numbers.json"
INDEX = PAPER / "provenance.json"
VERIFY = "--verify" in sys.argv

SUMMARY = c.ROOT / "docs/EXP-028-summary.json"
PLAN = c.ROOT / "experiments/trainability/EXP-028-plan.json"
AUDIT = c.DATA / "audits/EXP-028/validation-001/audit.json"
CURVE = c.EXPLORE / "a3-ns-curve/s2-28011/ns-curve.json"
STAIR_S2 = c.EXPLORE / "a3-staircase/s2-28011-out-0.0856/staircase.json"
STAIR_S1R = c.EXPLORE / "a3-staircase/s1r-28002-out-0.9385/staircase.json"
RAYS = c.EXPLORE / "a2-survey/rays.json"
SURVEY = c.EXPLORE / "a2-survey/survey.json"
ORDER = c.EXPLORE / "a3-well-ordering/well-ordering.json"
FIG6 = PAPER / "figures/f6_route_portraits.provenance.json"
FIG2 = PAPER / "figures/f2_rotation_map.provenance.json"
FIG7 = PAPER / "figures/f7_criticality.provenance.json"
FIG8 = PAPER / "figures/f8_ancestry.provenance.json"
FIG9 = PAPER / "figures/f9_generality.provenance.json"
GPU = {"main": c.EXPLORE / "a3-gpu-rotation/s2-28011-1024", "right": c.EXPLORE / "a3-gpu-rotation/s2-28011-right"}
MERGED = c.EXPLORE / "a3-ns-curve/s2-28011-merged/ns-curve.json"
CRIT = c.EXPLORE / "criticality"
CRIT_CURVES = [CRIT / "s2-28011-curve/curve.json", CRIT / "s2-28011-steep/curve.json"]
CRIT_BOUNDARY = CRIT / "boundary.json"
CRIT_S1R = CRIT / "s1r-28002-curve/curve.json"
ANC = c.EXPLORE / "ancestry"
ANC_LEVELS, ANC_LEFT = ANC / "levels-right-attempt2/levels.json", ANC / "levels-left/levels.json"
ANC_CONTROLS, ANC_CONNECT = ANC / "controls-002/controls.json", ANC / "connect-right/connect.json"
RD = c.EXPLORE / "realdata"
RD_NS = RD / "w8-n1024-halves-wd1e-3/continue-0.42/continue--0.42000.json"
RD_SWEEP = RD / "w8-n1024-halves-wd1e-3/sweep-ns/sweep.json"
RD_SHARP = RD / "w8-n1024-halves-wd1e-3/branch-2/branch.json"
RD_PARITY = [RD / "w8-n1024-parity-wd1e-3/continue-0.4067/continue--0.40670.json",
             RD / "w8-n1024-parity-wd1e-3/continue-0.4521/continue--0.45210.json"]
RD_OTHER_SUBSET = [RD / "s28102-w8-n1024-halves-wd1e-3/continue-0.7092/continue--0.70920.json",
                   RD / "s28102-w8-n1024-halves-wd1e-3/continue-0.7546/continue--0.75460.json"]
MM = c.EXPLORE / "minimal-model"
MM_EDGE, MM_EDGE_WD = MM / "hofmann/edge.json", MM / "hofmann/edge-lam0.01.json"
MM_CIRCLE_WD, MM_CIRCLE = MM / "hofmann/circle-lam0.01-hysteresis.json", MM / "hofmann/circle-lam0.0.json"
MM_STAIR, MM_FINE = MM / "hofmann/staircase-lam0.01.json", MM / "hofmann/staircase-lam0.01-fine08.json"
MM_GRID, MM_HELDOUT, MM_PRED = MM / "grid.json", MM / "heldout.json", MM / "predictions.json"
CONS = c.EXPLORE / "consequence/analysis.json"
CONS_POOLED = c.EXPLORE / "consequence/analysis-s2-pooled-robust.json"
PRE = {"STwo": c.EXPLORE / "consequence/precursor-s2-28011-ar/precursor.json",
       "SOneR": c.EXPLORE / "consequence/precursor-s1r-28002-ar/precursor.json"}
S1R_TRACES = [c.EXPLORE / "a3-ns-curve/s1r-28002/ns-curve.json",
              c.EXPLORE / "a3-ns-curve/s1r-28002-start0.95/ns-curve.json"]
CONS_TABLE = PAPER / "generated-consequence-table.tex"
CONS_OUTCOMES = [("repro_func_T40k_d1e-8", "function disagreement, 40k updates, start $10^{-8}$ apart (log$_{10}$)"),
                 ("repro_after_anneal_d1e-8", "function disagreement after the step-decay anneal (log$_{10}$)"),
                 ("stopping_time_spread", "spread over 64 anneal start times (log$_{10}$)"),
                 ("tail_train_loss", "training loss, last 8{,}192 updates (log$_{10}$)"),
                 ("tail_test_mse", "held-out loss, last 8{,}192 updates (log$_{10}$)"),
                 ("annealed_test_mse", "held-out loss after the anneal (log$_{10}$)"),
                 ("loss_line_fraction", "loss power in spectral lines")]
CONS_REGIMES = [("period2", "period 2"), ("quasiperiodic", "quasi-periodic"), ("locked", "locked"),
                ("chaotic", "chaotic")]

FROZEN = "prospectively frozen (EXP-028), independently audited"
EXPLORATORY = "exploratory"
COARSE = ["S1-28001", "S1-28002", "S1R-28001", "S1R-28002", "S2-28011", "S2-28012"]


def load(path):
    return json.loads(Path(path).read_text())


def integer(n):
    return f"{n:,}".replace(",", "{,}")


def percent(fraction, digits=1, widen=True):
    """Percent text; ``widen`` adds decimals near 100 so rounding never reads as all."""
    value = 100 * fraction
    if widen and 99.5 < value < 100 and digits < 2:
        digits = 2
    return f"{value:.{digits}f}\\%"


def fixed(x, digits):
    return f"{x:.{digits}f}"


def signed(x, digits=3):
    """Signed fixed-point text without a negative zero."""
    x = 0.0 if abs(x) < 0.5 * 10.0 ** -digits else x
    return f"{x:+.{digits}f}"


def power(x):
    """Exact power of ten as TeX math, for tolerances such as 1e-10."""
    exponent = int(round(np.log10(x)))
    assert abs(x - 10.0 ** exponent) <= 1e-12 * 10.0 ** exponent, x
    return f"10^{{{exponent}}}"


def sci(x, digits=1):
    """Scientific notation as TeX math, for example 4.2\\times10^{-9}."""
    exponent = int(np.floor(np.log10(abs(x))))
    return f"{x / 10.0 ** exponent:.{digits}f}\\times10^{{{exponent}}}"


def portable(path):
    """Checkout-independent source label: repo:<relative> or data:<relative>."""
    resolved = Path(path).resolve()
    for prefix, base in (("repo:", c.ROOT), ("data:", c.DATA)):
        try:
            return prefix + str(resolved.relative_to(Path(base).resolve()))
        except ValueError:
            continue
    return str(resolved)


def located(label):
    """Filesystem path for a portable source label."""
    for prefix, base in (("repo:", c.ROOT), ("data:", c.DATA)):
        if label.startswith(prefix):
            return Path(base) / label[len(prefix):]
    return Path(label)


class Numbers:
    def __init__(self):
        self.entries = []

    def add(self, macro, value, tex, evidence, sources, computation):
        assert macro.isalpha(), macro
        assert macro not in {e["macro"] for e in self.entries}, macro
        self.entries.append({"macro": macro, "value": json.loads(json.dumps(value, default=float)), "tex": tex,
                             "evidence": evidence, "sources": [portable(s) for s in sources],
                             "computation": computation})


def farey_pairs(fractions):
    pairs = list(zip(fractions[:-1], fractions[1:]))
    return sum(abs(a.numerator * b.denominator - b.numerator * a.denominator) == 1 for a, b in pairs), len(pairs)


def farey_null(fractions, draws=20000, seed=20261007):
    """Expected Farey-neighbour share of consecutive pairs for random reduced fractions.

    Same number of fractions, drawn uniformly without replacement from all
    reduced m/q with q up to the largest observed denominator and value inside
    the observed range, then sorted.
    """
    qmax = max(f.denominator for f in fractions)
    lo, hi = min(fractions), max(fractions)
    pool = sorted({Fraction(m, q) for q in range(2, qmax + 1) for m in range(1, q)
                   if gcd(m, q) == 1 and lo <= Fraction(m, q) <= hi})
    rng = np.random.default_rng(seed)
    shares = []
    for _ in range(draws):
        pick = sorted(pool[i] for i in rng.choice(len(pool), size=len(fractions), replace=False))
        hits, total = farey_pairs(pick)
        shares.append(hits / total)
    return float(np.mean(shares)), len(pool)


def build():
    n = Numbers()
    summary, plan, audit = load(SUMMARY), load(PLAN), load(AUDIT)
    it = plan["iteration"]
    n.add("nUpdates", it["steps"], integer(it["steps"]), FROZEN, [PLAN], "iteration.steps")
    n.add("nTail", it["tail_steps"], integer(it["tail_steps"]), FROZEN, [PLAN], "iteration.tail_steps")
    n.add("nMaxPeriod", it["max_period"], str(it["max_period"]), FROZEN, [PLAN], "iteration.max_period")
    n.add("nChaosThreshold", it["chaos_threshold"], f"{it['chaos_threshold']:g}", FROZEN, [PLAN],
          "iteration.chaos_threshold")
    cells = {m["cells"] for m in summary["maps"].values()}
    assert len(cells) == 1
    n.add("nMaps", len(summary["maps"]), str(len(summary["maps"])), FROZEN, [SUMMARY], "len(maps)")
    n.add("nCellsPerMap", cells.pop(), integer(next(iter({m["cells"] for m in summary["maps"].values()}))), FROZEN,
          [SUMMARY], "maps[*].cells (identical)")
    producer = {(q["map_id"], q["component"]) for q in summary["gate"]["qualifying"]}
    audited = {(q["map_id"], q["component"]) for q in audit["audit_gate"]["qualifying"]}
    confirmed = sorted(producer & audited)
    n.add("nProducerQualifying", len(producer), str(len(producer)), FROZEN, [SUMMARY], "len(gate.qualifying)")
    n.add("nAuditQualifying", len(audited), str(len(audited)), FROZEN, [AUDIT], "len(audit_gate.qualifying)")
    n.add("nConfirmed", len(confirmed), str(len(confirmed)), FROZEN, [SUMMARY, AUDIT],
          "|producer qualifying & audit qualifying| by (map_id, component)")
    certs = summary["certificates"]
    n.add("nCertificates", len(certs), str(len(certs)), FROZEN, [SUMMARY], "len(certificates)")
    n.add("nCertificatesVerified", sum(bool(r["verified"]) for r in certs),
          str(sum(bool(r["verified"]) for r in certs)), FROZEN, [SUMMARY], "certificates with verified = true")
    periods = {}
    for r in certs:
        if (r["map_id"], r["component"]) in set(confirmed):
            periods[(r["map_id"], r["component"])] = r["primitive_period"]
    values = sorted(periods.values())
    assert len(values) == len(confirmed)
    odd = sorted({p for p in values if p % 2})
    n.add("nPeriodMin", values[0], str(values[0]), FROZEN, [SUMMARY, AUDIT], "min primitive_period, confirmed")
    n.add("nPeriodMax", values[-1], str(values[-1]), FROZEN, [SUMMARY, AUDIT], "max primitive_period, confirmed")
    n.add("nOddWindows", sum(p % 2 for p in values), str(sum(p % 2 for p in values)), FROZEN, [SUMMARY, AUDIT],
          "confirmed windows with odd primitive period")
    n.add("tOddPeriods", odd, ", ".join(map(str, odd[:-1])) + f" and {odd[-1]}", FROZEN, [SUMMARY, AUDIT],
          "distinct odd primitive periods, confirmed")
    by_system = {s: sum(m.startswith(s + "-") for m, _ in confirmed) for s in ("S1R", "S2")}
    n.add("nConfirmedSOneR", by_system["S1R"], str(by_system["S1R"]), FROZEN, [SUMMARY, AUDIT],
          "confirmed windows in S1R maps")
    n.add("nConfirmedSTwo", by_system["S2"], str(by_system["S2"]), FROZEN, [SUMMARY, AUDIT],
          "confirmed windows in S2 maps")
    n.add("nRingReruns", summary["ring_reruns"], integer(summary["ring_reruns"]), FROZEN, [SUMMARY], "ring_reruns")
    rerun = sum(v["rerun"] for v in summary["random_chaos_diagnostics"].values())
    kept = sum(v["confirmed"] for v in summary["random_chaos_diagnostics"].values())
    n.add("nChaosRerun", rerun, str(rerun), FROZEN, [SUMMARY], "sum random_chaos_diagnostics[*].rerun")
    n.add("nChaosKept", kept, str(kept), FROZEN, [SUMMARY], "sum random_chaos_diagnostics[*].confirmed")
    shares = {}
    for m in COARSE:
        conv = summary["maps"][m]["sd_crosstab"]["1000"]["sd_convergent"]
        shares[m] = 1 - conv["fixed_point_candidate"] / sum(conv.values())
    n.add("pNotFixedMin", min(shares.values()), percent(min(shares.values())), FROZEN, [SUMMARY],
          "min over coarse maps of 1 - fixed / score-convergent (1,000-update score)")
    n.add("pNotFixedMax", max(shares.values()), percent(max(shares.values())), FROZEN, [SUMMARY],
          "max over coarse maps of the same share")
    s1 = sum(summary["maps"][m]["counts"]["periodic_candidate"] for m in ("S1-28001", "S1-28002"))
    n.add("nSOnePeriodic", s1, str(s1), FROZEN, [SUMMARY], "periodic cells in S1-28001 and S1-28002 maps")

    runs = c.ns_segments(CURVE)
    rows = [r for run in runs for r in run]
    traced = load(CURVE)["rows"]
    theta = [r["theta_over_2pi"] for r in rows]
    n.add("nNSSlices", len(traced), str(len(traced)), EXPLORATORY, [CURVE], "len(rows)")
    n.add("nNSCrossings", len(rows), str(len(rows)), EXPLORATORY, [CURVE], "rows with type neimark_sacker")
    others = len(traced) - len(rows)
    n.add("nNSNotPeriodTwoStart", others, str(others), EXPLORATORY, [CURVE], "rows whose start was not period 2")
    n.add("xBirthMin", min(theta), fixed(min(theta), 4), EXPLORATORY, [CURVE], "min theta_over_2pi")
    n.add("xBirthMax", max(theta), fixed(max(theta), 4), EXPLORATORY, [CURVE], "max theta_over_2pi")
    low = min(r["log10_hidden_boundary"] for r in rows)
    n.add("xNSLowest", low, fixed(low, 3), EXPLORATORY, [CURVE], "min log10_hidden_boundary")
    roots = sorted({r["fraction"] for r in c.ns_roots(runs, max_den=13)})
    n.add("tNSRoots", [str(f) for f in roots], ", ".join(f"{f.numerator}/{f.denominator}" for f in roots[:-1])
          + f" and {roots[-1].numerator}/{roots[-1].denominator}", EXPLORATORY, [CURVE],
          "distinct m/q (q <= 13) crossed by theta_over_2pi along each unbroken run")

    stair = load(STAIR_S2)
    fixed_out = stair["args"]["fixed"]
    pts = sorted(rows, key=lambda r: r["log10_output"])
    left = max((r for r in pts if r["log10_output"] <= fixed_out), key=lambda r: r["log10_output"])
    right = min((r for r in pts if r["log10_output"] >= fixed_out), key=lambda r: r["log10_output"])
    t = (fixed_out - left["log10_output"]) / (right["log10_output"] - left["log10_output"])
    birth = left["theta_over_2pi"] + t * (right["theta_over_2pi"] - left["theta_over_2pi"])
    boundary = left["log10_hidden_boundary"] + t * (right["log10_hidden_boundary"] - left["log10_hidden_boundary"])
    n.add("xBirthAtSlice", birth, fixed(birth, 4), EXPLORATORY, [CURVE, STAIR_S2],
          "theta_over_2pi linearly interpolated in log10 output to the staircase slice")
    n.add("xNSAtSlice", boundary, fixed(boundary, 3), EXPLORATORY, [CURVE, STAIR_S2],
          "log10_hidden_boundary interpolated to the staircase slice")
    first_qp = next(r for r in stair["rows"] if c.quasi_periodic(r))
    n.add("xOnsetAtSlice", first_qp["log10_varied"], fixed(first_qp["log10_varied"], 3), EXPLORATORY, [STAIR_S2],
          "first quasi-periodic cell along the slice")
    n.add("xRhoOnset", first_qp["rho"], fixed(first_qp["rho"], 4), EXPLORATORY, [STAIR_S2],
          "rotation number of the first quasi-periodic cell")
    for label, path in (("STwo", STAIR_S2), ("SOneR", STAIR_S1R)):
        data = load(path)
        merged = [p for p in c.merged_plateaus(data["rows"]) if p["cells"] >= 8]
        best = {}
        for p in merged:
            if p["cells"] > best.get((p["m"], p["q"]), {"cells": 0})["cells"]:
                best[(p["m"], p["q"])] = p
        fractions = sorted(Fraction(m, q) for m, q in best)
        hits, total = farey_pairs(fractions)
        null, pool = farey_null(fractions)
        n.add(f"tPlateaus{label}", [str(f) for f in fractions],
              ", ".join(f"{f.numerator}/{f.denominator}" for f in fractions[:-1])
              + f" and {fractions[-1].numerator}/{fractions[-1].denominator}", EXPLORATORY, [path],
              "distinct m/q of merged locked plateaus with >= 8 cells (common.merged_plateaus), by value")
        n.add(f"nFarey{label}", hits, str(hits), EXPLORATORY, [path],
              "consecutive plateau pairs (by value) with |m1 q2 - m2 q1| = 1")
        n.add(f"nFareyPairs{label}", total, str(total), EXPLORATORY, [path], "consecutive plateau pairs")
        n.add(f"pFareyNull{label}", null, percent(null, 0, widen=False), EXPLORATORY, [path],
              f"mean Farey share for sorted random reduced fractions from a pool of {pool} (seed 20261007)")
        if label == "STwo":
            lock = best[(2, 9)]
            n.add("xTwoNinthsStart", lock["start"], fixed(lock["start"], 3), EXPLORATORY, [path],
                  "start of the longest merged 2/9 plateau")
            n.add("xTwoNinthsEnd", lock["end"], fixed(lock["end"], 3), EXPLORATORY, [path],
                  "end of the longest merged 2/9 plateau")

    rays = load(RAYS)
    totals = {"fold": 0, "flip": 0, "neimark_sacker": 0, "other": 0}
    fold_flip = ns8 = 0
    for w in rays.values():
        k = {"fold": 0, "flip": 0, "neimark_sacker": 0, "other": 0}
        for r in w["rays"]:
            k[r["type"] if r["type"] in k else "other"] += 1
        for t_ in totals:
            totals[t_] += k[t_]
        fold_flip += k["fold"] >= 8 and k["flip"] >= 8
        ns8 += k["neimark_sacker"] >= 8
    for macro, key in (("nRaysFold", "fold"), ("nRaysFlip", "flip"), ("nRaysNS", "neimark_sacker"),
                       ("nRaysOther", "other")):
        n.add(macro, totals[key], integer(totals[key]), EXPLORATORY, [RAYS], f"rays of type {key}, all windows")
    n.add("nRaysTotal", sum(totals.values()), integer(sum(totals.values())), EXPLORATORY, [RAYS], "all rays")
    n.add("nWindowsFoldFlip", fold_flip, str(fold_flip), EXPLORATORY, [RAYS], "windows with fold >= 8 and flip >= 8")
    n.add("nWindowsNS", ns8, str(ns8), EXPLORATORY, [RAYS], "windows with Neimark-Sacker >= 8")
    survey = load(SURVEY)
    passing = sum(bool(r["shrimp_consistent_draft"]) for r in survey)
    n.add("nDraftShrimp", passing, str(passing), EXPLORATORY, [SURVEY], "shrimp_consistent_draft true")
    w587 = next(r for r in survey if r["key"] == "S2-28011-zoom:587")["metrics"]
    win = rays["S2-28011-zoom:587"]
    k587 = {t_: sum(r["type"] == t_ for r in win["rays"]) for t_ in ("fold", "flip", "neimark_sacker")}
    n.add("nWFold", k587["fold"], str(k587["fold"]), EXPLORATORY, [RAYS], "window 587 fold rays")
    n.add("nWFlip", k587["flip"], str(k587["flip"]), EXPLORATORY, [RAYS], "window 587 flip rays")
    n.add("nWNS", k587["neimark_sacker"], str(k587["neimark_sacker"]), EXPLORATORY, [RAYS], "window 587 NS rays")
    n.add("nWEdgeAgree", w587["edge_agreement"][0], str(w587["edge_agreement"][0]), EXPLORATORY, [SURVEY],
          "window 587 edge agreement (agreeing rays)")
    n.add("nWEdgeCompared", w587["edge_agreement"][1], str(w587["edge_agreement"][1]), EXPLORATORY, [SURVEY],
          "window 587 edge agreement (compared rays)")
    n.add("nWFlipTwoP", w587["flip_to_2p"][0], str(w587["flip_to_2p"][0]), EXPLORATORY, [SURVEY],
          "window 587 flip rays opening onto period 36")

    order = load(ORDER)
    groups = {
        "AllPlane": ([r for r in order if r["points"] >= 5], "plane"),
        "ConfPlane": ([r for r in order if r["confirmed"]], "plane"),
        "ConfRefined": ([r for r in order if r["confirmed"] and any(t_["q"] >= 5 for t_ in r["refined"]["tests"])],
                        "refined"),
        "AllRefined": ([r for r in order if any(t_["q"] >= 5 for t_ in r["refined"]["tests"])], "refined"),
        "Odd": ([r for r in order if r["confirmed"] and r["period"] % 2], "plane"),
    }
    for label, (sub, kind) in groups.items():
        if kind == "plane":
            observed, expected = sum(r["well_ordered"] for r in sub), sum(r["null_pass_rate"] for r in sub)
        else:
            observed = sum(r["refined"]["passed_q5"] for r in sub)
            expected = sum(r["refined"]["null_q5"] for r in sub)
        n.add(f"nWO{label}", observed, str(observed), EXPLORATORY, [ORDER], f"well-ordered ({kind}) in group {label}")
        n.add(f"nWO{label}Of", len(sub), str(len(sub)), EXPLORATORY, [ORDER], f"group size {label}")
        n.add(f"xWO{label}Null", expected, fixed(expected, 2), EXPLORATORY, [ORDER], f"summed null rate {label}")
    passed = [r for r in order if r["confirmed"] and r.get("refined", {}).get("passed_q5")]
    doubled = sum(r["refined"]["base"]["level"] > 0 for r in passed)
    n.add("nDoubledTongue", doubled, str(doubled), EXPLORATORY, [ORDER],
          "confirmed windows passing first at a period-doubled level")

    cert = plan["certification"]
    for macro, key in (("xCertNewton", "newton_tolerance"), ("xCertAtol", "atol"), ("xCertRtol", "rtol"),
                       ("xCertCandidate", "candidate_tolerance"), ("xCertDivisor", "divisor_separation"),
                       ("xCertMargin", "stability_margin")):
        n.add(macro, cert[key], power(cert[key]), FROZEN, [PLAN], f"certification.{key}")
    n.add("nCertIterations", cert["max_iterations"], str(cert["max_iterations"]), FROZEN, [PLAN],
          "certification.max_iterations")
    n.add("nMinComponent", cert["min_component_size"], str(cert["min_component_size"]), FROZEN, [PLAN],
          "certification.min_component_size")
    n.add("nGateMinPeriod", plan["gate"]["min_period"], str(plan["gate"]["min_period"]), FROZEN, [PLAN],
          "gate.min_period")
    n.add("pRingFraction", plan["gate"]["min_confirmed_ring_fraction"],
          percent(plan["gate"]["min_confirmed_ring_fraction"], 0, widen=False), FROZEN, [PLAN],
          "gate.min_confirmed_ring_fraction")
    n.add("xScreenAtol", it["atol"], power(it["atol"]), FROZEN, [PLAN], "iteration.atol")
    n.add("xScreenRtol", it["rtol"], power(it["rtol"]), FROZEN, [PLAN], "iteration.rtol")
    n.add("nMinRepeats", it["min_repeats"], str(it["min_repeats"]), FROZEN, [PLAN], "iteration.min_repeats")
    n.add("nFtleSteps", it["ftle_steps"], integer(it["ftle_steps"]), FROZEN, [PLAN], "iteration.ftle_steps")
    n.add("nFtleAlign", it["ftle_align"], integer(it["ftle_align"]), FROZEN, [PLAN], "iteration.ftle_align")
    n.add("pDrift", it["drift_threshold"], percent(it["drift_threshold"], 0, widen=False), FROZEN, [PLAN],
          "iteration.drift_threshold")
    n.add("xDivergence", it["divergence_threshold"], power(it["divergence_threshold"]), FROZEN, [PLAN],
          "iteration.divergence_threshold")
    n.add("nScoreHorizon", max(it["sd_horizons"]), integer(max(it["sd_horizons"])), FROZEN, [PLAN],
          "max(iteration.sd_horizons), the horizon used for the score comparison")
    ring = plan["ring_confirmation"]
    n.add("nRingSteps", ring["steps"], integer(ring["steps"]), FROZEN, [PLAN], "ring_confirmation.steps")
    n.add("nRingFtle", ring["ftle_steps"], integer(ring["ftle_steps"]), FROZEN, [PLAN],
          "ring_confirmation.ftle_steps")
    n.add("nRingCap", ring["max_cells_per_component"], str(ring["max_cells_per_component"]), FROZEN, [PLAN],
          "ring_confirmation.max_cells_per_component")
    n.add("nChaosPerMap", plan["chaos_diagnostics"]["count"], str(plan["chaos_diagnostics"]["count"]), FROZEN,
          [PLAN], "chaos_diagnostics.count")
    components = [k for m in audit["checks"]["maps"].values() for k in m.get("components", [])]
    checks = ("closure_ok", "radius_ok", "primitive_ok", "attracting_consistent", "seed_ok", "rates_from_axes_ok")
    checked = sum(bool(k["checked"]) for k in components)
    n.add("nAuditChecked", checked, str(checked), FROZEN, [AUDIT], "audit components with checked = true")
    passing = sum(all(k[x] for x in checks) for k in components)
    n.add("nAuditAllOk", passing, str(passing), FROZEN, [AUDIT], f"audit components with all of {list(checks)}")
    worst = max(k["radius_difference"] for k in components)
    n.add("xAuditMaxRadiusDiff", worst, sci(worst), FROZEN, [AUDIT],
          "max |finite-difference minus exact-Jacobian spectral radius| over audited certificates")
    receipt = audit["checks"]["receipt"]
    n.add("nAuditReceiptFiles", receipt["files"], str(receipt["files"]), FROZEN, [AUDIT], "checks.receipt.files")
    n.add("nAuditMismatched", len(receipt["mismatched"]), str(len(receipt["mismatched"])), FROZEN, [AUDIT],
          "len(checks.receipt.mismatched)")

    n.add("nCertPThree", len(order), str(len(order)), EXPLORATORY, [ORDER],
          "verified certified orbits with period >= 3 (rows of the well-ordering receipt)")
    trivial = [r for r in order if r["confirmed"] and r["points"] <= 3]
    n.add("nConfTrivial", len(trivial), str(len(trivial)), EXPLORATORY, [ORDER],
          "confirmed windows with at most 3 test points (pass the plane test by construction)")
    trivial_pass = sum(r["well_ordered"] for r in trivial)
    n.add("nConfTrivialPass", trivial_pass, str(trivial_pass), EXPLORATORY, [ORDER], "of those, passing the plane test")
    excluded = sorted(r["period"] for r in order if r["confirmed"]
                      and not any(t_["q"] >= 5 for t_ in r["refined"]["tests"]))
    n.add("nConfRefinedExcluded", len(excluded), str(len(excluded)), EXPLORATORY, [ORDER],
          "confirmed windows with no refined test on at least 5 points")
    n.add("tConfRefinedExcluded", excluded, ", ".join(map(str, excluded[:-1])) + f" and {excluded[-1]}",
          EXPLORATORY, [ORDER], "their periods")
    level0 = [r for r in passed if r["refined"]["base"]["level"] == 0]
    n.add("nLevelZeroPass", len(level0), str(len(level0)), EXPLORATORY, [ORDER],
          "passing confirmed windows whose first passing level is the two-step map")
    shifts = {}
    for r in level0:
        m_ = int(r["refined"]["base"]["rotation"].split("/")[0])
        shifts[m_] = shifts.get(m_, 0) + 1
    words = [f"$m={m_}$ for {k}" for m_, k in sorted(shifts.items())]
    n.add("tShiftCounts", {str(k): v for k, v in sorted(shifts.items())},
          (", ".join(words[:-1]) + f" and {words[-1]}") if len(words) > 1 else words[0], EXPLORATORY, [ORDER],
          "measured angular shift m (folded to m <= q/2) among windows passing at the two-step level")

    rows_s2 = load(STAIR_S2)["rows"]
    merged = c.merged_plateaus(rows_s2)
    two9 = max((p for p in merged if (p["m"], p["q"]) == (2, 9)), key=lambda p: p["cells"])
    quarter = max((p for p in merged if (p["m"], p["q"]) == (1, 4)), key=lambda p: p["cells"])
    band = [r for r in rows_s2 if c.quasi_periodic(r) and r["log10_varied"] < two9["start"]]
    low = min(band, key=lambda r: r["rho"])
    n.add("xRhoMinSlice", low["rho"], fixed(low["rho"], 4), EXPLORATORY, [STAIR_S2],
          "minimum rotation number of quasi-periodic cells before the 2/9 plateau")
    n.add("xRhoMinAt", low["log10_varied"], fixed(low["log10_varied"], 3), EXPLORATORY, [STAIR_S2],
          "its log10 hidden rate")
    between = [r for r in rows_s2 if two9["end"] < r["log10_varied"] < quarter["start"]]
    chaotic = [r for r in between if r["status"] == 3]
    n.add("nBetweenCells", len(between), str(len(between)), EXPLORATORY, [STAIR_S2],
          "cells strictly between the 2/9 and 1/4 plateaus")
    n.add("nBetweenChaotic", len(chaotic), str(len(chaotic)), EXPLORATORY, [STAIR_S2], "of those, chaotic")
    n.add("xQuarterStart", quarter["start"], fixed(quarter["start"], 3), EXPLORATORY, [STAIR_S2],
          "start of the longest merged 1/4 plateau")
    n.add("xQuarterEnd", quarter["end"], fixed(quarter["end"], 3), EXPLORATORY, [STAIR_S2],
          "end of the longest merged 1/4 plateau")

    # Rotation maps with the merged curve (figure 2)
    merged_rows = [r for r in load(MERGED)["rows"] if r.get("type") == "neimark_sacker"]
    n.add("nMergedPoints", len(merged_rows), str(len(merged_rows)), EXPLORATORY, [MERGED],
          "points of the merged complex-pair crossing trace")
    theta_m = [r["theta_over_2pi"] for r in merged_rows]
    n.add("xMergedBirthMin", min(theta_m), fixed(min(theta_m), 4), EXPLORATORY, [MERGED], "min crossing angle, merged")
    n.add("xMergedBirthMax", max(theta_m), fixed(max(theta_m), 4), EXPLORATORY, [MERGED], "max crossing angle, merged")
    fig2 = load(FIG2)["derived"]
    for tag, folder in GPU.items():
        first = load(folder / "analysis.json")["continuity"][0]
        name = {"main": "Main", "right": "Right"}[tag]
        meta = load(folder / "metadata.json")["args"]["resolution"]
        n.add(f"tMapSize{name}", meta, f"{meta[0]:,}".replace(",", "{,}") + "\\times" + f"{meta[1]:,}".replace(",", "{,}"),
              EXPLORATORY, [folder / "metadata.json"], "map resolution (hidden x output cells), math mode")
        n.add(f"xContOffset{name}", first["offset_log10_hidden"], f"{first['offset_log10_hidden']:g}", EXPLORATORY,
              [folder / "analysis.json"], "continuity offset beyond the curve (decades)")
        n.add(f"nContPoints{name}", first["points"], str(first["points"]), EXPLORATORY, [folder / "analysis.json"],
              "curve points compared at that offset")
        n.add(f"xContMedian{name}", first["median_abs_diff"], sci(first["median_abs_diff"]), EXPLORATORY,
              [folder / "analysis.json"], "median |rotation - crossing angle|")
        n.add(f"xContMax{name}", first["max_abs_diff"], sci(first["max_abs_diff"]), EXPLORATORY,
              [folder / "analysis.json"], "max |rotation - crossing angle|")
        label = {"main": "main region", "right": "right wing"}[tag]
        n.add(f"nRootsInside{name}", fig2[label]["roots_inside"], str(fig2[label]["roots_inside"]), EXPLORATORY,
              [FIG2, folder / "tongue-roots.json"], "resonance points inside the map")
        n.add(f"nRootsNear{name}", fig2[label]["roots_near"], str(fig2[label]["roots_near"]), EXPLORATORY,
              [FIG2, folder / "tongue-roots.json"], "of those, null fraction at least as close <= 0.05")

    # Criticality (figure 7)
    crit = load(FIG7)["derived"]
    n.add("nCritPoints", crit["points"], str(crit["points"]), EXPLORATORY, [FIG7, *CRIT_CURVES], "slice-wise crossings with d; cross-output full-state branch identity unverified")
    n.add("nCritSuper", crit["supercritical"], str(crit["supercritical"]), EXPLORATORY, [FIG7, *CRIT_CURVES], "d < 0")
    n.add("nCritSub", crit["subcritical"], str(crit["subcritical"]), EXPLORATORY, [FIG7, *CRIT_CURVES], "d > 0")
    changes = crit["sign_changes"]
    n.add("nChenciner", len(changes), str(len(changes)) if len(changes) != 3 else "three", EXPLORATORY,
          [FIG7, *CRIT_CURVES], "sign transitions between adjacent output-ordered rows; branch identity and generalized-NS nondegeneracy unverified")
    n.add("tChenciner", changes, ", ".join(f"${z:+.3f}$" for z in changes[:-1]) + f" and ${changes[-1]:+.3f}$",
          EXPLORATORY, [FIG7, *CRIT_CURVES], "midpoints between opposite-sign adjacent rows (log10 output); not established branchwise roots")
    n.add("xCritMostNegative", crit["most_negative_d"], fixed(crit["most_negative_d"], 2), EXPLORATORY,
          [FIG7, *CRIT_CURVES], "most negative d")
    checks_ = crit["dynamical_checks"]
    n.add("nDynChecks", len(checks_), str(len(checks_)), EXPLORATORY, [FIG7], "dynamical check points")
    n.add("nDynConsistent", sum(r["consistent"] for r in checks_), str(sum(r["consistent"] for r in checks_)),
          EXPLORATORY, [FIG7], "points whose behaviour matches the sign of d (rule in f7_criticality.py)")
    errs = [r["max_amplitude_error"] for r in checks_ if r["max_amplitude_error"] is not None]
    n.add("pDynMaxError", max(errs), percent(max(errs), 1, widen=False), EXPLORATORY, [FIG7],
          "largest relative amplitude error at supercritical points")
    n.add("pDynMedianError", float(np.median(errs)), percent(float(np.median(errs)), 1, widen=False), EXPLORATORY,
          [FIG7], "median over supercritical points of the per-point largest relative error")
    boundary = load(CRIT_BOUNDARY)
    divergent_below = [b for b in boundary if b["labels_just_below"]
                       and all(x == "divergent" for x in b["labels_just_below"])]
    n.add("nBasinPoints", len(divergent_below), str(len(divergent_below)), EXPLORATORY, [CRIT_BOUNDARY],
          "curve points whose fixed-start map is divergent just below the curve (where the 2-cycle is stable)")
    supercrit = sum(b["d"] is not None and b["d"] < 0 for b in divergent_below)
    n.add("nBasinSuper", supercrit, str(supercrit), EXPLORATORY, [CRIT_BOUNDARY], "of those, supercritical")
    s1r_rows = [r for r in load(CRIT_S1R)["rows"] if r.get("d") is not None]
    n.add("nCritSOneR", len(s1r_rows), str(len(s1r_rows)), EXPLORATORY, [CRIT_S1R], "S1R points with d")
    n.add("nCritSOneRSuper", sum(r["d"] < 0 for r in s1r_rows), str(sum(r["d"] < 0 for r in s1r_rows)),
          EXPLORATORY, [CRIT_S1R], "S1R points with d < 0")

    # Resonance ancestry (figure 8)
    anc = load(FIG8)["derived"]
    n.add("nAncLevels", len(anc["deltas"]), str(len(anc["deltas"])), EXPLORATORY, [FIG8, ANC_LEVELS],
          "cross-sections with two fold edges")
    n.add("tAncOutcome", anc["frozen_outcome"], anc["frozen_outcome"].replace("_", " "), EXPLORATORY,
          [FIG8, ANC_LEVELS], "frozen near-root outcome")
    n.add("xAncIntercept", anc["extrapolated_center"], sci(anc["extrapolated_center"]), EXPLORATORY,
          [FIG8, ANC_LEVELS], "frozen estimator intercept (decades)")
    n.add("xAncTolerance", anc["root_tolerance"], power(anc["root_tolerance"]), EXPLORATORY, [FIG8, ANC_LEVELS],
          "frozen root tolerance (decades), math mode")
    n.add("xAncWidthSlope", anc["posthoc_width_slope"], fixed(anc["posthoc_width_slope"], 3), "post hoc",
          [FIG8, ANC_LEVELS], "log-log slope of width over the deepest three levels")
    n.add("xAncAmpSlope", anc["posthoc_amplitude_slope"], fixed(anc["posthoc_amplitude_slope"], 3), "post hoc",
          [FIG8, ANC_LEVELS], "log-log slope of amplitude over the deepest three levels")
    words = {"pass": "passed", "alternative_ancestry": "was refused (alternative ancestry)"}
    for key, macro in (("positive", "tAncPositive"), ("specificity", "tAncSpecificity"), ("negative", "tAncNegative")):
        n.add(macro, anc["controls"][key], words.get(anc["controls"][key], anc["controls"][key].replace("_", " ")),
              EXPLORATORY, [FIG8, ANC_CONTROLS], f"{key} control outcome")
    n.add("tAncConnection", anc["connection_outcome"], anc["connection_outcome"].replace("_", " "), EXPLORATORY,
          [FIG8, ANC_CONNECT], "connection-to-window outcome")
    left = load(ANC_LEFT)["outcome"]
    n.add("tAncLeft", left, left.replace("_", " "), EXPLORATORY, [ANC_LEFT], "left-root outcome")
    for side, word in (("right", "Right"), ("left", "Left")):
        path = ANC / f"root-{side}/root.json"
        crit_ = load(path)["criteria"]
        flags = [v for k, v in crit_.items() if isinstance(v, bool)]
        n.add(f"nAncOnset{word}", sum(flags), str(sum(flags)), EXPLORATORY, [path], f"{side} root onset criteria passed")
        n.add(f"nAncOnset{word}Of", len(flags), str(len(flags)), EXPLORATORY, [path], f"{side} root onset criteria")
        n.add(f"xAncRadius{word}", crit_["radius_scaling_exponent"], fixed(crit_["radius_scaling_exponent"], 3),
              EXPLORATORY, [path], f"{side} root circle-radius scaling exponent (frozen band 0.3-0.7)")
    margin = anc["extrapolated_center"] / anc["root_tolerance"] - 1
    n.add("pAncMargin", margin, percent(margin, 0, widen=False), EXPLORATORY, [FIG8, ANC_LEVELS],
          "intercept above tolerance, relative")

    # Real data and minimal models (figure 9)
    rd = load(RD_NS)
    n.add("xRDCrossing", rd["log10_eta_boundary"], fixed(rd["log10_eta_boundary"], 4), EXPLORATORY, [RD_NS],
          "log10 learning rate of the complex-pair crossing")
    n.add("xRDPredicted", rd["theta_over_2pi"], fixed(rd["theta_over_2pi"], 7), EXPLORATORY, [RD_NS],
          "crossing angle (predicted rotation)")
    n.add("tRDMultiplier", rd["dominant"], f"{rd['dominant'][0]:.6f} + {rd['dominant'][1]:.6f}i", EXPLORATORY,
          [RD_NS], "dominant two-step multiplier at the crossing")
    sweep = load(RD_SWEEP)["rows"]
    near = min((r for r in sweep if r["finite"] and r["rho2"] is not None),
               key=lambda r: abs(r["log10_eta"] - rd["log10_eta_boundary"]))
    n.add("xRDMeasured", near["rho2"], fixed(near["rho2"], 6), EXPLORATORY, [RD_SWEEP],
          "measured rotation at the sweep point nearest the crossing")
    n.add("xRDMeasuredFtle", near["ftle"], fixed(near["ftle"], 4), EXPLORATORY, [RD_SWEEP], "its FTLE")
    sharp = [a for a in load(RD_SHARP)["attempts"] if a.get("type")]
    types = [rd["type"]] + [a["type"] for a in sharp] + [load(p)["type"] for p in RD_PARITY]
    n.add("nRDCertifiedBranches", len(types), str(len(types)), EXPLORATORY, [RD_NS, RD_SHARP, *RD_PARITY],
          "certified period-2 branches continued to their loss of stability")
    n.add("nRDNSBranches", types.count("neimark_sacker"), str(types.count("neimark_sacker")), EXPLORATORY,
          [RD_NS, RD_SHARP, *RD_PARITY], "of those, complex-pair crossing")
    n.add("nRDFoldBranches", types.count("fold"), str(types.count("fold")), EXPLORATORY,
          [RD_NS, RD_SHARP, *RD_PARITY], "of those, fold")
    other = [load(p)["type"] for p in RD_OTHER_SUBSET]
    n.add("nRDUncertified", sum(t == "start_not_attracting" for t in other),
          str(sum(t == "start_not_attracting" for t in other)), EXPLORATORY, RD_OTHER_SUBSET,
          "starts on the second subset where Newton did not certify a period-2 orbit")
    edge = load(MM_EDGE)
    n.add("xMMEdgeError", max(abs(e["crossing_minus_predicted"]) for e in edge),
          sci(max(abs(e["crossing_minus_predicted"]) for e in edge)), EXPLORATORY, [MM_EDGE],
          "max |numerical crossing - a(s1+s2)=2| over ratios, no weight decay")
    n.add("tMMBirth", [e["theta_over_2pi"] for e in edge], ", ".join(f"{e['theta_over_2pi']:.4f}" for e in edge[:-1])
          + f" and {edge[-1]['theta_over_2pi']:.4f}", EXPLORATORY, [MM_EDGE], "crossing angles for s2/s1 = 0.7, 0.8, 0.9")
    dm = [e["dmodulus_da"] for e in edge]
    n.add("tMMTransversality", dm, f"{min(dm):.2f}--{max(dm):.2f}", EXPLORATORY, [MM_EDGE],
          "range of d|mu|/da at the crossing")
    fig9 = load(FIG9)["derived"]
    wd = [v for k, v in fig9["amplitude_slopes"].items() if k.startswith("weight")]
    nowd = [v for k, v in fig9["amplitude_slopes"].items() if k.startswith("no")]
    n.add("tMMSlopeWD", wd, f"{min(wd):.2f}" if f"{min(wd):.2f}" == f"{max(wd):.2f}" else f"{min(wd):.2f}--{max(wd):.2f}",
          EXPLORATORY, [FIG9, MM_CIRCLE_WD], "amplitude exponent with weight decay 0.01")
    n.add("tMMSlopeNoWD", nowd, f"{min(nowd):.2f}--{max(nowd):.2f}", EXPLORATORY, [FIG9, MM_CIRCLE],
          "amplitude exponent without weight decay")
    circ = load(MM_CIRCLE_WD)
    closest = {}
    for r in circ:
        if r["delta"] > 0 and (r["s2_over_s1"] not in closest or r["delta"] < closest[r["s2_over_s1"]]["delta"]):
            closest[r["s2_over_s1"]] = r
    gap = max(abs(r["rotation"] - r["birth_theta_over_2pi"]) for r in closest.values())
    n.add("xMMRotationGap", gap, sci(gap), EXPLORATORY, [MM_CIRCLE_WD],
          "max |rotation - crossing angle| at the smallest offset, weight decay 0.01")
    n.add("xMMSmallestOffset", min(r["delta"] for r in closest.values()),
          power(min(r["delta"] for r in closest.values())), EXPLORATORY, [MM_CIRCLE_WD], "that offset, math mode")
    locked = fig9["hofmann_locked"]
    seven = ", ".join(f"{m}/{p // 2} (period {p})" for p, m in sorted(locked.get("0.7", [])))
    eight = ", ".join(f"{m}/{p // 2} (period {p})" for p, m in sorted(locked.get("0.8", [])))
    n.add("tMMLockedSeven", locked.get("0.7"), seven, EXPLORATORY, [FIG9, MM_STAIR], "locked cells at s2/s1 = 0.7")
    n.add("tMMLockedEight", locked.get("0.8"), eight, EXPLORATORY, [FIG9, MM_FINE], "locked cells at s2/s1 = 0.8")
    grid = load(MM_GRID)
    ns_grid = [g for g in grid if g.get("type") == "neimark_sacker"]
    n.add("nMMGridNS", len(ns_grid), str(len(ns_grid)), EXPLORATORY, [MM_GRID], "four-weight grid cases ending in NS")
    n.add("nMMGrid", len(grid), str(len(grid)), EXPLORATORY, [MM_GRID], "four-weight grid cases")
    two = next(g for g in ns_grid if g["c"] == 0.0)
    runs = two["criticality"]["runs"]
    n.add("xMMTwoWeightAngle", two["theta_over_2pi"], fixed(two["theta_over_2pi"], 4), EXPLORATORY, [MM_GRID],
          "crossing angle of the two-weight factorization case (c = 0, weight decay 0.2)")
    ratio = runs[1]["amplitude"] / runs[0]["amplitude"]
    n.add("xMMTwoWeightAmpRatio", ratio, fixed(ratio, 2), EXPLORATORY, [MM_GRID],
          "circle amplitude ratio for a fourfold offset (square-root scaling gives 2)")
    held, pred = load(MM_HELDOUT), load(MM_PRED)
    key = lambda r: (r["c"], r["lam"], round(r["ratio"], 6))
    pred_types = {key(p): p["predicted_type"] for p in pred}
    correct = sum(pred_types.get(key(h)) == h["type"] for h in held)
    n.add("nMMHeldoutCorrect", correct, str(correct), EXPLORATORY, [MM_HELDOUT, MM_PRED], "held-out types predicted")
    n.add("nMMHeldout", len(held), str(len(held)), EXPLORATORY, [MM_HELDOUT], "held-out cases")

    # S1R coexisting period-2 branches
    for i, path in enumerate(S1R_TRACES):
        rows_ = load(path)["rows"]
        tag = "A" if i == 0 else "B"
        for kind, word in (("neimark_sacker", "NS"), ("fold", "Fold"), ("start_not_period_2", "NotTwo")):
            k = sum(r.get("type") == kind for r in rows_)
            n.add(f"nSOneRTrace{tag}{word}", k, str(k), EXPLORATORY, [path], f"S1R trace {tag}: {kind}")
        n.add(f"nSOneRTrace{tag}", len(rows_), str(len(rows_)), EXPLORATORY, [path], f"S1R trace {tag}: output rates")

    # Consequence pilot
    cons = load(CONS)
    s2 = cons["s2-28011-out-0.0856"]
    s1r = cons["s1r-28002-out-0.9413"]
    for label, data in (("STwo", s2), ("SOneR", s1r)):
        qp_hit, qp_n = data["loss_peak_matches_rotation_quasiperiodic"]
        lk_hit, lk_n = data["loss_peak_matches_rotation_locked"]
        n.add(f"nPeakQP{label}", qp_hit, str(qp_hit), EXPLORATORY, [CONS], "QP cells whose loss peak equals rotation")
        n.add(f"nPeakQP{label}Of", qp_n, str(qp_n), EXPLORATORY, [CONS], "QP cells")
        n.add(f"nPeakLock{label}", lk_hit, str(lk_hit), EXPLORATORY, [CONS], "locked cells whose loss peak equals rotation")
        n.add(f"nPeakLock{label}Of", lk_n, str(lk_n), EXPLORATORY, [CONS], "locked cells")
    n.add("nConsCells", s2["metadata"]["cells"], str(s2["metadata"]["cells"]), EXPLORATORY, [CONS], "S2 slice cells")
    beyond = s2["nested_models"]["repro_func_T40k_d1e-8"]["beyond_eos"]["rows"]
    n.add("nConsRows", beyond, str(beyond), EXPLORATORY, [CONS], "S2 cells beyond the edge used in nested models")
    gains = []
    for outcome, _ in CONS_OUTCOMES:
        m = s2["nested_models"][outcome]["beyond_eos"]
        gains.append((outcome, m["delta_r2"], m["delta_r2_ci"]))
    above = [g for g in gains if g[2][0] > 0]
    n.add("nConsAbove", len(above), str(len(above)), EXPLORATORY, [CONS],
          "outcomes whose block-bootstrap interval for the incremental R2 lies above zero (S2 slice)")
    n.add("nConsOutcomes", len(gains), str(len(gains)), EXPLORATORY, [CONS], "outcomes tested")
    pooled = load(CONS_POOLED)["pooled"]["nested_models"]
    slope_ = pooled["repro_slope_vs_delta"]["beyond_eos_slice_slopes_transient"]
    stop_ = pooled["stopping_time_spread"]["beyond_eos_slice_slopes"]
    for macro, m in (("DoseGain", slope_), ("StopGain", stop_)):
        n.add(f"x{macro}", m["delta_r2"], f"{m['delta_r2']:+.3f}", "adapted", [CONS_POOLED], f"pooled incremental R2 ({macro})")
        n.add(f"t{macro}CI", m["delta_r2_ci"], f"[{m['delta_r2_ci'][0]:+.3f}, {m['delta_r2_ci'][1]:+.3f}]", "adapted",
              [CONS_POOLED], f"its block-bootstrap interval ({macro})")
    for label, path in PRE.items():
        rows_ = [r for r in load(path)["rows"] if r.get("period2")]
        errs = []
        for r in rows_:
            for noise in ("noise_1e-07", "noise_1e-05"):
                if r.get(noise) and r[noise].get("ar_frequency") is not None:
                    errs.append(abs(r[noise]["ar_frequency"] - r["predicted_frequency"]))
        n.add(f"nAR{label}", len(rows_), str(len(rows_)), "adapted", [path], "period-2 cells in the AR precursor test")
        n.add(f"xARMedian{label}", float(np.median(errs)), sci(float(np.median(errs))), "adapted", [path],
              "median |AR angle - multiplier angle| over cells and noise levels")
        n.add(f"xARMax{label}", float(np.max(errs)), sci(float(np.max(errs))), "adapted", [path],
              "max |AR angle - multiplier angle|")

    fig6 = load(FIG6)["derived"]
    for key, name in (("B", "QP"), ("C", "Lock"), ("D", "Chaos")):
        n.add(f"xFigSix{name}Rho", fig6[key]["rho"], fixed(fig6[key]["rho"], 4), EXPLORATORY, [FIG6, STAIR_S2],
              f"figure 6 panel {key} rotation number (staircase receipt)")
        n.add(f"xFigSix{name}Ftle", fig6[key]["ftle"], fixed(fig6[key]["ftle"], 4), EXPLORATORY, [FIG6, STAIR_S2],
              f"figure 6 panel {key} FTLE (staircase receipt)")
    # Independently reviewed exploratory pilots, with raw/source bindings in receipts.
    scalar_path = c.ROOT / "docs/explore-minimal-model-scalar-receipt.json"
    scalar = load(scalar_path)
    for macro, value, computation in [
        ("nScalarNS", scalar["counts"]["neimark_sacker"], "counts.neimark_sacker"),
        ("nScalarCases", len(scalar["table"]), "len(table)"),
        ("nScalarFlips", scalar["counts"]["flip"], "counts.flip")]:
        n.add(macro, value, str(value), "exploratory; rung 0", [scalar_path], computation)
    for macro, key, digits in [("xScalarEta", "eta", 9), ("xScalarRotation", "rotation", 6),
                               ("xScalarCoeff", "coefficient", 6), ("xScalarAmp", "amplitude_exponent", 5)]:
        value = scalar["representative"][key]
        n.add(macro, value, fixed(value, digits), "exploratory; rung 0", [scalar_path], "representative." + key)
    warm_receipt = c.ROOT / "docs/warmup-intervention-receipt.json"
    warm_path = c.EXPLORE / "warmup-intervention/attempt-002/summary.json"
    expected = next(r["sha256"] for r in load(warm_receipt)["raw_files"] if r["path"] == "summary.json")
    assert c.sha256(warm_path) == expected
    warm = load(warm_path)
    for point, label in [(0, "Hard"), (1, "Soft")]:
        rows = [r for r in warm if r["point"] == point and r["start"] == "near_orbit"]
        common = [r for r in rows if r["policy"] == "keep_top_rates"]
        peaks = [r["common_after_crossing"]["peak_loss_ratio_to_frozen_orbit"] for r in common]
        n.add("tWarm" + label + "Peak", [min(peaks), max(peaks)], f"{min(peaks):.2f}--{max(peaks):.2f}",
              "exploratory; rung 1", [warm_receipt, warm_path], "near-orbit common postcrossing peak range over dwell speeds")
        halve = [r["after_intervention"]["terminal_loss_ratio_to_initial_orbit_mean"]
                 for r in rows if r["policy"] == "halve_both_rates"]
        assert len({f"{v:.3f}" for v in halve}) == 1
        n.add("xWarm" + label + "Halve", halve, f"{halve[0]:.3f}", "exploratory; rung 1",
              [warm_receipt, warm_path], "near-orbit halving final-loss ratios over dwell speeds; common three-decimal display")
    certificate_paths = [c.ROOT / "certificates" / ("c1-" + row + "-receipt.json") for row in ("row35", "row121")]
    certificates = [load(path) for path in certificate_paths]
    assert all(r["passed"] and r["evidence_rung"] == 3 for r in certificates)
    assert len({r["statement"]["primitive_period"] for r in certificates}) == 1
    assert len({r["attraction_witness_power"] for r in certificates}) == 1
    proof = "computer-assisted proof, rung 3; exploratory selection; exact-real map"
    for macro, value, calc in [("nCOneOrbits", len(certificates), "number of distinct authoritative orbit receipts"),
                               ("nCOnePeriod", certificates[0]["statement"]["primitive_period"], "shared primitive_period"),
                               ("nCOnePower", certificates[0]["attraction_witness_power"], "shared attraction_witness_power")]:
        n.add(macro, value, str(value), proof, certificate_paths, calc)
    for path, cert, label in zip(certificate_paths, certificates, ["Window", "Feasibility"]):
        bound = cert["attraction_norm_upper"]
        assert bound["rad"] == [0, 0]
        mantissa, exponent = bound["mid"]
        exact = Fraction(mantissa) * Fraction(2) ** exponent
        rounded = Fraction((exact.numerator * 1000 + exact.denominator - 1) // exact.denominator, 1000)
        assert exact < rounded < 1
        n.add("xCOne" + label + "Bound", str(rounded), f"{float(rounded):.3f}", proof, [path],
              "rigorous exact-dyadic monodromy-power upper bound rounded upward to three decimal places")
    # Full 72D exact-real orbit: bind both the proof and its separate saved-outcome audit.
    s1r_cert_path = c.ROOT / "certificates/s1r-period2-receipt.json"
    s1r_cert = load(s1r_cert_path)
    assert s1r_cert["passed"] and s1r_cert["evidence_rung"] == 3
    s1r_cert_raw = Path(s1r_cert["external_root"]) / "certificate.json"
    assert c.sha256(s1r_cert_raw) == s1r_cert["files"]["certificate.json"]["sha256"]
    assert load(s1r_cert_raw)["result"]["exact_statement"] == s1r_cert["statement"]
    proof_audit_root = c.DATA / "audits/s1r-period2/attempt-001"
    proof_audit_manifest = proof_audit_root / "manifest.json"
    assert c.sha256(proof_audit_manifest) == "81c708ca8fab815483136d72b13438fa7b6846ae34335e6808a334bad4f67281"
    proof_audit_path = proof_audit_root / "audit.json"
    assert c.sha256(proof_audit_path) == load(proof_audit_manifest)["files"]["audit.json"]["sha256"]
    proof_audit = load(proof_audit_path)
    assert proof_audit["status"] == "passed" and proof_audit["mathematical_certificate_passed"]
    assert proof_audit["selected_state_proximity_passed"] and Fraction(proof_audit["normalized_max_distance"]) <= Fraction(1, 10_000_000)
    proof_bindings_path = proof_audit_root / "stage-01-bindings.json"
    assert c.sha256(proof_bindings_path) == load(proof_audit_manifest)["files"]["stage-01-bindings.json"]["sha256"]
    assert c.sha256(s1r_cert_path) == load(proof_bindings_path)["nested_receipt"]["sha256"]
    proof_replay_path = proof_audit_root / "stage-02-verifier.json"
    assert c.sha256(proof_replay_path) == load(proof_audit_manifest)["files"]["stage-02-verifier.json"]["sha256"]
    proof_replay = load(proof_replay_path)
    assert proof_replay["passed"] and proof_replay["evidence_rung"] == 3
    assert proof_replay["certificate_sha256"] == c.sha256(s1r_cert_raw)
    assert proof_replay["statement"] == s1r_cert["statement"]
    proof_sources = [s1r_cert_path, s1r_cert_raw, proof_audit_manifest, proof_audit_path,
                     proof_bindings_path, proof_replay_path,
                     c.ROOT / "docs/reviews/s1r72-certificate-outcome-receipt.json"]
    for macro, value, calc in [
        ("nSOneRCertDimension", s1r_cert["statement"]["dimension"], "full-state dimension"),
        ("nSOneRCertPeriod", s1r_cert["statement"]["primitive_period"], "primitive period"),
        ("nSOneRCertPower", s1r_cert["attraction_witness_power"], "monodromy-power witness")]:
        n.add(macro, value, str(value), proof, proof_sources, calc)
    bound = s1r_cert["attraction_norm_upper"]
    assert bound["rad"] == [0, 0]
    exact = Fraction(bound["mid"][0]) * Fraction(2) ** bound["mid"][1]
    rounded = Fraction((exact.numerator * 1000 + exact.denominator - 1) // exact.denominator, 1000)
    assert exact < rounded < 1
    n.add("xSOneRCertBound", str(rounded), f"{float(rounded):.3f}", proof, proof_sources,
          "exact-dyadic monodromy-power norm upper bound rounded upward to three decimals")

    # Selected period-18 center: saved proof/review bindings only; no enclosure rerun.
    p18_path = c.ROOT / "certificates/s2-period18-receipt.json"
    p18_prod_path = c.ROOT / "docs/reviews/s2-period18-producer-receipt.json"
    p18_review_path = c.ROOT / "docs/reviews/s2-period18-independent-review.json"
    p18, p18_prod, p18_review = load(p18_path), load(p18_prod_path), load(p18_review_path)
    assert p18["passed"] and p18["evidence_rung"] == 3
    assert p18_review["verdict"] == "passed_for_stated_exact_real_orbit_claim"
    assert p18_review["result"]["status"] == "passed" and not p18_review["blocking_findings"]
    assert p18["statement"] == p18_prod["mathematical_statement"] == p18_review["result"]["statement"]
    assert p18["statement"]["primitive_period"] == 18 and p18["statement"]["dimension"] == 13
    assert all(p18["statement"][k] for k in ("attraction", "existence_and_local_uniqueness", "primitive_period_excluded_divisors"))
    assert c.sha256(p18_path) == p18_prod["canonical_receipt"]["sha256"] == p18_review["producer_c1_receipt"]["sha256"]
    assert p18["source_commit"] == p18_prod["execution_source_commit"] == p18_review["producer_source_commit"] == "15125d2b6b01a9fe94ce89f5aa10ec723ddc27c8"
    assert p18_review["review_execution_commit"] == "880a757a05025363829d24fefa08dff211b4fbd2"
    p18_sources = [p18_path, p18_prod_path, p18_review_path]
    def bound_inventory(mf, pin):
        assert c.sha256(mf) == pin
        members = load(mf); members = members.get("files", members)
        assert set(members) == {str(q.relative_to(mf.parent)) for q in mf.parent.rglob("*") if q.is_file() and q != mf}
        for name, ident in members.items():
            q = mf.parent / name
            assert q.is_file() and not q.is_symlink() and q.stat().st_size == ident["bytes"] and c.sha256(q) == ident["sha256"]
        return [mf] + [mf.parent / name for name in members]
    for field in ("producer_manifest", "review_manifest", "launcher_manifest"):
        ident = p18_review[field]
        p18_sources += bound_inventory(Path(ident["path"]), ident["sha256"])
    for inventory in (p18_prod["source_hashes"], p18_review["review_source_and_runtime"]["sources"]):
        for name, pin in inventory.items():
            q = c.ROOT / name
            assert c.sha256(q) == pin
            p18_sources.append(q)
    p18_raw = Path(p18["external_root"]) / "certificate.json"
    assert c.sha256(p18_raw) == p18["files"]["certificate.json"]["sha256"]
    assert load(p18_raw)["result"]["exact_statement"] == p18["statement"]
    assert p18_review["nested_proof"]["passed"] and p18_review["nested_proof"]["strictly_inside_original_box"]
    assert all(row["map_enclosed"] and row["jacobian_enclosed"] for row in p18_review["independent_AD"]["samples"])
    assert Fraction(p18_review["whole_box_proximity"]["exact"]) <= Fraction(p18_review["whole_box_proximity"]["threshold"])
    assert p18_prod["geometry_binding"]["summary"]["key"] == "S2-28011-zoom:587"
    assert p18_prod["proper_divisors"] == p18_review["stored_witness_checks"]["divisors"] == [1, 2, 3, 6, 9]
    assert p18["attraction_witness_power"] == p18_review["stored_witness_checks"]["attraction_witness_power"]
    bounds = [p18["attraction_norm_upper"], p18_review["stored_witness_checks"]["attraction_norm_upper"]]
    assert all(b["rad"] == [0, 0] for b in bounds)
    exact = max(Fraction(b["mid"][0]) * Fraction(2) ** b["mid"][1] for b in bounds)
    rounded = Fraction((exact.numerator * 1000 + exact.denominator - 1) // exact.denominator, 1000)
    assert exact < rounded < 1
    for macro, value, display, calc in [
        ("nPEighteenPeriod", p18["statement"]["primitive_period"], str(p18["statement"]["primitive_period"]), "selected anatomy-center primitive period"),
        ("nPEighteenPower", p18["attraction_witness_power"], str(p18["attraction_witness_power"]), "attraction monodromy-power witness"),
        ("xPEighteenBound", str(rounded), f"{float(rounded):.3f}", "maximum producer/review exact-dyadic power bound rounded upward to three decimals"),
        ("nCertifiedOrbits", len(certificates) + 2, str(len(certificates) + 2), "two distinct period-10 receipts plus one 72D period-2 and one period-18 receipt")]:
        n.add(macro, value, display, proof, p18_sources + certificate_paths + [s1r_cert_path], calc)

    # Local selected-ray fold: floating scalar diagnostics, not interval bifurcation proof.
    fold_path = c.ROOT / "docs/reviews/s2-period18-fold-outcome.json"
    fold = load(fold_path); fold_root = Path(fold["raw_root"])
    fold_sources = [fold_path]
    for root, pin in ((fold_root, fold["raw_manifest_sha256"]), (Path(fold["launcher_root"]), fold["launcher_manifest_sha256"])):
        fold_sources += bound_inventory(root / "manifest.json", pin)
    for name, pin in fold["source_sha256"].items():
        q = c.ROOT / name; assert c.sha256(q) == pin; fold_sources.append(q)
    fold_raw = load(fold_root / "outcome.json")
    assert c.sha256(fold_root / "outcome.json") == fold["outcome_sha256"]
    assert fold_raw["accepted"] == fold["accepted"] is True
    assert fold_raw["target"] == fold["target"] == "S2-28011-zoom:587" and fold["ray_index"] == 0
    assert fold_raw["period"] == fold["period"] == p18["statement"]["primitive_period"]
    assert fold_raw["solver"] == fold["solver"] and fold["solver"]["success"]
    for phase, raw_key in (("phase0", "diagnostics"), ("phase1", "next_phase_diagnostics")):
        q = fold[phase]; full = fold_raw[raw_key]
        assert all(full[k] == v for k, v in q.items())
        assert all(q["gates"].values()) and full["nondegenerate_fold_numerically_supported"]
        rows = q["finite_difference_series"]
        assert [row["exponent"] for row in rows] == [14, 16, 18, 20, 22]
        for key in ("a", "c"):
            left, right = rows[-2][key], rows[-1][key]
            assert left * right > 0 and abs(right) > 1e-7 and abs(left-right) / max(abs(left), abs(right)) < 1e-3
    assert fold_raw["proper_divisor_separations"] == fold["proper_divisor_separations"]
    assert set(fold["proper_divisor_separations"]) == {"1", "2", "3", "6", "9"} and min(fold["proper_divisor_separations"].values()) > 1e-6
    fold_review_path = c.ROOT / "docs/reviews/s2-period18-fold-independent-review.json"
    fold_review = load(fold_review_path)
    assert fold_review["verdict"] == "numerical_point_derivative_checks_passed"
    assert fold_review["producer_commit"] == fold["source_commit"]
    assert fold_review["producer_manifest"]["sha256"] == fold["raw_manifest_sha256"]
    assert fold_review["review_execution_commit"] == "3c723630e13b9c707a445f96180ac1d6640cf426"
    fold_sources.append(fold_review_path)
    for field in ("review_manifest", "launcher_manifest"):
        ident = fold_review[field]
        fold_sources += bound_inventory(Path(ident["path"]), ident["sha256"])
    ident = fold_review["result_identity"]
    assert c.sha256(Path(ident["path"])) == ident["sha256"]
    reviewed = load(ident["path"])
    assert {k: v for k, v in reviewed.items() if k != "retained_evaluations"} == fold_review["results"] and reviewed["passed"]
    assert reviewed["retained_evaluations"] == fold_review["retained_evaluations"]
    assert reviewed["new_eigen_SVD_solve_continuation_calls"] == 0
    for row in reviewed["phases"]:
        assert [d["exponent"] for d in row["fd"]] == [18, 20]
        original = fold["phase" + str(row["phase"])]["finite_difference_series"]
        for d in row["fd"]:
            match = next(q for q in original if q["exponent"] == d["exponent"])
            assert all(abs(d[k]-match[k]) < 1e-6 for k in ("a", "c"))
    for name, pin in fold_review["source_and_runtime"]["sources"].items():
        q = c.ROOT / name; assert c.sha256(q) == pin; fold_sources.append(q)
    for macro, value, display, calc in [
        ("xPEighteenFoldA", fold["phase0"]["finite_difference_series"][-1]["a"], fixed(fold["phase0"]["finite_difference_series"][-1]["a"], 6), "phase0 w^T R_s; smallest saved finite-difference step; normalized historical-ray coordinate"),
        ("xPEighteenFoldC", fold["phase0"]["finite_difference_series"][-1]["c"], fixed(fold["phase0"]["finite_difference_series"][-1]["c"], 6), "phase0 w^T D^2R[v,v], without factor 1/2; unit v and w^T v=1"),
        ("xPEighteenFoldOther", max(fold[q]["other_radius"] for q in ("phase0", "phase1")), fixed(max(fold[q]["other_radius"] for q in ("phase0", "phase1")), 4), "maximum remaining Floquet radius across two adjacent checked phases; floating diagnostic")]:
        n.add(macro, value, display, "exploratory numerical local fold diagnostics; no interval proof", fold_sources, calc)

    # Numerical association remains distinct from the exact-real orbit proof.
    bootstrap_path = c.ROOT / "docs/s1r-bootstrap-association-receipt.json"
    bootstrap = load(bootstrap_path)
    bootstrap_root = Path(bootstrap["raw_root"])
    bootstrap_raw = bootstrap_root / "result.json"
    assert c.sha256(bootstrap_root / "manifest.json") == bootstrap["raw_manifest"]["sha256"]
    assert c.sha256(bootstrap_raw) == bootstrap["raw_result"]["sha256"]
    br = load(bootstrap_raw)
    assert br == bootstrap["result"] and br["status"] == "completed_common_control_comparison"
    bootstrap_audit_root = c.DATA / "audits/s1r-bootstrap-association/attempt-001"
    bootstrap_audit_manifest = bootstrap_audit_root / "manifest.json"
    assert c.sha256(bootstrap_audit_manifest) == "12891e617dd05cdb7bb9d00b4dd8a25878fcd9cd08935433e8204831623f2f35"
    bootstrap_audit_path = bootstrap_audit_root / "result.json"
    assert c.sha256(bootstrap_audit_path) == load(bootstrap_audit_manifest)["files"]["result.json"]["sha256"]
    ba = load(bootstrap_audit_path)
    assert ba["status"] == "passed_saved_data_consistency"
    assert ba["collection_manifest_sha256"] == bootstrap["raw_manifest"]["sha256"]
    assert ba["witness_graph"]["association"] == br["association"]
    counts = {w["accepted"] for w in ba["warmups"]}
    assert len(counts) == 1 and len(ba["warmups"]) == 2
    assert all(w["attempted"] == w["accepted"] == w["completed"] for w in ba["warmups"])
    updates = counts.pop()
    links = ba["witness_graph"]["accepted_links"]
    assert links == ba["witness_graph"]["links"] == len(br["bridge"])
    association = br["association"]
    assert association["status"] == "same_pair_within_tolerance" and association["match"]["passed"]
    mismatch = min(association["match"]["costs"])
    bootstrap_sources = [bootstrap_path, bootstrap_raw, bootstrap_audit_manifest, bootstrap_audit_path,
                         c.ROOT / "docs/reviews/s1r-bootstrap-outcome-receipt.json"]
    label = "exploratory numerical association; saved-data audit; no historical full-state ancestry"
    for macro, value, display, calc in [
        ("nSOneRBootstrapUpdates", updates, integer(updates), "recorded accepted updates per bootstrap, counters checked; unsaved updates not replayed"),
        ("nSOneRBootstrapLinks", links, integer(links), "accepted witnessed hidden-rate bridge links"),
        ("xSOneRBootstrapMatch", mismatch, sci(mismatch, 2), "minimum normalized common-control full-pair discrepancy over cyclic phases")]:
        n.add(macro, value, display, label, bootstrap_sources, calc)
    lower_receipt_path = c.ROOT / "docs/reviews/s1r-lower-independent-outcome-receipt.json"
    lower_receipt = load(lower_receipt_path)
    lower_root = Path(lower_receipt["inputs"]["collection"])
    lower_manifest = lower_root / "manifest.json"
    assert c.sha256(lower_manifest) == lower_receipt["inputs"]["manifest_sha256"]
    lower_raw_path = lower_root / "result.json"
    assert c.sha256(lower_raw_path) == load(lower_manifest)["files"]["result.json"]["sha256"]
    lower = load(lower_raw_path)
    lower_audit_root = Path(lower_receipt["audit"]["path"])
    lower_audit_manifest = lower_audit_root / "manifest.json"
    lower_audit_path = lower_audit_root / "result.json"
    assert c.sha256(lower_audit_manifest) == lower_receipt["audit"]["manifest"]["sha256"]
    assert c.sha256(lower_audit_path) == lower_receipt["audit"]["result"]["sha256"]
    lower_audit = load(lower_audit_path)
    assert lower_audit["status"] == "passed_saved_data_consistency"
    assert lower_audit["collection_manifest_sha256"] == c.sha256(lower_manifest)
    assert lower["status"] == lower_audit["producer_status"] == "bridge_gap"
    assert lower["association"] is lower_audit["witness_graph"]["association"] is None
    completed = sum(row["status"] == "connected" for row in lower["bridge"])
    assert completed == lower["completed_macro_links"] == lower_receipt["outcome"]["completed_macro_segments"]
    assert len(lower["witnessed_edge_ids"]) == lower_audit["witness_graph"]["accepted_links"]
    assert lower["bridge"][-2]["endpoint"] == lower["last_witnessed_node"]
    assert lower["bridge"][-2]["target"] == lower_receipt["outcome"]["last_witnessed_controls"]
    assert lower["bridge"][-1]["target"] == lower_receipt["outcome"]["failed_macro_target"]
    assert lower["bridge"][-1]["halvings"] == lower_receipt["outcome"]["first_phase_corrector_failures"]
    lower_sources = [lower_receipt_path, lower_manifest, lower_raw_path, lower_audit_manifest, lower_audit_path]
    for macro, value, display, calc in [
        ("nSOneRLowerCompleted", completed, str(completed), "completed connected macro steps"),
        ("nSOneRLowerPlanned", len(lower["macro_targets"]), str(len(lower["macro_targets"])), "planned macro targets"),
        ("nSOneRLowerLinks", len(lower["witnessed_edge_ids"]), str(len(lower["witnessed_edge_ids"])), "accepted witnessed links including subdivisions"),
        ("nSOneRLowerHalvings", lower["bridge"][-1]["halvings"]-1, str(lower["bridge"][-1]["halvings"]-1), "actual halvings after initial failed level; historical field counts failed levels"),
        ("xSOneRLowerLast", lower["bridge"][-2]["target"][0], fixed(lower["bridge"][-2]["target"][0], 3), "last connected log hidden rate"),
        ("xSOneRLowerTarget", lower["bridge"][-1]["target"][0], fixed(lower["bridge"][-1]["target"][0], 3), "failed next macro target log hidden rate")]:
        n.add(macro, value, display, "exploratory finite witnessed path ending at a gap; independent saved-data audit", lower_sources, calc)

    # Fixed-state gap diagnosis: bind saved outcomes, not new model/SVD evaluations.
    gap_receipt_path = c.ROOT / "docs/reviews/s1r-lower-gap-outcome-receipt.json"
    gap_audit_receipt_path = c.ROOT / "docs/reviews/s1r-lower-gap-independent-outcome-receipt.json"
    gap_receipt, gap_audit_receipt = load(gap_receipt_path), load(gap_audit_receipt_path)
    assert gap_audit_receipt["status"] == "pass_saved_arithmetic_and_provenance"
    gap_sources = [gap_receipt_path, gap_audit_receipt_path]
    gap_root = Path(gap_receipt["raw"]["attempt-001"]["root"])
    gap_launcher_root = Path(gap_receipt["raw"]["attempt-001-launcher"]["root"])
    gap_audit_root = Path(gap_audit_receipt["audit_root"])
    gap_audit_launcher_root = Path(gap_audit_receipt["launcher_root"])
    for root, pinned in [
        (gap_root, gap_audit_receipt["producer_pins"]["raw_manifest_sha256"]),
        (gap_launcher_root, gap_audit_receipt["producer_pins"]["launcher_manifest_sha256"]),
        (gap_audit_root, gap_audit_receipt["raw_manifest_sha256"]),
        (gap_audit_launcher_root, gap_audit_receipt["launcher_manifest_sha256"])]:
        mf = root / "manifest.json"
        assert c.sha256(mf) == pinned
        entries = load(mf)
        entries = entries.get("files", entries)
        assert set(entries) == {p.name for p in root.iterdir()} - {"manifest.json"}
        assert all((root / name).is_file() and not (root / name).is_symlink()
                   and (root / name).stat().st_size == row["bytes"]
                   and c.sha256(root / name) == row["sha256"] for name, row in entries.items())
        gap_sources.append(mf)
    gap_path, gap_audit_path = gap_root / "result.json", gap_audit_root / "result.json"
    gap_checks_path, gap_provenance_path = gap_audit_root / "arithmetic-audit.json", gap_audit_root / "provenance.json"
    gap, gap_audit = load(gap_path), load(gap_audit_path)
    gap_checks = load(gap_checks_path)
    gap_sources.extend([gap_path, gap_audit_path, gap_checks_path, gap_provenance_path,
                        gap_audit_root / "historical-inventory.json", gap_audit_root / "record-audit.json",
                        gap_launcher_root / "receipt.json", gap_audit_launcher_root / "receipt.json"])
    assert c.sha256(gap_path) == gap_receipt["result_identity"]["sha256"]
    assert gap["source_commit"] == gap_receipt["execution_commit"] == gap_audit["pins"]["execution_commit"] == "8143a886534a6f3671f8893406bd66737e973922"
    assert gap["review_commit"] == gap_audit["pins"]["source_review_commit"] == "38d7141a08c07b2e3b9006492dd739f99a2fea43"
    assert gap_audit["audit_source_commit"] == gap_audit_receipt["audit_source_commit"] == "80972cb9e0311ddd07d54cdd95d3cd0dfced38d1"
    assert gap_audit_receipt["source_review_commit"] == "0e24ae8cc1f3ec4bb94e4d81e0a60716067b3e7b"
    assert gap["status"] == gap_audit["producer_status"] == "diagnostic_complete"
    assert gap_audit["status"] == "finished" and gap_audit["verdict"] == "saved_outcome_audit_pass"
    assert gap_audit["arithmetic_qualified"] and gap_audit["checks"] == gap_checks
    assert not gap_checks["unvalidated_partial_roles"] and len(gap_checks["validated_roles"]) == 6
    assert not gap["accepted_orbit"] and not gap["applied_correction"] and not gap["fold_verified"]
    for root, exit_key in [(gap_launcher_root, "process_exit"), (gap_audit_launcher_root, "exit_code")]:
        launcher = load(root / "receipt.json")
        assert launcher[exit_key] == 0 and not launcher["owned_live_pids_remaining"]
    for name, row in (gap["sources"] | gap_audit["audit_source_inventory"]).items():
        path = c.ROOT / name
        assert path.stat().st_size == row["bytes"] and c.sha256(path) == row["sha256"]
        gap_sources.append(path)
    gap_input_path = c.ROOT / "docs/plans/s1r-lower-gap-inputs.json"
    gap_inputs = load(gap_input_path)
    historical_root = Path(gap_inputs["collection_root"])
    historical_manifest = historical_root / "manifest.json"
    assert c.sha256(historical_manifest) == gap_inputs["collection_manifest"]["sha256"] == gap_audit_receipt["historical_inputs"]["manifest_sha256"]
    gap_sources.extend([gap_input_path, historical_manifest])
    historical_files = dict(gap_inputs["collection_metadata"])
    for name, row in gap_inputs["selected_records"].items():
        historical_files[name] = row["json"]
        historical_files[row["arrays_file"]] = row["arrays"]
    for name, row in historical_files.items():
        path = historical_root / name
        assert path.stat().st_size == row["bytes"] and c.sha256(path) == row["sha256"]
        gap_sources.append(path)
    gap_roles = gap_checks["roles"]
    assert set(gap_roles) == set(gap["completed_roles"])
    assert all(row["assembly_checked"] and len(row["checked_decompositions"]) == 4 for row in gap_roles.values())
    for role in ("predecessor_native_phase_0", "predecessor_native_phase_1", "last_native_phase_0", "last_native_phase_1"):
        assert gap_roles[role]["agreement"]["qualified_endpoint"]
        assert all(g["rank"]["status"] == "full_row_rank_resolved" for g in gap_roles[role]["geometry"].values())
    failed = gap_roles["nearest_failure_state"]
    assert not failed["agreement"]["qualified_endpoint"]
    equation = failed["linear_equation"]
    assert equation["verdict"] == gap["saved_linear_equation"]["verdict"] == "linear_equation_resolved"
    for key, value in gap["saved_linear_equation"].items():
        if isinstance(value, (float, int)):
            assert np.isclose(value, equation[key], rtol=1e-12, atol=0)
    trials = gap_checks["saved_trials"]
    assert trials == gap["saved_line_search"] == gap_audit_receipt["saved_trials"]
    assert [row["halving"] for row in trials] == list(range(13))
    assert all(not row["accepted"] and row["ratio"] > 1 and row["alpha"] == 2.**-row["halving"] for row in trials)
    raw_h = failed["geometry"]["raw_B"]["null_tangent"]["h_component"]
    scaled_h = failed["geometry"]["scaled_B"]["null_tangent"]["h_component"]
    assert raw_h > 0 > scaled_h
    gap_sources = list(dict.fromkeys(gap_sources))
    for macro, value, display, calc in [
        ("xSOneRGapForcing", equation["true_relative_forcing"], sci(equation["true_relative_forcing"], 3), "saved true Euclidean linear residual divided by nonlinear residual norm"),
        ("xSOneRGapBackward", equation["normwise_backward_error_frobenius"], sci(equation["normwise_backward_error_frobenius"], 2), "saved Frobenius normwise backward error of the local Newton equation"),
        ("nSOneRGapTrials", len(trials), integer(len(trials)), "all retained historical trial decisions; no new trial"),
        ("xSOneRGapMinRatio", min(row["ratio"] for row in trials), fixed(min(row["ratio"] for row in trials), 5), "minimum original infinity-norm residual ratio among all retained trials"),
        ("xSOneRGapCondition", failed["geometry"]["raw_A"]["rank"]["condition"], sci(failed["geometry"]["raw_A"]["rank"]["condition"], 2), "raw-coordinate condition estimate at the nonclosed failed state; not a fold gate")]:
        n.add(macro, value, display, "exploratory fixed-state diagnostic; independent saved-arithmetic audit using shared derivative inputs; no fold or continuation proof", gap_sources, calc)

    # Fixed curvature pilot: saved scalar decisions only, no model or decomposition rerun.
    curvature_receipt_path = c.ROOT / "docs/reviews/curvature-six-reference-outcome-receipt.json"
    curvature_receipt = load(curvature_receipt_path)
    curvature_sources = [curvature_receipt_path]
    for item in curvature_receipt["inventory"].values():
        root = Path(item["path"])
        assert c.sha256(root / "manifest.json") == item["manifest"]["sha256"]
        assert set(item["files"]) == {p.name for p in root.iterdir()}
        for name, pin in item["files"].items():
            path = root / name
            assert path.is_file() and not path.is_symlink()
            assert path.stat().st_size == pin["bytes"] and c.sha256(path) == pin["sha256"]
        members = load(root / "manifest.json")["files"]
        assert members == {k: v for k, v in item["files"].items() if k != "manifest.json"}
        curvature_sources.append(root / "manifest.json")
    curvature_root = Path(curvature_receipt["inventory"]["attempt-001-panel"]["path"])
    curvature = load(curvature_root / "result.json")
    assert curvature["source_commit"] == curvature_receipt["source_commit"] == "f0d556d475b1416809ff654e3e00dc28ab8deaf2"
    assert curvature["review_commit"] == curvature_receipt["review_commit"] == "fbd0bce0efcbf975f1361994c94b864f12e83333"
    assert curvature["status"] == "panel_complete"
    assert curvature["sources"] == curvature_receipt["sources"]
    for name, pin in curvature["sources"].items():
        path = c.ROOT / name
        assert path.stat().st_size == pin["bytes"] and c.sha256(path) == pin["sha256"]
        curvature_sources.append(path)
    rows, decision = curvature["slots"], curvature["aggregate"]
    assert rows == curvature_receipt["slots"] and decision == curvature_receipt["aggregate"]
    expected_slots = [r["slot"] for r in load(c.ROOT / "docs/plans/curvature-subspace-input-inventory.json")["panel"]]
    assert [r["slot"] for r in rows] == expected_slots and len(rows) == 6
    assert all(r["qualified"] and r["qualification"]["physical_passed"] and r["historical_spectrum"]["passed"] for r in rows)
    errors = [r["pair_error"] for r in rows]
    threshold = .02  # Fixed pair_accuracy threshold in the bound curvature_pilot source/card.
    assert all(np.isfinite(x) and x > threshold for x in errors)
    real_rows = [r for r in rows if r["stratum"] == "stable-real"]
    false_complex = sum(r["target_kind"] == "real_pair" and r["modeled_kind"] == "complex_conjugate_pair" for r in real_rows)
    assert len(real_rows) == false_complex == 2
    s1complex = next(r for r in rows if r["slot"] == "S1R-28002-stable-complex")
    assert s1complex["target_kind"] == "complex_conjugate_pair" and s1complex["modeled_kind"] == "real_pair"
    assert abs(sum(errors)/len(rows)-decision["mean_pair_error"]) < 1e-14
    assert abs(sum(r["commuting_error"] for r in rows)/len(rows)-decision["mean_oracle_commuting_error"]) < 1e-14
    assert decision["gates"] == dict(all_six_slots_qualified=True, angles=False, character=False, complex_attribution=False, mean_gain=True, pair_accuracy=False)
    assert not decision["passed"]
    for kind in ("attempt-001-profile-launcher", "attempt-001-panel-launcher"):
        root = Path(curvature_receipt["inventory"][kind]["path"])
        launcher = load(root / "receipt.json")
        assert launcher["successful"] and launcher["process_exit"] == 0 and not launcher["owned_live_pids_remaining"]
        curvature_sources.append(root / "receipt.json")
    curvature_sources += [curvature_root / "result.json", curvature_root / "runtime-start.json", curvature_root / "journal.jsonl"]
    slot_records = [r for r in (json.loads(line) for line in (curvature_root / "journal.jsonl").read_text().splitlines()) if r["kind"] == "slot_complete"]
    for record, row in zip(slot_records, rows, strict=True):
        path = curvature_root / record["file"]; saved = load(path)
        # Historical matching arrays use NPZ descriptors in the journal, lists in result.json.
        for key in ("slot", "qualified", "reasons", "qualification", "target_kind", "modeled_kind", "pair_error", "angle_error", "commuting_error", "attribution", "stratum"):
            assert saved["payload"][key] == row[key]
        assert saved["payload"]["historical_spectrum"]["passed"] == row["historical_spectrum"]["passed"]
        curvature_sources.append(path)
        if "arrays_file" in saved:
            curvature_sources.append(curvature_root / saved["arrays_file"])
    for macro, value, display, calculation in [
        ("nCurvatureSlots", len(rows), integer(len(rows)), "all fixed qualified references; phases are not replicates"),
        ("nCurvatureFalseComplex", false_complex, integer(false_complex), "real-control slots falsely predicted to have a complex leading pair"),
        ("xCurvatureMinError", min(errors), fixed(min(errors), 4), "minimum normalized unordered leading-pair error across all six slots"),
        ("xCurvatureMaxError", max(errors), fixed(max(errors), 4), "maximum normalized unordered leading-pair error across all six slots"),
        ("xCurvatureErrorGate", threshold, fixed(threshold, 2), "unchanged exploratory normalized pair-error threshold"),
        ("xCurvatureMeanError", decision["mean_pair_error"], fixed(decision["mean_pair_error"], 4), "equal-weight mean normalized leading-pair error across all slots"),
        ("xCurvatureComparator", decision["mean_oracle_commuting_error"], fixed(decision["mean_oracle_commuting_error"], 4), "mean of the better aligned/orthogonal commuting error at each fixed reference")]:
        n.add(macro, value, display, "exploratory fixed-panel approximation failure; producer outcome and saved scalar decision checks; no utility claim", curvature_sources, calculation)

    # Local pseudo-arclength: bind audited saved records; no new continuation or decomposition.
    pal_receipt_path = c.ROOT / "docs/reviews/s1r-pseudoarclength-outcome-receipt.json"
    pal_audit_receipt_path = c.ROOT / "docs/reviews/s1r-pseudoarclength-independent-outcome-receipt.json"
    pal_receipt, pal_ar = load(pal_receipt_path), load(pal_audit_receipt_path)
    assert pal_ar["status"] == "passed_saved_outcome_audit" and pal_ar["arithmetic_qualified"]
    pal_sources = [pal_receipt_path, pal_audit_receipt_path]
    pal_root = Path(pal_receipt["raw"]["attempt-001"]["root"])
    pal_launch = Path(pal_receipt["raw"]["attempt-001-launcher"]["root"])
    pal_audit_root = Path(pal_ar["inventories"]["attempt-001"]["root"])
    pal_audit_launch = Path(pal_ar["inventories"]["attempt-001-launcher"]["root"])
    for root, pin in [
        (pal_root, pal_ar["producer_pins"]["raw_manifest_sha256"]),
        (pal_launch, pal_ar["producer_pins"]["launcher_manifest_sha256"]),
        (pal_audit_root, pal_ar["inventories"]["attempt-001"]["files"]["manifest.json"]["sha256"]),
        (pal_audit_launch, pal_ar["inventories"]["attempt-001-launcher"]["files"]["manifest.json"]["sha256"])]:
        mf = root / "manifest.json"
        assert c.sha256(mf) == pin
        members = load(mf); members = members.get("files", members)
        assert set(members) == {p.name for p in root.iterdir()} - {"manifest.json"}
        assert all((root / name).is_file() and not (root / name).is_symlink()
                   and (root / name).stat().st_size == row["bytes"] and c.sha256(root / name) == row["sha256"]
                   for name, row in members.items())
        pal_sources.append(mf)
    pal, pal_audit = load(pal_root / "result.json"), load(pal_audit_root / "result.json")
    pal_checks = load(pal_audit_root / "arithmetic-audit.json")
    assert pal["source_commit"] == pal_receipt["execution_commit"] == pal_audit["pins"]["execution_commit"] == "cc3976e349cae1b9e050dd20ef32e5187657db50"
    assert pal["review_commit"] == pal_ar["producer_pins"]["source_review_commit"] == "9df5e138fbba1b283c18a8d47f49ebe9a69b15ae"
    assert pal_receipt["reviewed_source"] == pal_ar["producer_reviewed_source"] == "482cf9c7720936503658206f98c960f074aed847"
    assert pal_audit["audit_source_commit"] == pal_ar["audit_source_commit"] == "3134d730af88bf4591554946eeb715fb7419318e"
    assert pal_ar["audit_source_review"] == "f6a27d8b75165b64fda5b26e833659c553eb6e88"
    assert pal["status"] == pal_audit["producer_status"] == "two_edges_complete"
    assert pal_audit["status"] == "finished" and pal_audit["verdict"] == "saved_outcome_audit_pass"
    assert pal_audit["arithmetic_qualified"] and pal_checks == pal_audit["checks"] == pal_ar["checks"]
    assert pal_checks["arithmetic_complete"] and pal_checks["events"] == 930
    assert not any(pal_checks[k] for k in ("failed_gates", "interrupted_record_files", "unvalidated_terminal_return", "terminal_checkpoint_lag", "terminal_edge_advancement_lag"))
    assert pal_checks["pending_operation"] is None
    assert pal["operation_attempted"] == pal["operation_returned"] == pal_checks["attempted"] == pal_checks["returned"]
    assert pal["accepted_edges"] == pal_checks["accepted_edges"] == [0, 1] and pal["last_witnessed"] == "node-0005"
    assert pal["stage_counts"] == dict(assemblies=66, common=4, correctors=8, edges=2, qualifiers=9)
    assert not pal["fold_verified"] and not pal["lower_association"]
    for root, key in ((pal_launch, "process_exit"), (pal_audit_launch, "exit_code")):
        launcher = load(root / "receipt.json")
        assert launcher[key] == 0 and not launcher["owned_live_pids_remaining"]
        pal_sources.append(root / "receipt.json")
    for name, row in (pal["sources"] | pal_audit["audit_source_inventory"]).items():
        path = c.ROOT / name
        assert path.stat().st_size == row["bytes"] and c.sha256(path) == row["sha256"]
        pal_sources.append(path)
    pal_sources += [pal_root / "result.json", pal_root / "runtime-start.json", pal_root / "input-plan.json", pal_root / "journal.jsonl",
                    pal_audit_root / "result.json", pal_audit_root / "arithmetic-audit.json",
                    pal_audit_root / "provenance.json", pal_audit_root / "historical-input-inventory.json"]
    def pal_payload(name):
        path = pal_root / name
        record = load(path); pal_sources.append(path)
        arrays = {}
        if "arrays_file" in record:
            archive = pal_root / record["arrays_file"]
            assert c.sha256(archive) == record["arrays_sha256"]
            pal_sources.append(archive)
            with np.load(archive, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
        used = set()
        def decode(v):
            if isinstance(v, dict) and "array" in v:
                a = arrays[v["array"]]; used.add(v["array"])
                assert a.dtype.str == v["dtype"] and list(a.shape) == v["shape"] and v["order"] == "C"
                import hashlib
                assert hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() == v["sha256"]
                return a
            if isinstance(v, dict): return {k: decode(x) for k, x in v.items()}
            if isinstance(v, list): return [decode(x) for x in v]
            return v
        value = decode(record["payload"])
        assert used == set(arrays)
        return value
    pal_nodes = {}
    for row in pal_receipt["nodes"]:
        assert c.sha256(pal_root / row["source_file"]) == row["source_identity"]["sha256"]
        node = pal_payload(row["source_file"])
        assert node["accepted"] and node["carried_phase"] == row["carried_phase"] == 0
        assert node["h"] == row["h"] and float(node["tangent"][-1]) == row["tangent_h"]
        assert node["state_checks"]["passed"] and max(node["state_checks"]["closure"] + node["state_checks"]["transitions"]) <= 1
        assert node["state_checks"]["separation"] >= 1e-6 and node["spectrum_error"] <= 1e-7
        assert all(s["stability"] == "unstable" and s["radius"] > 1+1e-7 for s in node["spectra"])
        pal_nodes[node["id"]] = node
    assert list(pal_nodes) == pal_checks["qualified_nodes"]
    for row, audited in zip(pal_receipt["edges"], pal_ar["edges"], strict=True):
        edge = pal_payload(row["source_file"])
        assert edge == audited["payload"] and edge["accepted"] and len(edge["nodes"]) == 4
    for row in pal_receipt["common"]:
        common = pal_payload(row["source_file"])
        assert common["passed"] and common["match"]["passed"] and common["state_checks"]["passed"]
        assert common["h_difference"] <= 1e-10
    first, last, midpoint = (pal_nodes[k] for k in ("node-0001", "node-0005", "node-0006"))
    assert first["tangent"][-1] < 0 < last["tangent"][-1] and midpoint["tangent"][-1] < 0
    assert midpoint["h"] < min(first["h"], last["h"])
    assert midpoint["id"] in pal_checks["qualified_unwitnessed_nodes"]
    pal_sources = list(dict.fromkeys(pal_sources))
    pal_label = "exploratory local numerical turning candidate on an unstable cycle segment; independent saved-arithmetic audit; no fold proof"
    for macro, value, display, calc in [
        ("nSOneRPALLinks", len(pal["accepted_edges"]), str(len(pal["accepted_edges"])), "durably witnessed direct/half/reverse links"),
        ("nSOneRPALPairs", len(pal_nodes), str(len(pal_nodes)), "all qualified native pairs, including auxiliary witnesses"),
        ("nSOneRPALAux", len(pal_checks["qualified_unwitnessed_nodes"]), str(len(pal_checks["qualified_unwitnessed_nodes"])), "qualified auxiliary pairs distinct from primary witnessed nodes"),
        ("xSOneRPALFirstH", first["h"], fixed(first["h"], 8), "second-edge parent log hidden rate"),
        ("xSOneRPALLastH", last["h"], fixed(last["h"], 8), "second-edge direct endpoint log hidden rate"),
        ("xSOneRPALMidH", midpoint["h"], fixed(midpoint["h"], 8), "second-edge qualified midpoint witness log hidden rate"),
        ("xSOneRPALFirstT", float(first["tangent"][-1]), signed(first["tangent"][-1], 6), "carried unit tangent hidden component in fixed raw Euclidean metric at edge parent"),
        ("xSOneRPALLastT", float(last["tangent"][-1]), signed(last["tangent"][-1], 6), "consistently oriented carried unit tangent hidden component at direct endpoint"),
        ("xSOneRPALMidT", float(midpoint["tangent"][-1]), signed(midpoint["tangent"][-1], 6), "same-metric carried tangent at qualified midpoint witness"),
        ("xSOneRPALSeconds", pal["elapsed_seconds"], fixed(pal["elapsed_seconds"], 2), "one-thread producer worker elapsed seconds; includes this fixed pilot only"),
        ("xSOneRPALAuditSeconds", pal_audit["elapsed_seconds"], fixed(pal_audit["elapsed_seconds"], 2), "one-thread independent saved-arithmetic audit worker elapsed seconds")]:
        n.add(macro, value, display, pal_label, pal_sources, calc)

    ns_path = c.ROOT / "certificates/scalar-ns-receipt.json"
    ns = load(ns_path)
    assert ns["summary"]["passed"] and all(ns["summary"]["gates"].values())
    ns_raw = Path(ns["external_root"]) / "certificate.json"
    assert c.sha256(ns_raw) == ns["files"]["certificate.json"]["sha256"]
    ns_result = load(ns_raw)["result"]
    for macro, ball, digits, calc in [
        ("tNSProofEta", ns["summary"]["eta_box"], 14, "enclosed crossing rate"),
        ("tNSProofTrace", ns_result["local"]["trace"], 8, "complete planar trace"),
        ("tNSProofSlope", ns["summary"]["modulus_slope_at_root"], 7, "modulus derivative along orbit branch"),
        ("tNSProofCoeff", ns["summary"]["coefficient"]["ell"], 6, "normalized resolvent criticality coefficient")]:
        mid = Fraction(ball["mid"][0]) * Fraction(2) ** ball["mid"][1]
        rad = Fraction(ball["rad"][0]) * Fraction(2) ** ball["rad"][1]
        scale = 10 ** digits
        low, high = (mid-rad)*scale, (mid+rad)*scale
        lower = low.numerator // low.denominator
        upper = -((-high.numerator) // high.denominator)
        assert Fraction(lower, scale) <= mid-rad <= mid+rad <= Fraction(upper, scale)
        display = "[" + format(Decimal(lower)/Decimal(scale), f".{digits}f") + ", " + format(Decimal(upper)/Decimal(scale), f".{digits}f") + "]"
        n.add(macro, {"ball": ball, "lower": str(Fraction(lower,scale)), "upper": str(Fraction(upper,scale))},
              display, proof, [ns_path, ns_raw], calc + "; exact dyadic endpoints rounded outward")
    lm_path = c.ROOT / "docs/explore-tiny-lm-receipt.json"
    lm = load(lm_path)
    for macro, value, calc in [
        ("nLMParameters", lm["parameters"], "parameters"),
        ("nLMTokens", lm["tokens_per_update"], "tokens_per_update"),
        ("nLMCases", sum(lm["main_case_counts"].values()), "sum(main_case_counts)"),
        ("nLMCompleted", lm["main_case_counts"]["completed"], "main_case_counts.completed"),
        ("nLMGuard", lm["main_case_counts"]["numerical_guard_stop"], "main_case_counts.numerical_guard_stop"),
        ("nLMUpdates", max(r["completed_steps"] for r in lm["rows"]), "max completed_steps")]:
        n.add(macro, value, integer(value), "exploratory; rung 0", [lm_path], calc)
    s1r_path = c.ROOT / "docs/s1r-branch-association-receipt.json"
    s1r_audit_path = c.ROOT / "docs/reviews/s1r-branch-outcome-receipt.json"
    s1r, s1r_audit = load(s1r_path), load(s1r_audit_path)
    assert s1r["status"] == "completed_local_segment" and s1r_audit["status"] == "passed"
    s1r_manifest = Path(s1r["external_root"]) / "manifest.json"
    assert c.sha256(s1r_manifest) == s1r["raw_files"]["manifest.json"]["sha256"] == s1r_audit["raw_manifest_sha256"]
    assert all(p["stability"] == ["attracting", "attracting"] for p in s1r["main_points"])
    radii = [v for p in s1r["main_points"] for v in p["spectral_radii"]]
    for macro, value, tex, calc in [
        ("nSOneRLocalPoints", len(s1r["main_points"]), integer(len(s1r["main_points"])), "len(main_points)"),
        ("xSOneRLocalMinRadius", min(radii), fixed(min(radii), 6), "minimum primary-phase spectral radius"),
        ("xSOneRLocalMaxRadius", max(radii), fixed(max(radii), 6), "maximum primary-phase spectral radius")]:
        n.add(macro, value, tex, "exploratory; local numerical continuation; independently audited",
              [s1r_path, s1r_audit_path, s1r_manifest], calc)
    cifar_path = c.ROOT / "docs/cifar-linear-diagnostic-receipt.json"
    cifar_audit_path = c.ROOT / "docs/reviews/cifar-linear-diagnostic-outcome-review.json"
    cifar, cifar_audit = load(cifar_path), load(cifar_audit_path)
    cifar_manifest = Path(cifar["artifacts"]["directory"]) / "manifest.json"
    assert c.sha256(cifar_manifest) == cifar["artifacts"]["manifest_sha256"] == cifar_audit["manifest"]["sha256"]
    assert cifar_audit["status"] == "passed_saved_array_audit"
    assert not cifar["accepted_orbit"] and not cifar["candidate_trial_indices"]
    assert not cifar_audit["accepted_orbit"] and not cifar_audit["candidates"]
    assert abs(cifar["linear"]["true_forcing_ratio"] - cifar_audit["recomputed_linear"]["true_forcing_ratio"]) < 1e-12
    for macro, value, tex, calc in [
        ("xCifarForcing", cifar["linear"]["true_forcing_ratio"], fixed(cifar["linear"]["true_forcing_ratio"], 4), "linear.true_forcing_ratio"),
        ("xCifarForcingLimit", cifar["settings"]["forcing_limit"], fixed(cifar["settings"]["forcing_limit"], 1), "settings.forcing_limit"),
        ("xCifarBestClosure", min(t["relative_residual"] for t in cifar["trials"]), sci(min(t["relative_residual"] for t in cifar["trials"]), 2), "minimum trial relative_residual"),
        ("xCifarClosureGate", cifar["settings"]["closure_tolerance"], power(cifar["settings"]["closure_tolerance"]), "settings.closure_tolerance")]:
        n.add(macro, value, tex, "exploratory; failed correction diagnostic; independently audited saved arrays",
              [cifar_path, cifar_audit_path, cifar_manifest], calc)
    # A later matched restart diagnostic is distinct from the original50x4 attempt.
    restart_receipt_path = c.ROOT / "docs/reviews/cifar-restart-outcome-receipt.json"
    restart_audit_receipt_path = c.ROOT / "docs/reviews/cifar-restart-independent-outcome-receipt.json"
    restart_receipt, restart_audit_receipt = load(restart_receipt_path), load(restart_audit_receipt_path)
    assert restart_audit_receipt["status"] == "passed_saved_arithmetic_and_provenance"
    restart_sources = [restart_receipt_path, restart_audit_receipt_path]
    restart_roots = {}
    for label in ("producer", "producer_launcher", "audit", "audit_launcher"):
        artifact = restart_audit_receipt["artifacts"][label]
        root = Path(artifact["path"])
        mf = root / "manifest.json"
        assert c.sha256(mf) == artifact["manifest_sha256"]
        entries = load(mf)
        entries = entries.get("files", entries)
        assert set(entries) == {p.name for p in root.iterdir()} - {"manifest.json"}
        assert all((root / name).is_file() and not (root / name).is_symlink()
                   and (root / name).stat().st_size == row["bytes"]
                   and c.sha256(root / name) == row["sha256"] for name, row in entries.items())
        assert len(entries)+1 == artifact["file_count"]
        assert sum(row["bytes"] for row in entries.values())+mf.stat().st_size == artifact["bytes"]
        restart_sources.append(mf)
        restart_roots[label] = root
    restart_root, restart_audit_root = restart_roots["producer"], restart_roots["audit"]
    restart_raw_path = restart_root / "result.json"
    restart_audit_path = restart_audit_root / "result.json"
    restart_linear_path = restart_root / "linear.json"
    restart_checks_path = restart_audit_root / "arithmetic-audit.json"
    restart_events_path = restart_audit_root / "events-audit.json"
    restart_launcher_path = restart_roots["producer_launcher"] / "receipt.json"
    restart_audit_launcher_path = restart_roots["audit_launcher"] / "receipt.json"
    restart_sources.extend([restart_raw_path, restart_audit_path, restart_linear_path, restart_checks_path,
                           restart_events_path, restart_launcher_path, restart_audit_launcher_path])
    restart, restart_audit = load(restart_raw_path), load(restart_audit_path)
    restart_linear, restart_checks = load(restart_linear_path), load(restart_checks_path)
    restart_events = load(restart_events_path)
    restart_launcher, restart_audit_launcher = load(restart_launcher_path), load(restart_audit_launcher_path)
    assert restart_receipt["execution_commit"] == restart["source_commit"] == restart_audit["pins"]["execution_commit"] == "6fea29cda810ebb72c234dbd0fc1533bbca4e5cb"
    assert restart["review_commit"] == restart_audit["pins"]["source_review_commit"] == "1f4f2abe8b4d4d4b8ad6e50cf1c9416d568d65f9"
    assert restart_audit["audit_source"] == restart_audit_receipt["source_and_execution_commit"] == "4de644d379884ba85ce71bf39611c7be6ab90d10"
    assert restart_audit_receipt["audit_source_review"] == "fa4fafffa394f82701233fd4d7e2a4371eac3c17"
    assert c.sha256(restart_root / "manifest.json") == restart_receipt["raw"]["attempt-001"]["manifest"]["sha256"] == restart_audit["pins"]["raw_manifest_sha256"]
    assert c.sha256(restart_roots["producer_launcher"] / "manifest.json") == restart_audit["pins"]["launcher_manifest_sha256"]
    assert c.sha256(restart_raw_path) == restart_receipt["result_identity"]["sha256"]
    assert restart["status"] == "diagnostic_complete" and restart["comparison_qualified"]
    assert restart_audit["status"] == "finished" and restart_audit["verdict"] == "saved_outcome_audit_pass"
    assert restart_audit["checks"] == restart_checks and restart_checks["control_failures"] == []
    assert restart_launcher["successful"] and restart_launcher["process_exit"] == 0 and not restart_launcher["owned_live_pids_remaining"]
    assert restart_audit_launcher["exit_code"] == 0 and restart_audit_launcher["error"] is None and not restart_audit_launcher["owned_live_pids_remaining"]
    assert restart["linear"] == restart_linear
    assert restart_linear["inner_iterations"] == cifar["linear"]["inner_iterations"] == len(restart_events["callbacks"]) == 200
    assert (cifar["linear"]["restart"], cifar["linear"]["cycles"]) == (50, 4)
    assert (restart_linear["restart"], restart_linear["cycles"]) == (200, 1)
    assert restart_events["attempted"] == restart_events["returned"] == restart["operation_attempted"] == restart["operation_returned"]
    assert restart_events["matvecs"] == restart_linear["matvec_calls"] == 202 and not restart_events["pending_operations"]
    assert restart_checks["prefix"] == restart_events["prefix"] and restart_checks["prefix"]["passed"] and restart_checks["prefix"]["compared"] == 50
    assert all(v["passed"] and v["max_absolute_error"] == 0 for v in restart_checks["reference_parity"].values())
    assert np.isclose(restart_checks["comparison"]["baseline_forcing_ratio"], cifar["linear"]["true_forcing_ratio"], rtol=1e-12, atol=1e-20)
    assert np.isclose(restart_checks["linear"]["true_forcing_ratio"], restart_linear["true_forcing_ratio"], rtol=1e-12, atol=1e-20)
    assert restart_linear["forcing_limit"] == cifar["settings"]["forcing_limit"] == .5
    assert .5 < restart_checks["linear"]["true_forcing_ratio"] < restart_checks["comparison"]["baseline_forcing_ratio"]
    assert restart_linear["gmres_info"] == 1 and not restart_linear["requested_tolerance_met"] and not restart_checks["linear"]["forcing_test_pass"]
    assert not restart["candidate_trial_indices"] and not restart["accepted_orbit"] and not restart["applied_to_continuation"] and not restart["spectrum_requested"]
    assert [t["alpha"] for t in restart["trials"]] == restart["settings"]["alphas"] == [1., .5, .25, .125]
    assert len(restart["trials"]) == len(restart_checks["trials"]) == 4
    for trial, independent in zip(restart["trials"], restart_checks["trials"]):
        assert trial["finite"] and independent["finite"] and trial["merit_ratio"] > 1 and independent["merit_ratio"] > 1
        assert np.isclose(trial["merit_ratio"], independent["merit_ratio"], rtol=1e-12, atol=1e-20)
        assert not trial["merit_decreased"] and not trial["armijo_pass"] and not trial["closure_screen"]
        assert not trial["inexact_step_candidate"] and not independent["inexact_step_candidate"]
        assert not trial["accepted_orbit"] and not trial["applied_to_continuation"]
    for macro, value, display, calc in [
        ("nCifarBaselineRestart", cifar["linear"]["restart"], integer(cifar["linear"]["restart"]), "original GMRES restart length"),
        ("nCifarBaselineCycles", cifar["linear"]["cycles"], integer(cifar["linear"]["cycles"]), "original GMRES cycle budget"),
        ("nCifarRestartInner", restart_linear["inner_iterations"], integer(restart_linear["inner_iterations"]), "matched200 inner iterations, original and restart diagnostic"),
        ("nCifarRestartPrefix", restart_checks["prefix"]["compared"], integer(restart_checks["prefix"]["compared"]), "independently audited matching saved prefix length"),
        ("xCifarRestartForcing", restart_checks["linear"]["true_forcing_ratio"], fixed(restart_checks["linear"]["true_forcing_ratio"], 4), "independent saved-vector true forcing ratio"),
        ("nCifarRestartTrials", len(restart_checks["trials"]), integer(len(restart_checks["trials"])), "all fixed trials: finite, merit increased, no accepted candidate")]:
        n.add(macro, value, display, "exploratory matched restart diagnostic; independently audited saved arrays; correction gates failed",
              restart_sources + [cifar_path, cifar_audit_path, cifar_manifest], calc)

    # Later diagnostic outcomes are kept distinct from the failed correction.
    span_receipt_path = c.ROOT / "docs/reviews/cifar-input-span-outcome-receipt.json"
    span_receipt = load(span_receipt_path)
    span_root = Path(span_receipt["inventories"]["attempt-001"]["root"])
    span_manifest = span_root / "manifest.json"
    span_raw_path = span_root / "result.json"
    assert c.sha256(span_manifest) == span_receipt["inventories"]["attempt-001"]["manifest"]["sha256"]
    assert c.sha256(span_raw_path) == span_receipt["result"]["sha256"] == load(span_manifest)["result.json"]["sha256"]
    span = load(span_raw_path)
    span_audit_root = c.DATA / "audits/cifar-input-span-independent/attempt-001"
    span_audit_manifest = span_audit_root / "manifest.json"
    span_audit_path = span_audit_root / "result.json"
    assert c.sha256(span_audit_manifest) == "0221596b14e01b6630f0d4014d5f34d29d23aa6011e9397ab7b9ee1d1097d82d"
    assert c.sha256(span_audit_path) == load(span_audit_manifest)["result.json"]["sha256"]
    span_audit = load(span_audit_path)
    assert span_audit["verdict"] == "saved_outcome_audit_pass"
    assert span_audit["pins"]["target_manifest_sha256"] == c.sha256(span_manifest)
    assert span["all_numerical_gates_passed"] and span_audit["all_numerical_gates_passed"]
    assert span["prioritize_reduced_solver"] is span_audit["prioritize_reduced_solver"] is False
    span_fraction = span["original_complementary_squared_residual_fraction"]
    assert abs(span_fraction-span_audit["original_complementary_squared_residual_fraction"]) <= 2e-14
    assert span_fraction < span["settings"]["triage_fraction"]
    span_sources = [span_receipt_path, span_manifest, span_raw_path, span_audit_manifest, span_audit_path,
                    c.ROOT / "docs/reviews/cifar-input-span-independent-outcome-receipt.json"]
    for macro, value, display, calc in [
        ("nCifarSpanColumns", span["settings"]["retained_columns"], integer(span["settings"]["retained_columns"]), "retained thin-QR columns, no rank claim"),
        ("nCifarFullDimension", span["dimensions"]["full"], integer(span["dimensions"]["full"]), "original full-state dimension"),
        ("nCifarActiveDimension", span["dimensions"]["active"], integer(span["dimensions"]["active"]), "coordinates after input-orthogonal decomposition"),
        ("xCifarComplementFraction", span_fraction, sci(span_fraction, 2), "original residual complementary squared-norm fraction"),
        ("xCifarSpanPriority", span["settings"]["triage_fraction"], fixed(span["settings"]["triage_fraction"], 1), "unchanged inclusive prioritization criterion")]:
        n.add(macro, value, display, "exploratory saved-state decomposition; independently audited; negative prioritization result", span_sources, calc)

    dose_receipt_path = c.ROOT / "docs/reviews/utility-dose-outcome-receipt.json"
    dose_receipt = load(dose_receipt_path)
    dose_root = Path(dose_receipt["raw"]["root"])
    dose_manifest = dose_root / "manifest.json"
    dose_raw_path = dose_root / "result.json"
    assert c.sha256(dose_manifest) == dose_receipt["raw"]["manifest"]["sha256"]
    dose_entries = load(dose_manifest)["files"]
    assert c.sha256(dose_raw_path) == dose_entries["result.json"]["sha256"] == dose_receipt["result"]["sha256"]
    dose = load(dose_raw_path)
    dose_audit_root = c.DATA / "audits/utility-dose-feasibility/attempt-001"
    dose_audit_manifest = dose_audit_root / "manifest.json"
    dose_audit_path = dose_audit_root / "result.json"
    assert c.sha256(dose_audit_manifest) == "4589dc9afeaa098537341adf27ae230a64110a8fdaf048baf7f8fef22c6d19be"
    assert c.sha256(dose_audit_path) == load(dose_audit_manifest)["files"]["result.json"]["sha256"]
    dose_audit = load(dose_audit_path)
    assert dose_audit["status"] == "audit_passed"
    assert dose_audit["raw_manifest_sha256"] == c.sha256(dose_manifest)
    assert dose["candidate_feasible"] is dose_audit["fixed_candidate_feasible"] is False
    assert dose["gradient_returns"] == dose_audit["reference_gradient_returns"] == len(dose["forks"]) == 4
    dose_prediction_paths = [dose_root/name for name in sorted(dose_entries) if name.endswith("-dose_prediction.json")]
    assert len(dose_prediction_paths) == 4
    dose_predictions = []
    for index, path in enumerate(dose_prediction_paths):
        assert c.sha256(path) == dose_entries[path.name]["sha256"]
        row = load(path)["payload"]
        assert row["index"] == index
        pred = row["prediction"]
        assert pred["status"] == dose_audit["forks"][index]["status"] == "infeasible"
        assert pred["reason"] == dose_audit["forks"][index]["reason"] == "fraction_cap"
        assert pred["fractions"]["output"] > pred["maximum_group_fraction"]
        dose_predictions.append(pred)
    uniform = {v["uniform_fraction"] for v in dose_predictions}
    cap = {v["maximum_group_fraction"] for v in dose_predictions}
    assert len(uniform) == len(cap) == 1
    uniform, cap = uniform.pop(), cap.pop()
    fractions = [v["fractions"]["output"] for v in dose_predictions]
    dose_sources = [dose_receipt_path, dose_manifest, dose_raw_path, dose_audit_manifest, dose_audit_path,
                    c.ROOT / "docs/reviews/utility-dose-independent-outcome-receipt.json"] + dose_prediction_paths
    for macro, value, display, calc in [
        ("nDoseForks", len(dose["forks"]), str(len(dose["forks"])), "saved states with exactly one gradient and no update"),
        ("xDoseUniform", uniform, fixed(uniform, 2), "declared uniform fractional rate reduction"),
        ("xDoseCap", cap, fixed(cap, 2), "declared maximum block fraction"),
        ("tDoseOutputFractions", fractions, ", ".join(fixed(v, 3) for v in fractions), "required output-only rate reductions for matched sham-relative dose, fixed fork order")]:
        n.add(macro, value, display, "exploratory algebraic feasibility; four independent analytic gradients; no intervention outcome", dose_sources, calc)

    broader_path = c.ROOT / "docs/explore-tiny-lm-broader-outcome-receipt.json"
    broader_audit_path = c.ROOT / "docs/reviews/tiny-lm-broader-outcome-audit.json"
    broader, broader_audit = load(broader_path), load(broader_audit_path)
    assert broader_audit["status"] == "passed" and broader_audit["arithmetic_passed"] and broader_audit["sampled_features_passed"]
    assert not broader["positive_forecast_interpretation"] and not broader_audit["positive_forecast_interpretation"]
    broader_analysis = Path(broader["analysis"]["path"])
    broader_recomputed = Path(broader_audit["raw_root"]) / "recomputed-analysis.json"
    assert c.sha256(broader_analysis) == broader["analysis"]["sha256"] == broader_audit["analysis_sha256"]
    assert c.sha256(broader_recomputed) == next(row["sha256"] for row in broader_audit["files"] if row["path"] == broader_recomputed.name)
    assert c.sha256(Path(broader["manifest"]["path"])) == broader["manifest"]["sha256"] == broader_audit["collection_manifest_sha256"]
    original, replay = load(broader_analysis), load(broader_recomputed)
    for macro, family in [("tLMBroaderBThree", "B3"), ("tLMBroaderBTwo", "B2"), ("tLMBroaderOrdinary", None)]:
        values = []
        for seed in ("28161", "28162"):
            selected = family or original["seeds"][seed]["selected_ordinary"]
            assert original["seeds"][seed]["selected_ordinary"] == replay["seeds"][seed]["selected_ordinary"]
            value = original["seeds"][seed]["scores"][selected]["256"]["test"]["normalized_mse"]
            audited = replay["seeds"][seed]["scores"][selected]["256"]["test"]["normalized_mse"]
            assert abs(value-audited) < 1e-10
            values.append(value)
        n.add(macro, values, ",\\; ".join(fixed(v, 4) for v in values),
              "exploratory; one fixed text pool, two initializations; independently audited saved-data comparison",
              [broader_path, broader_audit_path, broader_analysis, broader_recomputed],
              "late expanded-arm normalized MSE in seed order28161/28162; " + (family or "tune-selected ordinary comparator"))
    return n


def design_table():
    """Design table TeX from the frozen plan and the frozen analysis summary."""
    plan, summary = load(PLAN), load(SUMMARY)
    systems = {}
    for m in plan["maps"]:
        systems.setdefault(m["system"], []).append(m)
    zooms = {k.split("-")[0] for k in summary["maps"] if k.endswith("-zoom")}
    spans = {round(v["log10_hidden_range"][1] - v["log10_hidden_range"][0], 4) for v in summary["maps"].values()}
    data = {"S1": "standard-normal", "S1R": "standard-normal", "S2": "from a tanh teacher"}
    rows = []
    for system, maps in systems.items():
        net = maps[0]["network"]
        seeds = ", ".join(str(m["seed"]) for m in maps)
        decay = "0" if net["ridge"] == 0 else f"${power(net['ridge'])}$"
        arch = (f"{net['d']} inputs, {net['width']} tanh units, " + ("biases" if net["bias"] else "no biases")
                + (", mean-field" if net["in_scale"] != 1.0 else ""))
        maps_text = f"{len(maps)} coarse" + (f", {len(maps)} zoom" if system in zooms else "")
        rows.append(f"{system} & {maps[0]['n_parameters']} & {maps[0]['n_examples']} {data[system]} & {arch} & "
                    f"{decay} & {seeds} & {maps_text} \\\\")
    it = plan["iteration"]
    resolution = plan["maps"][0]["log10_hidden"]["resolution"]
    lines = [
        "% Generated by paper/gd-route/scripts/build_numbers.py from the frozen plan and summary; do not edit.",
        "\\begin{table}[htbp]", "\\centering\\small", "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{@{}l r >{\\raggedright\\arraybackslash}p{2.6cm} >{\\raggedright\\arraybackslash}p{3.6cm} l "
        ">{\\raggedright\\arraybackslash}p{1.4cm} >{\\raggedright\\arraybackslash}p{1.7cm}@{}}", "\\toprule",
        "Network & Parameters & Training points & Architecture & Weight decay & Seeds & Maps \\\\", "\\midrule",
        *rows, "\\bottomrule", "\\end{tabular}",
        f"\\caption{{\\textbf{{Design of the map study.}} Every map has ${resolution}\\times{resolution}$ "
        f"logarithmically spaced pairs of hidden- and output-layer rates; coarse maps span {max(spans):g} decades in "
        f"each rate and zoom maps {min(spans):.2f} decades. Every cell runs {integer(it['steps'])} full-batch updates in "
        f"{it['precision']} from its seed's initial parameters. S1 is the published configuration of the "
        f"fractal-trainability study and is reported descriptively.}}",
        "\\label{tab:design}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def consequence_table():
    """Consequence table TeX: per-regime medians and incremental R2 on the S2 slice (exploratory)."""
    s2 = load(CONS)["s2-28011-out-0.0856"]
    counts = s2["regime_counts"]
    head = " & ".join(label for _, label in CONS_REGIMES)
    sizes = " & ".join(f"({counts[key]} cells)" for key, _ in CONS_REGIMES)
    lines = [
        "% Generated by paper/gd-route/scripts/build_numbers.py from the consequence receipts; do not edit.",
        "\\begin{table}[htbp]", "\\centering\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{4.3cm} rrrr r@{}}", "\\toprule",
        f"Outcome (median per regime) & {head} & $\\Delta R^2$ [95\\%] \\\\",
        f" & {sizes} & \\\\", "\\midrule"]
    for outcome, label in CONS_OUTCOMES:
        per = s2["per_regime"][outcome]
        cells = " & ".join(f"${per[key]['median']:.2f}$" if key in per else "--" for key, _ in CONS_REGIMES)
        m = s2["nested_models"][outcome]["beyond_eos"]
        gain = "$" + signed(m["delta_r2"]) + "$ [$" + signed(m["delta_r2_ci"][0]) + "$, $" + signed(m["delta_r2_ci"][1]) + "$]"
        lines.append(f"{label} & {cells} & {gain} \\\\")
    rows = s2["nested_models"][CONS_OUTCOMES[0][0]]["beyond_eos"]["rows"]
    lines += [
        "\\bottomrule", "\\end{tabular}",
        f"\\caption{{\\textbf{{Consequences per regime (exploratory, S2-28011, one slice of "
        f"{s2['metadata']['cells']} cells).}} Columns give medians per regime (cell counts in parentheses). "
        f"$\\Delta R^2$ is the cross-validated gain from adding locking and rotation features (locked and "
        f"quasi-periodic indicators, rotation number, distance to the nearest rational, $1/q$) to a base model of "
        f"FTLE, sharpness, rate and oscillation amplitude, over the {rows} cells beyond the edge of stability, with "
        f"moving-block bootstrap intervals. Regime differences are large; the incremental value of rotation "
        f"information is not.}}",
        "\\label{tab:consequence}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def render(n):
    lines = ["% Generated by paper/gd-route/scripts/build_numbers.py from saved evidence; do not edit by hand.",
             "% Each macro's source, SHA-256 and computation are listed in data/numbers.json."]
    lines += [f"\\newcommand{{\\{e['macro']}}}{{{e['tex']}}}" for e in n.entries]
    return "\n".join(lines) + "\n"


def manifest(n):
    sources = sorted({s for e in n.entries for s in e["sources"]})
    return {"generator": str(Path(__file__).resolve().relative_to(c.ROOT)),
            "generator_sha256": c.sha256(__file__),
            "sources": {s: c.sha256(located(s)) for s in sources},
            "numbers": n.entries}


def index():
    figures = sorted((PAPER / "figures").glob("*.provenance.json"))
    return {"manuscript": "paper/gd-route/main.tex",
            "numbers_manifest": {"path": "data/numbers.json", "sha256": c.sha256(MANIFEST)},
            "figure_receipts": [{"path": f"figures/{p.name}", "sha256": c.sha256(p),
                                 "evidence_label": load(p)["evidence_label"]} for p in figures],
            "verification": "python paper/gd-route/scripts/verify.py"}


def main():
    n = build()
    tex, data, table, consequences = render(n), manifest(n), design_table(), consequence_table()
    if VERIFY:
        problems = []
        recorded = load(MANIFEST)
        if json.dumps(recorded["numbers"], sort_keys=True) != json.dumps(data["numbers"], sort_keys=True):
            problems.append("recomputed numbers differ from data/numbers.json")
        if recorded["sources"] != data["sources"]:
            problems.append("a source file changed since data/numbers.json was written")
        if recorded["generator_sha256"] != data["generator_sha256"]:
            problems.append("generator changed since data/numbers.json was written")
        if TEX.read_text() != tex:
            problems.append("generated-numbers.tex differs from the recomputed macros")
        if TABLE.read_text() != table:
            problems.append("generated-design-table.tex differs from the recomputed table")
        if CONS_TABLE.read_text() != consequences:
            problems.append("generated-consequence-table.tex differs from the recomputed table")
        if load(INDEX) != index():
            problems.append("provenance.json index is stale")
        if problems:
            raise SystemExit("numbers: VERIFY FAILED: " + "; ".join(problems))
        print(json.dumps({"numbers": len(n.entries), "verified": True}))
        return
    MANIFEST.parent.mkdir(exist_ok=True)
    TEX.write_text(tex)
    TABLE.write_text(table)
    CONS_TABLE.write_text(consequences)
    MANIFEST.write_text(json.dumps(data, indent=1) + "\n")
    INDEX.write_text(json.dumps(index(), indent=1) + "\n")
    print(json.dumps({"numbers": len(n.entries), "tex": str(TEX)}))


if __name__ == "__main__":
    main()
