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
        ("xPEighteenBound", str(rounded), f"{float(rounded):.3f}", "maximum producer/review exact-dyadic power bound rounded upward to three decimals")]:
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

    # Exact local period-18 fold: saved proof/review only, no kernel execution.
    local_path = c.ROOT / "certificates/s2-period18-fold-receipt.json"
    local_review_path = c.ROOT / "docs/reviews/s2-period18-fold-certificate-independent-review.json"
    local, local_review = load(local_path), load(local_review_path)
    local_root, local_audit = Path(local["external_root"]), Path(local_review["raw_root"])
    local_sources = [local_path, local_review_path]
    assert local["passed"] and local["evidence_rung"] == 3 and all(local["gates"].values())
    assert local_review["status"] == "passed"
    assert local["execution_commit"] == local_review["producer_adaptive_execution_commit"] == "c37dcb4e56a64fa7567cc2e03fe5917195a9f0fc"
    assert local_review["review_execution_commit"] == "f2284ebb5d5aa8cfa816da65ad15d04a2ec38224"
    local_sources += bound_inventory(local_root / "manifest.json", local["manifest"]["sha256"])
    local_sources += bound_inventory(local_audit / "manifest.json", local_review["raw_manifest_sha256"])
    assert load(local_root / "manifest.json")["files"] == local["files"]
    assert load(local_audit / "manifest.json")["files"] == local_review["file_inventory"]
    for inventory in (local["source_sha256"], local_review["review_source_sha256"]):
        for name, pin in inventory.items():
            q = c.ROOT / name; assert c.sha256(q) == pin; local_sources.append(q)
    run = local_root / "run"
    local_raw = load(run / "certificate.json")
    lr = local_raw["result"]
    assert lr["passed"] and lr["gates"] == local["gates"] and not lr["failures"]
    assert lr["settings"] == local["settings"] == dict(max_power=4096, newton_steps=5, precision=256, radius_exponent=-100)
    assert lr["exact_statement"] == local["exact_statement"] and lr["exact_statement"]["period"] == 18 and lr["exact_statement"]["dimension"] == 13
    assert local_raw["runtime"]["source_commit"] == local["execution_commit"]
    inp = load(run / "inputs.json")
    assert c.sha256(run / "inputs.json") == local_raw["inputs_sha256"]
    assert inp["map_inputs"]["lr_output"] == local["fixed_output_rate"] == "1848862009810323/2251799813685248"
    assert inp["map_inputs"]["network"] == dict(activation="tanh", bias=True, d=2, in_scale="1/1", out_scale="1/1", loss_scale="1/2", ridge="1152921504606847/1152921504606846976", width=3)
    assert [d["divisor"] for d in lr["divisors"]] == local["excluded_proper_divisors"] == [1, 2, 3, 6, 9]
    assert all(d["excluded"] for d in lr["divisors"])
    # Bind the preserved first attempt and the subsequent review's exact replay.
    prior_path = c.ROOT / local["prior_attempt_receipt"]["path"]
    assert c.sha256(prior_path) == local["prior_attempt_receipt"]["sha256"]
    local_sources.append(prior_path)
    prior_root = local_root.parent / "attempt-001-target"
    pin = local_review["bound_producer_attempts"]["attempt-001-target"]
    local_sources += bound_inventory(prior_root / "manifest.json", pin["manifest_sha256"])
    prior = load(prior_root / "run/certificate.json")["result"]
    assert prior["gates"] == local_review["preserved_failed_attempt"]["gates"]
    assert not prior["gates"]["other_modes_attracting"] and sum(prior["gates"].values()) == 6
    assert prior["settings"] == dict(max_power=4096, newton_steps=5, precision=256, radius_exponent=-80)
    reviewed = local_review["review"]
    assert reviewed["byte_identical_replay"] and reviewed["source_and_exact_input_binding"]
    assert load(local_audit / "run/result.json")["status"] == "passed"
    assert load(local_audit / "run/04-replay-result.json") == lr
    assert load(local_audit / "run/05-replay-summary.json")["byte_identical"]
    nested = load(local_audit / "run/08-nested-result.json")
    ns = load(local_audit / "run/09-nested-summary.json")
    assert nested["passed"] and all(nested["gates"].values()) and ns["passed"] and ns["strictly_nested"]
    assert nested["settings"] == ns["settings"] == reviewed["nested_proof"]["settings"] == dict(max_power=4096, newton_steps=5, precision=384, radius_exponent=-110)
    assert reviewed["nested_proof"]["strictly_inside_original_27d_box"]
    for name, raw_ball in (("hidden_rate", lr["box"][13]), ("a", lr["a"]),
                           ("quadratic_normal_form_coefficient_c_over_2", lr["quadratic_normal_form_coefficient_c_over_2"]),
                           ("complement_power_norm", lr["power_bounds"][-1]["norm_inf_upper"])):
        assert local["bounds"][name]["ball"] == raw_ball
    def ball_endpoints(ball):
        mid = Fraction(ball["mid"][0]) * Fraction(2) ** ball["mid"][1]
        rad = Fraction(ball["rad"][0]) * Fraction(2) ** ball["rad"][1]
        assert rad >= 0
        return mid - rad, mid + rad
    def exact_decimal(value, digits):
        # value is an integer number of decimal ticks: no float formatting.
        sign = "-" if value < 0 else ""
        value = abs(value)
        return f"{sign}{value // 10**digits}.{value % 10**digits:0{digits}d}"
    label = "rung 3 local exact-real primitive18 fold; adaptive radius amendment; independently reviewed shared Arb kernel"
    for macro, name, digits in (("tFoldProofEta", "hidden_rate", 14), ("tFoldProofA", "a", 12),
                                ("tFoldProofQuadratic", "quadratic_normal_form_coefficient_c_over_2", 9)):
        lo, hi = ball_endpoints(local["bounds"][name]["ball"])
        assert [lo, hi] == [Fraction(local["bounds"][name]["exact_lower"]), Fraction(local["bounds"][name]["exact_upper"])]
        scale = 10**digits
        flo = lo.numerator * scale // lo.denominator
        cei = -((-hi.numerator * scale) // hi.denominator)
        assert Fraction(flo, scale) <= lo <= hi <= Fraction(cei, scale)
        text = "[" + exact_decimal(flo, digits) + ", " + exact_decimal(cei, digits) + "]"
        n.add(macro, [str(lo), str(hi)], text, label, local_sources, "serialized dyadic ball endpoints rounded outward by exact Fraction floor/ceiling")
    output = Fraction(local["fixed_output_rate"])
    n.add("tFoldProofOutput", str(output), rf"\frac{{{output.numerator}}}{{2^{{51}}}}", label, local_sources, "fixed exact output-rate dyadic; denominator verified as 2^51")
    assert output.denominator == 2**51
    fold_power = lr["complement_witness_power"]
    assert fold_power == local["complement_witness_power"] == lr["power_bounds"][-1]["power"]
    _, exact = ball_endpoints(local["bounds"]["complement_power_norm"]["ball"])
    upper = -((-exact.numerator * 1000) // exact.denominator)
    assert exact < Fraction(upper, 1000) < 1
    n.add("nFoldProofPower", fold_power, str(fold_power), label, local_sources, "deflated-monodromy repeated-squaring witness power")
    n.add("xFoldProofBound", str(Fraction(upper, 1000)), exact_decimal(upper, 3), label, local_sources, "exact dyadic deflated-power norm bound rounded upward to three decimals")

    # Exact log-ray period-18 flip: source-bound saved enclosures, no proof rerun.
    flip_path = c.ROOT / "certificates/s2-period18-flip-receipt.json"
    flip = load(flip_path)
    flip_root = Path(flip["external_root"])
    flip_sources = [flip_path] + bound_inventory(flip_root / "manifest.json", flip["manifest"]["sha256"])
    assert load(flip_root / "manifest.json")["files"] == flip["files"]
    assert flip["passed"] and flip["evidence_rung"] == 3 and all(flip["gates"].values())
    assert flip["execution_commit"] == "e840d9357de9de30648f44084920cbdfa45d3f10"
    assert flip["criticality"] == "subcritical" and flip["attempt_count"] == 1
    for name, pin in flip["source_sha256"].items():
        q = c.ROOT / name; assert c.sha256(q) == pin; flip_sources.append(q)
    flip_run = flip_root / "run"
    flip_raw = load(flip_run / "certificate.json")
    fr = flip_raw["result"]
    assert c.sha256(flip_run / "certificate.json") == flip["certificate_sha256"]
    assert c.sha256(flip_run / "manifest.json") == flip["run_manifest_sha256"]
    assert c.sha256(flip_run / "inputs.json") == flip_raw["inputs_sha256"] == flip["inputs_sha256"]
    assert fr["passed"] and fr["gates"] == flip["gates"] and not fr["failures"]
    assert fr["settings"] == flip["settings"] == dict(max_power=4096, newton_steps=5, precision=256, radius_exponent=-100)
    assert fr["exact_statement"] == flip["exact_statement"]
    assert fr["criticality"] == flip["criticality"] and fr["doubled_side_in_s"] == flip["doubled_side_in_s"]
    assert flip_raw["runtime"]["source_commit"] == flip["execution_commit"]
    assert [d["divisor"] for d in fr["divisors"]] == flip["excluded_proper_divisors"] == [1, 2, 3, 6, 9]
    assert all(d["excluded"] for d in fr["divisors"])
    flip_input = load(flip_run / "inputs.json")
    assert flip_input["rate_family"] == flip["rate_family"]
    assert flip_input["map_inputs"]["network"] == inp["map_inputs"]["network"]
    assert all(flip_input["map_inputs"][k] == inp["map_inputs"][k] for k in ("x", "y"))
    flip_review_path = c.ROOT / "docs/reviews/s2-period18-flip-independent-review.json"
    flip_review = load(flip_review_path); flip_sources.append(flip_review_path)
    assert flip_review["status"] == "passed"
    assert flip_review["producer_execution_commit"] == flip["execution_commit"]
    assert flip_review["producer_source_commit"] == flip["reviewed_source_commit"]
    assert flip_review["review_source_commit"] == "2c20281c439a1d233cee3344cc85484106009010"
    for name, pin in flip_review["source_hashes"].items():
        q = c.ROOT / name; assert c.sha256(q) == pin; flip_sources.append(q)
    for name, pin in flip_review["producer_pins"].items():
        path = flip_root.parent / name
        flip_sources += bound_inventory(path / "manifest.json", pin["manifest_sha256"])
        assert c.sha256(path / "run/manifest.json") == pin["run_manifest_sha256"]
    for attempt in flip_review["attempts"]:
        flip_sources += bound_inventory(Path(attempt["manifest"]["path"]), attempt["manifest"]["sha256"])
        assert load(Path(attempt["path"]) / "run/result.json") == attempt["result"]
    first_review, final_review = flip_review["attempts"]
    assert first_review["exit_code"] == 2 and first_review["result"]["evaluations_retained"] == 0
    assert first_review["result"]["status"] == "failed_or_incomplete"
    assert final_review["exit_code"] == 0 and final_review["result"]["status"] == "passed"
    flip_audit = Path(final_review["path"]) / "run"
    assert load(flip_audit / "04-replay-result.json") == fr
    assert flip_review["checks"]["byte_identical_original_replay"]
    assert flip_review["checks"]["full_exact_source_and_input_binding"]
    assert load(flip_audit / "06-saved-arithmetic.json") == flip_review["checks"]["saved_ball_arithmetic"]
    assert Fraction(flip_review["checks"]["exact_whole_box_proximity"]["normalized_distance"]) == Fraction(flip["whole_box_proximity"]["normalized_max_distance"])
    flip_nested = load(flip_audit / "08-nested-result.json")
    flip_nested_summary = load(flip_audit / "09-nested-summary.json")
    assert flip_nested["passed"] and all(flip_nested["gates"].values())
    assert flip_nested_summary["passed"] and flip_nested_summary["strictly_nested"]
    assert c.sha256(flip_audit / "08-nested-result.json") == flip_review["checks"]["nested_proof"]["result_sha256"]
    assert flip_nested["settings"] == flip_nested_summary["settings"] == dict(max_power=4096, newton_steps=5, precision=384, radius_exponent=-110)
    assert len(fr["box"]) == len(flip_nested["box"]) == 27
    for outer, inner in zip(fr["box"], flip_nested["box"]):
        olo, ohi = ball_endpoints(outer); ilo, ihi = ball_endpoints(inner)
        assert olo < ilo <= ihi < ohi
    flip_label = "rung 3 local exact-real subcritical primitive18 flip; exploratory selection; independently reviewed shared Arb kernel"
    for macro, name, ball, digits in (
            ("tFlipProofS", "ray_parameter", fr["box"][13], 16),
            ("tFlipProofAlpha", "alpha", fr["alpha"], 7),
            ("tFlipProofCubic", "cubic_b", fr["cubic_b"], 6)):
        assert flip["bounds"][name]["ball"] == ball
        lo, hi = ball_endpoints(ball)
        assert [lo, hi] == [Fraction(flip["bounds"][name]["exact_lower"]), Fraction(flip["bounds"][name]["exact_upper"])]
        scale = 10**digits
        flo, cei = lo.numerator * scale // lo.denominator, -((-hi.numerator * scale) // hi.denominator)
        assert Fraction(flo, scale) <= lo <= hi <= Fraction(cei, scale)
        text = "[" + exact_decimal(flo, digits) + ", " + exact_decimal(cei, digits) + "]"
        n.add(macro, [str(lo), str(hi)], text, flip_label, flip_sources, "serialized dyadic ball endpoints rounded outward by exact Fraction floor/ceiling")
    assert ball_endpoints(fr["alpha"])[1] < 0 and ball_endpoints(fr["cubic_b"])[1] < 0
    for macro, field, index_ in (("tFlipEllHidden", "ell", 0), ("tFlipEllOutput", "ell", 1),
                                ("tFlipDirHidden", "direction", 0), ("tFlipDirOutput", "direction", 1)):
        value = Fraction(flip["rate_family"][field][index_])
        den = value.denominator
        assert den > 0 and den & (den - 1) == 0
        tex = rf"\frac{{{value.numerator}}}{{2^{{{den.bit_length()-1}}}}}"
        n.add(macro, str(value), tex, flip_label, flip_sources, "exact dyadic historical log-ray coefficient; direction is the recorded rounded binary64 product")
    flip_power = fr["complement_witness_power"]
    assert flip_power == flip["complement_witness_power"] == fr["power_bounds"][-1]["power"]
    flip_bound_ball = fr["power_bounds"][-1]["norm_inf_upper"]
    assert flip["bounds"]["complement_power_norm"]["ball"] == flip_bound_ball
    _, flip_exact_bound = ball_endpoints(flip_bound_ball)
    flip_upper = -((-flip_exact_bound.numerator * 1000000) // flip_exact_bound.denominator)
    assert flip_exact_bound < Fraction(flip_upper, 1000000) < 1
    n.add("nFlipProofPower", flip_power, str(flip_power), flip_label, flip_sources, "deflated-monodromy repeated-squaring witness power")
    n.add("xFlipProofBound", str(Fraction(flip_upper, 1000000)), exact_decimal(flip_upper, 6), flip_label, flip_sources, "exact dyadic deflated-power norm bound rounded upward to six decimals")

    # Ray51: distinguish local exponential-family flip from fixed-dyadic36 map.
    r51_paths = {"flip": c.ROOT / "certificates/s2-ray51-flip-receipt.json", "period36": c.ROOT / "certificates/s2-period36-receipt.json"}
    r51_review_path = c.ROOT / "docs/reviews/s2-ray51-certificates-independent-review.json"
    r51_review = load(r51_review_path); r51_audit = Path(r51_review["raw_root"])
    r51_sources = [r51_review_path] + list(r51_paths.values())
    assert r51_review["status"] == "passed" and r51_review["review_source_commit"] == "888d2531fd1360700d5e1e8627769a4c6db79e4c"
    assert r51_review["producer_source_commit"] == "04048350fdaff3bba5a5d9e82b059f10761a0dab"
    r51_sources += bound_inventory(r51_audit / "manifest.json", r51_review["manifest_sha256"])
    assert load(r51_audit / "manifest.json")["files"] == r51_review["file_inventory"]
    assert load(r51_audit / "run/result.json") == r51_review["runtime"]
    for name, pin in r51_review["source_sha256"].items():
        q = c.ROOT / name; assert c.sha256(q) == pin; r51_sources.append(q)
    r51_proofs, r51_inputs, r51_receipts = {}, {}, {}
    for kind, path in r51_paths.items():
        receipt = load(path); r51_receipts[kind] = receipt
        assert receipt["kind"] == kind and receipt["evidence_rung"] == 3
        assert receipt["source_commit"] == receipt["execution_commit"] == r51_review["producer_source_commit"]
        for name, pin in receipt["source_hashes"].items():
            q = c.ROOT / name; assert c.sha256(q) == pin; r51_sources.append(q)
        for item, review_key in (("profile", "profiles"), ("target", "targets")):
            bundle = receipt[item]; br = Path(bundle["root"])
            assert bundle["manifest"]["sha256"] == r51_review["pins"][review_key][kind]["manifest_sha256"]
            assert str(br) == r51_review["pins"][review_key][kind]["root"]
            r51_sources += bound_inventory(br / "manifest.json", bundle["manifest"]["sha256"])
            assert c.sha256(br / "run/manifest.json") == bundle["run_manifest"]["sha256"] == r51_review["pins"][review_key][kind]["run_manifest_sha256"]
            assert load(br / "run/outcome.json") == bundle["outcome"] and bundle["outcome"]["passed"]
        target = Path(receipt["target"]["root"]) / "run"
        env = load(target / "certificate.json"); proof51 = env["result"]; input51 = load(target / "inputs.json")
        assert c.sha256(target / "certificate.json") == receipt["target"]["certificate"]["sha256"]
        assert c.sha256(target / "inputs.json") == env["inputs_sha256"] == receipt["target"]["inputs"]["sha256"]
        assert env["kind"] == kind and env["runtime"]["source_commit"] == receipt["execution_commit"]
        assert proof51["passed"] and proof51["settings"] == receipt["settings"]
        assert env["whole_box_proximity"] == receipt["whole_box_proximity"]
        assert load(r51_audit / f"run/{kind}-03-replay.json") == proof51
        stronger51 = load(r51_audit / f"run/{kind}-06-nested.json")
        summary51 = load(r51_audit / f"run/{kind}-07-summary.json")
        assert summary51 == r51_review["results"][kind]["summary"]
        assert summary51["passed"] and summary51["byte_identical_replay"] and summary51["strictly_nested"]
        assert stronger51["passed"] and stronger51["settings"] == summary51["nested_settings"]
        assert c.sha256(r51_audit / f"run/{kind}-06-nested.json") == summary51["nested_sha256"]
        for outer, inner in zip(proof51["box"], stronger51["box"]):
            olo, ohi = ball_endpoints(outer); ilo, ihi = ball_endpoints(inner); assert olo < ilo <= ihi < ohi
        assert input51["provenance"] == receipt["input_provenance"]
        r51_proofs[kind], r51_inputs[kind] = proof51, input51
    sf, sp = r51_receipts["flip"], r51_proofs["flip"]
    p36, op36 = r51_receipts["period36"], r51_proofs["period36"]
    assert all(sp["gates"].values()) and sp["gates"] == sf["gates"] and sp["criticality"] == sf["criticality"] == "supercritical"
    assert sp["doubled_side_in_s"] == sf["doubled_side_in_s"] == "greater"
    assert sf["exact_family"] == r51_inputs["flip"]["rate_family"] and sf["exact_family"]["ell"] == flip["rate_family"]["ell"]
    assert [d["divisor"] for d in sp["divisors"]] == [1, 2, 3, 6, 9] and all(d["excluded"] for d in sp["divisors"])
    assert [d["divisor"] for d in op36["divisors"]] == p36["proper_divisors"] == [1, 2, 3, 4, 6, 9, 12, 18]
    assert all(d["excluded"] for d in op36["divisors"]) and op36["exact_statement"] == p36["exact_statement"]
    label51 = "rung 3 local supercritical18 flip and separate fixed-dyadic attracting36; independently reviewed shared kernels"
    for macro, name, ball, digits in (("tSuperFlipS", "parameter_s", sp["box"][13], 16), ("tSuperFlipAlpha", "alpha", sp["alpha"], 8), ("tSuperFlipCubic", "cubic_b", sp["cubic_b"], 6)):
        assert sf[name]["ball"] == ball
        lo, hi = ball_endpoints(ball); assert (lo, hi) == (Fraction(sf[name]["lower"]), Fraction(sf[name]["upper"]))
        scale = 10**digits; flo = lo.numerator * scale // lo.denominator; cei = -((-hi.numerator * scale) // hi.denominator)
        n.add(macro, [str(lo), str(hi)], "[" + exact_decimal(flo, digits) + ", " + exact_decimal(cei, digits) + "]", label51, r51_sources, "exact dyadic endpoints rounded outward by Fraction floor/ceiling")
    assert ball_endpoints(sp["alpha"])[1] < 0 and ball_endpoints(sp["cubic_b"])[0] > 0
    for macro, value in (("tSuperFlipDirHidden", sf["exact_family"]["direction"][0]), ("tSuperFlipDirOutput", sf["exact_family"]["direction"][1]), ("tPThirtySixHidden", p36["exact_fixed_rates"]["hidden"]), ("tPThirtySixOutput", p36["exact_fixed_rates"]["output"])):
        v = Fraction(value); assert v.denominator & (v.denominator - 1) == 0
        n.add(macro, str(v), rf"\frac{{{v.numerator}}}{{2^{{{v.denominator.bit_length()-1}}}}}", label51, r51_sources, "exact dyadic family direction or fixed actual rate; conventions distinguished in text")
    assert r51_inputs["period36"]["lr_hidden"] == p36["exact_fixed_rates"]["hidden"] and r51_inputs["period36"]["lr_output"] == p36["exact_fixed_rates"]["output"]
    for prefix, proof51, witness in (("SuperFlip", sp, sf["complement"]), ("PThirtySix", op36, p36["attraction"])):
        last = proof51["power_bounds"][-1]; assert last["power"] == witness["power"] and last["norm_inf_upper"] == witness["norm_upper"]
        _, upper51 = ball_endpoints(witness["norm_upper"]); ticks51 = -((-upper51.numerator * 1000000) // upper51.denominator)
        assert upper51 < Fraction(ticks51, 1000000) < 1
        n.add("n" + prefix + "Power", witness["power"], str(witness["power"]), label51, r51_sources, "bounded repeated-squaring monodromy/complement power witness")
        n.add("x" + prefix + "Bound", str(Fraction(ticks51, 1000000)), exact_decimal(ticks51, 6), label51, r51_sources, "exact dyadic power bound rounded upward to six decimals")
    n.add("nPThirtySixPeriod", op36["exact_statement"]["primitive_period"], "36", label51, r51_sources, "primitive period after excluding all eight proper divisors")

    # Saved-only binding of the joined interior branch and its separate stability cover.
    joined_path = c.ROOT / "certificates/s2-period18-connection-receipt.json"
    joined_review_path = c.ROOT / "docs/reviews/s2-period18-joined-branch-independent-review.json"
    joined, jr = load(joined_path), load(joined_review_path)
    assert joined["producer_passed"] and jr["passed"] and jr["review_status"] == "passed"
    js, rs = joined["statement"], jr["exact_statement"]
    assert js["period"] == rs["period"] == 18 and js["phase_dimension"] == rs["phase_dimension"] == 13
    assert js["shooting_dimension"] == rs["shooting_dimension"] == 234
    assert js["continuous_branch"] and js["primitive_period_throughout"] and js["full_state_attraction_throughout"]
    assert rs["uniform_attraction"] and rs["uniform_primitive_period"] and rs["same_original_center_orbit"]
    assert not rs["boundary_connection_proved"] and not rs["full_shrimp_geometry_proved"]
    assert js["new_individual_orbit_certificates"] == 0 and rs["proper_divisors_excluded"] == [1,2,3,6,9]
    assert js["fixed_output_rate"] == rs["fixed_output_rate"] == local["fixed_output_rate"]
    ends = [Fraction(v) for v in js["exact_hidden_rate_interval"]]
    assert js["exact_hidden_rate_interval"] == rs["hidden_rate_interval"] and ends[0] < ends[1]
    width = ends[1] - ends[0]
    assert width == Fraction(js["exact_interval_width"]) == Fraction(rs["hidden_rate_interval_width"]) == Fraction(1,1024)
    assert ends[0] < Fraction(js["original_anchor_rate"]) < ends[1]
    assert rs["original_anchor_coordinates_checked"] == rs["shared_endpoint_coordinates_checked"] == 234
    joined_sources = [joined_path, joined_review_path]
    for component in ("existence_primitive_join", "stability"):
        item = joined[component]; path = c.ROOT / item["receipt"]
        assert c.sha256(path) == item["receipt_identity"]["sha256"]
        joined_sources.append(path)
    assert c.sha256(c.ROOT / joined["anchor"]["canonical_receipt"]) == joined["anchor"]["sha256"] == jr["producer_original_receipt"]["sha256"]
    for root, pin in [(Path(joined["existence_primitive_join"]["external_root"]), jr["wider_charts"]["run_manifest"]),
                      (Path(joined["stability"]["external_root"]), jr["sliced_stability"]["run_manifest"]),
                      (Path(jr["audit"]["root"]), jr["audit"]["manifest"]["sha256"]),
                      (Path(jr["audit"]["root"]) / "run", jr["audit"]["run_manifest"]["sha256"])]:
        manifest = root / "manifest.json"; assert c.sha256(manifest) == pin
        joined_sources.append(manifest)
    jout_path = Path(jr["audit"]["root"]) / "run/outcome.json"; assert c.sha256(jout_path) == jr["audit"]["outcome"]["sha256"]
    jout = load(jout_path); assert jout["passed"] and jout["exact_interval"] == js["exact_hidden_rate_interval"]
    sliced_path = Path(joined["stability"]["external_root"]) / "result.json"
    assert c.sha256(sliced_path) == joined["stability"]["result"]["sha256"]
    sliced = load(sliced_path); assert sliced["passed"] and sliced["parent_certificate_sha256"] == joined["existence_primitive_join"]["certificate_sha256"]
    assert sliced["connected_parameter_endpoints"] == js["exact_hidden_rate_interval"]
    units = sliced["units"]; assert len(units) == joined["stability"]["periodic_metric_witnesses"] == jr["sliced_stability"]["complete_slice_count"] == 32
    assert all(u["passed"] and u["kind"] == "slice" for u in units)
    intervals = [[Fraction(x) for x in unit["exact_endpoints"]] for unit in units]
    assert intervals[0][0] == ends[0] and intervals[-1][1] == ends[1]
    assert all(a < b for a,b in intervals) and all(a[1] == b[0] for a,b in zip(intervals, intervals[1:]))
    assert jout["slice_model_steps"] == jr["sliced_stability"]["independent_native_derivative_calls"] == len(units)*18
    joined_sources.extend([jout_path, sliced_path])
    joined_label = "rung 3 uniformly attracting primitive18 interior branch; independently reviewed shared analytic map/Arb"
    def fraction_tex(v):
        return rf"\frac{{{v.numerator}}}{{{v.denominator}}}"
    n.add("tConnectedInterval", [str(v) for v in ends], "[" + ", ".join(fraction_tex(v) for v in ends) + "]", joined_label, joined_sources, "exact common-parameter joined shooting domains, original anchor and independent review agree")
    n.add("tConnectedWidth", str(width), fraction_tex(width), joined_label, joined_sources, "exact Fraction upper-minus-lower endpoint")
    n.add("nConnectedSlices", len(units), str(len(units)), joined_label, joined_sources, "all passed closed stability slices, exact adjacent endpoint coverage and independent derivative review")

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
    # Independent saved-data perturbation response, not an online forecast.
    lmrp=c.ROOT/"docs/reviews/tiny-lm-response-outcome.json"
    lmra=c.ROOT/"docs/reviews/tiny-lm-response-independent.json"
    lmp,lma=load(lmrp),load(lmra)
    lms=[lmrp,lmra]+bound_inventory(Path(lmp["raw_root"])/"manifest.json",lmp["outer_manifest_sha256"])
    lms+=bound_inventory(Path(lma["raw_root"])/"manifest.json",lma["outer"]["manifest_sha256"])
    lmo=load(Path(lmp["raw_root"])/"run/result.json");lmao=load(Path(lma["raw_root"])/"run/result.json")
    assert lmo==lmp["outcome"] and lmao==lma["result"] and lmao["passed"] and lmo["narrow_frozen_curvature_inadequacy"] and lmao["narrow_frozen_curvature_inadequacy"]
    assert lmao["rows_recomputed"]==24 and lmo["model_parameters"]==15776
    for arm,label in [(8,"Small"),(256,"Expanded")]:
        selected=[]
        for seed in [28161,28162]:
            prod=next(x for x in lmo["cases"]if x["seed"]==seed and x["arm"]==arm)
            aud=next(x for x in lmao["cases"]if x["seed"]==seed and x["arm"]==arm)
            row=next(x for x in prod["rows"]if x["direction"]=="gradient"and x["horizon"]==16)
            auditrow=next(x for x in aud["rows"]if x["direction"]=="gradient"and x["horizon"]==16)
            assert row["mechanism_qualified"] and auditrow["mechanism_qualified"] and row["training_ce_slope_resolved"] and auditrow["training_ce_slope_resolved"]
            for key in ["response_gain","frozen_vector_error","transported_vector_error"]:assert abs(row[key]-auditrow[key])<1e-8*max(1,abs(row[key]))
            selected.append(auditrow)
        for suffix,key in [("Gain","response_gain"),("Frozen","frozen_vector_error"),("Transport","transported_vector_error")]:
            vals=[x[key]for x in selected]
            display=r",\; ".join(fixed(x,3)if suffix=="Gain"else sci(x,2)for x in vals)
            n.add("tLMResponse"+label+suffix,vals,display,"exploratory four consumed checkpoints; independent saved arithmetic, shared native derivatives; no prospective forecast or intervention",lms,"gradient direction16-step response in seed order28161/28162; "+key)
    # Reviewed local attraction probes and separate Clark-method development pilots.
    locpath = c.ROOT / "docs/reviews/s2-period18-localization-independent-review.json"
    loc = load(locpath); assert loc["passed"] and loc["checks"]["original_model_anchor_and_parent_joins"]
    finite = [v for u in loc["units"] if u["mode"] == "probe" for v in u["probes"]]
    assert len(finite) == loc["checks"]["actual_finite_intervals"] == 2 and all(v["passed"] for v in finite)
    widths = [Fraction(v["exact_endpoints"][1])-Fraction(v["exact_endpoints"][0]) for v in finite]
    assert set(widths) == {Fraction(loc["checks"]["finite_interval_exact_width"])}
    n.add("nLocalizedIntervals", len(finite), str(len(finite)), proof, [locpath], "independently checked finite outer attraction intervals")
    n.add("tLocalizedWidth", str(widths[0]), str(widths[0]), proof, [locpath], "exact endpoint difference of each of the two outer intervals")
    responsepath = c.ROOT / "docs/reviews/clark-response-outcome-receipt.json"
    response = load(responsepath); rsroot = Path(response["raw_summary_root"])
    rsources = [responsepath] + bound_inventory(rsroot/"manifest.json", response["raw_summary_manifest"]["sha256"])
    summary = load(rsroot/"summary.json")
    assert summary["rows"] == summary["solve_qualified"] == response["numerical_solve_rows"] == response["solve_residual_screen_passes"]
    n.add("nClarkResponseSolves", summary["rows"], str(summary["rows"]), EXPLORATORY, rsources, "all regularized direct solve rows, both saved13D cocycles")
    limits = {v["maxiter"] for v in summary["matrixfree"]}
    assert len(limits)==1 and all(v["istop"]==7 and v["iterations"]==v["maxiter"] for v in summary["matrixfree"])
    limit = limits.pop()
    n.add("nClarkResponseIterations", limit, str(limit), EXPLORATORY, rsources, "both single-impulse matrix-free probes reach declared LSMR iteration limit")
    rnnpath = c.ROOT / "docs/reviews/clark-rnn-independent-review.json"
    rnn = load(rnnpath); assert rnn["passed"] and rnn["dominant_p5_component"]["touches_grid_edge"]
    refpath = c.ROOT / "docs/reviews/clark-rnn-refinement-outcome.json"; ref = load(refpath)
    refroot = Path(ref["raw"]); refmanifest = refroot/"manifest.json"
    assert c.sha256(refmanifest) == ref["raw_manifest_sha256"] == next(u["inventory"]["manifest"] for u in rnn["input_units"] if u["unit"]=="refine-p5-001")
    designpath = refroot/"run/design.json"; inp = load(designpath)
    assert c.sha256(designpath) == load(refmanifest)["files"]["run/design.json"]["sha256"]
    assert inp["n"]==4 and inp["panels"][0]["kind"] == ref["result"]["panels"][0]["kind"] == "p5"
    sources = [rnnpath,refpath,refmanifest,designpath]
    n.add("nClarkRNNDimension", inp["n"], str(inp["n"]), EXPLORATORY, sources, "native activity dimension of the fixed refined random network")
    n.add("nClarkRNNPeriod", int(inp["panels"][0]["kind"][1:]), inp["panels"][0]["kind"][1:], EXPLORATORY, sources, "primary primitive-period refinement; saved native cycle checks independently reviewed")
    # Reviewed native activity proof and primitive branch extension, separate scopes.
    activity_path=c.ROOT/"certificates/clark-rnn-p5-receipt.json"
    activity_review_path=c.ROOT/"docs/reviews/clark-rnn-p5-independent-review.json"
    activity=load(activity_path);activity_review=load(activity_review_path)
    assert activity["proof"]["passed"] and activity_review["passed"] and activity_review["result"]["independent_exp_proof"]
    activity_sources=[activity_path,activity_review_path]+bound_inventory(Path(activity["raw"])/"manifest.json",activity["raw_manifest_sha256"])+bound_inventory(Path(activity_review["raw"])/"manifest.json",activity_review["outer_manifest"]["sha256"])
    assert activity["inputs"]["period"]==activity_review["result"]["primitive_period"]==5
    assert activity["proof"]["odd_companion_distinct"] and activity_review["checks"]["strict_nested_box_in_original"]
    assert activity["proof"]["attraction_witness_power"]==1
    ub=max(sum(max(abs(v)for v in ball_endpoints(b))for b in row)for row in activity["proof"]["attraction_witness"])
    rounded=Fraction(-((-ub.numerator*10**6)//ub.denominator),10**6)
    assert ub<rounded<1
    n.add("xClarkActivityBound",str(rounded),fixed(float(rounded),6),proof,activity_sources,"exact rational interval row-sum norm upper bound rounded upward to six decimals")
    for name,key,digits in [("xClarkActivityGain","gain",2),("xClarkActivityLeak","delta",3)]:
        value=Fraction(activity["inputs"][key]);n.add(name,str(value),fixed(float(value),digits),proof,activity_sources,"decimal display of exact binary64-dyadic "+key+"; full rational input in receipt")
    weighted_path=c.ROOT/"docs/reviews/s2-period18-weighted-independent-review.json";weighted=load(weighted_path)
    assert weighted["passed"];w=next(v for v in weighted["units"]if v["name"]=="weighted")
    width=Fraction(w["exact_parameter_endpoints"][1])-Fraction(w["exact_parameter_endpoints"][0]);assert w["passed"] and width==Fraction(1,16)
    n.add("tWeightedPrimitiveWidth",str(width),str(width),proof,[weighted_path],"exact hidden-rate width of independently reviewed weighted primitive branch; no attraction claim")
    arcpath=c.ROOT/"docs/reviews/s2-period18-arclength-independent-review.json";arc=load(arcpath)
    arcsources=[arcpath]+bound_inventory(Path(arc["raw"])/"manifest.json",arc["outer_manifest"]["sha256"])
    assert arc["passed"] and arc["checks"]["first_anchor_full_correlated_gauge_containment"] and arc["checks"]["independent384_bit_all_coordinate_inclusions"]
    count=arc["result"]["chart_count"];assert count==len(arc["result"]["charts"]) and all(v["passed"]for v in arc["result"]["charts"])
    n.add("nArcReviewedCharts",count,str(count),proof,arcsources,"independently reviewed initial arclength charts, distinct from later producer-only extensions")
    endpoints=arc["result"]["exact_gauge_endpoints"]
    n.add("tArcGaugeInterval",endpoints,"["+", ".join(endpoints)+"]",proof,arcsources,"exact affine gauge interval; not a hidden-rate interval")
    cusp_review_path=c.ROOT/"docs/reviews/s2-cusps-independent-review.json";cusp_review=load(cusp_review_path)
    assert cusp_review["status"]=="passed_with_preserved_review_failures"
    cusp_sources=[cusp_review_path]
    for attempt in cusp_review["attempts"].values():
        mf=attempt["manifest"];cusp_sources+=bound_inventory(Path(mf["path"]),mf["sha256"])
    for name,pin in cusp_review["producer_canonical_receipts"].items():
        path=c.ROOT/name;assert c.sha256(path)==pin["sha256"];receipt=load(path)
        cusp_sources+=[path]+bound_inventory(Path(receipt["external_root"])/"manifest.json",receipt["manifest_sha256"])
    assert set(cusp_review["panels"])=={"cusp-a","cusp-b"}
    n.add("nLocalCusps",2,"2",proof,cusp_sources,"separate primitive18 local cusp roots, independent review; no attraction or whole-window claim")
    for panel,prefix in [("cusp-a","CuspA"),("cusp-b","CuspB")]:
        row=cusp_review["panels"][panel];assert row["summary"]["passed"] and row["summary"]["replay_identical"] and row["nested"]["strictly_contained_in_original_full41D_box"]
        ident=row["artifacts"]["nested-proof.json"];assert c.sha256(ident["path"])==ident["sha256"];nested=load(ident["path"])
        assert nested["passed"] and all(nested["gates"].values())
        assert row["independent_spectral_discs"]["outside"]==1 and row["independent_spectral_discs"]["inside"]==12
        for suffix,index_,digits in [("Hidden",13,6),("Output",14,9)]:
            lo,hi=ball_endpoints(nested["box"][index_]);mid=(lo+hi)/2
            n.add("x"+prefix+suffix,str(mid),fixed(float(mid),digits),proof,cusp_sources,"approximate decimal locator of nested exact rate box; not a replacement for enclosure")
        for suffix,field in [("Cubic","cubic_coefficient"),("Unfolding","unfolding_determinant")]:
            assert nested[field]==row["nested"][field]
            lo,hi=ball_endpoints(nested[field]);assert hi<0 or lo>0
            digits=6;scale=10**digits;flo=lo.numerator*scale//lo.denominator;cei=-((-hi.numerator*scale)//hi.denominator)
            value="["+exact_decimal(flo,digits)+", "+exact_decimal(cei,digits)+"]"
            n.add("t"+prefix+suffix,[str(lo),str(hi)],value,proof,cusp_sources,"exact nested dyadic endpoints rounded outward by rational floor/ceiling")
    # Independently audited continued branch and frozen-checkpoint inference.
    chainpath=c.ROOT/"docs/reviews/s2-newgauge-chain-independent.json";chain=load(chainpath)
    assert chain["passed"] and len(chain["units"])==2
    chainsources=[chainpath]
    for unit in chain["units"]:
        assert unit["outcome"]["passed"] and unit["launcher"]["exit_code"]==0
        chainsources+=bound_inventory(Path(unit["root"])/"manifest.json",unit["outer_manifest"]["sha256"])
    assert chain["units"][0]["outcome"]["total_qualified_charts"]==129
    total=chain["units"][-1]["outcome"]["total_qualified_charts"]
    assert total==chain["total_qualified_charts"]==144 and len(chain["units"][-1]["outcome"]["charts"])==15
    n.add("nArcTotalReviewed",total,integer(total),proof,chainsources,"total native joined charts across the original and correlated gauges; no full-path attraction or fold membership")
    inferencepath=c.ROOT/"docs/reviews/saved-lm-inference-independent.json";irec=load(inferencepath);inf=irec["result"]
    assert irec["passed"] and inf["passed"] and inf["matched_changed"]==inf["rollout_changed"]==0
    auditpath=Path(irec["audit_file"]["path"]);assert c.sha256(auditpath)==irec["audit_file"]["sha256"] and load(auditpath)==inf
    iraw=Path(inf["raw_root"]);insources=[inferencepath,auditpath]+bound_inventory(iraw/"manifest.json",inf["outer_manifest"]["sha256"])
    binding=load(iraw/"run/input-binding.json");prefixes=len(binding["prefix_offsets"])
    with np.load(iraw/"run/seed-28151-factor-0.8.npz",allow_pickle=False) as arr:steps=arr["tokens32"].shape[0]
    assert len(inf["cases"])*prefixes*steps==inf["decisions"]==1536
    label="exploratory saved-checkpoint precision comparison; independently audited outputs; no unseen-text outcomes"
    for macro,value in [("nInferenceCases",len(inf["cases"])),("nInferencePrefixes",prefixes),("nInferenceSteps",steps),("nInferenceDecisions",inf["decisions"])]:
        n.add(macro,value,integer(value),label,insources,"all fixed calibration checkpoints and already consumed prefixes; correlated finite-horizon decisions")
    for macro,key in [("xInferenceMaxLogit","max_logit_difference"),("xInferenceMinMargin","minimum_margin")]:
        n.add(macro,inf[key],sci(inf[key],3),label,insources,"measured binary32/binary64 pair at the same rounded weights; descriptive value, not rigorous error bound")
    c400path=c.ROOT/"docs/reviews/cifar400-saved-outcome-review.json";c400=load(c400path);cr=c400["result"]
    assert cr["passed"] and cr["forcing"]<=.5 and cr["slope"]<0 and not any(t["armijo_pass"] for t in cr["trials"])
    c400sources=[c400path]+bound_inventory(Path(c400["raw_root"])/"manifest.json",c400["manifest_sha256"])
    raw400=c.EXPLORE/"cifar/large-space/attempt-002";assert c.sha256(raw400/"manifest.json")==cr["raw_manifest"]
    c400sources+=[raw400/"manifest.json",raw400/"linear.json"]
    linear400=load(raw400/"linear.json");assert linear400["inner_iterations"]==400 and linear400["forcing_test_pass"]
    n.add("nCifarLargerInner",400,"400",EXPLORATORY,c400sources,"single unrestarted Krylov solve with matched first200 callbacks")
    n.add("xCifarLargerForcing",cr["forcing"],fixed(cr["forcing"],4),EXPLORATORY,c400sources,"independently reduced true residual forcing; all original four trial merits still increase")
    # Independently joined canonical fold and locally attracting near-fold endpoint.
    fcpath=c.ROOT/"docs/reviews/s2-period18-fold-connection-independent.json";fc=load(fcpath);fo=fc["outcome"]
    assert fo["passed"] and fo["fold_to_center_branch_identity"] and fo["matched_phase"]==0
    fcsources=[fcpath]+bound_inventory(Path(fc["raw_root"])/"manifest.json",fc["manifest_sha256"])
    assert fo["reverse_charts"]==len(fo["charts"])==31 and fo["forward_charts"]==156
    for macro,key in [("nFoldConnectedForward","forward_charts"),("nFoldConnectedReverse","reverse_charts")]:
        n.add(macro,fo[key],integer(fo[key]),proof,fcsources,"independently checked full-state primitive branch connection; not all-path attraction")
    apath=c.ROOT/"docs/reviews/s2-near-fold-endpoint-attraction.json";at=load(apath)
    assert at["local_attraction"] and at["independent_outcome"]["passed"] and at["interval_phases_passed"]==18
    asources=[apath]+bound_inventory(Path(at["independent_root"])/"manifest.json",at["independent_manifest_sha256"])
    n.add("nNearFoldAttractionPhases",at["interval_phases_passed"],integer(at["interval_phases_passed"]),proof,asources,"direct384-bit cyclic interval Lyapunov inequalities at the same localized near-fold endpoint")
    dpath=c.ROOT/"docs/reviews/cifar-damped-outcome.json";dr=load(dpath);dc=dr["saved_check"]
    assert dc["passed"] and dc["accepted_steps"]==1 and len(dc["steps"])==2
    dsources=[dpath]
    for raw,entry in dr["artifacts"].items():dsources+=bound_inventory(Path(raw)/"manifest.json",entry["manifest_sha256"])
    first,second=dc["steps"];trial=first["trials"][first["accepted_index"]]
    assert trial["accepted_local_step"] and trial["alpha"]==1/32 and first["forcing"]<=.5<second["forcing"]
    n.add("tCifarAcceptedAlpha","1/32","1/32",EXPLORATORY,dsources,"first accepted fixed Armijo damping at the subsequent saved state; not an orbit")
    for macro,value in [("xCifarAcceptedClosure",dc["final_relative_residual"]),("xCifarNextForcing",second["forcing"])]:
        n.add(macro,value,sci(value,3) if value<.001 else fixed(value,4),EXPLORATORY,dsources,"saved-array outcome check of one accepted local step followed by failed forcing")
    for method,macro in [("diagonal","xCifarDiagonalForcing"),("gram","xCifarGramForcing")]:
        path=c.ROOT/("docs/reviews/cifar-"+method+"-outcome.json");rec=load(path);chk=rec["saved_check"]
        assert chk["passed"] and rec["outcome"]["accepted_steps"]==rec["outcome"]["trials"]==0
        value=chk["actual_forcing"];assert value>.5 and abs(value-rec["linear"]["true_forcing_ratio"])<1e-12
        sources=[path]
        for raw,entry in rec["artifacts"].items():sources+=bound_inventory(Path(raw)/"manifest.json",entry["manifest_sha256"])
        n.add(macro,value,fixed(value,4),EXPLORATORY,sources,"same-state fixed400 true physical forcing; producer-agent saved arithmetic, not independent model reevaluation")
    # Continuous attraction and separately checked negative transfer/utility pilots.
    acpath=c.ROOT/"docs/reviews/s2-correlated-stability-chain-independent.json";ac=load(acpath)
    assert ac["status"]=="passed" and ac["native_parameters"]==13 and ac["primitive_period"]==18
    sources=[acpath];intervals=[]
    for unit in ac["units"]:
        raw=Path(unit["raw_root"]);sources+=bound_inventory(raw/"manifest.json",unit["manifest_sha256"])
        outcome=load(raw/"run/outcome.json");assert outcome==unit["audit"] and outcome["passed"]
        intervals.extend(tuple(map(Fraction,row["exact_endpoints"])) for row in outcome["slices"] if row["passed"])
    intervals.sort();assert len(intervals)==ac["closed_slices"]==24 and len(ac["units"])==ac["reviewed_charts"]==3
    assert all(a[1]==b[0] for a,b in zip(intervals,intervals[1:]))
    ends=[str(intervals[0][0]),str(intervals[-1][1])];assert ends==ac["exact_continuation_gauge"]
    n.add("nCorrelatedSlices",len(intervals),integer(len(intervals)),proof,sources,"independently audited closed adjacent attraction slices on the same primitive18 branch")
    n.add("tCorrelatedGauge",ends,"["+", ".join(ends)+"]",proof,sources,"later affine full-state continuation coordinate; not a hidden-rate interval")
    # Larger connected attraction coverage, including its exact physical-rate projection.
    covpath=c.ROOT/"docs/reviews/s2-attraction-coverage-independent.json";cov=load(covpath);cv=cov["result"]
    cs=[covpath]+bound_inventory(Path(cov["raw_root"])/"manifest.json",cov["manifest_sha256"])
    assert cv==load(Path(cov["raw_root"])/"result.json") and cv["passed"]
    ends=[];count=0
    for name,unit in cv["independent_units"].items():
        raw=c.DATA/"audits"/name;cs+=bound_inventory(raw/"manifest.json",unit["manifest_sha256"])
        out=load(raw/"run/outcome.json");assert out["passed"] and c.sha256(raw/"run/outcome.json")==unit["outcome_sha256"]
        assert all(x["passed"] for x in out["slices"]) and len(out["slices"])==unit["slice_count"]
        ends.append(tuple(map(Fraction,unit["coverage"])));count+=unit["slice_count"]
    ends.sort();assert count==cv["slices"]==68 and len(ends)==cv["charts"]==9
    assert all(a[1]==b[0] for a,b in zip(ends,ends[1:]))
    assert [str(ends[0][0]),str(ends[-1][1])]==cv["exact_full_state_gauge"]
    parent=c.DATA/"audits/s2-period18-small-march/attempt-001";cs+=bound_inventory(parent/"manifest.json",cv["same_branch_parent_manifest"])
    left,right=[load(parent/p) for p in cv["endpoint_proof_files"]]
    assert left["fresh_endpoints"][0][-1]==cv["left_hidden_rate_ball"] and right["fresh_endpoints"][1][-1]==cv["right_hidden_rate_ball"]
    lo,hi=map(Fraction,cv["rigorous_hidden_rate_interval"])
    assert ball_endpoints(cv["left_hidden_rate_ball"])[1]<=lo<hi<=ball_endpoints(cv["right_hidden_rate_ball"])[0]
    assert Fraction(cv["exact_output_rate"])==output
    displays=[exact_decimal(int(v*10**6),6) for v in (lo,hi)];assert displays==cv["display_hidden_rate_interval"]
    n.add("nAttractingSlices",count,integer(count),proof,cs,"all closed slices in nine independently audited joined charts")
    n.add("tAttractingGauge",cv["exact_full_state_gauge"],"["+", ".join(cv["exact_full_state_gauge"])+"]",proof,cs,"exact connected affine full-state gauge")
    n.add("tAttractingHiddenRate",cv["rigorous_hidden_rate_interval"],"["+", ".join(displays)+"]",proof,cs,"inward decimal rounding and continuity; at least one attracting primitive18 cycle at each hidden rate; monotonicity not assumed")
    tp=c.ROOT/"docs/reviews/s2-attraction-tail-independent-outcome.json";tw=load(tp)
    wp=c.ROOT/tw["previous_receipt"];w=load(wp)
    assert c.sha256(wp)==tw["previous_receipt_identity"]["sha256"] and w["status"]=="independent_pass" and tw["status"]=="passed"
    ws=cs+[tp,wp]+bound_inventory(Path(w["root"])/"manifest.json",w["outer_manifest"]["sha256"])
    ws+=bound_inventory(Path(tw["tail_batch_root"])/"manifest.json",tw["tail_batch_outer_manifest"]["sha256"])
    ws+=bound_inventory(Path(tw["reused_piece"]["root"])/"manifest.json",tw["reused_piece"]["outer_manifest"]["sha256"])
    wider=list(ends);total=count
    for row in w["charts"]+tw["tail_charts"]:
        out=row["outcome"];assert out["passed"] and out==load(Path(row["root"])/"run/outcome.json")
        wider.append(tuple(map(Fraction,out["coverage"])));total+=out["completed_slices"]
    out=tw["reused_piece"]["outcome"];assert out["passed"] and out==load(Path(tw["reused_piece"]["root"])/"run/outcome.json")
    wider.append(tuple(map(Fraction,out["coverage"])));total+=out["completed_slices"];wider.sort()
    assert total==tw["total_connected_independent_slices"]==112 and all(a[1]==b[0]for a,b in zip(wider,wider[1:]))
    wideends=[str(wider[0][0]),str(wider[-1][1])];assert wideends==tw["total_connected_exact_gauge_interval"]
    physical=[]
    for row in tw["physical_endpoint_proofs"]:
        pp=Path(row["path"]);assert c.sha256(pp)==row["identity"]["sha256"];ws.append(pp);pv=load(pp)
        if row["selector"] is not None:pv=pv[row["selector"]]
        assert pv["passed"] and pv["fresh_endpoints"][row["endpoint_index"]][-1]==row["hidden_rate_enclosure"]
        physical.append(ball_endpoints(row["hidden_rate_enclosure"]))
    lo,hi=map(Fraction,tw["guaranteed_hidden_rate_interval"]);assert physical[0][1]<=lo<hi<=physical[1][0] and Fraction(tw["fixed_output_rate"])==output
    displays=[exact_decimal(int(v*10**6),6)for v in(lo,hi)]
    n.add("nWideAttractingSlices",total,integer(total),proof,ws,"all112independently checked closed adjoining native attraction pieces")
    n.add("tWideAttractingGauge",wideends,"["+", ".join(wideends)+"]",proof,ws,"full closed connected gauge coverage after tail audit")
    n.add("tWideAttractingHiddenRate",tw["guaranteed_hidden_rate_interval"],"["+", ".join(displays)+"]",proof,ws,"exact rational endpoint bounds and IVT; no monotonicity assumption")
    # Independently proved double+1 Jordan root and strictly attracting quotient.
    jcpath=c.ROOT/"docs/reviews/generalized-chain-independent-outcome.json";jc=load(jcpath);jo=jc["outcome"]
    js=[jcpath]+bound_inventory(Path(jc["raw_root"])/"manifest.json",jc["outer"]["manifest_sha256"])
    assert jo["passed"] and jo["root_passed"] and jo["positive_actual_rates"] and jo["exact_algebraic_multiplicity_two"] and jo["complement_attracting"]
    assert jo["primitive_period"]==18 and jo["strict_inclusions"]==41
    quotient=jc["quotient"];assert quotient["dimension"]==len(quotient["discs"])==11 and quotient["all_discs_strictly_inside"]
    for macro,ball,digits in zip(["tJordanHiddenRate","tJordanOutputRate"],jo["actual_rate_enclosures"],[12,12]):
        lo,hi=ball_endpoints(ball);scale=10**digits;flo=lo.numerator*scale//lo.denominator;cei=-((-hi.numerator*scale)//hi.denominator)
        n.add(macro,[str(lo),str(hi)],"["+exact_decimal(flo,digits)+", "+exact_decimal(cei,digits)+"]",proof,js,"exact independently certified native rate enclosures; outward rational rounding")
    bound=max(ball_endpoints(x["recomputed_modulus_upper"])[1] for x in quotient["discs"])
    upper=Fraction(-((-bound*10**9)//1),10**9);assert bound<=upper<1
    n.add("xJordanComplementBound",str(upper),exact_decimal(int(upper*10**9),9),proof,js,"outward rounded maximum of eleven rigorous quotient Gershgorin modulus bounds")
    n.add("nJordanComplement",11,"11",proof,js,"strictly attracting complementary dimensions; remaining two form exactly one length2 Jordan block at+1")
    # Entire native +1 locus and exact same-root membership independently proved.
    ixp=c.ROOT/"docs/reviews/eigenpair-connected107-proof-index.json";ix=load(ixp)
    assert ix["same_native_component"] and ix["period"]==18 and ix["total_connected_charts"]==len(ix["charts"])==107
    ims=[ixp]
    for i,row in enumerate(ix["charts"],1):
        assert row["global_chart"]==i
        for key in ["independent256","independent384"]:
            ref=row[key];path=Path(ref["path"]);assert c.sha256(path)==ref["sha256"] and path.stat().st_size==ref["bytes"]
            assert load(path)["passed"];ims.append(path)
        for ref in row["join"].get("localization_proofs",[]):
            path=Path(ref["path"]);assert c.sha256(path)==ref["sha256"] and load(path)["passed"];ims.append(path)
    mem=ix["membership"];assert mem["outcome"]["passed"] and mem["outcome"]["native_coordinate_inclusions"]==470
    for key in ["native_domain_check","membership","input_bindings"]:
        ref=mem[key];path=Path(ref["path"]);assert c.sha256(path)==ref["sha256"] and path.stat().st_size==ref["bytes"];ims.append(path)
    n.add("nConnectedUnitCharts",len(ix["charts"]),integer(len(ix["charts"])),proof,ims,"every original256/fresh384 chart and full470 known-root membership in the connected normalized+1 locus; no uniform generic-fold assertion")
    # Finite complex-unit branch independently attached to the same exact root.
    fp=c.ROOT/"docs/reviews/native-unit-pair-independent-outcome.json";fr=load(fp);fo=fr["outcome"]
    fs=[fp]+bound_inventory(Path(fr["root"])/"manifest.json",fr["outer_manifest"]["sha256"])
    assert fo["passed"] and fo["same_independently_connected_root"] and fo["primitive_period"]==18
    assert all(fr["gates"].values()) and fo["strict_inclusions"]==fo["anchor_coordinates"]==704
    assert fo["parameter_endpoints"]==["0","1/32768"]
    pf=Path(fr["root"])/fr["proof_path"];assert c.sha256(pf)==fr["proof_identity"]["sha256"]
    pr=load(pf);assert pr["passed"] and pr["contraction"]==fo["contraction"]
    upper=ball_endpoints(fo["contraction"])[1];rounded=Fraction(-((-upper*10**9)//1),10**9);assert upper<=rounded<1
    n.add("tFiniteUnitParameter",fo["parameter_endpoints"],r"[0,\,2^{-15}]",proof,fs,"exact independently proved nonnegative parameter interval t=angle squared")
    n.add("nFiniteUnitUnknowns",fo["strict_inclusions"],integer(fo["strict_inclusions"]),proof,fs,"all full native state/mode/rate coordinate inclusions and exact-root anchor inclusions")
    n.add("xFiniteUnitContraction",str(rounded),exact_decimal(int(rounded*10**9),9),proof,fs,"outward decimal upper bound on independent384 weighted full shooting defect norm")
    # Generic physical-rate unfolding, independently reduced at the same proved root.
    upath=c.ROOT/"docs/reviews/chain-unfolding-independent.json";ur=load(upath);uo=ur["outcome"];uv=uo["reduction"]
    us=[upath]+bound_inventory(Path(ur["raw_root"])/"manifest.json",ur["manifest_sha256"])
    assert uo==load(Path(ur["raw_root"])/"run/outcome.json") and uo["passed"] and uo["precision"]==512
    assert uv["quadratic_nondegenerate"] and uv["resonant_nondegenerate"] and uv["unfolding_transverse"]
    assert uo["native_inventory"]["manifest_sha256"]==jc["outer"]["manifest_sha256"]
    for macro,key,digits,sign in [("tJordanNormalA","a",5,-1),("tJordanNormalB","b",3,-1),("tJordanNormalDifference","b_minus_a",3,-1),("tJordanNormalS","s",3,1),("tJordanRateDeterminant","unfolding_determinant",4,-1)]:
        lo,hi=ball_endpoints(uv[key]);assert (hi<0 if sign<0 else lo>0)
        scale=10**digits;flo=lo.numerator*scale//lo.denominator;cei=-((-hi.numerator*scale)//hi.denominator)
        n.add(macro,[str(lo),str(hi)],"["+exact_decimal(flo,digits)+", "+exact_decimal(cei,digits)+"]",proof,us,"independent512bit same-root coefficient/actual-rate unfolding; outward rational rounding")
    rupath=c.ROOT/"docs/reviews/response-utility-saved-audit.json";ru=load(rupath);rr=ru["result"]
    assert ru["status"]=="passed_saved_arithmetic" and rr["passed"]
    rus=[rupath]+bound_inventory(Path(ru["raw_root"])/"manifest.json",ru["manifest_sha256"])
    fractions=[row["fraction"] for row in rr["utility"]["rows"]];assert len(fractions)==4
    label="exploratory common-center saved arithmetic; conditional on saved analytic gradients/losses; no geometry-specific benefit"
    n.add("nCommonCenterCases",len(fractions),str(len(fractions)),label,rus,"all fixed paired loss cases")
    for macro,value in [("xCommonCenterMinFraction",min(fractions)),("xCommonCenterMaxFraction",max(fractions))]:
        n.add(macro,value,sci(value,2),label,rus,"absolute centered first-order residual divided by absolute actual paired loss difference")
    lines=rr["response"][1]["rows"];assert len(lines)==2 and not any(x["endpoint_pass"] for x in lines)
    for macro,row in zip(["xResponseLineErrorTen","xResponseLineErrorEighteen"],lines):
        n.add(macro,row["readout_error"],fixed(row["readout_error"],5),EXPLORATORY,rus,"independent terminal fsum reduction of coordinate-time-line preconditioned response; fixed action ceiling exhausted")
    npath=c.ROOT/"docs/reviews/clark-rnn-n12-panel.json";nr=load(npath);nt=nr["totals"]
    assert nr["result"]["status"]=="completed" and nt["qualification_attempts"]==nt["qualified_seed_cases"]==116
    ns=[npath]+bound_inventory(Path(nr["raw"])/"manifest.json",nr["outer_manifest_sha256"])
    label="exploratory fixed-weight N12 activity panel; source/inventory checked, floating native qualifications; no interval/whole-window proof"
    for macro,value in [("nClarkPanelMatrices",len(nr["result"]["networks"])),("nClarkPanelDimension",12),("nClarkPanelTrajectories",nt["trajectories"]),("nClarkPanelComponents",nt["consensus_components"]),("nClarkPanelQualified",nt["qualified_seed_cases"])]:
        n.add(macro,value,integer(value),label,ns,"all fixed development matrices and both starts; selected qualifications only")
    # Different large-model residual-merit solver; independently checked saved arithmetic.
    cmp=c.ROOT/"docs/reviews/cifar-merit-saved-independent.json";cm=load(cmp);cr=cm["result"]
    cms=[cmp]+bound_inventory(Path(cm["raw_root"])/"manifest.json",cm["outer"]["sha256"])
    assert cr==load(Path(cm["raw_root"])/"run/result.json") and cr["passed"] and not cr["closure_screen"] and cr["old_forcing_passes"]==0
    assert len(cr["steps"])==6 and all(z["accepted_local_step"] for z in cr["steps"])
    label="exploratory native CIFAR correction, independent saved-arithmetic audit; no accepted orbit"
    n.add("nCifarMeritSteps",6,"6",label,cms,"all fixed full-step acceptance gates independently checked")
    n.add("xCifarMeritClosure",cr["final_relative_residual"],sci(cr["final_relative_residual"],3),label,cms,"relative norm of final retained two-step return residual")
    n.add("pCifarMeritImprovement",cr["relative_residual_fractional_reduction"],fixed(cr["relative_residual_fractional_reduction"]*100,2)+r"\%",label,cms,"relative closure reduction from the same development state")
    cp=c.ROOT/"docs/reviews/clark-rnn-n12-flip-cubic-outcome.json";cc=load(cp)
    cs=[cp]+bound_inventory(Path(cc["raw_root"])/"manifest.json",cc["outer"]["manifest_sha256"])
    assert cc["status"]=="completed_numerical_cubic_bracket" and cc["midpoint_count"]==24 and cc["separate_endpoint_join_count"]==48
    br=cc["bracket"];assert len(br)==2 and all(v["passed"] and all(v["gates"].values())for v in br) and br[0]["cubic"]>0>br[1]["cubic"]
    lo,hi=[Fraction.from_float(v["gain"])for v in br];assert lo<hi
    digits=10;scale=10**digits;flo=lo.numerator*scale//lo.denominator;cei=-((-hi.numerator*scale)//hi.denominator)
    n.add("tNTwelveCubicGain",[float(lo),float(hi)],"["+exact_decimal(flo,digits)+", "+exact_decimal(cei,digits)+"]",EXPLORATORY,cs,"outward decimal display of numerical same-branch cubic-sign bracket; no interval proof")
    smp=c.ROOT/"docs/reviews/cifar-saved-shooting-merit-outcome.json";sm=load(smp);sr=sm["outcome"]
    sms=[smp]+bound_inventory(Path(sm["raw_root"])/"manifest.json",sm["manifest_sha256"])
    assert sr==load(Path(sm["raw_root"])/"result.json") and sr["comparison_qualified"] and not sr["accepted_orbit"] and sr["accepted_index"]==4
    assert sr["trials"][4]["accepted_local_step"] and not sr["direction_gate"]["old_newton_forcing_pass"]
    scp=c.ROOT/"docs/reviews/cifar-saved-step-independent.json";sca=load(scp);scar=Path(sca["raw_root"])
    sms+=[scp]+bound_inventory(scar/"manifest.json",sca["outer_manifest_sha256"])
    assert sca["outcome"]==load(scar/"run/result.json")and sca["outcome"]["passed"]and not sca["outcome"]["closure_screen"]
    label="exploratory native CIFAR residual-merit correction; independent saved acceptance arithmetic passed; no accepted orbit"
    n.add("xCifarSavedMeritClosure",sr["final_relative_residual"],sci(sr["final_relative_residual"],3),label,sms,"last accepted saved-direction two-step relative closure")
    n.add("xCifarShootingForcing",sr["direction_gate"]["true_forcing"],fixed(sr["direction_gate"]["true_forcing"],4),label,sms,"true eliminated Newton forcing of the retained1600MINRES direction; fixed.5gate fails")
    # Same-root local unit-pair/simple-flip junction; connectivity separate.
    jpath=c.ROOT/"docs/reviews/native-unit-pair-flip-independent.json";jr=load(jpath);jroot=Path(jr["raw_root"])
    js=[jpath]+bound_inventory(jroot/"manifest.json",jr["outer_manifest_sha256"])
    jo=load(jroot/"run/outcome.json");assert jo==jr["result"]and jo["passed"]and jo["same_unique_root"]and jo["native_calls"]==162
    ji=load(jroot/"run/root-identity.json");assert len(ji["fresh_inside_original"])==55 and all(ji["fresh_inside_original"])
    for which in ["saved","fresh"]:
        v=jo[which];assert v["inclusions"]==55 and v["proper_divisors"]==[1,2,3,6,9]and ball_endpoints(v["contraction"])[1]<1
        assert len(v["complement_bounds"])==10 and max(ball_endpoints(x)[1]for x in v["complement_bounds"])<Fraction(93,100)
        assert ball_endpoints(v["crossing"])[1]<0
    n.add("xUnitFlipComplement","93/100","0.930",proof,js,"strict upper bound on all ten complementary multiplier moduli, saved and fresh independently validated quotients")
    cross=ball_endpoints(jo["fresh"]["crossing"]);scale=10000;lo=(cross[0]*scale)//1;hi=-((-cross[1]*scale)//1)
    n.add("tUnitFlipCrossing",[str(x)for x in cross],"["+exact_decimal(int(lo),4)+", "+exact_decimal(int(hi),4)+"]",proof,js,"outward four-decimal interval for total hidden-rate derivative of the simple minus-one multiplier along the complex-unit section")
    rootj=load(jroot/"run/fresh-root.json");assert rootj["dimension"]==13 and rootj["period"]==18
    for index,macro in [(13,"xUnitFlipHidden"),(14,"xUnitFlipOutput")]:
        lo,hi=ball_endpoints(rootj["box"][index]);n.add(macro,str((lo+hi)/2),fixed(float((lo+hi)/2),8),proof,js,"display approximation to exact same-root physical-rate enclosure midpoint")
    # Highest-period agreed census representative: independent native72D proof.
    p58p=c.ROOT/"docs/reviews/census-period58-tight-outcome.json"
    p58a=c.ROOT/"docs/reviews/census-period58-independent-outcome.json"
    p58c=c.ROOT/"certificates/s1r-period58-receipt.json"
    producer,review,receipt=load(p58p),load(p58a),load(p58c)
    pr,ar=Path(producer["raw_root"]),Path(review["root"])
    sources58=[p58p,p58a,p58c]+bound_inventory(pr/"manifest.json",producer["outer"]["manifest"]["sha256"])
    sources58+=bound_inventory(ar/"manifest.json",review["outer_manifest"]["sha256"])
    outcome=load(ar/"run/outcome.json");assert outcome==review["outcome"] and outcome["passed"] and outcome["same_unique_root"]
    statement=receipt["statement"];assert statement==producer["outcome"]["certificate_receipt"]["statement"]
    assert statement["dimension"]==outcome["dimension"]==72 and statement["primitive_period"]==outcome["primitive_period"]==58
    assert all(statement[k] for k in ["attraction","existence_and_local_uniqueness","primitive_period_excluded_divisors"])
    identity58=load(ar/"run/root-identity.json");assert len(identity58["original_box_strictly_contained_in_fresh"])==72 and all(identity58["original_box_strictly_contained_in_fresh"])
    runtime58=load(ar/"run/runtime.json");assert runtime58["precision"]==768
    for which in ["saved","fresh"]:
        value=outcome[which];assert value["strict_inclusions"]==72 and value["phase_jacobians"]==58 and value["attraction_power"]==8
        assert [x["divisor"]for x in value["divisors"]]==[1,2,29] and all(x["coordinates"]for x in value["divisors"])
        assert ball_endpoints(value["contraction"])[1]<1
    power_bound=max(ball_endpoints(outcome[k]["attraction_bound"])[1]for k in ["saved","fresh"])
    rounded58=Fraction(-((-power_bound*1000)//1),1000);assert power_bound<rounded58<1
    for macro,value,calc in [("nPFiftyEightPeriod",58,"primitive period after all proper divisor exclusions"),("nPFiftyEightDimension",72,"full native trainable parameter dimension"),("nPFiftyEightPower",8,"monodromy-power attraction witness"),("nPFiftyEightPrecision",768,"independent interval arithmetic and replay precision in bits")]:
        n.add(macro,value,str(value),proof,sources58,calc)
    n.add("xPFiftyEightBound",str(rounded58),exact_decimal(int(rounded58*1000),3),proof,sources58,"maximum independent saved/fresh exact dyadic power bound rounded upward to three decimals")
    input58=load(ar/"run/fresh-inputs.json")
    for key,macro in [("lr_hidden","xPFiftyEightHiddenRate"),("lr_output","xPFiftyEightOutputRate")]:
        value=Fraction(input58[key]);n.add(macro,str(value),fixed(float(value),8),proof,sources58,"display approximation to supplied exact dyadic physical learning rate")
    # Every13D agreed-census representative has an exact attracting orbit proof.
    c13p=c.ROOT/"certificates/census13-orbits-receipt.json";c13a=c.ROOT/"docs/reviews/census13-ad-independent-outcome.json"
    c13,ac13=load(c13p),load(c13a);cp13,ca13=Path(c13["producer"]["root"]),Path(c13["independent"]["root"])
    # Verify every raw file, then bind the complete inventories by their hashes.
    # Repeating thousands of native acquisitions in each macro bloats the index.
    bound_inventory(cp13/"manifest.json",c13["producer"]["outer"])
    bound_inventory(ca13/"manifest.json",c13["independent"]["outer"])
    c13s=[c13p,c13a,cp13/"manifest.json",ca13/"manifest.json"]
    ao13=load(ca13/"run/outcome.json");assert ao13==ac13["outcome"]and ao13["passed"]and ao13["proofs_passed"]==26 and ao13["native_returned"]==1050
    assert c13["passed"]and c13["dimension"]==13 and c13["certified_13d_representatives"]==28 and c13["new_independent_ad_certificates"]==26
    inputbind=load(cp13/"run/input-binding.json");assert inputbind["already_proved"]==c13["reused"]and len(inputbind["all13"])==28 and [v["index"]for v in c13["new"]]==inputbind["new"]
    periods=[];newpairs=[]
    for row in c13["new"]:
        idx=row["index"];ap=ca13/f"run/row-{idx}/proof.json";pp=cp13/f"run/row-{idx}/proof.json";ip=cp13/f"run/row-{idx}/inputs.json"
        assert c.sha256(ap)==row["independent_proof_sha256"]and c.sha256(pp)==row["producer_proof_sha256"]and c.sha256(ip)==row["input_sha256"]
        aa,bb,ii=load(ap),load(pp),load(ip)
        assert aa["passed"]and all(aa["gates"].values())and aa["box"]==bb["box"]and aa["same_exact_original_box"]and aa["dimension"]==13 and aa["period"]==ii["period"]==row["period"]
        assert len(aa["inclusions"])==13 and all(aa["inclusions"])and ball_endpoints(aa["contraction"])[1]<1 and ball_endpoints(aa["power_bounds"][-1]["bound"])[1]<1
        assert [d["divisor"]for d in aa["divisors"]]==[k for k in range(1,ii["period"])if ii["period"]%k==0]and all(d["separating_coordinates"]for d in aa["divisors"])
        periods.append(ii["period"]);newpairs.append((ii["provenance"]["map_id"],ii["lr_hidden"],ii["lr_output"]))
    assert len(set(newpairs))==26
    # The older separate36-cycle is not the newly certified census36 setting.
    assert all((Fraction(row["lr_hidden"]),Fraction(row["lr_output"]))!=(Fraction(p36["exact_fixed_rates"]["hidden"]),Fraction(p36["exact_fixed_rates"]["output"]))for row in c13["new"]if row["period"]==36)
    for row in c13["reused"]:
        oldroot=Path(row["root"]);assert c.sha256(oldroot/"certificate.json")==row["proof"]["sha256"];old=load(oldroot/"certificate.json")["result"];assert old["passed"];periods.append(old["exact_statement"]["primitive_period"])
    for macro,value,calc in [("nCensusThirteen",28,"all13D representatives in fixed61-way agreed census"),("nCensusThirteenNew",26,"new same-box independent scalar-loss AD proofs"),("nCensusThirteenPrecision",1024,"independent AD and interval proof precision bits")]:n.add(macro,value,str(value),proof,c13s,calc)
    n.add("tCensusThirteenPeriods",sorted(set(periods)),", ".join(map(str,sorted(set(periods)))),proof,c13s,"all distinct certified primitive periods in complete13D representative subset")
    def exact_map_identity(inp):
        def normalize(v):
            if isinstance(v,dict):return {k:normalize(a)for k,a in v.items()}
            if isinstance(v,list):return [normalize(a)for a in v]
            if isinstance(v,str):
                try:return str(Fraction(v))
                except ValueError:return v
            return v
        return json.dumps(normalize({k:inp[k]for k in ["network","x","y","lr_hidden","lr_output"]}),sort_keys=True)
    previous_maps=[load(Path(z["external_root"])/"inputs.json")for z in certificates]
    previous_maps+=[load(Path(load(c.ROOT/"certificates/s2-period18-receipt.json")["external_root"])/"inputs.json"),r51_inputs["period36"],load(Path(s1r_cert["external_root"])/"inputs.json"),input58]
    # Complete full-state proof coverage of every72D agreed census representative.
    import gzip
    c72p=c.ROOT/"certificates/census72-orbits-receipt.json";c72a=c.ROOT/"docs/reviews/census72-ad-independent-outcome.json"
    c72,ac72=load(c72p),load(c72a);cp72,ca72=Path(c72["producer"]["root"]),Path(c72["independent"]["root"])
    bound_inventory(cp72/"manifest.json",c72["producer"]["outer"]);bound_inventory(ca72/"manifest.json",c72["independent"]["outer"])
    c72s=[c72p,c72a,cp72/"manifest.json",ca72/"manifest.json"]
    ao72=load(ca72/"run/outcome.json");assert ao72==ac72["outcome"]and ao72["passed"]and ao72["proofs_passed"]==32 and ao72["native_attempted"]==ao72["native_returned"]==1680
    assert c72["passed"]and c72["dimension"]==72 and c72["certified_72d_representatives"]==33 and c72["independent"]["precision"]==1024
    ib72=load(cp72/"run/input-binding.json");assert c72["reused"]==ib72["already_proved"]and len(ib72["all72"])==33 and [v["index"]for v in c72["new"]]==ib72["new"]
    periods72=[]
    for row in c72["new"]:
        idx=row["index"];ap=ca72/f"run/row-{idx}/proof.json.gz";pp=cp72/f"run/row-{idx}/proof.json.gz";ip=cp72/f"run/row-{idx}/inputs.json"
        assert c.sha256(ap)==row["independent_proof_sha256"]and c.sha256(pp)==row["producer_proof_sha256"]and c.sha256(ip)==row["input_sha256"]
        with gzip.open(ap,"rt")as f:aa=json.load(f)
        with gzip.open(pp,"rt")as f:bb=json.load(f)
        ii=load(ip);assert aa["passed"]and all(aa["gates"].values())and aa["same_exact_original_box"]and aa["box"]==bb["box"]and aa["dimension"]==72 and aa["period"]==row["period"]==ii["period"]
        assert len(aa["inclusions"])==72 and all(aa["inclusions"])and ball_endpoints(aa["contraction"])[1]<1 and ball_endpoints(aa["power_bounds"][-1]["bound"])[1]<1
        assert [d["divisor"]for d in aa["divisors"]]==[k for k in range(1,ii["period"])if ii["period"]%k==0]and all(d["separating_coordinates"]for d in aa["divisors"])
        periods72.append(ii["period"])
    assert sum(periods72)==840 and len(c72["reused"])==1 and c72["reused"][0]["index"]==86
    old72=Path(c72["reused"][0]["root"])/"certificate.json";assert c.sha256(old72)==c72["reused"][0]["proof"]["sha256"];periods72.append(58)
    indices13={v["index"]for v in c13["new"]+c13["reused"]};indices72=set(ib72["all72"])
    assert indices13.isdisjoint(indices72)and len(indices13|indices72)==c72["agreed_census_representatives"]==61
    for macro,value,calc in [("nCensusAll",61,"complete disjoint28 thirteen-dimensional plus33 seventy-two-dimensional representative indices"),("nCensusSeventyTwo",33,"all72D agreed census representatives including previously proved58-cycle"),("nCensusSeventyTwoNew",32,"new full72D same-box scalar-lossAD proofs"),("nCensusADPhases",1680,"fresh center and original-box native scalar-lossAD phases")]:n.add(macro,value,str(value),proof,c13s+c72s,calc)
    n.add("tCensusSeventyTwoPeriods",sorted(set(periods72)),", ".join(map(str,sorted(set(periods72)))),proof,c72s,"all distinct primitive periods in complete72D census subset")
    allmaps=previous_maps+[load(cp13/f"run/row-{z['index']}/inputs.json")for z in c13["new"]]+[load(cp72/f"run/row-{z['index']}/inputs.json")for z in c72["new"]]
    assert len(allmaps)==64 and len({exact_map_identity(z)for z in allmaps})==64
    n.add("nCertifiedOrbits",len(allmaps),str(len(allmaps)),"individually certified attracting training settings; neutral bifurcations, same-orbit tubes and fixed-weight activity excluded",p18_sources+certificate_paths+[s1r_cert_path]+r51_sources+sources58+c13s+c72s,"six previous exact attracting settings plus26 distinct new13D and32 distinct new72D census settings")
    # Independent705 uniform curve and same-root endpoint connection.
    palp=c.ROOT/"docs/reviews/native705-independent-outcome.json";palj=c.ROOT/"docs/reviews/native705-endpoint-independent-outcome.json"
    pp,pj=load(palp),load(palj);prs=Path(pp["root"]);pjs=Path(pj["root"])
    pals=[palp,palj]+bound_inventory(prs/"manifest.json",pp["outer_manifest_sha256"])+bound_inventory(pjs/"manifest.json",pj["outer_manifest_sha256"])
    po=load(prs/"run/outcome.json");jo=load(pjs/"run/outcome.json");assert po==pp["outcome"]and jo==pj["outcome"]and po["passed"]and jo["passed"]and po["inclusions"]==jo["anchor_coordinates"]==705 and jo["connected"]
    assert jo["parameter_endpoints"]==["-1/4096","63/4096"]and ball_endpoints(po["contraction"])[1]<Fraction(462,1000)
    n.add("nPALUnknowns",705,"705",proof,pals,"full18phase state andtwo-mode shooting plus two physical rates and squared angle")
    n.add("xPALContraction","231/500","0.462",proof,pals,"independent384 strict uniform weighted contraction rounded upward")
    n.add("tPALParameter",jo["parameter_endpoints"],r"[-1/4096,\;63/4096]",proof,pals,"exact affine-gauge interval with independent480 same-root endpoint membership")
    # Uniform complementary stability on the already connected finite704 arc.
    ucp=c.ROOT/"docs/reviews/native704-complement-independent.json";uca=load(ucp);ucr=Path(uca["raw_roots"][-1]["root"])
    bound_inventory(ucr/"manifest.json",uca["outer_manifest"]["sha256"])
    uco=load(ucr/"run/outcome.json");assert uco==uca["outcome"]and uco["passed"]and uco["localization_passed"]and uco["domain_coverage"]and uco["complement_dimension"]==11 and uco["primitive_period"]==18 and uco["parameter_endpoints"]==["0","1/32768"]
    assert ball_endpoints(uco["complementary_modulus_upper"])[1]<Fraction(843,1000)
    n.add("xFiniteUnitComplement","843/1000","0.843",proof,[ucp,ucr/"manifest.json"],"independent480 common-parameter Lyapunov bound on all11 complementary multiplier moduli, rounded strictly upward")
    # Four independently recomputed additional charts and every full-state join.
    cbp=c.ROOT/"docs/reviews/native705-batch-independent-outcome.json";cba=load(cbp);cbr=Path(cba["root"])
    bound_inventory(cbr/"manifest.json",cba["outer_manifest_sha256"]);cbo=load(cbr/"run/outcome.json")
    assert cbo==cba["outcome"]and cbo["passed"]and cbo["connected"]and cbo["connected_count"]==4 and len(cbo["proofs"])==len(cbo["joins"])==4
    right=jo["parameter_endpoints"][1]
    for k,(pr,jr)in enumerate(zip(cbo["proofs"],cbo["joins"],strict=True),1):
        assert pr["unit"]==k and pr["inclusions"]==705 and ball_endpoints(pr["contraction"])[1]<Fraction(483,1000)
        assert jr["passed"]and jr["coordinates"]==705 and jr["parameter"]==right==pr["parameter_endpoints"][0];right=pr["parameter_endpoints"][1]
    assert cbo["connected_parameter_endpoints"]==[jo["parameter_endpoints"][0],right]==["-1/4096","319/4096"]
    cbs=[cbp,cbr/"manifest.json",palp,palj]
    n.add("nPALCharts",5,"5",proof,cbs,"one original independently connected quadratic705 chart plus four independently recomputed exact-gauge joins")
    n.add("xPALConnectedContraction","483/1000","0.483",proof,cbs,"strict upper bound across all five independently proved705 charts")
    n.add("tPALConnectedParameter",cbo["connected_parameter_endpoints"],r"[-1/4096,\;319/4096]",proof,cbs,"union of all five adjoining independently proved705 chart intervals")
    # First five705 charts now independently have all eleven transverse modes stable.
    scp=c.ROOT/"docs/reviews/native705-complement-independent.json";scb=c.ROOT/"docs/reviews/native705-complement-batch-independent.json"
    sc,bc=load(scp),load(scb);stable_sources=[scp,scb];stable_rows=[]
    for rec in [sc,bc]:
        root=Path(rec["root"]);pin=next(z["manifest_sha256"]for z in rec["inventories"]if z["root"]==str(root))
        bound_inventory(root/"manifest.json",pin);stable_sources.append(root/"manifest.json")
        o=load(root/"run/outcome.json");assert o==rec["outcome"]and o["passed"]and o["native_calls"]==0
        stable_rows.extend(o.get("rows",[o]))
    assert len(stable_rows)==5
    endpoint="-1/4096"
    for row in stable_rows:
        assert row["passed"]and row["primitive_period"]==18 and row["complement_dimension"]==11
        assert row["domain_coverage"]and row["own_localization"]and row["own_correlated_tangent"]
        assert row["parameter_endpoints"][0]==endpoint;endpoint=row["parameter_endpoints"][1]
        assert ball_endpoints(row["complementary_modulus_upper"])[1]<Fraction(944,1000)
    assert endpoint=="319/4096"
    n.add("xPALStableComplement","118/125","0.944",proof,stable_sources,"strict upward bound on all eleven complementary multipliers across first five independently connected705 charts")
    # One wide successor plus three rotated eight-chart packets, every join independently proved.
    extra_sources=[];extra_count=0
    names=["native705-forward-independent-outcome"]+[f"native705-rotated-independent-{i:03d}-outcome"for i in range(1,4)]
    previous_root=cbr
    for packet,name in enumerate(names):
        rp=c.ROOT/"docs/reviews"/(name+".json");rec=load(rp);root=Path(rec["root"])
        bound_inventory(root/"manifest.json",rec["outer_manifest_sha256"]);extra_sources.extend([rp,root/"manifest.json"])
        o=load(root/"run/outcome.json");assert o==rec["outcome"]and o["passed"]and o["connected"]and o["native_calls"]==0
        count=1 if packet==0 else 8;assert o["connected_count"]==len(o["proofs"])==len(o["joins"])==count
        binding=load(root/"run/input-binding.json");assert binding["card"]["prior"]["root"]==str(previous_root)
        assert c.sha256(previous_root/"manifest.json")==binding["card"]["prior"]["outer"]
        if packet:assert o["localized_prior"]and o["joins"][0]["parameter_inside"]and not o["proofs"][0]["same_gauge"]
        for pr,jr in zip(o["proofs"],o["joins"],strict=True):
            assert pr["inclusions"]==jr["coordinates"]==705 and jr["passed"]and ball_endpoints(pr["contraction"])[1]<1
        extra_count+=count;previous_root=root
    assert extra_count==25
    n.add("nPALTotalCharts",5+extra_count,str(5+extra_count),proof,cbs+extra_sources,"five original charts, one wider successor and three independently connected eight-chart rotated-gauge packets; gauge coordinates are local")
    n.add("nPALRotatedCharts",24,"24",proof,extra_sources,"three fixed eight-chart packets with independently reconstructed prior localization and full705-coordinate cross-gauge membership")
    # Independent correlated705 larger-domain root and projected known-root join.
    corpath=c.ROOT/"docs/reviews/native705-correlated-independent-outcome.json";cor=load(corpath);corr=Path(cor["root"])
    corp=next(v["manifest_sha256"]for v in cor["inventories"]if v["root"]==str(corr));bound_inventory(corr/"manifest.json",corp)
    co=load(corr/"run/outcome.json");assert cor["passed"]and co==cor["outcome"]and co["passed"]and co["connected"]and co["strict_inclusions"]==co["anchor_coordinates"]==705 and co["primitive_period"]==18
    assert co["parameter_endpoints"]==["-49/2048","79/2048"]and ball_endpoints(co["contraction"])[1]<Fraction(66721,1000000)
    cors=[corpath,corr/"manifest.json"]
    n.add("xPALCorrelatedContraction","66721/1000000","0.066721",proof,cors,"independent480 all705 weighted contraction upper bound rounded upward")
    n.add("tPALCorrelatedParameter",co["parameter_endpoints"],r"[-49/2048,\;79/2048]",proof,cors,"exact new local-gauge domain joined by full705 known-root projection")
    # Three new connected cubic705 charts, independently rebuilt from native arrays.
    cubp=c.ROOT/"docs/reviews/native705-cubic-prefix-independent.json";cub=load(cubp);cubr=Path(cub["root"])
    bound_inventory(cubr/"manifest.json",cub["inventories"][0]["manifest_sha256"])
    cuo=load(cubr/"run/outcome.json");assert cuo==cub["outcome"]and cuo["passed"]and cuo["connected_count"]==3 and cuo["connected_right"]=="115/512"and cuo["native_calls"]==0
    for i,row in enumerate(cuo["charts"],1):
        pp=load(cubr/f"run/chart{i:03d}/independent-proof.json")
        assert row["passed"]and row["strict_inclusions"]==row["anchor_coordinates"]==705 and pp["passed"]and all(pp["gates"].values())
        assert len(pp["root"]["coordinate_inclusions"])==705 and all(pp["root"]["coordinate_inclusions"])and ball_endpoints(row["contraction"])[1]<Fraction(230168,1000000)
    cubs=[cubp,cubr/"manifest.json"]
    n.add("nPALCubicCharts",3,"3",proof,cubs,"three ordered independently480 proved cubic705 tubes and full705 joins")

    n.add("xPALCubicContraction","230168/1000000","0.230168",proof,cubs,"strict upward bound on all three independent weighted contractions")
    n.add("tPALCubicRight","115/512",r"115/512",proof,cubs,"right endpoint in the unchanged local affine gauge shared with the correlated predecessor")
    # Later rotated root and three maintained roots, every connection independently proved.
    more_sources=[];previous=cubr
    for name,count in [("native705-rotated-independent",1),("native705-maintained-independent",3)]:
        rp=c.ROOT/"docs/reviews"/(name+".json");rec=load(rp);root=Path(rec["root"]);pin=rec["inventories"][0]["manifest"]["sha256"]
        bound_inventory(root/"manifest.json",pin);out=load(root/"run/outcome.json");assert out==rec["outcome"]and out["passed"]and out["connected"]and out["native_calls"]==0
        binding=load(root/"run/input-binding.json");prior=binding["card"]["old_independent"if count==1 else "anchor"]
        assert prior["root"]==str(previous)and prior["outer"]==c.sha256(previous/"manifest.json")
        if count==1:
            assert out["strict_inclusions"]==out["anchor_coordinates"]==705 and out["primitive_period"]==18 and ball_endpoints(out["contraction"])[1]<Fraction(100102,1000000)
            pr=load(root/"run/new-root/independent-proof.json");assert pr["passed"]and all(pr["root"]["coordinate_inclusions"])and pr["membership"]["coordinates"]==705
        else:
            assert out["connected_count"]==3 and out["connected_right"]=="63/512"and [x["attempt"]for x in out["charts"]]==[1,2,4]
            for k,row in enumerate(out["charts"],1):
                assert row["passed"]and row["strict_inclusions"]==row["anchor_coordinates"]==705 and ball_endpoints(row["contraction"])[1]<Fraction(251429,1000000)
                pr=load(root/f"run/chart{k:03d}/independent-proof.json");assert pr["passed"]and all(pr["gates"].values())and all(pr["root"]["coordinate_inclusions"])
        previous=root;more_sources += [rp,root/"manifest.json"]

    n.add("xPALMaintainedContraction","251429/1000000","0.251429",proof,more_sources,"strict upward bound across the rotated root and three independently reconstructed maintained roots")
    # Sixth original chart has independently covered closed halves and stable quotient.
    hp=c.ROOT/"docs/reviews/native705-halves-independent.json";hr=load(hp);hroot=Path(hr["root"]);bound_inventory(hroot/"manifest.json",hr["inventories"][0]["manifest_sha256"])
    ho=load(hroot/"run/outcome.json");assert ho==hr["outcome"]and ho["passed"]and ho["native_calls"]==0
    assert ho["parameter_endpoints"]==["319/4096","447/4096"]and len(ho["rows"])==2
    assert [v["parameter_endpoints"]for v in ho["rows"]]==[["319/4096","383/4096"],["383/4096","447/4096"]]
    assert all(v["passed"]and v["same_root"]and v["primitive_period"]==18 and v["complement_dimension"]==11 and ball_endpoints(v["complementary_modulus_upper"])[1]<Fraction(241,250)for v in ho["rows"])
    n.add("xPALSixthComplement","241/250","0.964",proof,[hp,hroot/"manifest.json"],"strict upper bound on both independent closed-half complementary multiplier enclosures")
    # Two independent physical derivative reductions at the same exact local junction.
    cp=c.ROOT/"docs/reviews/unit-flip-corner-independent.json";cr=load(cp);croot=Path(cr["root"]);bound_inventory(croot/"manifest.json",cr["inventories"][0]["manifest_sha256"])
    co=load(croot/"run/outcome.json");assert co==cr["outcome"]and co["passed"]and co["same_unique_root"]and co["native_calls"]==0
    assert ball_endpoints(co["determinant"])[1]<0 and ball_endpoints(co["directional_derivatives"][0])[1]<0 and ball_endpoints(co["directional_derivatives"][1])[0]>0
    assert all(lo<=exact<=hi and hi-lo<Fraction(2)**-500 for (lo,hi),exact in zip(map(ball_endpoints,co["direction"]),[Fraction(-1),Fraction(11,20000)],strict=True))
    assert all(ball_endpoints(v)[1]<1 for v in co["complement_bounds"])
    for macro,v,digits in [("tCornerDeterminant",co["determinant"],2),("tCornerSigmaDirection",co["directional_derivatives"][0],8),("tCornerTauDirection",co["directional_derivatives"][1],5)]:
        lo,hi=ball_endpoints(v);scale=10**digits;a=(lo*scale).numerator//(lo*scale).denominator;b=-((-hi*scale).numerator//(-hi*scale).denominator)
        display="["+format(Decimal(a)/Decimal(scale),f".{digits}f")+", "+format(Decimal(b)/Decimal(scale),f".{digits}f")+"]"
        n.add(macro,{"ball":v,"lower":str(Fraction(a,scale)),"upper":str(Fraction(b,scale))},display,proof,[cp,croot/"manifest.json"],"outward decimal enclosure of independent512 physical derivative; direction selected after producer result")
    # Whole connected geometry and independent complementary stability are separate counts.
    mp=c.ROOT/"docs/reviews/native705-minimum-independent.json";mr=load(mp);mroot=Path(mr["root"])
    bound_inventory(mroot/"manifest.json",mr["inventories"][0]["manifest_sha256"])
    mo=load(mroot/"run/outcome.json");assert mo==mr["outcome"]and mo["passed"]and mo["connected"]and mo["native_calls"]==0
    mb=load(mroot/"run/input-binding.json");assert mb["card"]["prior"]["root"]==str(previous)and mb["card"]["prior"]["outer"]==c.sha256(previous/"manifest.json")
    assert len(mo["charts"])==2
    for i,row in enumerate(mo["charts"],1):
        assert row["passed"]and row["strict_inclusions"]==row["anchor_coordinates"]==705 and ball_endpoints(row["contraction"])[1]<Fraction(134647,1000000)
        pr=load(mroot/f"run/chart{i:03d}/independent-proof.json");assert pr["passed"]and all(pr["gates"].values())and all(pr["root"]["coordinate_inclusions"])and pr["anchor"]["passed"]
    minimum=mo["minimum"];assert minimum["passed"]and minimum["coordinate"]=="hidden_learning_rate"
    edges=[[Fraction(v)for v in row]for row in minimum["endpoint_intervals"]]
    assert edges[0][0]-edges[1][1]==Fraction(minimum["left_gap"])>Fraction(5018993,10**9)
    assert edges[2][0]-edges[1][1]==Fraction(minimum["right_gap"])>Fraction(1016317,10**9)
    minsources=[mp,mroot/"manifest.json"]
    tailp=c.ROOT/"docs/reviews/native705-maintained-tail-independent.json";tail=load(tailp);tailroot=Path(tail["root"])
    bound_inventory(tailroot/"manifest.json",tail["inventories"][0]["manifest_sha256"])
    tailout=load(tailroot/"run/outcome.json");assert tailout==tail["outcome"]and tailout["passed"]and tailout["connected"]and tailout["connected_count"]==19 and tailout["native_calls"]==0
    tailbind=load(tailroot/"run/input-binding.json");assert tailbind["card"]["anchor"]["root"]==str(mroot)and tailbind["card"]["anchor"]["outer"]==c.sha256(mroot/"manifest.json")
    for i in range(1,20):
        pr=load(tailroot/f"run/chart{i:03d}/independent-proof.json");assert pr["passed"]and all(pr["gates"].values())and len(pr["root"]["coordinate_inclusions"])==705 and all(pr["root"]["coordinate_inclusions"])and pr["anchor"]["passed"]and pr["anchor"]["coordinates"]==705
    thirdp=c.ROOT/"docs/reviews/native705-maintained003-independent.json";third=load(thirdp);thirdroot=Path(third["root"])
    bound_inventory(thirdroot/"manifest.json",third["inventories"][0]["manifest_sha256"])
    tho=load(thirdroot/"run/outcome.json");assert tho==third["outcome"]and tho["passed"]and tho["connected"]and tho["connected_count"]==24 and tho["native_calls"]==0
    thb=load(thirdroot/"run/input-binding.json");assert thb["card"]["anchor"]["root"]==str(tailroot)and thb["card"]["anchor"]["outer"]==c.sha256(tailroot/"manifest.json")
    for i in range(1,25):
        pr=load(thirdroot/f"run/chart{i:03d}/independent-proof.json");assert pr["passed"]and all(pr["gates"].values())and len(pr["root"]["coordinate_inclusions"])==705 and all(pr["root"]["coordinate_inclusions"])and pr["anchor"]["passed"]and pr["anchor"]["coordinates"]==705
    angle=load(thirdroot/"run/angle-maximum.json");assert angle==tho["interior_angle_maximum"]and angle["passed"]and angle["endpoint_indices"]==[16,17,18]and all(ball_endpoints(v)[0]>0 for v in angle["gaps"])
    junctionp=c.ROOT/"docs/reviews/native705-junction-chain-independent.json";junction=load(junctionp);jroot=Path(junction["root"])
    bound_inventory(jroot/"manifest.json",junction["inventories"][0]["manifest_sha256"])
    jo=load(jroot/"run/outcome.json");assert jo==junction["outcome"]and all(jo[k]for k in ["passed","connected","junction_connected"])and jo["connected_count"]==18 and jo["membership_coordinates"]==705 and jo["native_calls"]==0
    jb=load(jroot/"run/input-binding.json");assert jb["card"]["anchor"]["root"]==str(thirdroot)and jb["card"]["anchor"]["outer"]==c.sha256(thirdroot/"manifest.json")
    for i in range(1,19):
        pr=load(jroot/f"run/chart{i:03d}/independent-proof.json");assert pr["passed"]and all(pr["gates"].values())and len(pr["root"]["coordinate_inclusions"])==705 and all(pr["root"]["coordinate_inclusions"])and pr["anchor"]["passed"]and pr["anchor"]["coordinates"]==705
    jm=load(jroot/"run/junction-membership.json");assert jm["passed"]and jm["parameter_inside"]and jm["coordinates"]==len(jm["coordinate_inclusions"])==705 and all(jm["coordinate_inclusions"])
    n.add("nPALVisualCharts",101,"101",proof,extra_sources+cors+cubs+more_sources+minsources+[tailp,tailroot/"manifest.json",thirdp,thirdroot/"manifest.json",junctionp,jroot/"manifest.json"],"101 independently joined positive-angle full705 charts connecting generic1:1 and the same complex-unit/simple-flip junction; strict705 known-root membership, initial704 arc excluded")
    quarterp=c.ROOT/"docs/reviews/native705-fresh-quarter-independent.json";quarter=load(quarterp);qroot=Path(quarter["root"]);bound_inventory(qroot/"manifest.json",quarter["outer"]["sha256"])
    qo=load(qroot/"run/outcome.json");assert qo==quarter["outcome"]and qo["passed"]and qo["native_calls"]==0 and qo["new_cells"]==16 and qo["total_cells"]==17 and qo["parameter_endpoints"]==["-49/2048","-17/2048"]and qo["complement_dimension"]==11
    assert len(qo["completed_cells"])==16 and all(v["passed"]and ball_endpoints(v["modulus"])[1]<Fraction(878041,10**6)for v in qo["completed_cells"])
    prior=load(Path(quarter["binding"]["prior"]["root"])/"run/outcome.json");assert prior["passed"]and prior["complement_dimension"]==11 and ball_endpoints(prior["complementary_modulus_upper"])[1]<Fraction(878041,10**6)
    union=load(qroot/"run/union.json");assert union["passed"]and union["parameter_endpoints"]==qo["parameter_endpoints"]and union["total_cells"]==17
    n.add("xFreshQuarterComplement","878041/1000000","0.878041",proof,[quarterp,qroot/"manifest.json"],"strict upward bound on all11 complementary multipliers on exact17-cell closed cover of chart31 first quarter; same root curve; full chart coverage and stable-prefix connection are now separately proved")
    secondp=c.ROOT/"docs/reviews/native705-second-quarter-independent.json";second=load(secondp);sr=Path(second["root"])
    bound_inventory(sr/"manifest.json",second["inventories"][0]["manifest"]["sha256"]);so=load(sr/"run/outcome.json")
    lastp=c.ROOT/"docs/reviews/native705-remaining-quarters-independent.json";last=load(lastp);lr=Path(last["root"])
    bound_inventory(lr/"manifest.json",last["inventories"][0]["manifest"]["sha256"]);las=load(lr/"run/outcome.json")
    assert so==second["outcome"]and las==last["outcome"]and so["passed"]and las["passed"]and so["native_calls"]==las["native_calls"]==0
    assert so["parameter_endpoints"]==["-17/2048","15/2048"]and las["parameter_endpoints"]==["15/2048","79/2048"]and las["whole_chart_endpoints"]==["-49/2048","79/2048"]
    assert so["total_cells"]==4 and las["total_cells"]==10 and all(v["passed"]and v["complement_dimension"]==11 and ball_endpoints(v["modulus"])[1]<Fraction(980311,10**6)for v in so["completed_cells"]+las["completed_cells"])
    whole31sources=[quarterp,qroot/"manifest.json",secondp,sr/"manifest.json",lastp,lr/"manifest.json"]
    n.add("xChartThirtyOneComplement","980311/1000000","0.980311",proof,whole31sources,"strict upward bound for all11 complementary moduli on exact whole chart31 closed cover")
    n.add("nChartThirtyOneCells",31,"31",proof,whole31sources,"17first-quarter+4second-quarter+10remaining-half independently checked closed cells")
    hmp=c.ROOT/"docs/reviews/native705-chart31-boundary-hypotheses.json";hm=load(hmp)
    assert hm["passed"]and hm["chart"]==31 and hm["independent_complement_cells"]==31 and hm["parameter_endpoints"]==las["whole_chart_endpoints"]and hm["authorities"]["last"]["root"]==str(lr)
    for k,spec in hm["authorities"].items():
        rr=Path(spec["root"]);bound_inventory(rr/"manifest.json",spec["outer"])
        for name,e in hm["artifacts"][k].items():assert c.sha256(rr/"run"/name)==e["sha256"]
    for name,e in hm["source_equations"].items():
        if (c.ROOT/name).exists():assert c.sha256(c.ROOT/name)==e["sha256"]
    import hashlib
    actual=(PAPER/"main.tex").read_text();a=actual.index(r"\begin{lemma}[Physical stability boundary]");b=actual.index("For our analytic training map",a)
    assert hashlib.sha256(actual[a:b].encode()).hexdigest()==hm["analytic_lemma"]["lemma_and_compact_argument_sha256"]
    flipadvp=c.ROOT/"docs/reviews/native470-advancing-independent.json";fa=load(flipadvp);far=Path(fa["root"])
    bound_inventory(far/"manifest.json",fa["inventories"][0]["manifest"]["sha256"]);fao=load(far/"run/outcome.json")
    assert fao==fa["outcome"]and fao["passed"]and fao["native_calls"]==0 and fao["root_coordinates"]==fao["anchor_coordinates"]==470 and ball_endpoints(fao["contraction"])[1]<Fraction(545203,10**6)
    n.add("xAdvancingFlipContraction","545203/1000000","0.545203",proof,[flipadvp,far/"manifest.json"],"strict upper full470 contraction on independently joined same-hidden-gauge interval of width1/16")
    freshp=c.ROOT/"docs/reviews/native705-fresh-center-independent.json";fresh=load(freshp);froot=Path(fresh["root"]);bound_inventory(froot/"manifest.json",fresh["outer"]["sha256"])
    fo=load(froot/"run/outcome.json");assert fo==fresh["outcome"]and fo["passed"]and fo["native_calls"]==0 and fo["complement_dimension"]==11 and fo["subinterval_only"]and fo["parameter_endpoints"]==["-17/1024","-1/64"]
    assert ball_endpoints(fo["complementary_modulus_upper"])[1]<Fraction(875102,10**6)
    n.add("xFreshCenterComplement","437551/500000","0.875102",proof,[freshp,froot/"manifest.json"],"strict upward bound for all11 complementary multipliers only on fixed chart31 center subinterval")
    currentp=c.ROOT/"docs/reviews/cifar-current512-independent.json";current=load(currentp);curoot=Path(current["root"]);bound_inventory(curoot/"manifest.json",current["inventories"][0]["manifest"]["sha256"])
    cu=load(curoot/"run/outcome.json");assert cu==current["outcome"]and cu["passed"]and cu["native_calls"]==0 and cu["recurrence_rows"]==511 and cu["numerical_action_rank"]==512
    correction=cu["correction"];assert correction["direction_gate"]["passed"]and correction["accepted_index"]==6 and len(correction["trials"])==7 and not correction["stationarity_qualified"]
    assert abs(correction["direction_gate"]["true_forcing"]-.419126460267590)<1e-12 and correction["trials"][-1]["alpha"]==1/64
    cs=[currentp,curoot/"manifest.json"]
    n.add("nCifarCurrentBasis",512,"512","exploratory, independently cross-checked",cs,"full same-current-state qualified action basis rank")
    n.add("xCifarCurrentForcing",correction["direction_gate"]["true_forcing"],"0.419126","exploratory, independently cross-checked",cs,"true physical forcing at actual current state, six decimal places")
    n.add("xCifarCurrentGradient",correction["relative_gradient"],"2.57082\\times10^{-4}","exploratory, independently cross-checked",cs,"accepted current-state relative gradient, six significant figures")
    n.add("xCifarCurrentMeritRatio",cu["merit_ratio"],"0.991753","exploratory, independently cross-checked",cs,"actual new over old squared gradient norm, six decimal places")

    n.add("xPALMinimumLeftGap","5018993/1000000000","0.005018993",proof,minsources,"strict downward bound on first endpoint lower minus middle endpoint upper, exact rational subtraction")
    n.add("xPALMinimumRightGap","1016317/1000000000","0.001016317",proof,minsources,"strict downward bound on last endpoint lower minus middle endpoint upper, exact rational subtraction")
    cp=c.ROOT/"docs/reviews/native705-march-independent.json";cr=load(cp);croot=Path(cr["root"])
    bound_inventory(croot/"manifest.json",cr["inventories"][0]["manifest_sha256"])
    co=load(croot/"run/outcome.json");unions=load(croot/"run/same-root-unions.json")
    assert co==cr["outcome"]and unions==cr["same_root_unions"]and co["passed"]and co["native_calls"]==0 and co["original_charts"]==24 and co["accepted_leaves"]==48
    assert unions["passed"]and [v["ordinal"]for v in unions["charts"]]==list(range(7,31))
    for original in unions["charts"]:
        rows=[r for r in co["rows"]if r["ordinal"]==original["ordinal"]];assert len(rows)==original["leaves"]and original["passed"]and original["same_gauge_within_original_chart"]
        pieces=sorted([[Fraction(x)for x in row["cell"]["endpoints"]]for row in rows]);assert pieces[0][0]==Fraction(original["parameter_endpoints"][0])and pieces[-1][1]==Fraction(original["parameter_endpoints"][1])and all(a[1]==b[0]for a,b in zip(pieces,pieces[1:]))
        for row in rows:assert row["passed"]and row["same_root"]and row["primitive_period"]==18 and row["complement_dimension"]==11 and ball_endpoints(row["complementary_modulus_upper"])[1]<Fraction(497679,500000)
    compsources=[cp,croot/"manifest.json",hp,hroot/"manifest.json"]
    n.add("nPALStableCharts",31,"31",proof,compsources+extra_sources+whole31sources+[hmp],"first30 plus independently covered wholechart31, exact prior-endpoint inclusion and all existing physical-stability-boundary lemma hypotheses")
    n.add("xPALMarchComplement","497679/500000","0.995358",proof,compsources,"strict upper bound across48 independently proved complementary leaves covering charts7through30")
    lp=c.ROOT/"docs/reviews/tiny-lm-ordering-independent.json";lr=load(lp);lroot=Path(lr["root"]);bound_inventory(lroot/"manifest.json",lr["inventories"][0]["manifest"]["sha256"])
    lo=load(lroot/"run/result.json");assert lo==lr["outcome"]and lo["passed"]and lo["native_calls"]==0 and lo["rows"]==16 and lo["retained_hvp_checked"]==160
    rows=lo["summary"]["selected_rows"];assert len(rows)==2 and all(v["qualified"]for v in rows)and not lo["summary"]["material_reversal_odd_in_both_expanded_gradients"]and lo["summary"]["material_threshold"]==.5
    n.add("tLMOrderingOddRatios",[v["odd_to_error"]for v in rows],"1.14\\times10^{-5},\\ 2.78\\times10^{-4}","exploratory, independently cross-checked",[lp,lroot/"manifest.json"],"two independently reduced odd-to-frozen-error ratios;3significant figures; original seed order")
    # Independently reconstructed causal horizon warnings on consumed LM prefixes.
    lhp=c.ROOT/"docs/reviews/tiny-lm-horizon-independent-outcome.json";lh=load(lhp);lhr=Path(lh["root"])
    lpin=next(v["manifest_sha256"]for v in lh["inventories"]if v["root"]==str(lhr));bound_inventory(lhr/"manifest.json",lpin)
    lho=load(lhr/"run/result.json");assert lho["passed"]and lho["native_calls"]==0 and lho["rows_recomputed"]==32 and lho["retained_native_operations_checked"]==dict(hvp=72,ce=32)
    assert {k:v for k,v in lho.items()if k!="rows"}==lh["outcome"]
    ls=lho["summary"];lc=ls["future_warning_counts"];assert ls["total_rows"]==ls["resolved_rows"]==32 and ls["future_eligible_rows"]==24 and ls["already_failing_rows"]==8
    assert lc["drift"]==lc["gradient_change"]==lc["baseline_composite"]==dict(true_warning=1,false_warning=7,missed_failure=0,correct_quiet=16)
    assert lc["static_curvature"]==dict(true_warning=1,false_warning=5,missed_failure=0,correct_quiet=18)
    assert lc["loss_change"]==lc["persistent_defect"]==dict(true_warning=0,false_warning=0,missed_failure=1,correct_quiet=23)and ls["drift_only_caught_failures"]==0
    lhs=[lhp,lhr/"manifest.json"];lhl="independently checked saved32 correlated consumed development prefixes; shared native acquisitions, no incremental utility"
    for macro,value,calc in [("nLHRows",ls["total_rows"],"all fixed prefixes"),("nLHEligible",ls["future_eligible_rows"],"all qualified prefixes below20percent frozen error at first two horizons"),("nLHAlready",ls["already_failing_rows"],"resolved prefixes already over threshold"),("nLHTrue",lc["drift"]["true_warning"],"sole eligible future failure also caught by gradient and curvature"),("nLHDriftFalse",lc["drift"]["false_warning"],"false drift warnings"),("nLHCurvatureFalse",lc["static_curvature"]["false_warning"],"false static-curvature warnings")]:n.add(macro,value,str(value),lhl,lhs,calc)
    # Independent negative development comparisons; no fresh holdout or utility claim.
    rp=c.ROOT/"docs/reviews/cifar-regularized-independent.json";rr=load(rp);root=Path(rr["root"])
    bound_inventory(root/"manifest.json",rr["inventories"][0]["manifest"]["sha256"]);ro=load(root/"run/outcome.json")
    assert ro==rr["outcome"]and ro["passed"]and ro["native_calls"]==0 and not ro["improves_same_start_merit"]and not ro["correction"]["stationarity_qualified"]
    assert ro["correction"]["accepted_index"]==5 and ro["correction"]["trials"][-1]["alpha"]==1/32 and ro["regularized_merit_ratio"]>ro["baseline_merit_ratio"]
    pp=c.ROOT/"docs/reviews/cifar-regularized001.json";pr=load(pp);proot=Path(pr["root"]);bound_inventory(proot/"manifest.json",pr["inventories"][0]["manifest"]["sha256"])
    normratio=pr["outcome"]["comparison"]["direction_norm_ratio"]
    label="exploratory, independently cross-checked same-start negative comparison"
    n.add("xCifarRegularizedNormRatio",normratio,fixed(normratio,6),label,[rp,root/"manifest.json",pp,proot/"manifest.json"],"physical direction norm divided by original same-start direction norm; separate independent solve agrees")
    n.add("xCifarRegularizedMeritRatio",ro["regularized_merit_ratio"],fixed(ro["regularized_merit_ratio"],6),label,[rp,root/"manifest.json"],"actual regularized new/old squared-gradient merit; larger accepted alpha does not imply better progress")
    mp=c.ROOT/"docs/reviews/tiny-lm-temporal-mean-independent.json";mr=load(mp);mroot=Path(mr["root"])
    bound_inventory(mroot/"manifest.json",mr["inventories"][0]["manifest"]["sha256"]);mo=load(mroot/"run/result.json")
    assert mo==mr["outcome"]and mo["passed"]and mo["native_calls"]==0 and mo["counts"]==dict(hvp=1088,gradient=162,ce=58)and mo["rows"]==12 and not mo["descriptive_gate"]
    assert [(z["seed"],z["rate_role"])for z in mo["cases"]]==[(28161,"original"),(28161,"quarter_step"),(28162,"original"),(28162,"quarter_step")]
    assert all(z["mechanism_qualified"]and z["training_ce_slope_resolved"]for case in mo["cases"]for z in case["rows"])
    assert all(case["small_step_monotone"]and max(case["objective_increments"])<=0 for case in mo["cases"]if case["rate_role"]=="quarter_step")
    label="exploratory, independently cross-checked full retained temporal-mean response; consumed development"
    for name,role,model in [("tLMMeanOriginalErrors","original","mean"),("tLMMeanControlErrors","quarter_step","mean"),("tLMMeanControlFrozen","quarter_step","frozen")]:
        values=[case["rows"][-1]["comparisons"][model]["vector_error"]for case in mo["cases"]if case["rate_role"]==role]
        n.add(name,values,",\\ ".join(fixed(v,5)for v in values),label,[mp,mroot/"manifest.json"],"horizon16 vector errors for two initializations in fixed seed order, "+role+" "+model)
    n.add("nLMMeanCalls",sum(mo["counts"].values()),str(sum(mo["counts"].values())),label,[mp,mroot/"manifest.json"],"every retained input and return pair independently associated with actual state/direction and recurrence")
    # Full-state parity ordering contrast: independently replayed fixed development cases.
    pp=c.ROOT/"docs/reviews/tiny-lm-parity-order-independent.json";pr=load(pp);proot=Path(pr["root"])
    bound_inventory(proot/"manifest.json",pr["inventories"][0]["manifest"]["sha256"]);po=load(proot/"run/result.json")
    assert po==pr["outcome"]and po["passed"]and po["native_calls"]==0 and po["retained_hvp_checked"]==224 and po["old_reverse_hvp_checked"]==32 and po["descriptive_gate"]
    assert [(z["seed"],z["rate_role"])for z in po["cases"]]==[(28161,"original"),(28161,"quarter_step"),(28162,"original"),(28162,"quarter_step")]
    assert all(z["qualified"]and z["inherited_response_qualified"]and z["chronological_replay"]["passed"]and max(x["error"]for x in z["chronological_replay"]["rows"])==0 for z in po["cases"])
    label="exploratory, independently cross-checked full15776D saved-Jacobian order intervention; consumed development"
    for macro,role,key in [("tLMParityOriginalEven","original","even_relative_ordered"),("tLMParityControlEven","quarter_step","even_relative_ordered"),("tLMParityFrozenShare","original","even_relative_frozen_error")]:
        vals=[z["comparison"][key]for z in po["cases"]if z["rate_role"]==role]
        n.add(macro,vals,",\\ ".join(fixed(v,6)for v in vals),label,[pp,proot/"manifest.json"],"same two seeds in order: "+role+" "+key+"; norm ratios are not additive attribution shares")
    for j,z in enumerate(po["cases"]):
        tag=["OneOriginal","OneQuarter","TwoOriginal","TwoQuarter"][j]
        for key,description in [("P","chronological response"),("Q","even-indices then odd-indices response")]:
            val=z["comparison"]["products"][key]["gain"]
            n.add("xLMParity"+tag+key,val,fixed(val,6),label,[pp,proot/"manifest.json"],description+" norm, unit original direction")
        val=z["comparison"]["even_relative_ordered"]
        n.add("xLMParity"+tag+"Even",val,fixed(val,6),label,[pp,proot/"manifest.json"],"reversal-even ordering contrast norm divided by chronological response norm")
    # Independently reduced same-start curved correction, first passing point retained.
    cp=c.ROOT/"docs/reviews/cifar-curved-path-independent.json";cr=load(cp);croot=Path(cr["root"])
    bound_inventory(croot/"manifest.json",cr["outer"]["sha256"]);co=load(croot/"run/result.json")
    assert co==cr["outcome"]and co["passed"]and co["native_calls"]==0 and co["counts"]==dict(hvp=1,gradient=2)and co["improves_same_start"]
    assert co["arithmetic"]["accepted_index"]==0 and all(z["accepted"]for z in co["arithmetic"]["trials"])
    assert co["arithmetic"]["trials"][0]["alpha"]==1/32 and co["arithmetic"]["trials"][1]["alpha"]==1/64
    label="exploratory, independently replayed same-start curved gradient-merit correction"
    for macro,val,calc in [("xCifarCurvedMeritRatio",co["chosen_merit_ratio"],"fixed first passingalpha1/32 squared-gradient merit ratio"),("xCifarCurvedSecondRatio",co["arithmetic"]["trials"][1]["merit_ratio"],"retained secondalpha1/64 merit ratio, not retrospectively selected"),("pCifarCurvedDecrease",100*(1-co["chosen_merit_ratio"]),"percent actual same-start merit decrease for selected curved trial"),("pCifarStraightDecrease",100*(1-co["baseline_merit_ratio"]),"percent actual same-start merit decrease for original straight trial")]:
        n.add(macro,val,fixed(val,2 if macro.startswith("p")else 6),label,[cp,croot/"manifest.json"],calc)
    # Exact generalized flip of the signed native RNN return, independently512 audited.
    crp=c.ROOT/"docs/reviews/clark-rnn-generalized-flip-independent.json";cr=load(crp);crr=Path(cr["root"])
    assert cr["passed"]and cr["source"]=="e62b17ec13d41a5dce79eab4a88d7052745f5b18"
    cpin=next(z["manifest_sha256"]for z in cr["inventories"]if z["root"]==str(crr))
    bound_inventory(crr/"manifest.json",cpin);cro=load(crr/"run/outcome.json")
    assert cro==cr["outcome"]and cro["passed"]and cro["primitive_period"]==6 and cro["complement_dimension"]==11
    assert cro["same_root"]["all26_original_boxes_strictly_inside"]and cro["calls"]["native_phases"]==54
    crt=load(crr/"run/runtime.json");assert crt["precision"]==512 and crt["source_commit"]==cr["source"]
    assert cr["launcher"]["exit_code"]==0 and not cr["launcher"]["owned_live_pids"]and cr["launcher"]["worker_pid_absent"]
    for name in ["saved-root","fresh-root"]:
        z=load(crr/"run"/(name+".json"));assert z["passed"]and len(z["coordinate_inclusions"])==26 and all(z["coordinate_inclusions"])and ball_endpoints(z["contraction"])[1]<1
    native=load(crr/"run/native-context.json");assert native["primitive_period"]==6 and native["complement_dimension"]==11 and set(native["divisors"])=={"1","2","3"}
    assert all(any(not(lo<=0<=hi)for lo,hi in map(ball_endpoints,row))for row in native["divisors"].values())
    assert cro["power_bounds"]==native["power_bounds"]and native["witness_power"]==2 and ball_endpoints(cro["power_bounds"][-1]["bound"])[1]<Fraction(479,1000)
    assert ball_endpoints(cro["quintic"])[0]>0 and ball_endpoints(cro["physical_determinant"])[0]>0 and ball_endpoints(cro["multiplier_leak_derivative"])[0]>0 and ball_endpoints(cro["critical_cubic_gain_derivative"])[1]<0
    crs=[crp,crr/"manifest.json"]
    for macro,key,digits in [("xRNNOrganizerGain","gain",12),("xRNNOrganizerLeak","leak",12),("xRNNOrganizerQuintic","quintic",9),("xRNNOrganizerDeterminant","physical_determinant",9),("xRNNOrganizerLambdaLeak","multiplier_leak_derivative",9),("xRNNOrganizerCubicGain","critical_cubic_gain_derivative",9)]:
        lo,hi=ball_endpoints(cro[key]);n.add(macro,str((lo+hi)/2),fixed(float((lo+hi)/2),digits),proof,crs,"midpoint display of independently512 enclosed "+key+"; strict signs validated from exact dyadic endpoints")
    n.add("xRNNOrganizerComplement","479/1000","0.479",proof,crs,"strict upward bound on norm of signed critical quotient squared; nativeF6 quotient has this square")
    n.add("nRNNOrganizerPrecision",512,"512",proof,crs,"independent exact same26-dimensional root and normal-form arithmetic precision bits")
    # Exact organizer-to-continued-critical-root bridge of fixed-weight activity.
    brp=c.ROOT/"docs/reviews/clark-critical-bridge-independent.json";br=load(brp);brr=Path(br["root"])
    bound_inventory(brr/"manifest.json",br["inventories"][0]["manifest_sha256"]);bo=load(brr/"run/outcome.json")
    assert bo==br["outcome"]and br["passed"]and bo["passed"]and bo["primitive_period"]==6 and bo["strict_inclusions"]==bo["organizer_join"]==bo["endpoint_join"]==bo["native_return_join"]==26 and bo["mode_orientation"]==1
    assert bo["critical_native_multiplier"]==1 and bo["critical_signed_multiplier"]==-1 and bo["native_calls"]==18 and bo["shared_augmented_calls"]==0
    assert ball_endpoints(bo["contraction"])[1]<Fraction(148017,1000000)and bo["whole_quotient_power"]["power"]==2 and ball_endpoints(bo["whole_quotient_power"]["bound"])[1]<Fraction(577,1000)
    brs=[brp,brr/"manifest.json"]
    n.add("xRNNBridgeContraction","148017/1000000","0.148017",proof,brs,"strict upward uniform26D critical-curve contraction bound")
    n.add("xRNNBridgeComplement","577/1000","0.577",proof,brs,"strict upward whole signed quotient-square norm bound; nativeF6 complementary attraction")
    # Positive-amplitude critical continuation reaches the original generic native6 fold.
    erp=c.ROOT/"docs/reviews/clark-correlated-extension-independent.json";er=load(erp);err=Path(er["root"]);bound_inventory(err/"manifest.json",er["outer"]["sha256"])
    eo=load(err/"run/outcome.json");assert eo==er["outcome"]and eo["passed"]and eo["native_calls"]==0 and eo["range"]==["0","1/2048"]and eo["new_independent_tubes"]==2 and eo["prior_independent_tubes"]==6 and eo["endpoint_coordinate_inclusions"]==51 and eo["endpoint_memberships"]==[1]and eo["new_native_contexts"]==12
    assert ball_endpoints(eo["leak_aa"])[1]<0
    lo,hi=ball_endpoints(eo["leak_bb"]);assert Fraction(-15859138005,10**12)<lo<hi<Fraction(-15859138003,10**12)
    eb=load(err/"run/input-binding.json");prp=c.ROOT/"docs/reviews/clark-correlated-prefix-independent.json";prefix=load(prp);assert eb["independent_prefix"]["root"]==prefix["root"]and eb["independent_prefix"]["outer"]==prefix["inventories"][0]["manifest_sha256"]
    bound_inventory(Path(prefix["root"])/"manifest.json",prefix["inventories"][0]["manifest_sha256"]);assert prefix["outcome"]["passed"]and len(prefix["accepted"])==6 and prefix["outcome"]["range"]==["0","3/8192"]
    es=[erp,err/"manifest.json",prp,Path(prefix["root"])/"manifest.json"]
    n.add("nRNNFoldTubes",8,"8",proof,es,"six plus two independently joined full51 critical tubes from the exact organizer to squared amplitude1/2048")
    n.add("tRNNFoldCurvature",[str(lo),str(hi)],r"[-0.015859138005,\,-0.015859138003]",proof,es,"outward-rounded independent total physical leak curvature with respect to squared amplitude, at original selected fixed gain")
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
