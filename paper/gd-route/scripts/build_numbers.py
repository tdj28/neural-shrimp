"""Derive every number quoted in the gd-route manuscript from saved evidence.

Writes four files next to the manuscript:

  generated-numbers.tex   one \\newcommand per quoted number (the text uses only these)
  generated-design-table.tex  the design table, built from the frozen plan and summary
  data/numbers.json       the manifest: each macro's value, evidence label, source
                          file(s) with SHA-256 and the computation that yields it
  provenance.json         index of the manifest and every figure receipt, by hash

With --verify nothing is written: every number is recomputed, and the run
fails if a value, a source hash, the generated TeX or the index differs.
"""

import json
import sys
from fractions import Fraction
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

    fig6 = load(FIG6)["derived"]
    for key, name in (("B", "QP"), ("C", "Lock"), ("D", "Chaos")):
        n.add(f"xFigSix{name}Rho", fig6[key]["rho"], fixed(fig6[key]["rho"], 4), EXPLORATORY, [FIG6, STAIR_S2],
              f"figure 6 panel {key} rotation number (staircase receipt)")
        n.add(f"xFigSix{name}Ftle", fig6[key]["ftle"], fixed(fig6[key]["ftle"], 4), EXPLORATORY, [FIG6, STAIR_S2],
              f"figure 6 panel {key} FTLE (staircase receipt)")
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
    tex, data, table = render(n), manifest(n), design_table()
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
        if load(INDEX) != index():
            problems.append("provenance.json index is stale")
        if problems:
            raise SystemExit("numbers: VERIFY FAILED: " + "; ".join(problems))
        print(json.dumps({"numbers": len(n.entries), "verified": True}))
        return
    MANIFEST.parent.mkdir(exist_ok=True)
    TEX.write_text(tex)
    TABLE.write_text(table)
    MANIFEST.write_text(json.dumps(data, indent=1) + "\n")
    INDEX.write_text(json.dumps(index(), indent=1) + "\n")
    print(json.dumps({"numbers": len(n.entries), "tex": str(TEX)}))


if __name__ == "__main__":
    main()
