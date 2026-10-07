"""Derive every number quoted in the gd-route manuscript from saved evidence.

Writes three files next to the manuscript:

  generated-numbers.tex   one \\newcommand per quoted number (the text uses only these)
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


class Numbers:
    def __init__(self):
        self.entries = []

    def add(self, macro, value, tex, evidence, sources, computation):
        assert macro.isalpha(), macro
        assert macro not in {e["macro"] for e in self.entries}, macro
        self.entries.append({"macro": macro, "value": json.loads(json.dumps(value, default=float)), "tex": tex,
                             "evidence": evidence, "sources": [str(Path(s)) for s in sources],
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

    fig6 = load(FIG6)["derived"]
    for key, name in (("B", "QP"), ("C", "Lock"), ("D", "Chaos")):
        n.add(f"xFigSix{name}Rho", fig6[key]["rho"], fixed(fig6[key]["rho"], 4), EXPLORATORY, [FIG6, STAIR_S2],
              f"figure 6 panel {key} rotation number (staircase receipt)")
        n.add(f"xFigSix{name}Ftle", fig6[key]["ftle"], fixed(fig6[key]["ftle"], 4), EXPLORATORY, [FIG6, STAIR_S2],
              f"figure 6 panel {key} FTLE (staircase receipt)")
    return n


def render(n):
    lines = ["% Generated by paper/gd-route/scripts/build_numbers.py from saved evidence; do not edit by hand.",
             "% Each macro's source, SHA-256 and computation are listed in data/numbers.json."]
    lines += [f"\\newcommand{{\\{e['macro']}}}{{{e['tex']}}}" for e in n.entries]
    return "\n".join(lines) + "\n"


def manifest(n):
    sources = sorted({s for e in n.entries for s in e["sources"]})
    return {"generator": str(Path(__file__).resolve().relative_to(c.ROOT)),
            "generator_sha256": c.sha256(__file__),
            "sources": {s: c.sha256(s) for s in sources},
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
    tex, data = render(n), manifest(n)
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
        if load(INDEX) != index():
            problems.append("provenance.json index is stale")
        if problems:
            raise SystemExit("numbers: VERIFY FAILED: " + "; ".join(problems))
        print(json.dumps({"numbers": len(n.entries), "verified": True}))
        return
    MANIFEST.parent.mkdir(exist_ok=True)
    TEX.write_text(tex)
    MANIFEST.write_text(json.dumps(data, indent=1) + "\n")
    INDEX.write_text(json.dumps(index(), indent=1) + "\n")
    print(json.dumps({"numbers": len(n.entries), "tex": str(TEX)}))


if __name__ == "__main__":
    main()
