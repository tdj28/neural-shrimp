"""Score the loop-free cheap-predictor test (run CP) as frozen in docs/start-state-predictor.md.

Per model (the 14 dense models): from llm-soft-loop/pod64/start-state/MODEL/start-state.json, the start-state
intermediate-entropy breadth at prefix 0 (the paper's prefix) and its median over the ten prefixes, log g_net at
prefix 0 and the one-step gain maximum at prefix 0. Outcome: chaotic decades of each model's unsteered sweep, as frozen
in the predictions file (FROZEN_DECADES, computed before that file); the values of the current scaling summary
(docs/scale-summary.json, with the later wide and upper sweeps) are reported as a sensitivity analysis, not scored.

CP1: Spearman(breadth at prefix 0, decades) >= 0.5. CP2: Spearman(median breadth, decades) >= 0.5. CP3: descriptive
Spearman of the decades with log g_net, the one-step gain maximum and log parameters.

Usage: python tools/start_state_score.py OUT.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
# model key, scaling-table family and label, parameters (billions), frozen chaotic decades
FROZEN = [
    ("qwen0.5b", "Qwen2.5", "0.5B", 0.49, 0.018), ("qwen1.5b", "Qwen2.5", "1.5B", 1.54, 0.231),
    ("qwen3b", "Qwen2.5", "3B", 3.09, 0.030), ("qwen7b", "Qwen2.5", "7B", 7.62, 0.165),
    ("qwen14b", "Qwen2.5", "14B", 14.77, 0.050), ("qwen32b", "Qwen2.5", "32B", 32.76, 0.010),
    ("qwen72b", "Qwen2.5", "72B", 72.71, 0.000), ("llama3.1-8b", "Llama-3.1", "8B", 8.03, 0.244),
    ("llama3.1-70b", "Llama-3.1", "70B", 70.55, 0.031), ("llama3.3-70b", "Llama-3.3", "70B", 70.55, 0.000),
    ("olmo2-1b", "OLMo-2", "1B", 1.48, 0.050), ("olmo2-7b", "OLMo-2", "7B", 7.30, 0.081),
    ("olmo2-13b", "OLMo-2", "13B", 13.72, 0.000), ("olmo2-32b", "OLMo-2", "32B", 32.23, 0.252),
]


def spearman(x, y):
    from scipy.stats import spearmanr

    r = spearmanr(x, y)
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": len(x)}


def main():
    import explore_llm_attractors as ela

    out = Path(sys.argv[1])
    current = {(r["family"], r["model"]): r for r in json.loads((ROOT / "docs" / "scale-summary.json").read_text())["rows"]}
    rows = []
    for key, fam, label, params, dec in FROZEN:
        p = ela.DATA / "explore" / "llm-soft-loop" / "pod64" / "start-state" / key / "start-state.json"
        if not p.exists():
            rows.append({"model": key, "missing": True})
            continue
        pref = json.loads(p.read_text())["prefixes"]
        if len(pref) != 10 or pref[0]["prefix_index"] != 0:
            raise SystemExit(f"{key}: expected ten prefixes starting at index 0")
        cur = current.get((fam, label))
        rows.append({"model": key, "missing": False, "params_b": params, "decades_frozen": dec,
                     "decades_current": None if cur is None or cur.get("missing") else cur["chaos_decades"],
                     "breadth_p0": pref[0]["breadth_decades"],
                     "breadth_median": float(np.median([q["breadth_decades"] for q in pref])),
                     "log_g_net_p0": pref[0]["log_g_net_median"], "one_step_max_p0": pref[0]["one_step_gain_max"],
                     "entropy_T1_p0": pref[0]["entropy_at_T1"]})
    ok = [r for r in rows if not r["missing"]]
    dec = [r["decades_frozen"] for r in ok]
    res = {"models_present": len(ok), "models_missing": [r["model"] for r in rows if r["missing"]],
           "CP1": spearman([r["breadth_p0"] for r in ok], dec), "CP2": spearman([r["breadth_median"] for r in ok], dec),
           "CP3": {"log_g_net_p0": spearman([r["log_g_net_p0"] for r in ok], dec),
                   "one_step_max_p0": spearman([r["one_step_max_p0"] for r in ok], dec),
                   "log_params": spearman([np.log(r["params_b"]) for r in ok], dec)}}
    res["CP1"]["holds"] = res["CP1"]["rho"] >= 0.5
    res["CP2"]["holds"] = res["CP2"]["rho"] >= 0.5
    cur = [r for r in ok if r["decades_current"] is not None]
    res["sensitivity_current_decades"] = {
        "breadth_p0": spearman([r["breadth_p0"] for r in cur], [r["decades_current"] for r in cur]),
        "breadth_median": spearman([r["breadth_median"] for r in cur], [r["decades_current"] for r in cur]),
        "log_params": spearman([np.log(r["params_b"]) for r in cur], [r["decades_current"] for r in cur])}
    out.write_text(json.dumps({"score": res, "rows": rows}, indent=1))
    for r in ok:
        print(f"{r['model']:13s} decades {r['decades_frozen']:.3f} (now {r['decades_current'] if r['decades_current'] is None else round(r['decades_current'], 3)})"
              f"  breadth p0 {r['breadth_p0']:.3f} median {r['breadth_median']:.3f}  log g_net {r['log_g_net_p0']:+.2f}"
              f"  one-step max {r['one_step_max_p0']:+.2f}  H(T=1) {r['entropy_T1_p0']:.2f}")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
