"""Cusp-unfolding pilot, stage 2 (docs/qwen2-cusp-unfolding-pilot.md), run on a GPU pod: continue the fold curve of a
stage-1 branch in both directions (float64), and at every parameter-tangent reversal run the cusp check with the
cubic coefficient on the five surrounding points and the unfolding test (tools/cusp_unfolding.py, frozen offsets,
predictions written before solving) at the flagged point's neighbour with the smallest |b|.

Usage: python tools/pilot_stage2.py BRANCH_RUN OUT_PREFIX STEPS MINUTES TLO THI ALO AHI
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

CURVE = ["curve", "--device", "cuda", "--dtype", "float64", "--reference-dtype", "none", "--no-exact-jvp",
         "--gmres-tol", "1e-10", "--tol", "1e-10", "--gtol", "1e-9", "--max-newton", "10", "--chunk", "896",
         "--kind", "fold", "--ds", "0.002", "--ds-max", "0.02"]


def run(cmd, minutes, log):
    with open(log, "w") as f:
        try:
            p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=60 * minutes, cwd=ROOT)
            return p.returncode
        except subprocess.TimeoutExpired:
            return 124


def main():
    import explore_llm_attractors as ela

    branch, prefix, steps, minutes = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    tlo, thi, alo, ahi = sys.argv[5:9]
    logs = ela.DATA / "explore" / "llm-soft-loop" / "pod64" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    summary = []
    for direction, tag in ((1, "plus"), (-1, "minus")):
        out = f"{prefix}-fold-{tag}"
        rc = run([py, "tools/explore_llm_soft_loop.py", *CURVE, "--start", branch, "--direction", str(direction),
                  "--steps", steps, "--log-temperature-bounds", tlo, thi, "--alpha-bounds", alo, ahi, "--out", out],
                 minutes, logs / f"pilot2-curve-{tag}.log")
        cfile = ela.DATA / "explore" / out / "curve.json"
        if not cfile.exists():
            summary.append({"curve": out, "status": rc, "points": 0, "flags": []})
            print(json.dumps(summary[-1]), flush=True)
            continue
        cv = json.loads(cfile.read_text())
        pts = cv["points"]
        flags = [i for i, p in enumerate(pts) if p.get("cusp_candidate")]
        rec = {"curve": out, "status": rc, "points": len(pts), "flags": flags, "stop": cv.get("stop"),
               "alpha_range": [min(p["alpha"] for p in pts), max(p["alpha"] for p in pts)],
               "log10_T_range": [min(p["log10_temperature"] for p in pts), max(p["log10_temperature"] for p in pts)],
               "cusps": []}
        for i in flags:
            idx = [j for j in range(i - 2, i + 3) if 0 <= j < len(pts)]
            chk = subprocess.run([py, "tools/cusp_check.py", out, ",".join(map(str, idx)), "cuda", "cubic"],
                                 capture_output=True, text=True, cwd=ROOT, timeout=3600)
            rows = [json.loads(line) for line in chk.stdout.splitlines() if line.startswith("{")]
            (ela.DATA / "explore" / out / f"cusp-cubic-{i}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows))
            near = [r for r in rows if r["index"] in (i - 1, i, i + 1)]
            best = min(near, key=lambda r: abs(r["b_oriented"]))["index"] if near else i
            unf = subprocess.run([py, "tools/cusp_unfolding.py", out, str(best), "cuda"], capture_output=True,
                                 text=True, cwd=ROOT, timeout=5400)
            rec["cusps"].append({"flag": i, "b_oriented": [round(r["b_oriented"], 3) for r in rows],
                                 "c": [r.get("c") for r in rows], "unfolding_index": best,
                                 "unfolding_stdout": unf.stdout[-3000:], "unfolding_rc": unf.returncode})
        summary.append(rec)
        print(json.dumps(rec)[:4000], flush=True)
    (ela.DATA / "explore" / f"{prefix}-stage2-summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
