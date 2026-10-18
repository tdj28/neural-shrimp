"""Rebuild curve.json of a bifurcation-curve run that was stopped by its time limit, from its progress.jsonl.

A curve run (tools/explore_llm_soft_loop.py curve) writes one progress line per accepted point ("HH:MM:SS {json}")
and its curve.json only at the end. When the job's time limit stops it first, the points are all in the progress log.
This tool parses them, refuses the rebuild unless every point passes the run's own gates (Newton residual below --tol,
|g| below --gtol, the critical multiplier real and within --mult-tol of +1 for a fold or -1 for a flip, the same type
at every point) and writes curve.json with the metadata of a finished sibling run (the same start, the other
direction), marked reconstructed_from_progress. The original progress log is left untouched.

Usage: python tools/curve_from_progress.py RUN_DIR --sibling SIBLING_RUN_DIR --direction 1 --stop "time limit; ..."
"""
import argparse
import json
from pathlib import Path


def parse(progress):
    pts = []
    for line in Path(progress).read_text().splitlines():
        if not line.strip():
            continue
        _, js = line.split(" ", 1)
        pts.append(json.loads(js))
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--sibling", type=Path, required=True, help="finished run of the same start (other direction)")
    ap.add_argument("--direction", type=int, required=True, choices=(1, -1))
    ap.add_argument("--stop", required=True)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--gtol", type=float, default=1e-9)
    ap.add_argument("--mult-tol", type=float, default=1e-6)
    args = ap.parse_args()
    out = args.run / "curve.json"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    sib = json.loads((args.sibling / "curve.json").read_text())
    pts = parse(args.run / "progress.jsonl")
    if not pts:
        raise SystemExit("no points in the progress log")
    target = {"fold": 1.0, "flip": -1.0}[sib["kind"]]
    bad = []
    for p in pts:
        mu = p["critical_multiplier"]
        if not (p["residual"] < args.tol and abs(p["g"]) < args.gtol and abs(mu[1]) == 0.0
                and abs(mu[0] - target) < args.mult_tol):
            bad.append(p["index"])
    if bad:
        raise SystemExit(f"points failing the gates ({sib['kind']}): {bad[:20]} ({len(bad)} in all)")
    if [p["index"] for p in pts] != list(range(len(pts))):
        raise SystemExit("point indices are not consecutive from 0")
    run_args = dict(sib["args"], direction=args.direction, out=str(sib["args"]["out"]).replace(args.sibling.name, args.run.name),
                    note="reconstructed from progress.jsonl by tools/curve_from_progress.py")
    rec = {"args": run_args, "kind": sib["kind"], "period": sib["period"], "start": sib["start"],
           "parameters": sib["parameters"], "points": pts, "stop": args.stop, "seconds": None,
           "reconstructed_from_progress": True,
           "gates": {"tol": args.tol, "gtol": args.gtol, "mult_tol": args.mult_tol, "points": len(pts),
                     "max_residual": max(p["residual"] for p in pts), "max_abs_g": max(abs(p["g"]) for p in pts),
                     "max_multiplier_offset": max(abs(p["critical_multiplier"][0] - target) for p in pts),
                     "max_other_spectral_radius": max(p["other_spectral_radius"] or 0.0 for p in pts),
                     "gmres_unconverged": sum(p["gmres_unconverged"] for p in pts),
                     "cusp_candidates": [p["index"] for p in pts if p.get("cusp_candidate")]}}
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps({"out": str(out), **rec["gates"]}))


if __name__ == "__main__":
    main()
