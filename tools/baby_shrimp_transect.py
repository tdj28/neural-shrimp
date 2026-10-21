"""Bounded Jones transect using the existing full-state BSP1 trajectory method."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[key]="8"
from baby_shrimp_trajectories import worker,write
from baby_shrimp_orbits import DATA,ROOT


def validate(plan):
    import numpy as np
    cells=plan["cells"]
    if len(cells)!=65 or plan["transient"]!=3000 or plan["record"]!=3000:
        raise ValueError("declared transect resource bounds required")
    if [c["name"] for c in cells]!=[f"jones-cut-{i:02d}" for i in range(65)]:
        raise ValueError("unique ordered transect cells required")
    if not np.array_equal([c["alpha"] for c in cells],np.linspace(.073045,.073140,65)):
        raise ValueError("transect axis differs")
    if any(c["log10_temperature"]!=-.15440845070422535 for c in cells):
        raise ValueError("fixed temperature differs")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",required=True);ap.add_argument("--out",required=True)
    ap.add_argument("--source-commit",required=True)
    ap.add_argument("--plan",default="docs/baby-shrimp-zoom/jones-transect-plan.json")
    args=ap.parse_args();path=ROOT/args.plan;plan=json.loads(path.read_text());validate(plan)
    seeds=DATA/"explore"/args.input
    if hashlib.sha256((seeds/"seeds.npz").read_bytes()).hexdigest()!=plan["reference_seed_sha256"]:
        raise ValueError("reference probe archive differs")
    out=DATA/"explore"/args.out;out.mkdir(parents=True,exist_ok=True)
    if (out/"status.json").exists() or (out/"states.npy").exists():raise ValueError("preserve prior transect attempt")
    status={"source_commit":args.source_commit,"plan_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "status":"running","started_utc":datetime.now(timezone.utc).isoformat(),
            "initialization":"Fresh opener initialization at each cell; reference seeds identify the target only."}
    write(out/"status.json",status)
    try:
        worker(SimpleNamespace(cells=path,out=out,transient=plan["transient"],record=plan["record"]))
        status["status"]="complete"
    except BaseException as error:
        status.update(status="failed",error=f"{type(error).__name__}: {error}");raise
    finally:
        status["finished_utc"]=datetime.now(timezone.utc).isoformat();write(out/"status.json",status)


if __name__=="__main__":main()
