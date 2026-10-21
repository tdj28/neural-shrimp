"""Bounded local BSP2 precision extension, preserving the original failed gate.

Reuse a saved float64 monodromy, check it against a fresh composed JVP on the
local device, and extend central differences to smaller offsets. No dense
Jacobian recomputation or paid resource is needed. Every old row is retained.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCES=("tools/baby_shrimp_precision_extension.py", "tools/baby_shrimp_orbits.py",
         "tools/bifurcation_curves.py", "tools/batch_fate_finish.py", "tools/batch_fate_summary.py",
         "tools/explore_llm_soft_loop.py", "tools/explore_llm_attractors.py", "tools/explore_llm_phase_map.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(args):
    import numpy as np
    import torch
    import explore_llm_soft_loop as soft
    import baby_shrimp_orbits as checks

    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.85)
    original=json.loads((args.orbit_run/"orbits.json").read_text())
    if original.get("complete") is not True:
        raise ValueError("original orbit attempt is incomplete")
    selected=[r for r in original["cells"] if r["name"]==args.cell]
    if len(selected)!=1:
        raise ValueError("requested orbit is absent or duplicated")
    row=selected[0]
    if not (row["primitive_gate"]["closed"] and row["primitive_gate"]["no_lower_period"]):
        raise ValueError("precision extension requires an already closed primitive orbit")
    previous=json.loads(json.dumps(row))
    loop=soft.SoftLoop("qwen0.5b",12,8,"open","Hello!",1,"float64","cuda","native")
    if loop.sm.engine.revision != "7ae557604adf67be50417f59c2c2f167def9a775":
        raise ValueError("unexpected model revision")
    norm=loop.model.model.norm
    test=torch.linspace(-1,1,norm.weight.numel(),dtype=torch.float64,device=loop.device)[None,None,:]
    sensitivity=float((norm(test*(1+1e-12))-norm(test)).abs().max())
    if not (0<sensitivity<1e-9 and hasattr(loop.model.model.rotary_emb,"_inv_freq64")):
        raise ValueError("local effective-precision gate failed")
    vector=soft.elpm.load_vectors("llm-phase-map/vectors-l12")["philosophy"]
    orb=soft.SoftOrbit(loop,row["alpha"]*vector,10.0**row["log10_temperature"],chunk=1,jacobian_method="ad")
    src=args.orbit_run/args.cell
    X=torch.as_tensor(np.load(src/"orbit-state.npy",allow_pickle=False),dtype=loop.state_dtype,device=loop.device)
    M=np.load(src/"monodromy.npy",allow_pickle=False)
    period=row["period_candidate"]
    points=orb.points(X,period)
    returns=orb.return_distances(X,period)
    local_primitive=checks.primitive_gate(returns,period)
    if not (local_primitive["closed"] and local_primitive["no_lower_period"]):
        raise ValueError("stored orbit does not close on the local device")

    def progress(stage,**kw):
        record={"stage":stage,**kw}
        with (args.out/"progress.jsonl").open("a") as f:
            f.write(json.dumps(record)+"\n")
        print(json.dumps(record),flush=True)

    precision=checks.precision_checks(orb,X,M,points[:-1],args.out,progress,
                                      relative_steps=[3e-5,1e-5,3e-6,1e-6,3e-7,1e-7,3e-8])
    extension={"source_commit":args.source_commit,"original_orbits_sha256":sha(args.orbit_run/"orbits.json"),
               "stored_monodromy_sha256":sha(src/"monodromy.npy"),"stored_state_sha256":sha(src/"orbit-state.npy"),
               "device":torch.cuda.get_device_name(),"torch":torch.__version__,"cuda":torch.version.cuda,
               "norm_sensitivity_to_1e_12":sensitivity,"rotary_float64":True,
               "local_primitive_gate":local_primitive,"derivative_precision":precision,
               "scope":"New local-device composed-map check; original A100 attempt and failed gate remain immutable."}
    checks.save(args.out/"extension.json",extension)
    row["original_derivative_precision"]=previous["derivative_precision"]
    row["derivative_precision"]=precision
    row["precision_extension"]=extension
    row["verified_primitive_orbit"]=precision["passed"]
    row["status"]="verified numerical primitive orbit after disclosed precision extension" if precision["passed"] else "derivative qualification still failed"
    original["precision_extension"]={"cell":args.cell,"source_commit":args.source_commit,
                                     "original_orbits_sha256":sha(args.orbit_run/"orbits.json")}
    checks.save(args.out/"orbits.json",original)
    print(json.dumps({"cell":args.cell,"passed":precision["passed"]}),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orbit-run",type=Path,required=True)
    ap.add_argument("--cell",required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--source-commit",required=True)
    ap.add_argument("--timeout",type=int,default=1800)
    ap.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=ap.parse_args()
    if not 0<args.timeout<=1800:
        ap.error("local precision extension must be bounded to at most1800seconds")
    if args.worker:
        worker(args)
        return
    import batch_fate_finish as guard
    from baby_shrimp_orbits import save
    args.out.mkdir(parents=True,exist_ok=False)
    commit=subprocess.check_output(["git","rev-parse",args.source_commit],cwd=ROOT,text=True).strip()
    hashes={}
    for source in SOURCES:
        blob=subprocess.check_output(["git","show",commit+":"+source],cwd=ROOT)
        hashes[source]=hashlib.sha256(blob).hexdigest()
        if hashes[source]!=sha(ROOT/source):
            raise ValueError(f"uncommitted source: {source}")
    for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
        os.environ[key]="8"
    os.environ["HF_HUB_OFFLINE"]="1"
    status={"source_commit":commit,"source_sha256":hashes,"started_utc":datetime.now(timezone.utc).isoformat(),"status":"running"}

    def event(state,**kw):
        with (args.out/"events.jsonl").open("a") as f:
            f.write(json.dumps({"time":datetime.now(timezone.utc).isoformat(),"state":state,**kw})+"\n")

    def interrupted(signum,frame):
        raise InterruptedError(f"precision parent received signal {signum}")

    signal.signal(signal.SIGTERM,interrupted)
    try:
        with guard.gpu_lock(guard.summary.DATA/"render-tmp/gpu.lock",timeout=1):
            status["resources"]=guard.resource_snapshot()
            save(args.out/"status.json",status)
            command=[sys.executable,str(Path(__file__).resolve()),"--worker","--orbit-run",str(args.orbit_run.resolve()),
                     "--cell",args.cell,"--out",str(args.out.resolve()),"--source-commit",commit,"--timeout",str(args.timeout)]
            guard.run_bounded(command,args.out/"worker.log",args.timeout,event,monitor_memory=True)
        status["status"]="complete"
    except BaseException as error:
        status.update(status="failed",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["finished_utc"]=datetime.now(timezone.utc).isoformat()
        save(args.out/"status.json",status)


if __name__=="__main__":
    main()
