"""Measured fixed-start close-up at the audited memory reconnection candidate.

The ordinary scan kernel, horizons and estimator are retained. Only the K=1
feedback rule changes; emitted token probabilities are the original readout.
The benchmark also compares this batched implementation with MemoryOrbit.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "8"
import numpy as np
import explore_llm_soft_loop as soft
from baby_shrimp_memory import MemoryOrbit

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["tools/baby_shrimp_memory_scan.py", "tools/baby_shrimp_memory.py",
           "tools/explore_llm_soft_loop.py", "tools/explore_llm_attractors.py",
           "tools/explore_llm_phase_map.py", "tools/explore_latent_loop.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


class MemoryScanLoop(soft.SoftLoop):
    def __init__(self, *args, beta, **kwargs):
        if not np.isfinite(beta) or not 0 <= beta <= .05:
            raise ValueError("scan requires finite 0 <= beta <= .05")
        super().__init__(*args, **kwargs)
        if self.K != 1:
            raise ValueError("memory scan is defined only for the one-token map")
        self.beta = float(beta)

    def step(self, x, temps):
        y, p = super().step(x, temps)
        return (y if self.beta == 0 else (1-self.beta)*y+self.beta*x), p

    def step_jvp(self, x, temps, v):
        y, p, jv = super().step_jvp(x, temps, v)
        if self.beta == 0:
            return y, p, jv
        return (1-self.beta)*y+self.beta*x, p, (1-self.beta)*jv+self.beta*v


def unretained_reference_step(loop, reference, x):
    """MemoryOrbit must wrap F, not the already mixed scan F_beta.

    SoftOrbit delegates its forward to loop.step. Temporarily using beta=0
    makes that delegate exactly the ordinary map without a second model copy.
    Restore the scan intervention even if reference evaluation raises.
    """
    beta = loop.beta
    try:
        loop.beta = 0.
        return reference.f(x)
    finally:
        loop.beta = beta


def scan_arguments(plan, out, device, chunk):
    keys = ["model", "layer", "threads", "prompt", "opener", "window", "dtype", "weights",
            "vectors", "concept", "alpha", "resolution", "transient", "samples", "record",
            "max_period", "tol", "eps", "chaos_level"]
    d = {k: plan[k] for k in keys}
    d.update(log_temperature=plan["log10_temperature"], command="scan", prefix=None,
             start_vector=None, init_states=None, save_states=False, projections=False,
             word_period=0, return_period=0, device=device, chunk=chunk, out=out,
             feedback_memory=plan["beta"])
    return argparse.Namespace(**d)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="docs/baby-shrimp-zoom/memory-junction-scan-plan.json")
    parser.add_argument("--out", required=True, help="relative to DATA/explore")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--mode", choices=["benchmark", "scan"], required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--chunk", type=int, default=128)
    args = parser.parse_args()
    plan_path = ROOT / args.plan
    plan = json.loads(plan_path.read_text())
    receipt_path = ROOT/plan['root_receipt']['path']
    if sha(receipt_path) != plan['root_receipt']['sha256']:
        raise ValueError('root receipt hash mismatch')
    receipt = json.loads(receipt_path.read_text())
    root = receipt['height_root']
    expected_state = (Path(receipt['raw_directory']) /
                      f"point-{root['cyclic_point_index']:04d}" / 'points.npy')
    if (plan['beta'] != root['parameter'] or
        plan['center'] != [root['alpha'],root['log10_temperature']] or
        plan['root_state']['path'] != str(expected_state) or
        plan['model_revision'] != receipt['scientific_metadata']['model_revision']):
        raise ValueError('scan candidate differs from audited root receipt')
    for record in [plan['reference_scan'], plan['steering_vectors']]:
        if sha(soft.DATA/record['path']) != record['sha256']:
            raise ValueError('bound input hash mismatch: '+record['path'])
    if args.chunk < 1 or args.chunk > plan["maximum_chunk"]:
        raise ValueError("chunk outside declared range")
    if Path(args.out).is_absolute() or ".." in Path(args.out).parts:
        raise ValueError("output must stay under external explore root")
    out = soft.DATA / "explore" / args.out
    if ROOT == out.resolve() or ROOT in out.resolve().parents:
        raise ValueError("raw output must be outside the repository")
    out.mkdir(parents=True, exist_ok=False)
    commit = args.source_commit or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_hashes = {p: sha(ROOT/p) for p in SOURCES}
    if (ROOT/".git").exists():
        for p in [*SOURCES, args.plan]:
            frozen = subprocess.check_output(["git", "show", f"{commit}:{p}"], cwd=ROOT)
            if hashlib.sha256(frozen).hexdigest() != sha(ROOT/p):
                raise ValueError("uncommitted execution input: " + p)
    save(out/"execution.json", {"source_commit":commit, "source_sha256":source_hashes,
         "plan_sha256":sha(plan_path), "mode":args.mode, "device":args.device, "chunk":args.chunk,
         "started_utc":datetime.now(timezone.utc).isoformat(), "pid":os.getpid()})
    import torch
    torch.set_num_threads(plan["threads"])
    torch.set_num_interop_threads(1)
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(.75)
    loop = MemoryScanLoop(plan["model"], plan["layer"], plan["threads"], plan["prompt"],
                          plan["opener"], plan["window"], plan["dtype"], args.device,
                          plan["weights"], beta=plan["beta"])
    if loop.sm.engine.revision != plan["model_revision"]:
        raise ValueError("model revision mismatch")
    norm = loop.model.model.norm
    x = torch.linspace(-1,1,norm.weight.numel(),dtype=torch.float64,device=loop.device)[None,None]
    sensitivity = float((norm(x*(1+1e-12))-norm(x)).abs().max())
    if not 0 < sensitivity < 1e-9 or not hasattr(loop.model.model.rotary_emb,"_inv_freq64"):
        raise ValueError("effective float64 gate failed")
    save(out/"runtime.json", {"torch":torch.__version__, "cuda":torch.version.cuda,
         "device":torch.cuda.get_device_name() if args.device=="cuda" else "cpu",
         "norm_sensitivity":sensitivity, "rotary_float64":True, "threads":torch.get_num_threads()})
    vector = soft.elpm.load_vectors(plan["vectors"])[plan["concept"]]
    ref = plan["root_state"]
    ref_path = soft.DATA/ref["path"]
    if sha(ref_path) != ref["sha256"]:
        raise ValueError("root state hash mismatch")
    state = np.load(ref_path, allow_pickle=False)[0]
    center = np.array(plan["center"])
    base = soft.SoftOrbit(loop, center[0]*vector, 10**center[1], chunk=4, jacobian_method="ad")
    reference = MemoryOrbit(base, plan["beta"])
    x = torch.as_tensor(state.reshape((1,1,-1)),dtype=torch.float64,device=loop.device)
    expected = unretained_reference_step(loop, reference, x)
    loop.prepare(center[0]*vector[None,:])
    actual, _ = loop.step(x, [10**center[1]])
    error = float((actual-expected).norm()/expected.norm())
    if not np.isfinite(error) or error > 1e-12:
        raise ValueError("batched scan map differs from audited MemoryOrbit")
    start = x.clone()
    for _ in range(18):
        x, _ = loop.step(x, [10**center[1]])
    closure = float((x-start).norm()/start.norm())
    if not np.isfinite(closure) or closure > 1e-10:
        raise ValueError("audited root does not close under scan implementation")
    save(out/"map-equivalence.json", {"one_step_relative_error":error,
         "saved_root_relative_closure":closure, "period":18, "passed":True})
    if args.mode == "benchmark":
        # Full measurement horizon on a timed batch; not a parameter map result.
        n=args.chunk
        steer=np.repeat((center[0]*vector)[None,:],2*n,axis=0)
        xx=loop.prepare(steer); temps=np.full(2*n,10**center[1])
        for _ in range(3): xx,_=loop.step(xx,temps)
        if args.device=="cuda": torch.cuda.synchronize()
        started=time.monotonic()
        for _ in range(plan["benchmark_steps"]): xx,_=loop.step(xx,temps)
        if args.device=="cuda": torch.cuda.synchronize()
        elapsed=time.monotonic()-started
        result={"cells_per_batch":n,"state_rows":2*n,"steps":plan["benchmark_steps"],
                "seconds":elapsed,"seconds_per_step":elapsed/plan["benchmark_steps"],
                "projected_scan_numerical_seconds":elapsed/plan["benchmark_steps"]*
                    (plan["transient"]+plan["samples"])*int(np.ceil(np.prod(plan["resolution"])/n)),
                "scope":"Timed step-only scaling; excludes prefix construction, recording and I/O. Not an image result."}
        save(out/"benchmark.json",result)
        print(json.dumps(result),flush=True)
    else:
        soft.scan(scan_arguments(plan,args.out,args.device,args.chunk),loop_factory=lambda *a,**kw:loop)
    save(out/"complete.json", {"complete":True,"mode":args.mode,
         "completed_utc":datetime.now(timezone.utc).isoformat(),
         "files":[{"name":p.name,"sha256":sha(p),"bytes":p.stat().st_size}
                  for p in sorted(out.iterdir()) if p.is_file()]})


if __name__ == "__main__":
    main()
