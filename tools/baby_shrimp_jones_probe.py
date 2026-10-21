"""Evaluate Jones-style critical candidates through the full float64 network.

The input is a validated state-curve lift, not an asserted invariant manifold.
Zeros belong to a projected six-step derivative. Critical trajectories retain
all 896 coordinates so that approximate scalar closure cannot masquerade as
an exact superstable orbit.
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
import time

ROOT=Path(__file__).resolve().parents[1]
SOURCES=("tools/baby_shrimp_jones_probe.py","tools/explore_llm_soft_loop.py",
         "tools/explore_llm_attractors.py","tools/explore_llm_phase_map.py",
         "tools/batch_fate_finish.py","tools/batch_fate_summary.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lift(coef, origin, u, derivative=False):
    import numpy as np
    from numpy.polynomial.chebyshev import chebder,chebvander
    u=np.atleast_1d(u)
    if derivative:
        c=2*chebder(coef,axis=0)
        return chebvander(2*u-1,len(c)-1)@c
    return chebvander(2*u-1,len(coef)-1)@coef+origin


def brackets(u, derivative):
    import numpy as np
    if not np.isfinite(derivative).all() or len(u)!=len(derivative):
        raise ValueError("finite matched derivative samples required")
    return [(float(u[i]),float(u[i+1])) for i in range(len(u)-1)
            if .02<u[i]<u[i+1]<.98 and derivative[i]*derivative[i+1]<0]


def worker(args):
    import numpy as np
    import torch
    import explore_llm_soft_loop as soft
    from baby_shrimp_orbits import save
    torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.85)
    report=json.loads((args.reconstruction/"jones.json").read_text())
    if sha(args.reconstruction/"reconstructions.npz")!=report["artifacts"]["reconstructions.npz"]["sha256"]:
        raise ValueError("reconstruction hash differs")
    if sha(args.trajectory/"states.npy")!=report["states_sha256"]:
        raise ValueError("trajectory hash differs")
    arrays=np.load(args.reconstruction/"reconstructions.npz",allow_pickle=False)
    trajectory=json.loads((args.trajectory/"trajectories.json").read_text())
    states=np.load(args.trajectory/"states.npy",mmap_mode="r",allow_pickle=False)
    core=next(c for c in trajectory["cells"] if c["name"]=="baby-core")
    selected=[("mother-chaos",3), ("baby-chaos-left",0), ("baby-chaos-right",0), ("baby-chaos-below",0)]
    cells=[]
    for name,phase in selected:
        row=next(r for r in report["rows"] if r["name"]==name and r["phase"]==phase)
        cells.append({"name":name,"source":name,"phase":phase,"alpha":row["alpha"],"log10_temperature":row["log10_temperature"]})
    for source in ("baby-chaos-below","baby-chaos-right"):
        cells.append({"name":"baby-core-from-"+source,"source":source,"phase":0,
                      "alpha":core["alpha"],"log10_temperature":core["log10_temperature"]})
    loop=soft.SoftLoop("qwen0.5b",12,8,"open","Hello!",1,"float64","cuda","native")
    if loop.sm.engine.revision!="7ae557604adf67be50417f59c2c2f167def9a775":
        raise ValueError("unexpected model revision")
    norm=loop.model.model.norm
    check=torch.linspace(-1,1,norm.weight.numel(),device=loop.device,dtype=torch.float64)[None,None,:]
    sensitivity=float((norm(check*(1+1e-12))-norm(check)).abs().max())
    if not (0<sensitivity<1e-9 and hasattr(loop.model.model.rotary_emb,"_inv_freq64")):
        raise ValueError("effective float64 check failed")
    vector=soft.elpm.load_vectors("llm-phase-map/vectors-l12")["philosophy"]
    started=time.monotonic();results=[]
    save(args.out/"protocol.json",{"source_commit":args.source_commit,"reconstruction_sha256":sha(args.reconstruction/"jones.json"),
         "model_revision":loop.sm.engine.revision,"norm_sensitivity":sensitivity,"rotary_float64":True,
         "grid_points":201,"batch_limit":32,"critical_bisections":24,"returns":7,"cells":cells,
         "scope":"Projected derivatives along fitted state curves; no invariant manifold or true superstability assumed."})

    def emit(stage,**kw):
        row={"stage":stage,"seconds":time.monotonic()-started,**kw}
        with (args.out/"progress.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
        print(json.dumps(row),flush=True)

    for cell in cells:
        name=cell["name"];dest=args.out/name;dest.mkdir()
        row=next(r for r in report["rows"] if r["name"]==cell["source"] and r["phase"]==cell["phase"])
        if not row["lift_passed"]:
            raise ValueError("selected full-state lift failed validation")
        prefix=f"{cell['source']}__{cell['phase']}__"
        coef,origin,axis,span=(arrays[prefix+k] for k in ("lift_coefficients","origin","axis","span"))
        span=float(span);a=cell["alpha"];temp=10**cell["log10_temperature"]
        orb=soft.SoftOrbit(loop,a*vector,temp,chunk=32,jacobian_method="ad")

        def evaluate(u, tangent=True):
            outputs=[];derivatives=[]
            u=np.atleast_1d(u)
            for k in range(0,len(u),32):
                v=u[k:k+32]
                X=torch.as_tensor(lift(coef,origin,v)[:,None,:],device=loop.device,dtype=torch.float64)
                if tangent:
                    V=torch.as_tensor(lift(coef,origin,v,True)[:,None,:],device=loop.device,dtype=torch.float64)
                    orb._use(len(v))
                    for step in range(6):X,_,V=loop.step_jvp(X,np.full(len(v),temp),V)
                    derivatives.append(V[:,0,:].cpu().numpy()@axis/span)
                else:
                    X=orb.iterate(X,6)
                outputs.append(X[:,0,:].cpu().numpy())
            Y=np.concatenate(outputs)
            return (Y-origin)@axis/span, np.concatenate(derivatives) if tangent else None, Y

        tick=time.monotonic();u=np.linspace(0,1,201)
        g,dg,Y=evaluate(u)
        np.savez_compressed(dest/"map.npz",u=u,image=g,derivative=dg,output_states=Y,
                            input_states=lift(coef,origin,u),origin=origin,axis=axis,span=span)
        emit("curve_complete",name=name,curve_seconds=time.monotonic()-tick)
        # Direct map values independently check the AD primal at three samples.
        chosen=np.array([.2,.5,.8]);primal,ad,Yad=evaluate(chosen)
        direct,_,Ydirect=evaluate(chosen,False)
        discrepancy=float(np.max(np.linalg.norm(Yad-Ydirect,axis=1)/np.maximum(np.linalg.norm(Ydirect,axis=1),1)))
        if discrepancy>=1e-11:
            raise ValueError("AD primal and plain full-map evaluation disagree")
        fd=[]
        for h in (1e-3,3e-4,1e-4,3e-5):
            plus,_,_=evaluate(chosen+h,False);minus,_,_=evaluate(chosen-h,False)
            errors=np.abs((plus-minus)/(2*h)-ad)/np.maximum(abs(ad),1e-12)
            fd.append({"h":h,"relative_errors":errors.tolist()})
        orders=np.log(np.array([v["relative_errors"] for v in fd[:-1]])/np.array([v["relative_errors"] for v in fd[1:]]))/np.log(np.array([v["h"] for v in fd[:-1]])/np.array([v["h"] for v in fd[1:]]))[:,None]
        derivative_ok=bool(np.all(np.min(np.array([v["relative_errors"] for v in fd]),axis=0)<1e-5)
                           and np.all(np.any((orders[:-1]>1.5)&(orders[:-1]<2.5)&(orders[1:]>1.5)&(orders[1:]<2.5),axis=0)))
        # Closure of a fitted lift is measured, not assumed even for stable maps.
        inside=(g>=0)&(g<=1)
        off=np.linalg.norm(Y[inside]-lift(coef,origin,g[inside]),axis=1)/span
        criticals=[]
        for lo,hi in brackets(u,dg):
            left=evaluate([lo])[1][0]
            history=[]
            for k in range(24):
                mid=(lo+hi)/2;value=evaluate([mid])[1][0]
                history.append([mid,float(value)])
                if left*value<=0:hi=mid
                else:lo=mid;left=value
            x=(lo+hi)/2;value,derivative,_=evaluate([x])
            X=torch.as_tensor(lift(coef,origin,[x])[:,None,:],device=loop.device,dtype=torch.float64)
            orbit=[X[0,0].cpu().numpy().copy()];tokens=[]
            for k in range(42):
                orb._use(1);X,prob=loop.step(X,np.array([temp]));tokens.append(int(prob.argmax(-1)[0]))
                if (k+1)%6==0:orbit.append(X[0,0].cpu().numpy().copy())
            orbit=np.array(orbit);coordinates=(orbit-origin)@axis/span
            returns=np.linalg.norm(orbit[1:]-orbit[0],axis=1)/max(np.linalg.norm(orbit[0]),1)
            label="C-minimum" if history[0][1]>left else "candidate"
            # Orientation comes from the original resolved derivative bracket.
            first_derivative=float(evaluate([history[0][0]-.003])[1][0])
            last_derivative=float(evaluate([history[0][0]+.003])[1][0])
            label="C-minimum" if first_derivative<0<last_derivative else "D-maximum" if first_derivative>0>last_derivative else "unresolved"
            n=len(criticals)
            np.savez_compressed(dest/f"critical-{n}.npz",orbit=orbit,coordinates=coordinates,tokens=tokens,
                                bisection=np.array(history),full_state_return=returns)
            criticals.append({"label":label,"coordinate":x,"projected_image":float(value[0]),"projected_derivative":float(derivative[0]),
                              "bracket_width":hi-lo,"full_state_returns_six_steps":returns.tolist(),"coordinates":coordinates.tolist(),
                              "exact_six_token_loop":bool(np.array_equal(tokens[6:],tokens[:-6]))})
            emit("critical_complete",name=name,critical=n)
        result={**cell,"derivative_qualified":derivative_ok,"plain_vs_ad_primal_error":discrepancy,
                "finite_differences":fd,"orders":orders.tolist(),"curve_outside_source_domain":int((~inside).sum()),
                "lift_invariance_rms_over_span":float(np.sqrt(np.mean(off**2))) if len(off) else None,
                "lift_invariance_max_over_span":float(off.max()) if len(off) else None,"critical_points":criticals}
        save(dest/"summary.json",result);results.append(result);save(args.out/"results.json",{"complete":False,"cells":results})
        emit("cell_complete",name=name,derivative_qualified=derivative_ok)
        del orb
    save(args.out/"results.json",{"complete":True,"source_commit":args.source_commit,"cells":results,"seconds":time.monotonic()-started})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reconstruction",type=Path,required=True);ap.add_argument("--trajectory",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True);ap.add_argument("--source-commit",required=True)
    ap.add_argument("--timeout",type=int,default=2700);ap.add_argument("--worker",action="store_true")
    args=ap.parse_args()
    if not 0<args.timeout<=2700:ap.error("maximum local runtime2700seconds")
    if args.worker:worker(args);return
    import batch_fate_finish as guard
    from baby_shrimp_orbits import save
    args.out.mkdir(parents=True,exist_ok=False)
    commit=subprocess.check_output(["git","rev-parse",args.source_commit],cwd=ROOT,text=True).strip();hashes={}
    for source in SOURCES:
        blob=subprocess.check_output(["git","show",commit+":"+source],cwd=ROOT)
        hashes[source]=hashlib.sha256(blob).hexdigest()
        if hashes[source]!=sha(ROOT/source):raise ValueError(f"uncommitted source: {source}")
    for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[key]="8"
    os.environ["HF_HUB_OFFLINE"]="1"
    status={"source_commit":commit,"sources":hashes,"started_utc":datetime.now(timezone.utc).isoformat(),"status":"running"}
    def event(state,**kw):
        with (args.out/"events.jsonl").open("a") as f:f.write(json.dumps({"time":datetime.now(timezone.utc).isoformat(),"state":state,**kw})+"\n")
    def interrupted(signum,frame):raise InterruptedError(f"Jones parent received signal{signum}")
    signal.signal(signal.SIGTERM,interrupted)
    try:
        with guard.gpu_lock(guard.summary.DATA/"render-tmp/gpu.lock",timeout=1):
            status["resources"]=guard.resource_snapshot();save(args.out/"status.json",status)
            command=[sys.executable,str(Path(__file__).resolve()),"--worker","--reconstruction",str(args.reconstruction),
                     "--trajectory",str(args.trajectory),"--out",str(args.out),"--source-commit",commit,"--timeout",str(args.timeout)]
            guard.run_bounded(command,args.out/"worker.log",args.timeout,event,monitor_memory=True)
        status["status"]="complete"
    except BaseException as error:
        status.update(status="failed",error=f"{type(error).__name__}: {error}");raise
    finally:
        status["finished_utc"]=datetime.now(timezone.utc).isoformat();save(args.out/"status.json",status)


if __name__=="__main__":main()
