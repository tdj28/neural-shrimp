#!/usr/bin/env python3
"""Bounded exploratory fixed-matrix panel, with outcomes and failures retained."""
import argparse,hashlib,json,os,platform,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from neural_shrimp.clark_rnn import inputs,simulate_batch,qualify_cycle,iterate
SOURCES=["src/neural_shrimp/clark_rnn.py","src/neural_shrimp/__init__.py","src/neural_shrimp/controls.py","src/neural_shrimp/models.py","tools/explore_clark_rnn.py","tools/launch_clark_rnn.py","docs/plans/clark-rnn-shrimp-pilot.md"]
SEEDS=[28170,28171];NS=[4,8,16]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,x):Path(path).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")
def source():
    if subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip():raise ValueError("dirty source")
    return {"commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"sha256":{p:sha(ROOT/p) for p in SOURCES}}

def manifest(out):
    files={str(p.relative_to(out)):{"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(out.rglob("*")) if p.is_file() and p.name!="manifest.json"}
    save(out/"manifest.json",{"schema":"clark-rnn-inventory-v1","files":files})

def profile_gate(path,pin,identity):
    path=Path(path)
    if sha(path/"manifest.json")!=pin:raise ValueError("profile pin")
    files=json.loads((path/"manifest.json").read_text())["files"]
    if set(files)!={str(p.relative_to(path)) for p in path.rglob("*") if p.is_file() and p.name!="manifest.json"}:raise ValueError("profile inventory")
    for name,d in files.items():
        p=path/name
        if Path(name).is_absolute() or ".." in Path(name).parts:raise ValueError("profile unsafe path")
        if p.is_symlink() or p.stat().st_size!=d["bytes"] or sha(p)!=d["sha256"]:raise ValueError("profile bytes")
    result=json.loads((path/"result.json").read_text());start=json.loads((path/"runtime-start.json").read_text())
    if result["status"]!="completed" or start["mode"]!="profile" or start["source"]["sha256"]!=identity["sha256"]:raise ValueError("profile source/status")
    if start["numpy"]!=np.__version__ or start["python"]!=sys.version or start["platform"]!=platform.platform():raise ValueError("profile runtime")
    if any(v!="1" for v in start["thread_environment"].values()):raise ValueError("profile threads")
    if not np.isfinite(result["peak_rss_kib"]) or not 0<result["peak_rss_kib"]<=512*1024:raise ValueError("profile RSS")
    a=result["batch_seconds"];b=result["period64_unit_seconds"]
    if not all(np.isfinite(z) and z>0 for z in [a,b,result["elapsed_seconds"]]):raise ValueError("profile time")
    projected=2*6*(441/81)*a+216*8*b+20
    peak=result["peak_rss_kib"]*1024+3*(513*882*16*8)+64*1024**2
    if projected>300 or peak>512*1024**2 or result["elapsed_seconds"]>45:raise ValueError("resource profile failed")
    return {"manifest_sha256":pin,"projected_seconds":projected,"conservative_peak_bytes":peak}

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--mode",choices=["profile","scan"],required=True);parser.add_argument("--out",type=Path,required=True);parser.add_argument("--profile",type=Path);parser.add_argument("--profile-sha256");args=parser.parse_args()
    identity=source();gate=None
    if args.mode=="scan":
        if args.profile is None or args.profile_sha256 is None:raise ValueError("profile required")
        gate=profile_gate(args.profile,args.profile_sha256,identity)
    out=args.out;out.mkdir(parents=True,exist_ok=False);started=time.monotonic();interrupted=False
    def term(_signum,_frame):
        nonlocal interrupted;interrupted=True
    signal.signal(signal.SIGTERM,term)
    def check():
        if interrupted:raise InterruptedError("SIGTERM")
        if time.monotonic()-started>(45 if args.mode=="profile" else 300):raise TimeoutError("internal cap")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>512*1024:raise MemoryError("512MiB RSS cap")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file())>300*1024**2:raise OSError("300MiB storage cap")
    runtime={"source":identity,"mode":args.mode,"pid":os.getpid(),"started_unix":time.time(),"python":sys.version,"numpy":np.__version__,"platform":platform.platform(),"nice":os.getpriority(os.PRIO_PROCESS,0),"thread_environment":{k:os.environ.get(k) for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]},"profile_gate":gate,"load_average":list(os.getloadavg())};save(out/"runtime-start.json",runtime)
    result={"status":"incomplete","networks":[],"evidence":"exploratory_float64_discrete_map_only"}
    try:
        axes=9 if args.mode=="profile" else 21;gains=np.geomspace(.6,20,axes);deltas=np.linspace(.05,1,axes)
        panel=[(16,28169)] if args.mode=="profile" else [(n,s) for n in NS for s in SEEDS]
        save(out/"design.json",{"panel":panel,"gains":gains.tolist(),"deltas":deltas.tolist(),"burn":6000,"tangent_align":1000,"measured":4096,"tail":513,"block":256,"max_period":64,"qualification":"first32 p>=3 in grid/IC order plus first4 p2 per network; no p1 correction","reserved_seeds_untouched":True})
        for n,seed in panel:
            check();folder=out/f"n{n}-s{seed}";folder.mkdir();j,x,v=inputs(n,seed);np.savez(folder/"inputs.npz",j=j,initials=x,tangents=v,gains=gains,deltas=deltas)
            def checkpoint(t,a,b):np.savez(folder/f"checkpoint-{t:05d}.npz",step=np.array(t),state=a,tangent=b)
            t=time.monotonic();data=simulate_batch(j,x,v,gains,deltas,check=check,checkpoint=checkpoint);batch_seconds=time.monotonic()-t
            je,jv=np.linalg.eig(j);zero=(1-data["grid"][:,1,None])+data["grid"][:,1,None]*data["grid"][:,0,None]*je[None,:]
            data.update(j0_eigenvalues=je,j0_eigenvectors=jv,zero_fixed_point_radius=np.max(np.abs(zero),axis=1),origin_final_maxnorm=np.max(np.abs(data["final"]),axis=1))
            np.savez(folder/"batch.npz",**data)
            periods=data["period_proposals"];inds=np.concatenate([np.flatnonzero(periods>=3)[:32],np.flatnonzero(periods==2)[:4]])
            records=[]
            for i in inds:
                check();g,d=data["grid"][i];np.savez(folder/f"cycle-{i:04d}-start.npz",theta=data["final"][i],period=periods[i],grid=data["grid"][i]);record=qualify_cycle(data["final"][i],j,float(g),float(d),int(periods[i]),check)
                arrays={k:v for k,v in record.items() if isinstance(v,np.ndarray)};summary={k:v for k,v in record.items() if not isinstance(v,np.ndarray)}
                np.savez(folder/f"cycle-{i:04d}.npz",**arrays);summary.update(index=int(i),gain=float(g),delta=float(d),initial=int(i%2));save(folder/f"cycle-{i:04d}.json",summary);records.append(summary)
            report={"n":n,"seed":seed,"batch_seconds":batch_seconds,"period_proposal_counts":{str(p):int(np.sum(periods==p)) for p in np.unique(periods)},"qualification_attempts":len(records),"qualification_indices":inds.tolist(),"qualified_seed_cases":sum(x["status"]=="numerical_attracting_primitive" for x in records),"tangent_collapse_cases":int(np.sum(data["tangent_collapse_count"]>0)),"positive_both_half_time_estimates":int(np.sum((data["tangent_collapse_count"]==0)&(np.min(data["growth_blocks"].reshape(2,8,-1).sum(axis=1)/2048/data["grid"][:,1],axis=0)>.002))),"qualified_status":"not_a_shrimp_geometry_claim"}
            result["networks"].append(report);save(folder/"summary.json",report);save(out/"progress.json",result)
            if args.mode=="profile":
                t=time.monotonic()
                for _ in range(10):
                    _,m,_,_=iterate(np.zeros(n),j,.8,.5,64);np.linalg.eig(m)
                result.update(batch_seconds=batch_seconds,period64_unit_seconds=(time.monotonic()-t)/10)
        check();result["status"]="completed"
    except BaseException as e:result.update(status="failed_or_interrupted",error=type(e).__name__+": "+str(e))
    finally:
        result.update(elapsed_seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save(out/"result.json",result);manifest(out)
    return 0 if result["status"]=="completed" else 2
if __name__=="__main__":sys.exit(main())
