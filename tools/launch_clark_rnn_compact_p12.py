#!/usr/bin/env python3
"""Owned subprocess and retained launch records for the bounded Clark pilot."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def save(p,x):p.write_text(json.dumps(x,indent=2)+"\n")
def group_alive(pgid):
    try:os.killpg(pgid,0);return True
    except ProcessLookupError:return False

def cleanup_group(pgid):
    if group_alive(pgid):
        os.killpg(pgid,signal.SIGTERM);deadline=time.monotonic()+1
        while group_alive(pgid) and time.monotonic()<deadline:time.sleep(.05)
    if group_alive(pgid):
        os.killpg(pgid,signal.SIGKILL);deadline=time.monotonic()+4
        while group_alive(pgid) and time.monotonic()<deadline:time.sleep(.05)
    return not group_alive(pgid)

def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
    if os.getpriority(os.PRIO_PROCESS,0)<10:raise ValueError("nice enforcement failed")
    a.out.mkdir(parents=True,exist_ok=False);env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
    args=[sys.executable,"-u",str(ROOT/"tools/refine_clark_rnn_compact_p12.py"),"--out",str(a.out/"run")]
    started=time.monotonic();record={"command":args,"launcher_pid":os.getpid(),"source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"started_unix":time.time(),"external_seconds":270,"kill_after_seconds":5,"nice":os.getpriority(os.PRIO_PROCESS,0),"mem_available_bytes":int(next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:")))*1024}
    with (a.out/"process.log").open("wb") as log:
        child=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);record["child_pid"]=child.pid;save(a.out/"launch-start.json",record)
        try:code=child.wait(timeout=record["external_seconds"])
        except subprocess.TimeoutExpired:
            record["timed_out"]=True;os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
    empty=cleanup_group(child.pid)
    if not empty:code=2
    record.update(owned_group_empty=empty,exit_code=code,elapsed_seconds=time.monotonic()-started,child_pid_absent=not Path(f"/proc/{child.pid}").exists());save(a.out/"launch-end.json",record)
    files={str(f.relative_to(a.out)):{"sha256":hashlib.sha256(f.read_bytes()).hexdigest(),"bytes":f.stat().st_size} for f in sorted(a.out.rglob("*")) if f.is_file()};save(a.out/"manifest.json",{"files":files})
    for f in a.out.rglob("*"):
        if f.is_file():f.chmod(0o444)
    for d in sorted([x for x in a.out.rglob("*") if x.is_dir()],key=lambda x:len(x.parts),reverse=True):d.chmod(0o555)
    a.out.chmod(0o555)
    print(json.dumps(record));return code
if __name__=="__main__":sys.exit(main())
