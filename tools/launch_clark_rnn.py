#!/usr/bin/env python3
"""Owned subprocess and retained launch records for the bounded Clark pilot."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def save(p,x):p.write_text(json.dumps(x,indent=2)+"\n")
def main():
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=["profile","scan"],required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--profile");p.add_argument("--profile-sha256");a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
    args=[sys.executable,"-u",str(ROOT/"tools/explore_clark_rnn.py"),"--mode",a.mode,"--out",str(a.out/"run")]
    if a.profile:args += ["--profile",a.profile,"--profile-sha256",a.profile_sha256]
    started=time.monotonic();record={"command":args,"launcher_pid":os.getpid(),"source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"started_unix":time.time(),"external_seconds":60 if a.mode=="profile" else 330,"kill_after_seconds":5}
    with (a.out/"process.log").open("wb") as log:
        child=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);record["child_pid"]=child.pid;save(a.out/"launch-start.json",record)
        try:code=child.wait(timeout=record["external_seconds"])
        except subprocess.TimeoutExpired:
            record["timed_out"]=True;os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
    record.update(exit_code=code,elapsed_seconds=time.monotonic()-started,child_pid_absent=not Path(f"/proc/{child.pid}").exists());save(a.out/"launch-end.json",record)
    files={str(f.relative_to(a.out)):{"sha256":hashlib.sha256(f.read_bytes()).hexdigest(),"bytes":f.stat().st_size} for f in sorted(a.out.rglob("*")) if f.is_file()};save(a.out/"manifest.json",{"files":files})
    for f in a.out.rglob("*"):
        if f.is_file():f.chmod(0o444)
    for d in sorted([x for x in a.out.rglob("*") if x.is_dir()],key=lambda x:len(x.parts),reverse=True):d.chmod(0o555)
    a.out.chmod(0o555)
    print(json.dumps(record));return code
if __name__=="__main__":sys.exit(main())
