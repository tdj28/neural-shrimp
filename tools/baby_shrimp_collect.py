"""Bounded collection of already-owned BSZ1/BSP3 jobs; never create resources.

Wait for the exact workers' retrieved, terminated outcomes, then run the
committed numerical audits and figures. This complements the independent pod
watchdogs and allows collection to continue during a conversation interruption.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT.parent/"neural-shrimp-data"
RUNS=DATA/"runpod/baby-shrimp-zoom"
BANDS=[("bsz1-20261021","neural-shrimp-bsz1-c00","jjxh5evklyq4g7"),
       ("bsz1-20261021","neural-shrimp-bsz1-c01","csdh3rdzw0rnns"),
       ("bsz1-secure2-20261021","neural-shrimp-bsz1-c02-secure2","ihz1dfg3hha41f"),
       ("bsz1-secure2-20261021","neural-shrimp-bsz1-c03-secure2","zpdw0hnx2koygq")]
SURVEY=("bsp3-20261021","neural-shrimp-bsz1-periods","ygvg7zs096c81t")
SOURCES=("tools/baby_shrimp_collect.py","tools/baby_shrimp_finish.py","tools/baby_shrimp_period_survey_finish.py")


def worker(spec):
    group,name,pid=spec
    matches=list((RUNS/group/"workers"/name).glob("*/runner-stdout.txt"))
    if len(matches)!=1:
        raise ValueError(f"expected exactly one attempt for {name}; found {len(matches)}")
    return matches[0].parent


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--timeout",type=int,default=5400)
    args=ap.parse_args()
    if not 0<args.timeout<=7200:
        ap.error("collection time limit must be at most7200seconds")
    args.out.mkdir(parents=True,exist_ok=False)
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    hashes={}
    for name in SOURCES:
        hashes[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        blob=subprocess.check_output(["git","show",commit+":"+name],cwd=ROOT)
        if hashlib.sha256(blob).hexdigest()!=hashes[name]:raise ValueError("commit collection source first")
    folders=[worker(s) for s in BANDS];survey=worker(SURVEY)
    env={**os.environ,**{k:"8" for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")}}
    status={"source_commit":commit,"source_sha256":hashes,"started_utc":datetime.now(timezone.utc).isoformat(),"completed":{}}
    deadline=time.monotonic()+args.timeout

    def ready(folder,pid,terminated):
        if not (folder/"result.json").exists() or pid not in terminated:return False
        row=json.loads((folder/"result.json").read_text())
        if row["pod_id"]!=pid or not row.get("hash_verified") or row.get("runner_returncode")!=0:
            raise ValueError(f"worker did not pass transfer/return-code gate: {pid}")
        return True

    def run(name,command):
        for source,sha in hashes.items():
            if hashlib.sha256((ROOT/source).read_bytes()).hexdigest()!=sha:raise ValueError("analysis source changed during collection")
        with (args.out/f"{name}.log").open("x") as f:
            subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=600)
        status["completed"][name]=datetime.now(timezone.utc).isoformat()
        print(json.dumps({"audit_complete":name}),flush=True)

    try:
        while len(status["completed"])<2:
            if time.monotonic()>=deadline:raise TimeoutError("collection wait expired; independent pod guards remain active")
            ledger=[json.loads(s) for s in (DATA/"runpod/pods.jsonl").read_text().splitlines()]
            terminated={r["pod_id"] for r in ledger if r["event"]=="terminated_verified"}
            if "periods" not in status["completed"] and ready(survey,SURVEY[2],terminated):
                run("periods",[sys.executable,str(ROOT/"tools/baby_shrimp_period_survey_finish.py"),"--worker",str(survey),
                               "--out",str(args.out.parent/"20261021-period-survey-audit-01")])
            if "zoom" not in status["completed"] and all(ready(f,s[2],terminated) for f,s in zip(folders,BANDS)):
                run("zoom",[sys.executable,str(ROOT/"tools/baby_shrimp_finish.py"),"--workers",*[str(f) for f in folders],
                            "--out",str(args.out.parent/"20261021-zoom-audit-01")])
            status["last_check_utc"]=datetime.now(timezone.utc).isoformat()
            status["retrieved_and_terminated"]=[s[2] for s,f in zip(BANDS+[SURVEY],folders+[survey]) if ready(f,s[2],terminated)]
            (args.out/"status.json").write_text(json.dumps(status,indent=2)+"\n")
            print(json.dumps({"time":status["last_check_utc"],"retrieved_and_terminated":len(status["retrieved_and_terminated"]),
                              "of":5,"completed_audits":list(status["completed"])}),flush=True)
            if len(status["completed"])<2:time.sleep(60)
    except BaseException as error:
        (args.out/"failed.json").write_text(json.dumps({"type":type(error).__name__,"error":str(error)})+"\n")
        raise


if __name__=="__main__":
    main()
