"""Owned-process30+5s launcher for the fixed saved-only gauge diagnosis."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,json,signal,subprocess,sys,time
from pathlib import Path
import analyze_s2_period18_tangent_variation as worker
io=worker.io


def launch(out):
    require=worker.require
    require(os.getpriority(os.PRIO_PROCESS,0)>=10,'launcher nice10')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=worker.ROOT,text=True).strip(),'clean source')
    source=Path(__file__).read_bytes()
    require(source==subprocess.check_output(['git','show','HEAD:tools/launch_s2_period18_tangent_variation.py'],cwd=worker.ROOT),'committed launcher')
    out.mkdir(parents=True,exist_ok=False);(out/'launcher-source.py').write_bytes(source)
    start=time.monotonic();proc=None
    cmd=['timeout','--signal=TERM','--kill-after=5s','30s',sys.executable,str(worker.ROOT/'tools/analyze_s2_period18_tangent_variation.py'),'--out',str(out/'run')]
    meta={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=worker.ROOT,text=True).strip(),'launcher_pid':os.getpid(),
        'argv':cmd,'started_utc':io.now(),'nice':os.getpriority(os.PRIO_PROCESS,0),'load':os.getloadavg(),'meminfo':Path('/proc/meminfo').read_text(),
        'launcher_source_sha256':io.sha256(source),'external_seconds':30,'kill_grace_seconds':5,'aggregate_raw_cap':9*1024**2}
    io.save(out/'launcher-start.json',meta)
    with (out/'process.log').open('xb') as log:
        try:
            proc=subprocess.Popen(cmd,cwd=worker.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            meta['owned_group_pid']=proc.pid;io.save(out/'launcher-start.json',meta);code=proc.wait(timeout=37)
        except BaseException as exc:meta['error']={'type':type(exc).__name__,'message':str(exc)};code=124
        finally:
            if proc is not None:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:code=124
    live=[]
    if proc is not None:
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:
                fields=(p/'stat').read_text().rsplit(')',1)[1].split()
                if int(fields[2])==proc.pid and fields[0]!='Z':live.append(int(p.name))
            except (OSError,ValueError):pass
    if live:code=124
    meta.update(exit_code=code,elapsed_seconds=time.monotonic()-start,finished_utc=io.now(),owned_live_pids=live)
    io.save(out/'launcher-result.json',meta)
    if (out/'run/manifest.json').exists():io.verify_inventory(out/'run',io.read(out/'run/manifest.json')['files'])
    io.save(out/'manifest.json',{'files':io.inventory(out),'within_raw_cap':io.total(out)<9*1024**2})
    for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o444 if p.is_file() else 0o555)
    out.chmod(0o555);require(io.total(out)<9*1024**2,'aggregate raw cap')
    print(json.dumps({'exit_code':code,'elapsed_seconds':meta['elapsed_seconds'],'manifest':io.entry(out/'manifest.json')}));return code


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();raise SystemExit(launch(a.out))
