"""Fixed owned-process supervisor for one saved-only mean-value outcome review."""
import os,json,hashlib,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path('/data2/neural-shrimp-data/audits/s2-period18-meanvalue/attempt-001')
BUNDLE='/data2/neural-shrimp-data/explore/certificates/s2-period18-connection-meanvalue-2026-10-09/attempt-001-target'
PIN='0e514532fed9d2a486e57c0341c5355c152e8c550cdd19aa180b1b2e6a2663e7'
EXECUTION='916c76d35c1786b906991bebcc16406864e62c25'

def entry(p):return {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def run():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    assert os.getpriority(os.PRIO_PROCESS,0)>=10
    OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();proc=None;code=2
    def save(name,obj):
        with(OUT/name).open('xb')as f:f.write((json.dumps(obj,sort_keys=True,indent=2)+'\n').encode());f.flush();os.fsync(f.fileno())
    command=['timeout','--signal=TERM','--kill-after=5s','45s','nice','-n','10',sys.executable,str(ROOT/'tools/reviews/review_meanvalue_outcome.py'),'--out',str(OUT/'run'),'--bundle',BUNDLE,'--manifest',PIN,'--execution',EXECUTION]
    meta={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'launcher_pid':os.getpid(),'argv':command,'load':os.getloadavg(),'meminfo':Path('/proc/meminfo').read_text(),'started_unix':time.time(),'external_seconds':45,'kill_grace_seconds':5}
    save('launcher-start.json',meta);(OUT/'launcher.py').write_bytes(Path(__file__).read_bytes())
    with(OUT/'process.log').open('xb')as log:
        try:proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);save('owned-process.json',{'group_pid':proc.pid});code=proc.wait(timeout=52)
        except BaseException as exc:meta['failure']=type(exc).__name__+': '+str(exc)
        finally:
            if proc is not None:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait()
    live=[]
    if proc is not None:
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:
                fields=(p/'stat').read_text().rsplit(')',1)[1].split()
                if int(fields[2])==proc.pid and fields[0]!='Z':live.append(int(p.name))
            except(OSError,ValueError):pass
    if live:code=2
    if sum(p.stat().st_size for p in OUT.rglob('*')if p.is_file())>9*1024**2:code=2
    meta.update(exit_code=code,elapsed_seconds=time.monotonic()-start,owned_group_pid=None if proc is None else proc.pid,owned_live_pids=live);save('launcher-result.json',meta)
    files={str(p.relative_to(OUT)):entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()};save('manifest.json',{'files':files,'total_bytes':sum(x['bytes']for x in files.values())})
    for p in sorted(OUT.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
    OUT.chmod(0o555);print(json.dumps({'out':str(OUT),'exit_code':code,'manifest':entry(OUT/'manifest.json')}),flush=True);return code
if __name__=='__main__':raise SystemExit(run())
