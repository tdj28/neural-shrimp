"""Exclusive owned CPU1 block-mean profile or separately admitted target."""
import argparse,hashlib,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/data2/neural-shrimp-data/explore/tiny-lm-block-mean')
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
CAP=514*1024**2

def entry(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1024**2):h.update(b)
 return dict(bytes=p.stat().st_size,sha256=h.hexdigest())
def owned(group):
 live=[];rss=0
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   fields=(p/'stat').read_text().rsplit(')',1)[1].split()
   if int(fields[2])==group and fields[0]!='Z':
    live.append(int(p.name));rss+=int(next((v.split()[1] for v in (p/'status').read_text().splitlines() if v.startswith('VmRSS:')),'0'))
  except(OSError,ValueError):pass
 return live,rss

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['profile','target'],required=True);args=parser.parse_args()
 if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip():raise ValueError('clean source')
 if os.getpriority(os.PRIO_PROCESS,0)<10:raise ValueError('nice10')
 available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))
 if available<10*1024**2:raise MemoryError('10GiB headroom')
 card=json.loads((ROOT/'docs/plans/tiny-lm-block-mean-inputs.json').read_bytes())
 if card['status']!=('profile_admitted' if args.mode=='profile' else 'target_admitted'):raise ValueError('mode-specific committed admission')
 resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,8*1024**3))
 out=BASE/('attempt001-profile' if args.mode=='profile' else 'attempt001-target');out.mkdir(parents=True,exist_ok=False);start=time.monotonic();proc=None;failure=None;peak=0
 def save(name,row):
  with(out/name).open('xb') as f:f.write((json.dumps(row,indent=2,sort_keys=True)+'\n').encode());f.flush();os.fsync(f.fileno())
 limit=80 if args.mode=='profile' else 1215
 argv=[sys.executable,str(ROOT/'tools/explore_tiny_lm_block_mean.py'),'--mode',args.mode,'--out',str(out/'run')]
 metadata=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),launcher_pid=os.getpid(),mode=args.mode,argv=argv,external_seconds=limit,kill_grace_seconds=5,worker_rss_kib=4*1024**2,worker_as_bytes=8*1024**3,output_cap_bytes=CAP,mem_available_kib=available,load=os.getloadavg(),started_unix=time.time(),threads=dict.fromkeys(THREADS,'1'))
 save('launcher-start.json',metadata);(out/'launcher.py').write_bytes(Path(__file__).read_bytes())
 env=dict(os.environ,**dict.fromkeys(THREADS,'1'),PYTHONDONTWRITEBYTECODE='1')
 def interrupted(*_):raise TimeoutError('owned launcher interrupted')
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 with(out/'process.log').open('xb') as log:
  try:
   proc=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);save('owned-process.json',dict(group_pid=proc.pid))
   while proc.poll() is None:
    _,rss=owned(proc.pid);peak=max(peak,rss)
    if peak>4*1024**2:raise MemoryError('owned4GiB group')
    if time.monotonic()-start>limit:raise TimeoutError('owned external cap')
    if (out/'process.log').stat().st_size>1024**2 or sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>CAP:raise OSError('owned output cap')
    time.sleep(.25)
  except BaseException as exc:failure=type(exc).__name__+': '+str(exc)
  finally:
   if proc is not None:
    if proc.poll() is None:
     try:os.killpg(proc.pid,signal.SIGTERM)
     except ProcessLookupError:pass
     try:proc.wait(timeout=5)
     except subprocess.TimeoutExpired:pass
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    proc.wait()
 live=[] if proc is None else owned(proc.pid)[0];code=2 if proc is None else proc.returncode
 if failure or live or sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>CAP:code=2
 metadata.update(exit_code=code,elapsed_seconds=time.monotonic()-start,owned_group_pid=None if proc is None else proc.pid,owned_live_pids=live,worker_pid_absent=proc is None or not Path('/proc',str(proc.pid)).exists(),maximum_group_rss_kib=peak,failure=failure)
 save('launcher-result.json',metadata);save('manifest.json',dict(files={str(p.relative_to(out)):entry(p) for p in out.rglob('*') if p.is_file()}))
 for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir() else 0o444)
 out.chmod(0o555);print(json.dumps(dict(root=str(out),exit_code=code,manifest=entry(out/'manifest.json'))));return code
if __name__=='__main__':raise SystemExit(main())
