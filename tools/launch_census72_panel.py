"""Owned external bounds and complete immutable census pilot inventory."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def entry(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  while b:=f.read(1024**2):h.update(b)
 return dict(bytes=p.stat().st_size,sha256=h.hexdigest())
def group(g):
 rows=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   x=(p/'stat').read_text().rsplit(')',1)[1].split()
   if int(x[2])==g and x[0]!='Z':rows.append((int(p.name),int(x[21])*os.sysconf('SC_PAGE_SIZE')))
  except(OSError,ValueError):pass
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['profile','target'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--profile-outer');p.add_argument('--profile-worker');a=p.parse_args()
 if a.out!=Path('/data2/neural-shrimp-data/explore/certificates/census72-panel')/f'attempt002-{a.mode}':raise ValueError('fixed owned output root')
 if os.getpriority(os.PRIO_PROCESS,0)<10 or subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip():raise ValueError('clean committed nice10 launch')
 mem=Path('/proc/meminfo').read_text();available=int(next(x.split()[1]for x in mem.splitlines()if x.startswith('MemAvailable:')))*1024
 if available<8*1024**3:raise MemoryError('8GiBheadroom')
 if a.mode=='target' and not(a.profile_outer and a.profile_worker):raise ValueError('literal profile pins required')
 a.out.mkdir(parents=True,exist_ok=False);external=200 if a.mode=='profile'else 5420;env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');args=['timeout','--signal=TERM','--kill-after=5s',str(external)+'s',sys.executable,str(ROOT/'tools/certify_census72_panel.py'),'--mode',a.mode]
 if a.mode=='target':args+=['--profile-outer',a.profile_outer,'--profile-worker',a.profile_worker]

 def save(n,v):
  with(a.out/n).open('x')as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 rec=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),mode=a.mode,command=args,meminfo=mem,load=os.getloadavg(),external_seconds=external,kill_grace=5,rss_cap_bytes=2*1024**3,peak_group_rss_bytes=0,started_unix=time.time());save('launcher-start.json',rec);start=time.monotonic();child=None;code=2
 try:
  with(a.out/'process.log').open('xb')as log:
   child=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);save('owned-process.json',{'pid':child.pid})
   while child.poll()is None:
    rows=group(child.pid);rss=sum(x[1]for x in rows);rec['peak_group_rss_bytes']=max(rec['peak_group_rss_bytes'],rss)
    if rss>rec['rss_cap_bytes']or time.monotonic()-start>external+7 or (a.out/'process.log').stat().st_size>1024**2:raise RuntimeError('owned resource ceiling')
    time.sleep(.1)
   code=child.returncode
 except BaseException as e:rec['failure']=type(e).__name__+': '+str(e)
 finally:
  if child is not None:
   try:os.killpg(child.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   child.wait()
  live=[]if child is None else group(child.pid);rec.update(exit_code=code,elapsed_seconds=time.monotonic()-start,owned_live_pids=[x[0]for x in live]);save('launcher-result.json',rec)
  files={str(p.relative_to(a.out)):entry(p)for p in a.out.rglob('*')if p.is_file()};save('manifest.json',{'files':files})
  for f in sorted(a.out.rglob('*'),key=lambda p:len(p.parts),reverse=True):f.chmod(0o555 if f.is_dir()else 0o444)
  a.out.chmod(0o555)
 print(json.dumps({'root':str(a.out),'code':code,'owned_live_pids':rec['owned_live_pids'],'manifest':entry(a.out/'manifest.json')}));return 0 if code==0 and not live and sum(x['bytes']for x in files.values())<16*1024**3+1024**2 else 2
if __name__=='__main__':raise SystemExit(main())
