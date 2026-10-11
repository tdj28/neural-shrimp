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
 out=Path('/data2/neural-shrimp-data/audits/census-period58-independent/attempt002')
 if os.getpriority(os.PRIO_PROCESS,0)<10 or subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip():raise ValueError('clean committed nice10 launch')
 mem=Path('/proc/meminfo').read_text();available=int(next(x.split()[1]for x in mem.splitlines()if x.startswith('MemAvailable:')))*1024
 if available<10*1024**3:raise MemoryError('10GiB headroom')
 out.mkdir(parents=True,exist_ok=False);(out/'launcher-source.py').write_bytes(Path(__file__).read_bytes());external=620;env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');args=['timeout','--signal=TERM','--kill-after=5s',str(external)+'s',sys.executable,str(ROOT/'tools/audit_census_long_period.py')]
 def save(n,v):
  with(out/n).open('x')as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 rec=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),command=args,meminfo=mem,load=os.getloadavg(),external_seconds=external,kill_grace=5,rss_cap_bytes=4*1024**3,peak_group_rss_bytes=0,started_unix=time.time());save('launcher-start.json',rec);start=time.monotonic();child=None;code=2
 try:
  with(out/'process.log').open('xb')as log:
   child=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);save('owned-process.json',{'pid':child.pid})
   while child.poll()is None:
    rows=group(child.pid);rss=sum(x[1]for x in rows);rec['peak_group_rss_bytes']=max(rec['peak_group_rss_bytes'],rss)
    if rss>rec['rss_cap_bytes']or time.monotonic()-start>external+7 or (out/'process.log').stat().st_size>1024**2:raise RuntimeError('owned resource ceiling')
    time.sleep(.1)
   code=child.returncode
 except BaseException as e:rec['failure']=type(e).__name__+': '+str(e)
 finally:
  if child is not None:
   try:os.killpg(child.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   child.wait()
  live=[]if child is None else group(child.pid);rec.update(exit_code=code,elapsed_seconds=time.monotonic()-start,owned_live_pids=[x[0]for x in live]);save('launcher-result.json',rec)
  files={str(p.relative_to(out)):entry(p)for p in out.rglob('*')if p.is_file()};save('manifest.json',{'files':files})
  for f in sorted(out.rglob('*'),key=lambda p:len(p.parts),reverse=True):f.chmod(0o555 if f.is_dir()else 0o444)
  out.chmod(0o555)
 print(json.dumps({'root':str(out),'code':code,'owned_live_pids':rec['owned_live_pids'],'manifest':entry(out/'manifest.json')}));return 0 if code==0 and not live and sum(x['bytes']for x in files.values())<513*1024**2 else 2
if __name__=='__main__':raise SystemExit(main())
