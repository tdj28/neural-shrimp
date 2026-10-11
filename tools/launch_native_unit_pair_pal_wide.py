#!/usr/bin/env python3
"""Four bounded independent705 CPU workers and ordered exact-gauge joins."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import gc,json,signal,subprocess,sys,time,resource
from pathlib import Path
import probe_native_unit_pair_pal_wide as B
from flint import ctx
ROOT=B.ROOT;OUT=B.BASE;entry=B.entry
PROFILE='11819b8985345f34fb68e2ed8c9da5384f33804dd3343671ae2cd0c6fd38cbcc'
PROFILE_OUTER='fa27b1e43c10cbe66040fa7bd80f58e4effdd44b72f31101ae04f92e5db7135f'

def owned(group):
 rows=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   fields=(p/'stat').read_text().rsplit(')',1)[1].split()
   if int(fields[2])==group and fields[0]!='Z':rows.append((int(p.name),int(next((s.split()[1]for s in (p/'status').read_text().splitlines()if s.startswith('VmRSS:')),'0'))))
  except(OSError,ValueError):pass
 return rows

def stop(group,sig):
 try:os.killpg(group,sig)
 except ProcessLookupError:pass

def main():
 B.require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source');B.require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
 mem=Path('/proc/meminfo').read_text();available=int(next(v.split()[1]for v in mem.splitlines()if v.startswith('MemAvailable:')));B.require(available>36*1024**2,'36GiB headroom for four capped workers')
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();jobs=[];peak_total=0;parent_peak=0;failure=None;result={'passed':False,'connected_count':0,'joins':[],'workers':[],'scope':'Four independently proposed705 tubes; only the consecutive prefix passing every full endpoint join is connected.'}
 source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 def save(name,value):
  raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();B.require(len(raw)<4*1024**2,'small parent metadata')
  with(OUT/name).open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 def parent_check():
  nonlocal parent_peak
  rss=int(next(v.split()[1]for v in Path('/proc/self/status').read_text().splitlines()if v.startswith('VmRSS:')))
  parent_peak=max(parent_peak,rss);B.require(rss<4*1024**2,'parent4GiB cap including ordered joins')
 def deadline(*_):raise TimeoutError('330s total batch including joins')
 signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGTERM,deadline);signal.setitimer(signal.ITIMER_REAL,330)
 save('launcher-start.json',{'commit':source,'launcher_pid':os.getpid(),'meminfo':mem,'load':os.getloadavg(),'worker_count':4,'worker_seconds':240,'owned_seconds':255,'grace_seconds':5,'worker_rss_kib':6*1024**2,'aggregate_worker_rss_kib':24*1024**2,'parent_rss_kib':4*1024**2,'total_seconds':330,'raw_cap_bytes':3076*1024**2,'started_unix':time.time()})
 try:
  # Proposals use only the same already accepted parent, so workers have no
  # sequential mathematical dependency. Connection is checked afterward.
  for unit in range(1,5):
   folder=OUT/f'chart{unit:03d}';folder.mkdir(exist_ok=False);log=(folder/'process.log').open('xb')
   cmd=[sys.executable,str(ROOT/'tools/probe_native_unit_pair_pal_wide.py'),'--unit',str(unit),'--out',str(folder/'run'),'--profile-manifest',PROFILE,'--profile-outer-manifest',PROFILE_OUTER]
   child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);jobs.append({'unit':unit,'child':child,'log':log,'started':time.monotonic(),'peak':0,'reason':None,'term':None,'folder':folder,'command':cmd})
  save('owned-processes.json',{'workers':[{'unit':j['unit'],'group':j['child'].pid,'command':j['command']}for j in jobs]})
  while any(j['child'].poll()is None for j in jobs):
   total=0
   for j in jobs:
    rss=sum(r[1]for r in owned(j['child'].pid));total+=rss;j['peak']=max(j['peak'],rss)
    if j['child'].poll()is not None:continue
    reason='6GiB workerRSS'if rss>=6*1024**2 else '255s owned deadline'if time.monotonic()-j['started']>=255 else '128KiB workerlog'if (j['folder']/'process.log').stat().st_size>=128*1024 else None
    if reason and j['term']is None:j['reason']=reason;j['term']=time.monotonic();stop(j['child'].pid,signal.SIGTERM)
    if j['term']is not None and time.monotonic()-j['term']>=5:stop(j['child'].pid,signal.SIGKILL)
   peak_total=max(peak_total,total);parent_peak=max(parent_peak,int(next(v.split()[1]for v in Path('/proc/self/status').read_text().splitlines()if v.startswith('VmRSS:'))))
   B.require(total<24*1024**2 and parent_peak<4*1024**2,'aggregate24GiB and parent4GiB caps')
   B.require(sum(p.stat().st_size for p in OUT.rglob('*')if p.is_file())<3076*1024**2,'total batch output cap')
   time.sleep(.2)
  for j in jobs:
   j['child'].wait();stop(j['child'].pid,signal.SIGKILL);j['log'].close();live=owned(j['child'].pid)
   row={'unit':j['unit'],'group':j['child'].pid,'exit_code':j['child'].returncode,'peak_group_rss_kib':j['peak'],'stop_reason':j['reason'],'owned_live_pids':[p for p,_ in live]};result['workers'].append(row)
   B.require(not live,'owned worker termination')
  # Every join is conditional on an independently proved original parent and
  # complete source-matched uniform roots, never on float path proximity.
  with ctx.workprec(384):
   model,previous,binding=B.bind_parent();parent_check();save('parent-binding.json',binding)
   keep={'period','dimension','parameter_center','parameter_endpoints','normal','origin','center','slope','curvature','coordinate_radii','endpoint_root_boxes','settings'}
   previous={k:v for k,v in previous.items()if k in keep};gc.collect()
   for row in result['workers']:
    unit=row['unit'];folder=OUT/f'chart{unit:03d}/run'
    if row['exit_code']!=0 or row['stop_reason']is not None:break
    pin=entry(folder/'manifest.json');B.inventory(folder,pin['sha256']);rt=B.read(folder/'runtime.json');done=B.read(folder/'outcome.json');inp=B.read(folder/'inputs.json')
    B.require(rt['commit']==source and rt['settings']==B.settings(unit)and done['unit']==unit,'exact indexed worker source/settings')
    B.require(done['passed']is True and done['status']=='uniform705_unjoined'and not done['connected']and done['ticks']==253 and done['events']=={'direction_return':144,'pal_chart_return':1,'pal_newton_proposal':2,'pal_path_assembly_return':3,'pal_path_phase_return':108,'pal_preconditioner_return':1},'complete full705 root-only acquisition')
    B.require(inp['map_inputs']==model and inp['bindings']==binding,'same literal map/proof/independent parent')
    for name,identity in rt['sources'].items():B.require(entry(ROOT/name)==entry(folder/'source'/name)==identity,'identical reviewed source '+name)
    current=B.read(folder/'certificate.json.gz');parent_check();B.require(B.uniform_admission(current,B.decode(current['whole_box'][-1])>0),'complete uniform mathematical proof')
    current={k:v for k,v in current.items()if k in keep};gc.collect();join=B.ordered_join(previous,current);parent_check();save(f'join-{unit:03d}.json',join)
    result['joins'].append({'unit':unit,'worker_manifest':pin,'passed':join['passed'],'coordinates':sum(join.get('coordinate_inclusions',[])),'right':current['parameter_endpoints'][1]})
    if not join['passed']:break
    result['connected_count']+=1;result['connected_right']=current['parameter_endpoints'][1];result['right_physical']=current['endpoint_root_boxes'][1][-3:];previous=current
   parent_check();result['passed']=result['connected_count']==4
 except BaseException as exc:failure=type(exc).__name__+': '+str(exc);result['failure']=failure
 finally:
  signal.setitimer(signal.ITIMER_REAL,0)
  for j in jobs:
   stop(j['child'].pid,signal.SIGKILL);j['child'].wait();j['log'].close()
  live=[p for j in jobs for p,_ in owned(j['child'].pid)];result.update(owned_live_pids=live,seconds=time.monotonic()-start,peak_worker_sum_rss_kib=peak_total,parent_peak_sampled_rss_kib=parent_peak,commit=source)
  if live or failure or result['seconds']>=330:result['passed']=False
  save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(OUT)):entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}})
  for p in sorted(OUT.rglob('*'),key=lambda p:len(p.parts),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
  OUT.chmod(0o555)
 print(json.dumps({'root':str(OUT),'passed':result['passed'],'connected_count':result['connected_count'],'manifest':entry(OUT/'manifest.json')}));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
