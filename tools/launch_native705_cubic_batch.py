#!/usr/bin/env python3
"""One source/card-bound four-chart cubic705 batch with full native ordered joins."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,gc,hashlib,json,resource,signal,subprocess,sys,time
from pathlib import Path
from fractions import Fraction
from flint import arb,ctx
import probe_native705_cubic_batch as W
import launch_native_unit_pair_pal_batch as L
ROOT=W.ROOT;BASE=W.BASE;need=W.require;entry=W.entry;read=W.read
KEEP={'dimension','period','shooting_dimension','normal','origin','center','slope','curvature','cubic','weights','settings','parameter_center','parameter_endpoints','endpoint_root_boxes','bound','gates'}

def save(root,name,value):
 raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();need(len(raw)<16*1024**2,'bounded parent metadata')
 with(root/name).open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())

def seal(root):
 save(root,'manifest.json',{'files':{str(p.relative_to(root)):entry(p)for p in sorted(root.rglob('*'))if p.is_file()}})
 for p in sorted(root.rglob('*'),key=lambda p:len(p.parts),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
 root.chmod(0o555)

def bound_root(pin):
 root=Path(pin['root']);W.inventory(root,pin['outer']);W.inventory(root/'run',pin['worker']);rt=W.C.W.C.archive_bound(root,pin['source']);return root,rt

def known_parent(plan,identity):
 parent=plan['parent']
 if parent['kind']=='independent-correlated':
  need(plan['batch']==1 and {k:v for k,v in parent.items()if k!='kind'}==W.APINS,'sole literal independent initial parent')
  ar,rt=bound_root(W.APINS);out=read(ar/'run/outcome.json');la=read(ar/'launcher-result.json');binding=read(ar/'run/input-binding.json')['card'];ep=read(ar/'run/connected-endpoint.json')
  need(out['passed']is True and out['connected']is True and out['strict_inclusions']==out['anchor_coordinates']==705 and out['primitive_period']==18 and out['native_calls']==0 and out['parameter_endpoints']==['-49/2048','79/2048']and la['exit_code']==0 and not la['owned_live_pids'],'complete independently connected705 root')
  need(entry(ar/'run/connected-endpoint.json')['sha256']==W.APINS['endpoint']and len(ep['endpoint'])==705 and ep['chart']['shooting_dimension']==705,'literal independent full endpoint')
  prod=binding['producer'];need(prod['source']=='99b6f71e61b85cc8e201fa0b0f198a15ebc3a5c2'and prod['outer']=='28393e38f8d3fadfa1355f759439d574639cb4255ca20e1d6971085caeb5971c','same independently audited initial producer')
  pr,prt=bound_root(prod);W.C.same_native_source(identity,prt['sources']);model=read(pr/'run/inputs.json')['map_inputs']
  fr,frt=bound_root(W.FPINS);W.C.same_native_source(identity,frt['sources']);fo=read(fr/'run/outcome.json')
  need(not fo['passed']and fo['status']=='completed_bounds_failed'and fo['ticks']==149 and fo['strict_inclusions']==0 and fo['anchor_inclusions']==705,'complete preserved failed wider proposal')
  need(entry(fr/'run/certificate.json.gz')['sha256']==W.FPINS['certificate'],'literal failed proposal certificate');proposal=read(fr/'run/certificate.json.gz')
  need(read(fr/'run/inputs.json')['map_inputs']==model and not proposal['bound']['passed'],'same model and no promotion of failed proposal')
  for k in ['normal','origin','parameter_center']:need(ep['chart'][k]==proposal[k],'same literal independent/proposal gauge '+k)
  anchor={'box':ep['endpoint'],'normal':ep['chart']['normal'],'origin':ep['chart']['origin'],'parameter':ep['chart']['parameter_endpoints'][1]}
  return model,proposal,anchor,dict(parent=parent,proposal=W.FPINS)
 need(parent['kind']=='completed-cubic-batch'and type(plan['batch'])is int and 2<=plan['batch']<=20,'bounded continuation parent kind')
 root=BASE/f"batch{plan['batch']-1:03d}";need(parent['root']==str(root),'immediate preceding batch only');W.inventory(root,parent['outer']);out=read(root/'outcome.json');card=read(root/'input-card.json')
 need(out['passed']is True and out['connected_count']==4 and out['source']==parent['source']==card['source']and not out['owned_live_pids']and len(out['joins'])==4 and all(v['passed']and v['coordinates']==705 for v in out['joins']),'entire completed prior branch joins')
 last=root/'chart004';W.inventory(last/'run',out['joins'][-1]['worker_manifest']['sha256']);rt=W.C.W.C.archive_bound(last,parent['source']);W.C.same_native_source(identity,rt['sources']);done=read(last/'run/outcome.json')
 need(done['passed']and done['ticks']==331 and done['events']==W.EXPECTED and done['strict_inclusions']==705 and done['status']=='uniform705_unjoined','exact complete last root')
 proposal=read(last/'run/certificate.json.gz');need(W.uniform(proposal),'complete prior mathematical root');ep=read(root/'connected-endpoint.json')
 need(ep['endpoint']==proposal['endpoint_root_boxes'][1]and ep['chart']=={k:proposal[k]for k in ['shooting_dimension','normal','origin','parameter_center','parameter_endpoints','settings']},'literal prior full endpoint')
 model=read(last/'run/inputs.json')['map_inputs'];need(model==card['map_inputs'],'same literal parent model')
 return model,proposal,{'box':ep['endpoint'],'normal':ep['chart']['normal'],'origin':ep['chart']['origin'],'parameter':ep['chart']['parameter_endpoints'][1]},dict(parent=parent,last_worker=entry(last/'run/manifest.json'))

def seeds(proposal,anchor):
 need(all(len(proposal[k])==705 for k in ['center','slope','curvature','cubic','weights'])and proposal['normal']==anchor['normal']and proposal['origin']==anchor['origin'],'full same-gauge finite proposal')
 c,s,a,b=[[W.decode(v).mid()for v in proposal[k]]for k in ['center','slope','curvature','cubic']];initials=[]
 for unit in range(1,5):
  d=W.exact(W.settings(anchor['parameter'],unit)['parameter'])-W.exact(proposal['parameter_center'])
  initials.append([W.exact_mid(c[i]+s[i]*d+a[i]*d*d/2+b[i]*d*d*d)for i in range(705)])
 return initials

def committed_plan(path):
 path=path.resolve();need(path.parent==ROOT/'docs/plans'and path.name.startswith('native705-cubic-batch-')and path.suffix=='.json','committed local batch plan only')
 rel=str(path.relative_to(ROOT));need(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT)).hexdigest()==entry(path)['sha256'],'exact committed pretarget card');p=read(path)
 need(p['schema']=='native705-cubic-batch-v1'and p['status']=='measured-reviewed'and type(p['batch'])is int and 1<=p['batch']<=20 and p['workers']==4 and p['halfwidth_exponent']==W.HALFWIDTH and p['radius_exponent']==-10 and p['overlap']=='1/2048','fixed reviewed domain and measured card')
 return p

def main(args):
 identity=W.sources();plan=committed_plan(args.card);source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
 mem=Path('/proc/meminfo').read_text();available=int(next(v.split()[1]for v in mem.splitlines()if v.startswith('MemAvailable:')));need(available>36*1024**2,'36GiB current headroom for24GiB workers plus parent/OS')
 root=BASE/f"batch{plan['batch']:03d}";root.mkdir(parents=True,exist_ok=False);start=time.monotonic();jobs=[];failure=None;peak=0;result={'passed':False,'connected_count':0,'joins':[],'workers':[],'source':source,'batch':plan['batch'],'scope':'Connected primitive18 unit-pair locus prefix only; complementary spectrum and global junction/window topology remain separate.'}
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,12*1024**3))
 def check():
  need(time.monotonic()-start<330,'330s whole batch');rss=int(next(v.split()[1]for v in Path('/proc/self/status').read_text().splitlines()if v.startswith('VmRSS:')));need(rss<4*1024**2,'4GiB current parent RSS');need(sum(p.stat().st_size for p in root.rglob('*')if p.is_file())<4100*1024**2-16*1024**2,'whole4100MiB raw with closing reserve')
 def halt(*_):raise TimeoutError('330s whole batch limit')
 signal.signal(signal.SIGALRM,halt);signal.signal(signal.SIGTERM,halt);signal.setitimer(signal.ITIMER_REAL,330)
 save(root,'launcher-start.json',{'source':source,'pid':os.getpid(),'card':entry(args.card),'plan':plan,'meminfo':mem,'load':os.getloadavg(),'workers':4,'worker_seconds':240,'owned_seconds':255,'kill_grace':5,'worker_rss_kib':6*1024**2,'aggregate_worker_rss_kib':24*1024**2,'parent_rss_kib':4*1024**2,'raw_cap_bytes':4100*1024**2})
 try:
  prof=W.profile(identity,plan['profile']);need(prof['forecast']==plan['forecast'],'literal measured continuation forecast');check()
  with ctx.workprec(384):model,proposal,anchor,binding=known_parent(plan,identity)
  with ctx.workprec(256):initials=seeds(proposal,anchor)
  card={'source':source,'batch':plan['batch'],'worker_count':4,'kernel':'fresh-cubic705-v1','map_inputs':model,'normal':anchor['normal'],'origin':anchor['origin'],'weights':proposal['weights'],'anchor':anchor,'initials':initials,'settings':[W.settings(anchor['parameter'],i)for i in range(1,5)],'parent':binding,'profile':plan['profile'],'committed_plan':entry(args.card),'forecast':prof['forecast']}
  W.validate_card(card,source,plan['batch']);save(root,'input-card.json',card);del proposal;gc.collect();check();cardsha=entry(root/'input-card.json')['sha256']
  for unit in range(1,5):
   folder=root/f'chart{unit:03d}';folder.mkdir();log=(folder/'process.log').open('xb');cmd=[sys.executable,str(ROOT/W.WORKER),'--batch',str(plan['batch']),'--unit',str(unit),'--out',str(folder/'run'),'--card-sha',cardsha]
   proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);jobs.append({'unit':unit,'proc':proc,'log':log,'folder':folder,'start':time.monotonic(),'peak':0,'reason':None,'term':None,'command':cmd})
  save(root,'owned-processes.json',{'workers':[{'unit':j['unit'],'group':j['proc'].pid,'command':j['command']}for j in jobs]})
  while any(j['proc'].poll()is None for j in jobs):
   check();total=0
   for j in jobs:
    rss=sum(v for _,v in L.owned(j['proc'].pid));total+=rss;j['peak']=max(j['peak'],rss)
    if j['proc'].poll()is not None:continue
    reason='6GiB worker RSS'if rss>=6*1024**2 else '255s owned worker time'if time.monotonic()-j['start']>=255 else '128KiB log cap'if(j['folder']/'process.log').stat().st_size>=128*1024 else None
    if reason and j['term']is None:j['reason']=reason;j['term']=time.monotonic();L.stop(j['proc'].pid,signal.SIGTERM)
    if j['term']is not None and time.monotonic()-j['term']>=5:L.stop(j['proc'].pid,signal.SIGKILL)
   peak=max(peak,total);need(total<24*1024**2,'24GiB aggregate worker RSS');time.sleep(.25)
  for j in jobs:
   j['proc'].wait();L.stop(j['proc'].pid,signal.SIGKILL);j['log'].close();live=L.owned(j['proc'].pid);need(not live,'empty owned group');result['workers'].append({'unit':j['unit'],'exit_code':j['proc'].returncode,'peak_rss_kib':j['peak'],'stop_reason':j['reason'],'owned_live_pids':[]})
  for owner in result['workers']:
   unit=owner['unit'];folder=root/f'chart{unit:03d}/run'
   if owner['exit_code']!=0 or owner['stop_reason']:break
   pin=entry(folder/'manifest.json');W.inventory(folder,pin['sha256']);rt=read(folder/'runtime.json');done=read(folder/'outcome.json');inp=read(folder/'inputs.json')
   need(rt['commit']==source and rt['settings']==card['settings'][unit-1]and done['batch']==plan['batch']and done['unit']==unit and done['passed']and not done['connected']and done['status']=='uniform705_unjoined'and done['ticks']==331 and done['events']==W.EXPECTED and done['strict_inclusions']==705,'complete indexed unjoined root')
   need(inp['map_inputs']==model and inp['card']==entry(root/'input-card.json')and inp['parent']==binding,'same actual model/card/known parent')
   for name,row in identity.items():need(row==rt['sources'][name]==entry(folder/'source'/name),'same archived source '+name)
   with ctx.workprec(384):
    chart=read(folder/'certificate.json.gz');need(W.uniform(chart),'all full root gates');membership=W.join(anchor,chart)
   save(root,f'join-{unit:03d}.json',membership);result['joins'].append({'unit':unit,'passed':membership['passed'],'coordinates':sum(membership['coordinate_inclusions']),'worker_manifest':pin})
   if not membership['passed']:break
   result['connected_count']+=1;result['connected_right']=chart['parameter_endpoints'][1];result['physical_right']=chart['endpoint_root_boxes'][1][-3:]
   anchor={'box':chart['endpoint_root_boxes'][1],'normal':chart['normal'],'origin':chart['origin'],'parameter':chart['parameter_endpoints'][1]};endpoint={'chart':{k:chart[k]for k in ['shooting_dimension','normal','origin','parameter_center','parameter_endpoints','settings']},'endpoint':anchor['box']};del chart;gc.collect();check()
  if result['connected_count']:save(root,'connected-endpoint.json',endpoint)
  result['passed']=result['connected_count']==4
 except BaseException as exc:failure=type(exc).__name__+': '+str(exc);result['failure']=failure
 finally:
  signal.setitimer(signal.ITIMER_REAL,0)
  for j in jobs:L.stop(j['proc'].pid,signal.SIGKILL);j['proc'].wait();j['log'].close()
  live=[p for j in jobs for p,_ in L.owned(j['proc'].pid)];result.update(seconds=time.monotonic()-start,owned_live_pids=live,peak_aggregate_worker_rss_kib=peak)
  if live or failure or result['seconds']>=330:result['passed']=False
  save(root,'outcome.json',result);seal(root)
 print(json.dumps({'root':str(root),'passed':result['passed'],'connected_count':result['connected_count'],'manifest':entry(root/'manifest.json')}));return 0 if result['passed']else 2

def parser():
 p=argparse.ArgumentParser();p.add_argument('--card',type=Path,required=True);return p
if __name__=='__main__':raise SystemExit(main(parser().parse_args()))
