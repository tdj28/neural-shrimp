#!/usr/bin/env python3
"""Fixed-budget endpoint-rotated continuation; every attempt is immutable."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,gc,hashlib,json,resource,signal,subprocess,sys,time
from pathlib import Path
from flint import arb,ctx
import probe_native705_adaptive as W
import launch_native705_cubic_batch as B
from neural_shrimp.cubic_rechart import rotate_cubic_endpoint,adaptive_width
from neural_shrimp.rigorous_pal_rechart import rechart_seed
ROOT=W.ROOT;BASE=W.BASE;need=W.require;entry=W.entry;read=W.read;save=B.save;seal=B.seal
PREFIX={'root':str(W.P.BASE/'narrow001/batch001'),'source':'6ae48eb09c199b7b2ce1a816b0355d35ce2265ec','outer':'4fe4f57aa561dec7230cc8586ca0f95ec4d4c1ebc015acad4d080d391740a289','endpoint':'e2474c9b851c01c431eb5d1fb29603101f45d783c6cf64ba8898acbaaa7ae4c3','workers':['e59c3f7a2e6d6f39ef418fd96e6bd9372b1dacc49a29e9f3b2fcc64f3120cbac','0ab5725415d448c63f962dc5321a347ca1c1568ca6691e3ed4ffbd67cdfe1696','7377b86d23234fde635bb9fb0ef6a2c6263d59757dd0b222c2ded32874802420','03233d51128f11bb02577325711894451ec13aa90fbe207d94be537521e035c1']}
LIMITS={'attempts':32,'ticks':32*331,'native_calls':32*324,'seconds':8400,'worker_seconds':240,'owned_seconds':255,'grace_seconds':5,'worker_rss_kib':6*1024**2,'parent_rss_kib':4*1024**2,'raw_bytes':8*1024**3,'available_kib':16*1024**2,'minimum_halfwidth':-9,'initial_halfwidth':-5,'radius_exponent':-10,'overlap':'1/2048'}
KEEP={'dimension','period','shooting_dimension','normal','origin','center','slope','curvature','cubic','weights','settings','parameter_center','parameter_endpoints','endpoint_root_boxes'}

def committed_policy(path):
 path=path.resolve();need(path==ROOT/'docs/plans/native705-adaptive-policy.json','one literal committed bounded policy')
 rel=str(path.relative_to(ROOT));need(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT)).hexdigest()==entry(path)['sha256'],'exact pretarget policy commit');p=read(path)
 need(p['schema']=='native705-adaptive-v1'and p['status']=='measured-reviewed'and p['limits']==LIMITS and p['initial_parent']==PREFIX,'fixed policy/source parent and total caps')
 return p

def initial_parent(identity):
 root=Path(PREFIX['root']);W.inventory(root,PREFIX['outer']);out=read(root/'outcome.json');card=read(root/'input-card.json');ep=read(root/'connected-endpoint.json')
 need(out['source']==card['source']==PREFIX['source']and out['passed']is False and out['connected_count']==3 and out['connected_right']=='115/512'and not out['owned_live_pids'],'actual three-chart prefix with fourth failure retained')
 need(len(out['joins'])==3 and all(v['passed']and v['coordinates']==705 and v['unit']==i and v['worker_manifest']['sha256']==PREFIX['workers'][i-1]for i,v in enumerate(out['joins'],1)),'complete ordered original705 membership')
 need(entry(root/'connected-endpoint.json')['sha256']==PREFIX['endpoint'],'literal full prefix endpoint')
 model=card['map_inputs'];last=None
 for i,pin in enumerate(PREFIX['workers'],1):
  folder=root/f'chart{i:03d}';W.inventory(folder/'run',pin);rt=read(folder/'run/runtime.json');done=read(folder/'run/outcome.json');inputs=read(folder/'run/inputs.json')
  need(rt['commit']==PREFIX['source']and inputs['map_inputs']==model and inputs['card']==entry(root/'input-card.json')and rt['settings']==card['settings'][i-1],'same indexed original model/card/settings')
  need(done['ticks']==331 and done['events']==W.EXPECTED and done['unit']==i and done['batch']==1,'complete original attempt including failed fourth')
  need(done['passed']==(i<=3)and done['status']==('uniform705_unjoined'if i<=3 else'completed_bounds_failed'),'preserved original evidence labels')
  for name,p in rt['sources'].items():need(entry(folder/'run/source'/name)==p and hashlib.sha256(subprocess.check_output(['git','show',PREFIX['source']+':'+name],cwd=ROOT)).hexdigest()==p['sha256'],'literal old source '+name)
  W.C.same_native_source(identity,rt['sources'])
  if i==3:
   last=read(folder/'run/certificate.json.gz');need(W.uniform(last),'complete original uniform root')
 need(ep['endpoint']==last['endpoint_root_boxes'][1]and ep['chart']=={k:last[k]for k in ['shooting_dimension','normal','origin','parameter_center','parameter_endpoints','settings']},'literal accepted full705 endpoint')
 return model,{k:last[k]for k in KEEP},{'box':ep['endpoint']},dict(kind='sealed-three-chart-prefix',**PREFIX)

def make_card(source,attempt,exponent,model,proposal,anchor,parent,profile,policy):
 with ctx.workprec(256):
  point=[W.decode(v)for v in anchor['box']];s,a,b=[[W.decode(v).mid()for v in proposal[k]]for k in ['slope','curvature','cubic']]
  rotation=rotate_cubic_endpoint(point,s,a,b,arb((1,proposal['settings']['halfwidth_exponent'])));cfg=W.settings(exponent);initial=rechart_seed(rotation,cfg['parameter'])
 return dict(source=source,attempt=attempt,kernel='fresh-cubic705-adaptive-v1',settings=cfg,map_inputs=model,normal=rotation['normal'],origin=rotation['origin'],weights=proposal['weights'],initial=initial,anchor=anchor,parent=parent,profile=profile,policy=policy,rotation=rotation)

def complete_result(done,chart,attempt,cfg):
 need(done['attempt']==attempt and done['ticks']==331 and done['events']==W.EXPECTED and not done['connected'],'complete native331 root-only worker')
 need(chart['shooting_dimension']==705 and chart['dimension']==13 and chart['period']==18,'literal complete native dimension')
 need(done['status']in ['uniform705_unjoined','completed_bounds_failed']and chart['settings']['halfwidth_exponent']==cfg['halfwidth_exponent']and chart['settings']['radius_exponent']==-10 and chart['parameter_center']==cfg['parameter'],'same acquired full domain')
 need(done['passed']==W.uniform(chart)and done['gates']==chart['gates']and done['strict_inclusions']==sum(chart['bound']['coordinate_inclusions'])and len(chart['bound']['coordinate_inclusions'])==705,'actual complete theorem gates')
 g=chart['gates'];return bool(g['primitive_period']and g['realified_frame_independent']and g['whole_positive_angle']and not g['anchor_identified'])

def run_worker(folder,index,cardsha,check):
 cmd=[sys.executable,str(ROOT/'tools/probe_native705_adaptive.py'),'--attempt',str(index),'--out',str(folder/'run'),'--card-sha',cardsha]
 log=(folder/'process.log').open('xb');proc=None;start=time.monotonic();peak=0;failure=None
 try:
  proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);save(folder,'owned-process.json',dict(group=proc.pid,command=cmd,source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
  while proc.poll()is None:
   check();rss=sum(v for _,v in B.L.owned(proc.pid));peak=max(peak,rss)
   need(rss<6*1024**2,'6GiB owned worker RSS');need(time.monotonic()-start<255,'255s owned worker cap');need((folder/'process.log').stat().st_size<128*1024,'128KiB bounded worker log');time.sleep(.25)
 except BaseException as exc:failure=type(exc).__name__+': '+str(exc)
 finally:
  if proc is not None:
   if proc.poll()is None:
    B.L.stop(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:pass
   B.L.stop(proc.pid,signal.SIGKILL);proc.wait()
  log.close()
 live=[]if proc is None else B.L.owned(proc.pid);result=dict(exit_code=None if proc is None else proc.returncode,peak_rss_kib=peak,seconds=time.monotonic()-start,stop_reason=failure,owned_live_pids=[p for p,_ in live]);save(folder,'launcher-result.json',result)
 need(not live and failure is None,'owned worker stopped without resource/infrastructure failure');return result

def main(args):
 identity=W.sources();policy=committed_policy(args.card);source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
 mem=Path('/proc/meminfo').read_text();available=int(next(v.split()[1]for v in mem.splitlines()if v.startswith('MemAvailable:')));need(available>LIMITS['available_kib'],'16GiB current headroom for one6GiB worker and4GiB parent')
 BASE.mkdir(parents=True,exist_ok=False);start=time.monotonic();current=[None];sealed_bytes=[0];peak=[0];failure=None
 result=dict(passed=False,source=source,connected_count=0,launched_attempts=0,attempts=[],owned_live_pids=[],status='incomplete',scope='Producer interval connected primitive18 complex-unit curve prefix only. No complete-window, junction-connection, complementary-spectrum or independent-audit claim.')
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,12*1024**3))
 def check():
  need(time.monotonic()-start<LIMITS['seconds'],'8400s total packet');rss=int(next(v.split()[1]for v in Path('/proc/self/status').read_text().splitlines()if v.startswith('VmRSS:')));peak[0]=max(peak[0],rss);need(rss<4*1024**2,'4GiB parent RSS')
  active=0 if current[0]is None else sum(p.stat().st_size for p in current[0].rglob('*')if p.is_file());need(sealed_bytes[0]+active<8*1024**3-16*1024**2,'8GiB total raw with16MiB final reserve')
 def halt(*_):raise TimeoutError('8400s adaptive packet cap')
 signal.signal(signal.SIGALRM,halt);signal.signal(signal.SIGTERM,halt);signal.setitimer(signal.ITIMER_REAL,LIMITS['seconds'])
 save(BASE,'policy.json',policy);save(BASE,'launcher-start.json',dict(source=source,pid=os.getpid(),plan=entry(args.card),limits=LIMITS,meminfo=mem,load=os.getloadavg(),started_unix=time.time(),threads=dict.fromkeys(W.THREADS,'1')))
 try:
  prof=W.profile(identity,policy['profile']);need(prof['forecast']==policy['forecast'],'same actual measured full331 forecast')
  need(32*prof['forecast']['one_fresh_chart_seconds']+240<8400 and 32*prof['forecast']['compressed_storage_bytes']+16*1024**2<8*1024**3,'complete32-attempt measured time/storage forecast');save(BASE,'profile-binding.json',prof)
  with ctx.workprec(384):model,proposal,anchor,parent=initial_parent(identity)
  save(BASE,'initial-parent.json',parent);exponent=-5;endpoint=None;sealed_bytes[0]=sum(p.stat().st_size for p in BASE.rglob('*')if p.is_file())
  for index in range(1,33):
   check()
   if LIMITS['seconds']-(time.monotonic()-start)<270:result['status']='insufficient_remaining_bounded_time';break
   folder=BASE/f'attempt{index:03d}';folder.mkdir(exist_ok=False);current[0]=folder
   try:
    card=make_card(source,index,exponent,model,proposal,anchor,parent,policy['profile'],entry(BASE/'policy.json'));W.validate_card(card,source,index);save(folder,'input-card.json',card);check()
    available=int(next(v.split()[1]for v in Path('/proc/meminfo').read_text().splitlines()if v.startswith('MemAvailable:')));need(available>10*1024**2,'fresh10GiB worker/OS admission')
    result['launched_attempts']+=1;owner=run_worker(folder,index,entry(folder/'input-card.json')['sha256'],check)
    need(owner['exit_code']in (0,2),'expected mathematical worker exit');pin=entry(folder/'run/manifest.json');W.inventory(folder/'run',pin['sha256']);rt=read(folder/'run/runtime.json');done=read(folder/'run/outcome.json');inp=read(folder/'run/inputs.json')
    need(rt['commit']==source and rt['settings']==card['settings']and inp['map_inputs']==model and inp['card']==entry(folder/'input-card.json')and inp['parent']==parent and inp['anchor']==anchor and inp['policy']==entry(BASE/'policy.json'),'literal indexed source/map/known-parent/card')
    need(rt['sources']==identity,'complete current source closure')
    for name,row in identity.items():need(entry(folder/'run/source'/name)==row,'same archived source '+name)
    with ctx.workprec(384):
     chart=read(folder/'run/certificate.json.gz');other=complete_result(done,chart,index,card['settings']);need(owner['exit_code']==(0 if done['passed']else 2),'actual worker mathematical exit');membership=W.projected_join(anchor,chart)if done['passed']else None
    if membership is not None:save(folder,'join.json',membership)
    joined=membership is not None and membership['passed']and len(membership['coordinate_inclusions'])==705 and all(membership['coordinate_inclusions'])
    next_width,action=adaptive_width(exponent,root_passed=bool(done['passed']),joined=bool(joined),other_gates=other)
    row=dict(attempt=index,settings=card['settings'],root_passed=done['passed'],joined=bool(joined),action=action,next_halfwidth=next_width,worker=pin,contraction=done['contraction'],strict_inclusions=done['strict_inclusions'],owner=owner)
    if action=='advance_rotate':
     result['connected_count']+=1;anchor={'box':chart['endpoint_root_boxes'][1]};proposal={k:chart[k]for k in KEEP};endpoint={'chart':{k:chart[k]for k in ['shooting_dimension','normal','origin','parameter_center','parameter_endpoints','settings']},'endpoint':anchor['box']};save(folder,'connected-endpoint.json',endpoint);row['physical_endpoint']=anchor['box'][-3:]
    save(folder,'decision.json',row);del chart;gc.collect();check();seal(folder)
    row['outer_manifest']=entry(folder/'manifest.json');result['attempts'].append(row)
    if action=='advance_rotate':parent=dict(kind='accepted-adaptive-endpoint',root=str(folder),source=source,outer=row['outer_manifest']['sha256'],worker=pin['sha256'],endpoint=entry(folder/'connected-endpoint.json')['sha256'])
    sealed_bytes[0]+=sum(p.stat().st_size for p in folder.rglob('*')if p.is_file());current[0]=None;exponent=next_width
    if action=='stop':result['status']='mathematical_stop';break
   except BaseException as exc:
    if folder.stat().st_mode&0o200:save(folder,'packet-error.json',dict(error=type(exc).__name__+': '+str(exc)));seal(folder)
    current[0]=None;raise
  else:result['status']='attempt_cap_reached';result['passed']=True
  if endpoint is not None:save(BASE,'connected-endpoint.json',endpoint)
 except BaseException as exc:failure=type(exc).__name__+': '+str(exc);result.update(status='failed_or_interrupted',failure=failure)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(seconds=time.monotonic()-start,parent_peak_rss_kib=peak[0],attempted=len(result['attempts']),max_inclusive_ticks=331*result['launched_attempts'],native_call_ceiling=324*result['launched_attempts'])
  if failure:result['passed']=False
  save(BASE,'outcome.json',result);seal(BASE)
 print(json.dumps(dict(root=str(BASE),status=result['status'],connected_count=result['connected_count'],manifest=entry(BASE/'manifest.json'))));return 0 if result['passed']else 2

def parser():
 p=argparse.ArgumentParser();p.add_argument('--card',type=Path,required=True);return p
if __name__=='__main__':raise SystemExit(main(parser().parse_args()))
