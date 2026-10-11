#!/usr/bin/env python3
"""Bounded maintained-gauge cubic continuation with measured width adaptation."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,gc,hashlib,json,resource,signal,subprocess,sys,time
from pathlib import Path
from fractions import Fraction as F
from flint import arb,ctx
import probe_native705_maintained as W
import launch_native705_adaptive as OLD
from neural_shrimp.cubic_continuation_policy import next_step
from neural_shrimp.cubic_rechart import rotate_cubic_endpoint
from neural_shrimp.rigorous_pal_rechart import rechart_seed
from neural_shrimp.rigorous_orbits import exact
from neural_shrimp.rigorous_eigenpair_locus import exact_mid
ROOT=W.ROOT;BASE=W.BASE;need=W.need;entry=W.entry;read=W.read;save=OLD.save;seal=OLD.seal;KEEP=OLD.KEEP
PARENT=dict(root=str(W.E.TARGET.parent),source='3c7309781fd29ae029f869f50c14f424f7fb7097',outer='65b642c67403d40b5867722b54ae2e36595b7ab462653d0aef3301b49b41ef20',worker='0aa0d0b28c62c9c3eea67e7e2e877e72cba435f6f1ab0ac9a06a75a913c19aa2',endpoint='d3cc4b496596c2064035c5a4254ae89baf934220679016eb633eb13a8741eca7')
LIMITS=dict(OLD.LIMITS,initial_halfwidth=-6,maximum_halfwidth=-3,strong_threshold='1/5',strong_passes=2,rotate_threshold='1/4')

def policy(path):
 need(path.resolve()==ROOT/'docs/plans/native705-maintained-policy.json','literal committed continuation policy');p=read(path);need(p['status']=='measured-reviewed'and p['limits']==LIMITS and p['initial_parent']==PARENT and p['schema']=='native705-maintained-v1','fixed measured bounded policy')
 need(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+str(path.resolve().relative_to(ROOT))],cwd=ROOT)).hexdigest()==entry(path)['sha256'],'same committed policy');return p

def initial_parent(identity,check):
 domain,old,chart,pc=W.E.C.bind(identity,lambda n,v:save(BASE,'parent-'+n,v),check);root=Path(PARENT['root']);W.inventory(root,PARENT['outer']);W.inventory(root/'run',PARENT['worker']);rt=read(root/'run/runtime.json');o=read(root/'run/outcome.json');owned=read(root/'launcher-result.json');j=read(root/'run/localized-endpoint-membership.json');ep=read(root/'run/connected-endpoint.json')
 need(rt['commit']==owned['commit']==PARENT['source']and owned['exit_code']==0 and not owned['owned_live_pids'],'exact completed endpoint residual owner')
 need(o['passed']and o['status']=='same_known_endpoint_joined'and o['localized_join_inclusions']==705 and o['native_calls']==72 and o['events']=={'direction_return':72},'complete fresh endpoint result')
 need(j['passed']and j['parameter_inside']and len(j['coordinate_inclusions'])==705 and all(j['coordinate_inclusions']),'strict actual full705 projected old-to-new identity')
 for n,row in rt['sources'].items():need(entry(root/'run/source'/n)==row and hashlib.sha256(subprocess.check_output(['git','show',PARENT['source']+':'+n],cwd=ROOT)).hexdigest()==row['sha256'],'exact endpoint source '+n)
 need(entry(root/'run/connected-endpoint.json')['sha256']==PARENT['endpoint']and ep['endpoint']==chart['endpoint_root_boxes'][1]and ep['chart']=={k:chart[k]for k in ['shooting_dimension','normal','origin','parameter_center','parameter_endpoints','settings']},'literal retained now-connected right endpoint')
 point=read(root/'run/point-input.json');need(point['model']==pc['map_inputs']and point['normal']==pc['normal']and point['origin']==pc['origin']and point['parameter']=='115/512','same native endpoint genealogy')
 for n in W.W.C.NATIVE_SOURCES:need(identity[n]==rt['sources'][n],'current actual native matches endpoint residual '+n)
 return pc['map_inputs'],{k:chart[k]for k in KEEP},anchor_from(chart),dict(kind='retained-root-connected-by-endpoint',**PARENT)

def anchor_from(chart):
 return dict(box=chart['endpoint_root_boxes'][1],normal=chart['normal'],origin=chart['origin'],exact_parameter=chart['parameter_endpoints'][1])

def make_card(source,index,exponent,mode,model,proposal,anchor,parent,profile,pin):
 with ctx.workprec(256):
  if mode=='same_gauge':
   need(anchor['normal']==proposal['normal']and anchor['origin']==proposal['origin']and anchor['exact_parameter']==proposal['parameter_endpoints'][1],'literal accepted same-gauge endpoint')
   cfg=W.settings(exponent,str(F(anchor['exact_parameter'])+F(2)**exponent-F(1,2048)));delta=exact(cfg['parameter'])-exact(proposal['parameter_center']);c,s,a,b=[[W.decode(v).mid()for v in proposal[k]]for k in ['center','slope','curvature','cubic']]
   initial=[exact_mid((c[i]+s[i]*delta+a[i]*delta**2/2+b[i]*delta**3).mid())for i in range(705)];normal=proposal['normal'];origin=proposal['origin'];rotation=None;known=anchor
  else:
   need(mode=='rotated_gauge','fixed continuation modes');point=[W.decode(v)for v in anchor['box']];s,a,b=[[W.decode(v).mid()for v in proposal[k]]for k in ['slope','curvature','cubic']]
   rotation=rotate_cubic_endpoint(point,s,a,b,arb((1,proposal['settings']['halfwidth_exponent'])));cfg=W.settings(exponent,str(F(2)**exponent-F(1,2048)));initial=rechart_seed(rotation,cfg['parameter']);normal=rotation['normal'];origin=rotation['origin'];known={'box':anchor['box']}
 return dict(source=source,attempt=index,kernel='fresh-cubic705-maintained-v1',mode=mode,settings=cfg,map_inputs=model,normal=normal,origin=origin,weights=proposal['weights'],initial=initial,anchor=known,parent=parent,profile=profile,policy=pin,rotation=rotation)

def contraction_upper(done,chart):
 need(done['contraction']==chart['bound']['contraction'] and done['contraction']['rad']==[0,0],'literal exact upper contraction bound')
 m,e=done['contraction']['mid'];q=F(m)*F(2)**e;need(0<=q<1,'strict passed contraction');return q

def run_worker(folder,index,cardsha,check):
 cmd=[sys.executable,str(ROOT/'tools/probe_native705_maintained.py'),'--attempt',str(index),'--out',str(folder/'run'),'--card-sha',cardsha];log=(folder/'process.log').open('xb');proc=None;start=time.monotonic();peak=0;failure=None
 try:
  proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);save(folder,'owned-process.json',dict(group=proc.pid,command=cmd))
  while proc.poll()is None:
   check();rss=sum(v for _,v in OLD.B.L.owned(proc.pid));peak=max(peak,rss);need(rss<6*1024**2 and time.monotonic()-start<255,'owned6GiB/255s worker');need((folder/'process.log').stat().st_size<128*1024,'bounded log');time.sleep(.25)
 except BaseException as exc:failure=type(exc).__name__+': '+str(exc)
 finally:
  if proc is not None:
   if proc.poll()is None:
    OLD.B.L.stop(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:pass
   OLD.B.L.stop(proc.pid,signal.SIGKILL);proc.wait()
  log.close()
 live=[]if proc is None else OLD.B.L.owned(proc.pid);o=dict(exit_code=None if proc is None else proc.returncode,peak_rss_kib=peak,seconds=time.monotonic()-start,stop_reason=failure,owned_live_pids=[p for p,_ in live]);save(folder,'launcher-result.json',o);need(not live and failure is None,'owned worker completed under caps');return o

def main(args):
 need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source');source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();identity={n:entry(ROOT/n)for n in W.SOURCES};p=policy(args.card);need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');mem=Path('/proc/meminfo').read_text();available=int(next(v.split()[1]for v in mem.splitlines()if v.startswith('MemAvailable:')));need(available>16*1024**2,'16GiB fresh shared headroom');BASE.mkdir(parents=True,exist_ok=False)
 start=time.monotonic();active=[None];used=[0];peak=[0];resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,12*1024**3));result=dict(passed=False,status='incomplete',source=source,connected_count=0,launched_attempts=0,attempts=[],owned_live_pids=[],scope='Connected primitive18 unit-pair locus only; complementary spectra, distant junction and full window remain separate.')
 def check():
  rss=int(next(v.split()[1]for v in Path('/proc/self/status').read_text().splitlines()if v.startswith('VmRSS:')));peak[0]=max(peak[0],rss);need(time.monotonic()-start<8400 and rss<4*1024**2,'8400s/4GiB parent');current=0 if active[0]is None else sum(v.stat().st_size for v in active[0].rglob('*')if v.is_file());need(used[0]+current<8*1024**3-16*1024**2,'8GiB with closing reserve')
 def halt(*_):raise TimeoutError('fixed maintained continuation deadline')
 signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGALRM,halt);signal.setitimer(signal.ITIMER_REAL,8400);save(BASE,'policy.json',p);save(BASE,'launcher-start.json',dict(source=source,pid=os.getpid(),policy=entry(args.card),limits=LIMITS,started_unix=time.time(),meminfo=mem,load=os.getloadavg()))
 try:
  prof=W.W.profile(identity,p['profile']);need(prof['forecast']==p['forecast']and 32*prof['forecast']['one_fresh_chart_seconds']+240<8400 and 32*prof['forecast']['compressed_storage_bytes']+16*1024**2<8*1024**3,'same complete measured32-attempt admission');save(BASE,'profile-binding.json',prof)
  with ctx.workprec(480):model,proposal,anchor,parent=initial_parent(identity,check)
  exponent=-6;strong=0;mode='same_gauge';endpoint=None;used[0]=sum(v.stat().st_size for v in BASE.rglob('*')if v.is_file())
  for index in range(1,33):
   check()
   if 8400-(time.monotonic()-start)<270:result['status']='insufficient_remaining_time';break
   folder=BASE/f'attempt{index:03d}';folder.mkdir(exist_ok=False);active[0]=folder
   try:
    card=make_card(source,index,exponent,mode,model,proposal,anchor,parent,p['profile'],entry(BASE/'policy.json'));save(folder,'input-card.json',card);W.validate_card(card,source,index);available=int(next(v.split()[1]for v in Path('/proc/meminfo').read_text().splitlines()if v.startswith('MemAvailable:')));need(available>10*1024**2,'fresh10GiB worker headroom');result['launched_attempts']+=1;owner=run_worker(folder,index,entry(folder/'input-card.json')['sha256'],check);need(owner['exit_code']in(0,2),'expected complete mathematical result')
    pin=entry(folder/'run/manifest.json');W.inventory(folder/'run',pin['sha256']);rt=read(folder/'run/runtime.json');done=read(folder/'run/outcome.json');inp=read(folder/'run/inputs.json');need(rt['commit']==source and rt['settings']==card['settings']and rt['sources']==identity and inp['card']==entry(folder/'input-card.json')and inp['parent']==parent and inp['anchor']==card['anchor']and inp['map_inputs']==model and inp['policy']==entry(BASE/'policy.json'),'literal full attempt/source/card/known-root bindings')
    for n,row in identity.items():need(entry(folder/'run/source'/n)==row,'same source archive '+n)
    with ctx.workprec(480):
     chart=read(folder/'run/certificate.json.gz');other=OLD.complete_result(done,chart,index,card['settings']);need(owner['exit_code']==(0 if done['passed']else 2),'unchanged root exit');join=W.join(card['anchor'],chart)if done['passed']else None
    if join is not None:save(folder,'join.json',join)
    joined=join is not None and join['passed']and len(join['coordinate_inclusions'])==705 and all(join['coordinate_inclusions']);q=contraction_upper(done,chart)if done['passed']else None
    transition=next_step(exponent,strong,root_passed=done['passed'],joined=bool(joined),other_gates=other,contraction=q);row=dict(attempt=index,settings=card['settings'],mode=mode,root_passed=done['passed'],joined=bool(joined),transition=transition,worker=pin,contraction=done['contraction'],strict_inclusions=done['strict_inclusions'],owner=owner)
    if transition['action']=='advance':
     result['connected_count']+=1;anchor=anchor_from(chart);proposal={k:chart[k]for k in KEEP};endpoint=dict(chart={k:chart[k]for k in ['shooting_dimension','normal','origin','parameter_center','parameter_endpoints','settings']},endpoint=anchor['box']);save(folder,'connected-endpoint.json',endpoint);row['physical_endpoint']=anchor['box'][-3:]
    save(folder,'decision.json',row);del chart;gc.collect();check();seal(folder);row['outer_manifest']=entry(folder/'manifest.json');result['attempts'].append(row)
    if transition['action']=='advance':parent=dict(kind='accepted-maintained-endpoint',root=str(folder),source=source,outer=row['outer_manifest']['sha256'],worker=pin['sha256'],endpoint=entry(folder/'connected-endpoint.json')['sha256']);mode='rotated_gauge'if transition['rotate']else'same_gauge'
    exponent=transition['halfwidth'];strong=transition['strong'];used[0]+=sum(v.stat().st_size for v in folder.rglob('*')if v.is_file());active[0]=None
    if transition['action']=='stop':result['status']='mathematical_stop';break
   except BaseException as exc:
    if folder.stat().st_mode&0o200:save(folder,'packet-error.json',dict(error=type(exc).__name__+': '+str(exc)));seal(folder)
    active[0]=None;raise
  else:result['status']='attempt_cap_reached';result['passed']=True
  if endpoint is not None:save(BASE,'connected-endpoint.json',endpoint)
 except BaseException as exc:result.update(status='failed_or_interrupted',failure=type(exc).__name__+': '+str(exc))
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(seconds=time.monotonic()-start,parent_peak_rss_kib=peak[0],attempted=len(result['attempts']),max_inclusive_ticks=331*result['launched_attempts'],native_call_ceiling=324*result['launched_attempts']);save(BASE,'outcome.json',result);seal(BASE)
 print(json.dumps(dict(root=str(BASE),status=result['status'],connected_count=result['connected_count'],manifest=entry(BASE/'manifest.json'))));return 0 if result['passed']else 2

def parser():
 p=argparse.ArgumentParser();p.add_argument('--card',type=Path,required=True);return p
if __name__=='__main__':raise SystemExit(main(parser().parse_args()))
