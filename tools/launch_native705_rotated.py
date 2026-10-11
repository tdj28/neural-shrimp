#!/usr/bin/env python3
"""At most ten batches of eight fresh705 proofs and gauge-aware native joins."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import gc,json,resource,signal,subprocess,sys,time
from pathlib import Path
from fractions import Fraction
from flint import arb,arb_mat,ctx
import probe_native705_rotated as W
import launch_native_unit_pair_pal_batch as L
from neural_shrimp.rigorous_pal_rechart import localize_endpoint,rotation_proposal,rechart_seed
from neural_shrimp.rigorous_unit_pair_pal import anchor_test
from neural_shrimp.rigorous_orbits import exact,vector_json
ROOT=W.ROOT;BASE=W.BASE;entry=W.entry;read=W.read;need=W.require
INPUTS=ROOT/'docs/plans/native705-rotated-inputs.json'
KEEP={'period','dimension','parameter_center','parameter_endpoints','normal','origin','center','slope','curvature','coordinate_radii','endpoint_root_boxes','settings'}
PROFILE='11819b8985345f34fb68e2ed8c9da5384f33804dd3343671ae2cd0c6fd38cbcc'
EXPECTED_EVENTS={'direction_return':144,'pal_chart_return':1,'pal_newton_proposal':2,'pal_path_assembly_return':3,'pal_path_phase_return':108,'pal_preconditioner_return':1}
PROFILE_OUTER='fa27b1e43c10cbe66040fa7bd80f58e4effdd44b72f31101ae04f92e5db7135f'

def save(root,name,value):
 raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();need(len(raw)<16*1024**2,'bounded parent metadata')
 with(root/name).open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())

def seal(root):
 save(root,'manifest.json',{'files':{str(p.relative_to(root)):entry(p)for p in sorted(root.rglob('*'))if p.is_file()}})
 for p in sorted(root.rglob('*'),key=lambda p:len(p.parts),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
 root.chmod(0o555)

def measured_forecast(rows):
 need(len(rows)==4 and all(r['ticks']==253 and r['events']==EXPECTED_EVENTS and r['status']in('uniform705_unjoined','completed_bounds_failed')for r in rows),'complete unchanged253-call measurements including failures')
 need(all(r['elapsed_seconds']>0 and r['peak_rss_kib']>0 and r['stored_bytes_before_final']>0 for r in rows),'positive measured costs')
 return {'seconds':3*max(r['elapsed_seconds']for r in rows)+20,'rss_kib':2*max(r['peak_rss_kib']for r in rows)+128*1024,'bytes':2*max(r['stored_bytes_before_final']for r in rows)+8*1024**2}


def initial_parent(card):
 need(card['status']=='fixed-reviewed-initial-inputs','complete initial independent prefix card')
 pr=Path(card['producer_batch']['root']);W.inventory(pr,card['producer_batch']['outer']);out=read(pr/'outcome.json')
 need(out['connected_count']==1 and out['connected_right']=='447/4096'and not out['passed']and not out['owned_live_pids'],'complete retained wider partial prefix')
 worker=pr/'chart001';W.inventory(worker/'run',out['joins'][0]['worker_manifest']['sha256']);W.C.archive_bound(worker,out['commit'])
 done=read(worker/'run/outcome.json');need(done['passed']and done['strict_inclusions']==705 and done['ticks']==253 and out['joins'][0]['coordinates']==705 and out['joins'][0]['passed'],'complete first wider native root/join')
 audit=card['independent'];ar=Path(audit['root']);W.inventory(ar,audit['outer']);W.inventory(ar/'run',audit['worker']);W.C.archive_bound(ar,audit['source'])
 ao=read(ar/'run/outcome.json');la=read(ar/'launcher-result.json');ab=read(ar/'run/input-binding.json')
 need(ao['passed']and ao['connected']and ao['connected_count']==1 and ao['connected_parameter_endpoints']==['319/4096','447/4096']and la['exit_code']==0 and not la['owned_live_pids'],'independent full wider-prefix root and connection')
 need(ab['card']['root']==str(pr)and ab['card']['outer']==card['producer_batch']['outer']and ab['card']['source']==out['commit']and ab['card']['qualified_prefix']==1 and ab['card']['prior']==card['prior_independent'],'exact independent producer binding')
 chart=read(worker/'run/certificate.json.gz');ep=read(ar/'run/connected-endpoint.json')
 need(ep['chart']['normal']==chart['normal']and ep['chart']['origin']==chart['origin']and ep['chart']['parameter_endpoints']==chart['parameter_endpoints']and len(ep['endpoint'])==705,'literal same independent wider endpoint')
 chart['endpoint_root_boxes'][1]=ep['endpoint'];need(W.uniform_admission(chart,W.decode(chart['whole_box'][-1])>0),'whole wider native root')
 model=read(worker/'run/inputs.json')['map_inputs'];weights=[str(2*Fraction(v))for v in chart['weights']]
 need(all(0<Fraction(v)<=1 for v in weights),'one fixed entire-packet weight proposal')
 measurements=[]
 for index in range(1,5):
  root=pr/f'chart{index:03d}/run';measurements.append(read(root/'outcome.json'))
 forecast=measured_forecast(measurements);need(forecast==card['forecast'],'literal independently recomputed measured forecast');need(forecast['seconds']<240 and forecast['rss_kib']<4*1024**2 and forecast['bytes']<W.CAP,'measured eight-worker admission')
 return model,chart,weights,{'root':str(worker),'manifest':entry(worker/'run/manifest.json'),'independent':audit},forecast

def endpoint_proposal(chart):
 need(chart['gates']['uniform_full_native_unit_pair_locus']and len(chart['coordinate_inclusions'])==705 and all(chart['coordinate_inclusions'])and chart['shooting_dimension']==705,'known full705 tube')
 c,s,a=[[W.decode(v)for v in chart[k]]for k in ('center','slope','curvature')]
 weights=[exact(v)for v in chart['weights']];h=arb((1,chart['settings']['halfwidth_exponent']));rho=arb((1,chart['settings']['radius_exponent']))
 matrix=lambda k:arb_mat([[W.decode(v)for v in row]for row in chart[k]])
 endpoint,proof=localize_endpoint(matrix('preconditioner'),matrix('whole_jacobian'),matrix('quadratic_center_coefficients'),matrix('quadratic_whole_coefficients'),c,s,a,weights,rho,h)
 # Intersect with the source-authenticated endpoint enclosure; both contain
 # exactly the same unique root, not just nearby floating approximations.
 known=[W.decode(v)for v in chart['endpoint_root_boxes'][1]]
 endpoint=[v.intersection(k)for v,k in zip(endpoint,known,strict=True)];need(all(v>0 for v in endpoint[-3:]),'known endpoint actual positive rates and angle');proof['retained_endpoint']=vector_json(endpoint)
 with ctx.workprec(256):proposal=rotation_proposal(endpoint,s,a,h)
 return endpoint,proof,proposal

def join(anchor,current,*,same_gauge=False):
 box=[W.decode(v)for v in anchor['box']]
 if same_gauge:
  need(anchor['normal']==current['normal']and anchor['origin']==current['origin'],'identical actual current gauge')
  need(Fraction(anchor['parameter'])==Fraction(current['parameter_endpoints'][0]),'exact adjacent current-gauge boundary')
  context={'box':box,'dimension':13,'period':18,'exact_parameter':anchor['parameter']}
 else:context={'box':box,'dimension':13,'period':18}
 need(current['settings']=={'precision':256,'newton_steps':2,'halfwidth_exponent':-7,'radius_exponent':-10},'unchanged safe narrow domain')
 return anchor_test(context,[W.decode(v)for v in current['center']],[W.decode(v)for v in current['slope']],[W.decode(v)for v in current['curvature']],[exact(v)for v in current['normal']],[exact(v)for v in current['origin']],exact(current['parameter_center']),arb((1,-7)),[W.decode(v)for v in current['coordinate_radii']],13,18)

def main():
 W.sources();need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2)
 source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();BASE.mkdir(parents=True,exist_ok=False);start=time.monotonic();livejobs=[];result={'passed':False,'connected_charts':0,'batches':[],'scope':'Bounded connected705 continuation only. No whole-window topology, complementary stability or finaljunction identity inferred.'};failure=None;active_root=None;active_record=None
 def check():
  need(time.monotonic()-start<3300,'3300s total packet');rss=int(next(s.split()[1]for s in Path('/proc/self/status').read_text().splitlines()if s.startswith('VmRSS:')));need(rss<4*1024**2,'4GiB current parent RSS')
  need(sum(p.stat().st_size for p in BASE.rglob('*')if p.is_file())<60*1024**3-16*1024**2,'60GiB complete raw cap/final reserve')
 def halt(*_):raise TimeoutError('3300s bounded rotated packet')
 signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGALRM,halt);signal.setitimer(signal.ITIMER_REAL,3300)
 save(BASE,'launcher-start.json',{'source':source,'pid':os.getpid(),'maximum_batches':10,'workers_per_batch':8,'maximum_charts':80,'worker_rss_kib':4*1024**2,'aggregate_worker_rss_kib':32*1024**2,'parent_rss_kib':4*1024**2,'worker_seconds':240,'owned_seconds':255,'kill_grace_seconds':5,'batch_seconds':330,'total_seconds':3300,'raw_bytes':60*1024**3,'initial_card':entry(INPUTS)})
 try:
  with ctx.workprec(384):model,previous,weights,parent_binding,forecast=initial_parent(read(INPUTS))
  for batch in range(1,11):
   check();mem=Path('/proc/meminfo').read_text();available=int(next(v.split()[1]for v in mem.splitlines()if v.startswith('MemAvailable:')));need(available>44*1024**2,'44GiB fresh headroom for eight4GiB workers')
   root=BASE/f'batch{batch:03d}';root.mkdir(exist_ok=False);bstart=time.monotonic();record={'batch':batch,'commit':source,'passed':False,'connected_count':0,'workers':[],'joins':[]};livejobs=[];active_root=root;active_record=record
   with ctx.workprec(384):endpoint,localization,proposal=endpoint_proposal(previous)
   save(root,'endpoint-localization.json',localization);del previous;gc.collect()
   with ctx.workprec(256):initials=[rechart_seed(proposal,W.settings(i)['parameter'])for i in range(1,9)]
   card={'batch':batch,'source':source,'worker_count':8,'kernel':'unchanged-full705-quadratic-v1','map_inputs':model,'settings':[W.settings(i)for i in range(1,9)],'weights':weights,'normal':proposal['normal'],'origin':proposal['origin'],'initials':initials,'anchor':vector_json(endpoint),'parent':parent_binding,'localization':entry(root/'endpoint-localization.json'),'proposal':proposal,'forecast':forecast}
   save(root,'input-card.json',card);cardsha=entry(root/'input-card.json')['sha256'];save(root,'resource-admission.json',{'available_kib':available,'load':os.getloadavg(),'forecast':forecast});check()
   for unit in range(1,9):
    folder=root/f'chart{unit:03d}';folder.mkdir();log=(folder/'process.log').open('xb');command=[sys.executable,str(ROOT/'tools/probe_native705_rotated.py'),'--batch',str(batch),'--unit',str(unit),'--out',str(folder/'run'),'--card-sha',cardsha,'--profile-manifest',PROFILE,'--profile-outer-manifest',PROFILE_OUTER]
    proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);livejobs.append({'unit':unit,'proc':proc,'log':log,'folder':folder,'start':time.monotonic(),'peak':0,'reason':None,'term':None,'command':command})
   save(root,'owned-processes.json',{'workers':[{'unit':j['unit'],'group':j['proc'].pid,'command':j['command']}for j in livejobs]});peak=0
   while any(j['proc'].poll()is None for j in livejobs):
    check();need(time.monotonic()-bstart<330,'330s complete batch');total=0
    for j in livejobs:
     rss=sum(v for _,v in L.owned(j['proc'].pid));total+=rss;j['peak']=max(j['peak'],rss)
     if j['proc'].poll()is not None:continue
     reason='4GiB workerRSS'if rss>=4*1024**2 else'255s owned time'if time.monotonic()-j['start']>=255 else'128KiB processlog'if(j['folder']/'process.log').stat().st_size>=128*1024 else None
     if reason and j['term']is None:j['reason']=reason;j['term']=time.monotonic();L.stop(j['proc'].pid,signal.SIGTERM)
     if j['term']is not None and time.monotonic()-j['term']>=5:L.stop(j['proc'].pid,signal.SIGKILL)
    peak=max(peak,total);need(total<32*1024**2,'32GiB aggregate workers');time.sleep(.25)
   for j in livejobs:
    j['proc'].wait();L.stop(j['proc'].pid,signal.SIGKILL);j['log'].close();live=L.owned(j['proc'].pid);need(not live,'all owned group members stopped')
    record['workers'].append({'unit':j['unit'],'exit_code':j['proc'].returncode,'group':j['proc'].pid,'peak_rss_kib':j['peak'],'stop_reason':j['reason'],'owned_live_pids':[v for v,_ in live]})
   anchor={'box':card['anchor']};lastpath=None
   for owner in record['workers']:
    unit=owner['unit'];folder=root/f'chart{unit:03d}/run'
    if owner['exit_code']!=0 or owner['stop_reason']is not None:break
    pin=entry(folder/'manifest.json');W.inventory(folder,pin['sha256']);rt=read(folder/'runtime.json');done=read(folder/'outcome.json');inputs=read(folder/'inputs.json')
    need(rt['commit']==source and rt['settings']==W.settings(unit)and done['batch']==batch and done['unit']==unit and done['passed']and done['status']=='uniform705_unjoined'and not done['connected']and done['ticks']==253 and done['strict_inclusions']==705 and done['events']==EXPECTED_EVENTS,'complete exact indexed705 root')
    need(inputs['card']==entry(root/'input-card.json')and inputs['map_inputs']==model and inputs['batch']==batch,'same literal model and finite proposal card')
    for name,row in rt['sources'].items():need(entry(ROOT/name)==entry(folder/'source'/name)==row,'same current reviewed source '+name)
    with ctx.workprec(384):
     proof=read(folder/'certificate.json.gz');need(W.uniform_admission(proof,W.decode(proof['whole_box'][-1])>0),'full primitive positive705 root');current={k:v for k,v in proof.items()if k in KEEP};del proof;gc.collect();membership=join(anchor,current,same_gauge=unit>1)
    save(root,f'join-{unit:03d}.json',membership);record['joins'].append({'unit':unit,'passed':membership['passed'],'coordinates':sum(membership.get('coordinate_inclusions',[])),'worker_manifest':pin})
    if not membership['passed']:break
    record['connected_count']+=1;result['connected_charts']+=1;lastpath=folder;record['last_physical']=current['endpoint_root_boxes'][1][-3:];record['right']=current['parameter_endpoints'][1]
    anchor={'box':current['endpoint_root_boxes'][1],'normal':current['normal'],'origin':current['origin'],'parameter':current['parameter_endpoints'][1]};check()
   record.update(passed=record['connected_count']==8,seconds=time.monotonic()-bstart,peak_aggregate_rss_kib=peak,owned_live_pids=[]);need(record['seconds']<330,'complete batch330s cap');save(root,'outcome.json',record);seal(root);result['batches'].append({'batch':batch,'root':str(root),'manifest':entry(root/'manifest.json'),**record});active_root=None;active_record=None
   if not record['passed']:break
   parent_binding={'root':str(lastpath.parent),'manifest':entry(lastpath/'manifest.json'),'batch_root':str(root),'batch_manifest':entry(root/'manifest.json'),'source':source};previous=read(lastpath/'certificate.json.gz');check()
  result['passed']=len(result['batches'])==10 and all(r['passed']for r in result['batches'])
 except BaseException as exc:failure=type(exc).__name__+': '+str(exc);result['failure']=failure
 finally:
  signal.setitimer(signal.ITIMER_REAL,0)
  for j in livejobs:
   L.stop(j['proc'].pid,signal.SIGKILL);j['proc'].wait();j['log'].close()
  live=[p for j in livejobs for p,_ in L.owned(j['proc'].pid)];result.update(seconds=time.monotonic()-start,owned_live_pids=live,source=source)
  if failure or live or result['seconds']>=3300:result['passed']=False
  if active_root is not None and not(active_root/'manifest.json').exists():
   active_record.update(passed=False,failure=failure,owned_live_pids=live,seconds=time.monotonic()-bstart)
   if not(active_root/'outcome.json').exists():save(active_root,'outcome.json',active_record)
   seal(active_root);result['batches'].append({'batch':active_record['batch'],'root':str(active_root),'manifest':entry(active_root/'manifest.json'),**active_record})
  save(BASE,'outcome.json',result);seal(BASE)
 print(json.dumps({'root':str(BASE),'passed':result['passed'],'connected_charts':result['connected_charts'],'manifest':entry(BASE/'manifest.json')}));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
