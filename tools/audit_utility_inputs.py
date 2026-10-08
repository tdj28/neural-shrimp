"""Read-only consumed-development inventory. No model or dynamics imports/evaluations."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,math,resource,signal,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/data2/neural-shrimp-data')
WARM=DATA/'explore/warmup-intervention/attempt-002'
EXP=DATA/'raw/EXP-028/attempt-002'
SLICE_NAMES=('s2-28011-out-0.0856','s2-28011-out-0.30')
SOURCES=('tools/audit_utility_inputs.py','tests/test_utility_input_audit.py','docs/plans/utility-input-qualification.md')
CAP=2*1024**2

def require(x,message):
    if not x:raise ValueError(message)
def digest(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_bytes())
def entry(p):
    b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':digest(b)}
def array_identity(x):
    a=np.ascontiguousarray(x,dtype='<f8');require(np.all(np.isfinite(a)),'finite state')
    return {'dtype':a.dtype.str,'shape':list(a.shape),'order':'C','sha256':digest(a.tobytes()),'binary64_hex':[format(int(v),'016x') for v in a.ravel().view('<u8')]}
def finite(x):return isinstance(x,(int,float,np.number)) and bool(np.isfinite(x))
def ratio(a,b):
    if not finite(a) or not finite(b) or min(a,b)<=0:return None
    return max(float(a)/float(b),float(b)/float(a))
def within_support(a,b):
    q=ratio(a,b);return q is not None and q<=2

def local_fork(trace,refs,cell,point):
    ks=np.flatnonzero(trace['alpha'][:,cell]>=0);require(len(ks)>0,'crossing missing');k=int(ks[0])
    require(k>=999 and k+1<len(trace['state']),'1000-pair history and next saved loss required')
    state=trace['state'][k,cell];history=trace['loss'][k-999:k+1,cell]
    require(history.shape==(1000,2) and np.all(np.isfinite(history)),'complete finite loss history')
    ref=refs[0];cross_rates=trace['rates'][k,cell]
    cross_ref=next(r for r in refs if r['alpha']==float(trace['alpha'][k,cell]))
    require(np.array_equal(cross_rates,[10**cross_ref['h'],10**cross_ref['o']]),'crossing rate binding')
    recovery=np.array([10**ref['h'],10**ref['o']]);loss=float(trace['loss'][k+1,cell,0]);require(finite(loss),'fork loss finite')
    tail=trace['state'][k-999:k+1,cell];previous=trace['state'][k-1000:k,cell] if k>=1000 else np.concatenate([trace['initial_state'][cell][None],trace['state'][:k,cell]])
    return {'point':point,'origin':('near_orbit','ordinary_initialization')[cell],'array_index':[k,cell],
      'accepted_updates_at_fork':2*(k+1),'state':array_identity(state),'state_values':state.tolist(),
      'prefix_pair_indices_inclusive':[k-999,k],'prefix_loss':array_identity(history),
      'prefix_mean_training_loss':float(history.mean()),'fork_training_loss':loss,
      'recorded_alpha':float(trace['alpha'][k,cell]),'cross_log_rates':[cross_ref['h'],cross_ref['o']],
      'cross_rates':array_identity(cross_rates),'recovery_log_rates':[ref['h'],ref['o']],
      'recovery_rates':array_identity(recovery),'recovery_rate_values':recovery.tolist(),
      'control_recovery_uniform_factor':float(recovery[0]/cross_rates[0]),
      'saved_recovery_reference_orbit':array_identity(ref['orbit']),
      'recovery_reference':{key:ref[key] for key in ('alpha','closure_max','primitive_separation','loss_mean','loss_max')},
      'crossing_distance':float(trace['distance'][k,cell]),'near_reference_within_0_01':bool(trace['distance'][k,cell]<.01),
      'prefix_pair_end_rms':float(np.sqrt(np.mean(np.sum((tail-tail.mean(0))**2,axis=1)))),
      'prefix_pair_net_step_rms':float(np.sqrt(np.mean(np.sum((tail-previous)**2,axis=1)))),
      'pair_metrics_are_not_one_step_or_all_phase_scales':True,'full_one_update_prefix_states_present':False,
      'gradient_at_fork_present':False,'dose_feasibility':'not_evaluated','recovery_full_spectrum_present':False}

def nominate(rows,forks):
    selected=[];used=set();used_controls=set()
    for point in (0,1):
      target=[x['prefix_mean_training_loss'] for x in forks if x['point']==point]
      for kind in ('periodic_candidate','chaotic_candidate'):
        pool=[]
        for row in rows:
          if row['class']!=kind or not row['distance_pass'] or row['id'] in used or (kind,tuple(row['log_rates'])) in used_controls:continue
          loss=row['tail_mean_training_loss'];rr=[ratio(loss,x) for x in target]
          if any(x is None for x in rr):continue
          score=max(math.log(x) for x in rr)
          pool.append((score,row['id'],row,rr))
        if not pool:selected.append({'point':point,'class':kind,'candidate':None});continue
        _,_,row,rr=min(pool,key=lambda x:(x[0],x[1]));used.add(row['id']);used_controls.add((kind,tuple(row['log_rates'])))
        selected.append({'point':point,'class':kind,'candidate':row['id'],'prefix_mean_loss_ratios':rr,
           'prefix_mean_loss_screen_pass':max(rr)<=2,'selection_is_not_qualification':True,
           'provisional_recovery_factor':next(f['control_recovery_uniform_factor'] for f in forks if f['point']==point)})
    return selected

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    require(a.out.resolve().parent==DATA/'audits/utility-inputs','audit parent')
    require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    identity={n:digest((ROOT/n).read_bytes()) for n in SOURCES}
    for n in SOURCES:require((ROOT/n).read_bytes()==subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT),'committed source')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    a.out.mkdir(parents=True,exist_ok=False);start=time.monotonic();inputs=[]
    def save(name,value):
        b=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
        require(sum(x.stat().st_size for x in a.out.iterdir() if x.is_file())+len(b)<=CAP,'2MiB cap')
        with (a.out/name).open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    def stop(*_):raise TimeoutError('readonly audit30s cap')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,30)
    result={'status':'running','source_commit':commit,'source_sha256':identity,'pid':os.getpid(),'cpu_threads':1,'nice':os.getpriority(os.PRIO_PROCESS,0),'dynamics_evaluations':0}
    save('runtime-start.json',result)
    try:
      def bind(path,expected=None):
        e=entry(path)
        if expected:require(e['bytes']==expected['bytes'] and e['sha256']==expected['sha256'],'input hash '+str(path))
        inputs.append(e);return path
      receipt=read(bind(ROOT/'docs/warmup-intervention-receipt.json'));man=bind(WARM/'manifest.json')
      require(entry(man)['sha256']==receipt['raw_manifest']['sha256'],'warmup manifest')
      wm=read(man);wi={e['path']:e for e in wm['files']};meta=read(bind(WARM/'metadata.json',wi['metadata.json']))
      require(meta['status']=='complete' and meta['source_commit']==receipt['source_commit'],'warmup complete/source')
      forks=[]
      for point in (0,1):
        refs=read(bind(WARM/f'reference-{point}.json',wi[f'reference-{point}.json']))['references']
        name=f'point-{point}-dwell-100-common.npz';file=bind(WARM/name,wi[name])
        with np.load(file,allow_pickle=False) as z:trace={k:z[k] for k in ('state','loss','alpha','rates','distance','initial_state')}
        for cell in (0,1):
          f=local_fork(trace,refs,cell,point);f['source_file']=str(file);f['reference_source_file']=str(WARM/f'reference-{point}.json');forks.append(f)
      save('local-forks.json',forks)
      plan_path=bind(ROOT/'experiments/trainability/EXP-028-plan.json');plan=read(plan_path)
      model=next(m for m in plan['maps'] if m['map_id']=='S2-28011')
      save('model-identity.json',{'plan':entry(plan_path),'map':model,'data_regenerated':False,'data_binding':'plan train-data/initialization hashes and historical source; no data values evaluated'})
      cross=np.array([f['cross_log_rates'] for f in forks[::2]])
      def distance(logs):return float(np.min(np.max(np.abs(cross-np.asarray(logs)),axis=1)))
      rows=[];schemas=[]
      for name in SLICE_NAMES:
        folder=DATA/'explore/consequence'/name;mp=bind(folder/'metadata.json');md=read(mp)
        require(md['system']=='S2' and md['seed']==28011,'S2 development only')
        path=bind(folder/'slice.npz')
        with np.load(path,allow_pickle=False) as z:
          schemas.append({'source':str(path),'members':list(z.files),'raw_state_or_loss_series_saved':False,'recorded_source_commit':md['commit'],'clean_source_recorded':False})
          z={k:z[k] for k in ('log10_hidden','status','period','ftle','separation','tail_train_mean','amplitude','sharpness_mean')}
        for i,h in enumerate(z['log10_hidden']):
          kind='periodic_candidate' if z['status'][i]==2 and z['period'][i]==2 else 'chaotic_candidate' if z['status'][i]==3 and np.all(z['ftle'][i]>.005) and z['separation'][i]>1e-6 else None
          if not kind:continue
          logs=[float(h),md['log10_output']];dist=distance(logs)
          if dist<.20:continue
          rows.append({'id':f'{name}:{i}','source':str(path),'row':i,'class':kind,'log_rates':logs,'distance_decades':dist,'distance_pass':True,
            'tail_mean_training_loss':float(z['tail_train_mean'][i]),'loss_summary_updates':md['config']['tail'],'amplitude_summary':float(z['amplitude'][i]),
            'ftle':z['ftle'][i].tolist(),'ftle_is_two_directions_at_one_horizon':True,'final_perturbation_separation':float(z['separation'][i]),
            'state_available':False,'complete_qualification':False,'full_prefix_losses_available':False,'full_prefix_states_available':False,
            'recovery_qualified':False,'ordinary_branch_independently_traced':False})
      er=read(bind(EXP/'receipt.json'));ei={v['path']:v for v in er['files']}
      certs=read(bind(EXP/'stage3/certificates.json',ei['stage3/certificates.json']))
      chaos=read(bind(EXP/'stage3/chaos-diagnostics.json',ei['stage3/chaos-diagnostics.json']))
      maps={}
      for label,sub in [('S2-28011','stage1/S2-28011.npz'),('S2-28011-zoom','stage2/S2-28011-zoom.npz')]:
        path=bind(EXP/sub,ei[sub])
        with np.load(path,allow_pickle=False) as z:maps[label]={k:z[k] for k in ('final_state','tail_mean_loss','last_evaluated_loss','lr_hidden','lr_output')}
      for category,records in [('periodic',certs),('chaos',chaos)]:
        for i,r in enumerate(records):
          if r['map_id'] not in maps:continue
          c=r.get('certificate',{})
          if category=='periodic' and not (r['kind']=='other' and c.get('verified') and c.get('attracting') and c.get('primitive_period')==2):continue
          if category=='chaos' and not (r['confirmed_chaotic'] and r['original_status']==3 and r['rerun_status']==3 and min(r['original_ftle'],r['rerun_ftle'])>.005):continue
          logs=[math.log10(r['lr_hidden']),math.log10(r['lr_output'])];dist=distance(logs)
          if dist<.20:continue
          cell=r['representative'] if category=='periodic' else r['cell'];grid=maps[r['map_id']]
          require(np.array_equal([r['lr_hidden'],r['lr_output']],[grid['lr_hidden'].ravel()[cell],grid['lr_output'].ravel()[cell]]),'EXP coordinate binding')
          row={'id':f'EXP028-{category}:{i}','source':str(EXP/('stage3/certificates.json' if category=='periodic' else 'stage3/chaos-diagnostics.json')),'row':i,'class':'periodic_candidate' if category=='periodic' else 'chaotic_candidate',
            'map_id':r['map_id'],'cell':cell,'log_rates':logs,'rates':array_identity([r['lr_hidden'],r['lr_output']]),'distance_decades':dist,'distance_pass':True,
            'tail_mean_training_loss':float(grid['tail_mean_loss'].ravel()[cell]),'loss_summary_updates':384,
            'state_available':True,'state':array_identity(grid['final_state'].reshape(-1,13)[cell]),'state_clock':'coarse/zoom final state after20000 updates; not the chaos30000-update rerun terminal',
            'complete_qualification':False,'full_prefix_losses_available':False,'full_prefix_states_available':False,'recovery_qualified':False,'ordinary_branch_independently_traced':False}
          if category=='periodic':row.update(periodic_orbit=array_identity(c['orbit']),primitive_period=2,spectral_radius=c['spectral_radius'],normalized_closure=c['normalized_closure'],final_relative_correction=c['final_relative_correction'],min_divisor_separation=c['min_divisor_separation'],candidate_distance=c['candidate_distance'],component_kind=r['kind'])
          else:row.update(original_ftle=r['original_ftle'],rerun_ftle=r['rerun_ftle'],ftle_windows=[4000,10000],ftle_blocks_saved=False,on_attractor_separation_saved=False,precision_refinement_saved=False)
          rows.append(row)
      save('control-candidates.json',rows);save('input-schemas.json',schemas)
      nominees=nominate(rows,forks);save('nominees.json',nominees)
      source_records=[]
      for commit_,names in [(meta['source_commit'],('src/neural_shrimp/warmup.py','tools/explore_warmup_intervention.py','src/neural_shrimp/models.py','src/neural_shrimp/trainability.py'))]+[(x['recorded_source_commit'],('tools/explore_consequence.py',)) for x in schemas]+[(er['git']['commit'],('src/neural_shrimp/gdmaps.py','src/neural_shrimp/orbits.py'))]:
        for name in names:
          raw=subprocess.check_output(['git','show',commit_+':'+name],cwd=ROOT);source_records.append({'commit':commit_,'path':name,'bytes':len(raw),'sha256':digest(raw)})
      save('source-bindings.json',source_records)
      # Counts from the immutable warmup run configuration, not a new benchmark.
      cfg=meta['args'];pair_updates=sum(2*(2*(cfg['pre_pairs']+cfg['levels']*d+cfg['top_pairs'])+6*(cfg['levels']*d+cfg['recovery_pairs'])) for d in cfg['dwell_pairs'])*len(cfg['outputs'])
      result.update(status='inventory_complete_test_not_ready',local_forks=4,eligible_candidates=len(rows),candidate_counts={kind:sum(r['class']==kind for r in rows) for kind in ('periodic_candidate','chaotic_candidate')},fully_qualified_controls=0,
        no_go_reasons=['Control history/support variables absent; no pulse permitted.','No independently traced ordinary branch.','No single chaos candidate has all multi-horizon/block/perturbation/precision qualifications.','Local intermediate update states/gradients and full recovery spectra absent.'],
        historical_warmup={'elapsed_seconds':meta['elapsed_seconds'],'counted_batched_cell_updates':pair_updates,'seconds_per_cell_update':meta['elapsed_seconds']/pair_updates,'not_a_new_profile_or_runtime_guarantee':True},input_count=len(inputs))
      save('inputs.json',inputs)
    except BaseException as e:
      result.update(status='failed_or_incomplete',error=type(e).__name__+': '+str(e));save('failure.json',result)
    finally:
      signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
      save('result.json',result)
      inventory={x.name:{k:v for k,v in entry(x).items() if k!='path'} for x in sorted(a.out.iterdir()) if x.is_file()}
      save('manifest.json',{'files':inventory,'status':result['status'],'total_bytes':sum(v['bytes'] for v in inventory.values())})
    print(json.dumps({k:result[k] for k in ('status','elapsed_seconds','max_rss_kib')}))
    return 0 if result['status']=='inventory_complete_test_not_ready' else 2
if __name__=='__main__':raise SystemExit(main())
