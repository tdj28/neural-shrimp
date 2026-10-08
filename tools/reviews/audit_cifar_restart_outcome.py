"""Prospective saved-only CIFAR restart audit; no model, solver, QR or derivative."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for name in THREADS:os.environ[name]='1'
import argparse,hashlib,json,math,platform,signal,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
RAW=Path('/data2/neural-shrimp-data/explore/cifar/restart-sensitivity/attempt-001')
LAUNCH=RAW.with_name('attempt-001-launcher')
DATA=Path('/data2/neural-shrimp-data/explore/cifar')
BUNDLE=DATA/'device-benchmark/reference-001';OLD=DATA/'corrector-only/attempt-001';BASELINE=DATA/'linear-diagnostic/attempt-001'
FROZEN_SOURCE='8aa146fdbeca2f7d1cd4957995ced9b8bda719ee'
SOURCE_REVIEW='1f4f2abe8b4d4d4b8ad6e50cf1c9416d568d65f9'
# Original immutable evidence; these hashes are not inferred from the new result.
PINS=dict(bundle='414e8f2b1491a7f28cd1e1522376f7b008d7d3ac57bd29f69f0d5f4b953add38',old='d02393831f0d0b207b06eec11f114fb798e459bfac6679f19dc5cd68ddf50dc8',baseline='d47983793f58b033389fca5fc4aedc9ad78798b38cef06b2f3a4cf0f731534a3',state='36066b4b1d657924bd6efdce83dc3fd2d06a7ab135ef78afb03fb87649dd9674')
SOURCES=('src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py','src/neural_shrimp/cifar.py','src/neural_shrimp/cifar_linear_diagnostic.py','src/neural_shrimp/cifar_restart_diagnostic.py','tools/diagnose_cifar_linear_solve.py','tools/diagnose_cifar_restart.py','tools/launch_cifar_restart.py','docs/plans/cifar-restart-sensitivity-decision.md','docs/plans/cifar-restart-sensitivity-decision.json','docs/plans/cifar-restart-sensitivity-implementation.md')
AUDIT_FILES=('tools/reviews/audit_cifar_restart_outcome.py','tools/reviews/launch_cifar_restart_audit.py','tests/reviews/test_cifar_restart_outcome_audit.py','docs/reviews/cifar-restart-outcome-audit-card.md')
CAPS=dict(action=202,j2vp=203,hvp=407,direct_gradient=11)
SETTINGS=dict(restart=200,cycles=1,rtol=1e-10,forcing_limit=.5,max_matvec=202,explicit_zero_start=True,alphas=[1.,.5,.25,.125],armijo=1e-4,closure_tolerance=1e-8,primitive_separation=1e-7,internal_seconds=45.,operation_caps=CAPS,store_bytes=31*1024**2,launcher_bytes=1024**2,rss_bytes=2*1024**3,rss_enforcement='monitored, not kernel-hard',prefix_rtol=1e-10,prefix_atol=1e-12,prefix_length=50)
PIN_KEYS={'execution_commit','source_review_commit','raw_manifest_sha256','launcher_manifest_sha256'}
GMRES_SHA='cf8316712dfd558ce3153637c6176d285161de1347309ab7f6139abc80122a06'
VERSIONS=dict(torch='2.14.1+cu130',numpy='2.5.3',scipy='1.18.1')

def require(v,m):
    if not v:raise ValueError(m)
def canonical(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def digest(b):return hashlib.sha256(b).hexdigest()
def entry(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return dict(bytes=p.stat().st_size,sha256=h.hexdigest())
def read(p):return json.loads(p.read_bytes())
def close(a,b,label):
    require(type(a) in (float,int) and type(b) in (float,int) and math.isfinite(a) and math.isfinite(b),'finite scalar '+label)
    require(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-20),'scalar disagreement '+label)
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes(order='C')==b.tobytes(order='C')
def vec(a,finite=True,n=98666):
    a=np.asarray(a);require(a.dtype==np.float64 and a.shape==(n,),'vector dtype/shape')
    if finite:require(np.isfinite(a).all(),'finite vector')
    return a
def norm(a):
    result=math.sqrt(math.fsum(float(x)*float(x) for x in np.asarray(a).flat));require(math.isfinite(result),'finite norm');return result
def dot(a,b):
    require(a.shape==b.shape,'dot shape');result=math.fsum(float(x)*float(y) for x,y in zip(a.flat,b.flat));require(math.isfinite(result),'finite dot');return result

def directory(root):
    require(root.is_dir() and not root.is_symlink(),'regular directory')
    require(all(p.is_file() and not p.is_symlink() for p in root.iterdir()),'regular files only')
    return {p.name:entry(p) for p in sorted(root.iterdir())}
def inventory(root,pin,wrapped=False):
    rows=directory(root);require(rows['manifest.json']['sha256']==pin,'pinned manifest')
    m=read(root/'manifest.json');m=m['files'] if wrapped else m
    require(m=={k:v for k,v in rows.items() if k!='manifest.json'},'complete file inventory')
    return rows

def counter_check(events,result):
    attempted=dict.fromkeys(CAPS,0);returned=dict.fromkeys(CAPS,0);stack=[];products=[];callbacks=[];trials=[]
    for e in events:
        kind=e['kind']
        if kind=='operation_started':
            op=e['operation'];require(op in CAPS,'operation name');attempted[op]+=1
            require(e['index']==attempted[op]<=CAPS[op],'attempted operation index/cap')
            parents=[v['operation'] for v in stack]
            allowed={'action':[[]],'direct_gradient':[[]],'j2vp':[[],['action']],'hvp':[[],['j2vp'],['action','j2vp']]}
            require(parents in allowed[op],'operation causal nesting')
            if not stack and op=='action':require(returned['direct_gradient']==3 and returned['j2vp']==returned['action']+1 and returned['hvp']==2*returned['j2vp']+1,'action after complete parity')
            if not stack and op=='j2vp':require(returned['direct_gradient']==3 and returned['hvp']==1 and attempted['action']==0,'top-level parity J2 order')
            if not stack and op=='hvp':require(returned['direct_gradient']==3 and returned['j2vp']==0 and attempted['action']==0 and attempted['hvp']==1,'top-level parity HVP order')
            if op=='direct_gradient' and attempted[op]>3:require(len(callbacks)>=50 and len(products)==len(callbacks)+2,'trial gradients after returned direction/true check')
            if stack:stack[-1]['children'].append(op)
            stack.append(dict(operation=op,children=[]))
        elif kind=='operation_returned':
            op=e['operation'];require(stack and stack[-1]['operation']==op,'nested return order');frame=stack.pop();returned[op]+=1
            require(frame['children']==({'action':['j2vp'],'j2vp':['hvp','hvp']}.get(op,[])),'complete operation children')
            require(e['index']==returned[op]<=attempted[op] and math.isfinite(e['seconds']) and e['seconds']>=0,'returned operation index/time')
        elif kind=='matvec':
            require(not stack and e['index']==len(products)+1<=CAPS['action'] and returned['action']==e['index'],'matvec sequence');require(math.isfinite(e['seconds']) and e['seconds']>=0,'matvec time');products.append(e)
        elif kind=='gmres_inner':
            require(not stack and e['iteration']==len(callbacks)+1<=200 and len(products)==e['iteration'],'callback causal sequence');require(math.isfinite(e['estimated_relative_residual']) and e['estimated_relative_residual']>=0,'callback finite/nonnegative');callbacks.append(e['estimated_relative_residual'])
        elif kind=='trial_result':
            require(not stack and e['index']==len(trials)<4 and returned['direct_gradient']==3+2*(e['index']+1),'trial event sequence');trials.append({k:v for k,v in e.items() if k!='kind'})
        else:raise ValueError('unknown event')
    require(result['operation_attempted']==attempted and result['operation_returned']==returned,'operation count mirrors')
    require(result['retained_prefix_values']==callbacks,'retained callback mirror')
    require(len(products)<=returned['action']<=attempted['action']<=len(products)+1,'action/matvec partial boundary')
    if result['status']=='diagnostic_complete':
        require(not stack and attempted==returned and len(callbacks)>=50,'completed operation scope')
        require(attempted['action']==len(callbacks)+2 and attempted['j2vp']==attempted['action']+1 and attempted['hvp']==2*attempted['j2vp']+1 and attempted['direct_gradient']==11,'complete one-cycle operation relations')
        require(trials==result['trials'] and len(trials)==4,'all four completed trials')
    return dict(attempted=attempted,returned=returned,pending_operations=stack,matvecs=len(products),callbacks=callbacks,trial_events=trials)

def prefix_check(history,baseline):
    require(len(history)<=200 and all(math.isfinite(x) and x>=0 for x in history),'finite nonnegative callback history')
    require(len(baseline)==200 and all(math.isfinite(x) and x>=0 for x in baseline),'fixed baseline200 callbacks')
    checks=[abs(x-baseline[i])<=1e-12+1e-10*abs(baseline[i]) for i,x in enumerate(history[:50])]
    return dict(compared=len(checks),passed=all(checks) and len(checks)==50,matching_retained_prefix=all(checks),first_mismatch=next((i+1 for i,x in enumerate(checks) if not x),None))

def parity_check(actual,expected,record=None):
    require(actual.shape==expected.shape,'parity shape');finite=bool(np.isfinite(actual).all())
    if not finite:
        require(record is None or record['passed'] is False,'nonfinite parity cannot pass');return dict(passed=False,finite=False)
    dif=np.abs(actual-expected);absolute=float(dif.max());scaled=float((dif/(1e-11+1e-9*np.abs(expected))).max())
    if not math.isfinite(absolute) or not math.isfinite(scaled):
        require(record is None or record['passed'] is False,'overflow parity cannot pass');return dict(passed=False,finite=True,error_overflow=True,max_absolute_error=None,max_scaled_error=None)
    values=dict(max_absolute_error=absolute,max_scaled_error=scaled,passed=scaled<=1.)
    if record is not None:
        close(absolute,record['max_absolute_error'],'parity absolute');close(scaled,record['max_scaled_error'],'parity scaled');require(record['passed'] is values['passed'],'parity decision')
    return values

def linear_check(r,p,ap,e,saved):
    require(same(e,ap+r),'exact e=Ap+R bytes');rhs=norm(r);require(rhs>0,'positive RHS');true=norm(e);q=true/rhs;slope=dot(r,ap)
    values=dict(rhs_norm=rhs,true_linear_residual_norm=true,true_forcing_ratio=q,merit_directional_derivative=slope)
    for k,v in values.items():close(v,saved[k],k)
    flags=dict(forcing_test_pass=q<=.5,descent_direction=slope<0,requested_tolerance_met=true<=1e-10*rhs)
    for k,v in flags.items():require(saved[k] is v,'linear decision '+k)
    require(type(saved['gmres_info']) is int and saved['gmres_info'] in (0,1),'one-cycle solver status');require(saved['gmres_reported_converged'] is (saved['gmres_info']==0),'solver status mirror')
    for k,v in dict(restart=200,cycles=1,max_matvec=202,forcing_limit=.5,requested_linear_tolerance=1e-10).items():require(saved[k]==v,'linear setting '+k)
    return dict(**values,**flags)

def trial_check(theta,r,p,alpha,arrays,row,linear):
    trial=arrays['theta'];require(same(trial,theta+alpha*p),'exact trial theta bytes')
    phi0=.5*dot(r,r);require(phi0>0,'positive original merit');upper=phi0+1e-4*alpha*linear['merit_directional_derivative']
    close(row['alpha'],alpha,'trial alpha');close(row['armijo_upper'],upper,'Armijo upper');close(row['relative_step_norm'],alpha*norm(p)/max(1.,norm(theta)),'step length')
    finite=bool(np.isfinite(trial).all());values={}
    if finite:
        require(set(arrays)=={'theta','map1','map2','residual'},'complete evaluated trial')
        one,two,res=(arrays[k] for k in ('map1','map2','residual'));require(same(res,two-trial),'exact trial residual bytes')
        finite=bool(np.isfinite(res).all() and np.isfinite(one).all())
        if finite:
            try:rn=norm(res);tn=norm(trial);sep=norm(one-trial)/max(1.,tn);phi=.5*dot(res,res);relative=rn/max(1.,tn)
            except (ValueError,OverflowError):finite=False
            if finite:
                values=dict(residual_norm=rn,relative_residual=relative,merit=phi,merit_ratio=phi/phi0,relative_separation=sep)
                for k,v in values.items():close(row[k],v,'trial '+k)
                flags=dict(closure_screen=relative<1e-8,primitive_screen=sep>=1e-7,merit_decreased=phi<phi0,armijo_pass=phi<=upper)
                for k,v in flags.items():require(row[k] is v,'trial decision '+k)
    else:require(set(arrays)=={'theta'},'unevaluated nonfinite state has no invented maps')
    flags=dict(finite=finite,inexact_step_candidate=bool(finite and linear['forcing_test_pass'] and linear['descent_direction'] and values['merit']<=upper),accepted_orbit=False,applied_to_continuation=False)
    for k,v in flags.items():require(row[k] is v,'trial flag '+k)
    return dict(alpha=alpha,**values,**flags)

def source_check(inventory,commit):
    require(set(inventory)==set(SOURCES),'complete producer sources')
    for name,sha in inventory.items():
        for revision in (commit,FROZEN_SOURCE):require(digest(subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT))==sha,'reviewed execution source '+name)

def resource_check(start,result,launcher_start,launcher_result,complete):
    for r in (result,launcher_result):
        require(type(r['elapsed_seconds']) in (int,float) and math.isfinite(r['elapsed_seconds']) and r['elapsed_seconds']>=0,'finite observed duration')
    stamps=[datetime.fromisoformat(v) for v in (launcher_start['started_utc'],start['started_utc'],result['finished_utc'],launcher_result['finished_utc'])]
    require(all(t.tzinfo is not None for t in stamps) and stamps==sorted(stamps),'ordered UTC execution endpoints')
    require(result['elapsed_seconds']<=launcher_result['elapsed_seconds'] and launcher_result['elapsed_seconds']<=66.1,'observed external/reap duration')
    require(type(result['max_rss_kib']) is int and result['max_rss_kib']>0 and type(launcher_result['max_group_rss_bytes']) is int and launcher_result['max_group_rss_bytes']>=0,'observed RSS finite')
    pid=start['runtime']['pid'];require(type(pid) is int and pid>0 and pid!=launcher_start['group_pid'],'worker/group identities')
    seen=launcher_result['observed_owned_pids'];require(isinstance(seen,list) and all(type(p) is int and p>0 for p in seen) and seen==sorted(set(seen)),'observed owned PID set')
    if complete:
        require(pid in seen,'completed worker observed in owned group')
        require(result['max_rss_kib']*1024<=2*1024**3 and launcher_result['max_group_rss_bytes']<=2*1024**3,'successful observed RSS cap')
    return dict(worker_pid=pid,worker_observed=pid in seen,worker_seconds=result['elapsed_seconds'],cooperative_internal_seconds=45.,finished_record_over_internal_seconds=max(0.,result['elapsed_seconds']-45.),launcher_seconds=launcher_result['elapsed_seconds'],worker_peak_rss_kib=result['max_rss_kib'],group_peak_rss_bytes=launcher_result['max_group_rss_bytes'])

def audit(pins,emit):
    require(pins['source_review_commit']==SOURCE_REVIEW,'fixed independent producer review')
    launchrows=inventory(LAUNCH,pins['launcher_manifest_sha256'],True);lr=read(LAUNCH/'receipt.json');ls=read(LAUNCH/'runtime-start.json')
    require(set(launchrows)=={'launcher-source.py','runtime-start.json','stdout.log','receipt.json','manifest.json'},'complete launcher files')
    require(ls['source_commit']==lr['source_commit']==pins['execution_commit'] and ls['review_commit']==lr['review_commit']==pins['source_review_commit'],'launcher source/review')
    source_check(ls['sources'],pins['execution_commit']);require(launchrows['launcher-source.py']['sha256']==ls['sources']['tools/launch_cifar_restart.py'],'archived exact launcher')
    require(ls['out']==str(RAW) and ls['launcher']==str(LAUNCH),'fixed artifact paths')
    require(ls['command']==['timeout','--signal=TERM','--kill-after=5s','60s','nice','-n','10','/data2/neural-shrimp/.venv/bin/python','tools/diagnose_cifar_restart.py','--out',str(RAW),'--review-commit',pins['source_review_commit']],'exact producer command')
    require(ls['nice']>=10 and all(ls['thread_env'][k]=='1' for k in THREADS) and ls['gpu_used'] is False,'launcher resource scope')
    for k,v in dict(external_seconds=60,kill_grace_seconds=5,supervisor_seconds=65.,rss_cap_bytes=2*1024**3,store_bytes=31*1024**2,launcher_bytes=1024**2,log_bytes=128*1024).items():require(ls[k]==v,'launcher cap '+k)
    require(ls['group_pid']==lr['group_pid'],'owned group PID mirror')
    if ls['group_pid'] is None:require(lr['process_exit'] is None and lr['successful'] is False,'unstarted process stays incomplete')
    else:require(type(ls['group_pid']) is int and ls['group_pid']>0,'owned group PID')
    require(type(lr['elapsed_seconds']) in (int,float) and math.isfinite(lr['elapsed_seconds']) and 0<=lr['elapsed_seconds']<=66.1,'launcher observed duration')
    require(datetime.fromisoformat(ls['started_utc'])<=datetime.fromisoformat(lr['finished_utc']),'launcher UTC order')
    require(type(lr['max_group_rss_bytes']) is int and lr['max_group_rss_bytes']>=0,'observed process-group RSS')
    require(not lr['owned_live_pids_remaining'],'owned group exited')
    rawrows=directory(RAW) if RAW.exists() else {};require(rawrows==lr['raw_inventory'] and len(rawrows)==lr['raw_file_count'],'launcher exact raw mirror')
    rawbytes=sum(v['bytes'] for v in rawrows.values());launchbytes=sum(v['bytes'] for v in launchrows.values())
    require(rawbytes==lr['raw_bytes']<=31*1024**2 and launchbytes<=1024**2 and rawbytes+launchbytes<=32*1024**2 and launchrows['stdout.log']['bytes']<=128*1024,'complete aggregate caps')
    require(lr['within_store_cap'] is True,'store cap mirror')
    if pins['raw_manifest_sha256'] is None:require('manifest.json' not in rawrows,'null pin only when no manifest exists')
    else:require(rawrows['manifest.json']['sha256']==pins['raw_manifest_sha256'],'scientific manifest bytes pin')
    if not lr['raw_manifest_verified']:
        require(lr['successful'] is False,'unverified scientific inventory cannot succeed')
        malformed=None
        if 'manifest.json' in rawrows:
            try:require(read(RAW/'manifest.json')['files']=={k:v for k,v in rawrows.items() if k!='manifest.json'},'scientific inventory inconsistent')
            except (ValueError,KeyError,TypeError) as e:malformed=type(e).__name__+': '+str(e)
            require(malformed is not None,'launcher manifest failure contradicts valid complete manifest')
        emit('provenance.json',dict(pins=pins,raw_inventory=rawrows,launcher_inventory=launchrows,launcher_receipt=lr,scientific_manifest_error=malformed))
        return dict(verdict='incomplete_inventory_only',producer_status='missing_or_malformed_scientific_manifest',comparison_qualified=False,raw_bytes=rawbytes,scope='launcher-bound retained bytes only; no scientific counter/arithmetic verdict')
    require(inventory(RAW,pins['raw_manifest_sha256'],True)==rawrows,'scientific manifest verified')
    result=read(RAW/'result.json');runtime=read(RAW/'runtime-start.json')
    source_check(runtime['source_sha256'],pins['execution_commit'])
    require(runtime['source_sha256']==ls['sources'] and runtime['source_commit']==pins['execution_commit'] and runtime['review_commit']==pins['source_review_commit'],'science source/review')
    for key,value in runtime.items():
        if key not in ('status','comparison_qualified'):require(result[key]==value,'runtime/result mirror '+key)
    require(runtime['status']=='running' and runtime['settings']==SETTINGS,'frozen settings')
    rt=runtime['runtime'];require(all(rt[k]==v for k,v in VERSIONS.items()) and rt['gmres_python_source_sha256']==GMRES_SHA,'frozen library identity')
    require(rt['device']=='cpu' and rt['gpu'] is False and rt['nice']>=10 and rt['torch_threads']==rt['torch_interop_threads']==1 and all(rt['thread_env'][k]=='1' for k in THREADS),'science thread/device')
    require(all(result[k] is False for k in ('accepted_orbit','applied_to_continuation','spectrum_requested')),'never an orbit or continuation')
    require(result['status'] in ('diagnostic_complete','failed_or_interrupted'),'actual status')
    complete=result['status']=='diagnostic_complete'
    require(result['comparison_qualified'] is complete and lr['scientific_complete'] is complete,'qualification status')
    if complete:require(lr['successful'] is True and lr['process_exit']==0 and lr['failure'] is None and not lr['log_truncated'],'completed launcher success')
    else:require(lr['successful'] is False,'incomplete launcher must fail')
    resources=resource_check(runtime,result,ls,lr,complete)
    originals={str(folder):inventory(folder,PINS[key]) for folder,key in ((BUNDLE,'bundle'),(OLD,'old'),(BASELINE,'baseline'))}
    basis=read(BUNDLE/'result.json');old=read(OLD/'orbit.json');baseline=read(BASELINE/'result.json');baseline_linear=read(BASELINE/'linear.json')
    require(basis['status']=='complete' and basis['source_commit']=='a3debab72195d5ac58bee5562cb59afd8b6466e8' and old['status']=='corrector_not_accepted' and old['source_commit']=='7de5dcfb5eb50ed7829bc3b69b43c6e2c4f889cd','original source/status')
    for original in (basis,old,baseline):
        for name,sha in original['source_sha256'].items():require(digest(subprocess.check_output(['git','show',original['source_commit']+':'+name],cwd=ROOT))==sha,'original source bytes')
    require(baseline['source_commit']=='a9830e714d259597a53cee6ef7be41a9f07bb205' and baseline['status']=='diagnostic_complete','baseline source/status')
    require(baseline_linear==baseline['linear'] and baseline_linear['restart']==50 and baseline_linear['cycles']==4 and baseline_linear['inner_iterations']==200 and baseline_linear['matvec_calls']==205,'baseline fixed comparison')
    require(all(baseline[k]==v for k,v in VERSIONS.items()) and baseline['gmres_python_source_sha256']==GMRES_SHA,'same baseline implementation')
    emit('provenance.json',dict(pins=pins,raw_inventory=rawrows,launcher_inventory=launchrows,original_inventories=originals,producer_result_status=result['status'],launcher_receipt=lr,resources=resources))
    events=[] if 'events.jsonl' not in rawrows else [json.loads(line) for line in (RAW/'events.jsonl').read_text().splitlines()]
    counts=counter_check(events,result);prefix=prefix_check(counts['callbacks'],baseline_linear['estimated_relative_residual_history']);emit('events-audit.json',dict(**counts,prefix=prefix))
    if complete:require(prefix['passed'],'matched first50 required')
    arrays={p.stem:vec(np.load(p,allow_pickle=False),False) for p in RAW.glob('*.npy')}
    names={'initial-'+k for k in ('theta','map1','map2','residual')}|{'reference-'+k for k in ('gradient','hvp','j2vp')}|{'direction','applied-direction','true-linear-residual'}|{f'trial-{i}-{k}' for i in range(4) for k in ('theta','map1','map2','residual')}
    require(set(arrays)<=names,'only declared saved vectors')
    json_names={'runtime-start.json','input-identity.json','initial.json','parity.json','returned-direction.json','linear.json','events.jsonl','result.json','manifest.json'}
    require(set(rawrows)<=json_names|{name+'.npy' for name in names},'exact declared output-name scope')
    control_failures=[]
    for name in ('initial-theta','initial-map1','initial-map2','initial-residual'):
        if name in arrays and not same(arrays[name],np.load(BASELINE/(name+'.npy'),allow_pickle=False)):control_failures.append('byte replay '+name)
    with np.load(BUNDLE/'reference.npz',allow_pickle=False) as z:
        require(set(z.files)=={'x','y','theta','directions','gradient','hvp','j2vp'},'reference inventory')
        reference={k:z[k] for k in z.files}
    state_file=OLD/old['last_corrector']['state_file'];require(state_file.name==old['last_corrector']['state_file'] and entry(state_file)['sha256']==PINS['state'],'historical saved state')
    require(same(reference['theta'],np.load(state_file,allow_pickle=False)),'reference exact state')
    for key in ('x','y'):
        a=reference[key];i=basis['data_identity'][key];require(a.dtype==np.float64 and list(a.shape)==i['shape'] and digest(np.ascontiguousarray(a).tobytes())==i['sha256'],'normalized data identity')
    parity={};saved_parity=read(RAW/'parity.json') if 'parity.json' in rawrows else {}
    for name in ('gradient','hvp','j2vp'):
        if 'reference-'+name in arrays:
            expected=reference[name] if name=='gradient' else reference[name][0]
            parity[name]=parity_check(arrays['reference-'+name],expected,saved_parity.get(name,result.get('reference_parity',{}).get(name)))
            if not parity[name]['passed']:control_failures.append('reference parity '+name)
    if 'parity.json' in rawrows:require(result['reference_parity']==saved_parity and set(saved_parity)=={'gradient','hvp','j2vp'},'parity mirror')
    if 'input-identity.json' in rawrows:
        identity=read(RAW/'input-identity.json');require(identity==result['input_identity'],'input identity mirror')
        require(identity==dict(bundle=str(BUNDLE),bundle_manifest_sha256=PINS['bundle'],old_corrector=str(OLD),old_manifest_sha256=PINS['old'],baseline=str(BASELINE),baseline_manifest_sha256=PINS['baseline'],baseline_source_commit=baseline['source_commit'],state_sha256=PINS['state'],data_identity=basis['data_identity'],model=basis['model'],eta=basis['eta'],lam=basis['lam'],reference_source_commit=basis['source_commit']),'exact input identity')
        require(identity['eta']==.00827753 and identity['lam']==.01 and identity['model']==dict(d=3072,width=32,k=10) and old['data_identity']==basis['data_identity'],'fixed model/rate')
    if 'initial.json' in rawrows:
        initial=read(RAW/'initial.json');require(initial==result['initial'],'initial mirror');theta=vec(arrays['initial-theta']);r=vec(arrays['initial-residual'])
        require(same(r,arrays['initial-map2']-theta),'initial residual exact bytes')
        for k,v in dict(residual_norm=norm(r),relative_residual=norm(r)/max(1.,norm(theta)),relative_separation=norm(arrays['initial-map1']-theta)/max(1.,norm(theta))).items():close(v,initial[k],k)
    if 'returned-direction.json' in rawrows:
        returned=read(RAW/'returned-direction.json');require(returned['shape']==[98666] and returned['finite'] is bool(np.isfinite(arrays['direction']).all()) and returned['pending_independent_true_check'] is True,'returned direction metadata')
    checks=dict(reference_parity=parity,control_failures=control_failures,prefix=prefix)
    if 'linear.json' in rawrows:
        require(not control_failures and prefix['passed'],'controls before linear qualification');linear=read(RAW/'linear.json');require(linear==result['linear'],'linear mirror')
        require(returned['gmres_info']==linear['gmres_info'] and linear['inner_iterations']==len(counts['callbacks']) and linear['matvec_calls']==counts['matvecs'] and linear['estimated_relative_residual_history']==counts['callbacks'],'linear status/history counts')
        checks['linear']=linear_check(vec(arrays['initial-residual']),vec(arrays['direction']),vec(arrays['applied-direction']),vec(arrays['true-linear-residual']),linear)
        trials=[]
        for row in counts['trial_events']:
            i=row['index'];parts={k:arrays[f'trial-{i}-{k}'] for k in ('theta','map1','map2','residual') if f'trial-{i}-{k}' in arrays}
            trials.append(trial_check(theta,r,arrays['direction'],(1.,.5,.25,.125)[i],parts,row,checks['linear']))
        checks['trials']=trials
    if complete:
        required={'runtime-start.json','input-identity.json','initial.json','parity.json','returned-direction.json','linear.json','events.jsonl','result.json','manifest.json'}
        require(set(rawrows)==required|{name+'.npy' for name in arrays} and {'initial-theta','initial-map1','initial-map2','initial-residual','reference-gradient','reference-hvp','reference-j2vp','direction','applied-direction','true-linear-residual'}<=set(arrays) and len(checks['trials'])==4 and not control_failures and set(parity)=={'gradient','hvp','j2vp'},'complete stage inventory')
        require(result['candidate_trial_indices']==[i for i,row in enumerate(checks['trials']) if row['inexact_step_candidate']],'candidate aggregate')
        baseline_forcing=norm(np.load(BASELINE/'true-linear-residual.npy',allow_pickle=False))/norm(np.load(BASELINE/'initial-residual.npy',allow_pickle=False))
        close(baseline_forcing,baseline_linear['true_forcing_ratio'],'independent baseline forcing')
        close(result['baseline_true_forcing_ratio'],baseline_forcing,'baseline forcing')
        checks['comparison']=dict(baseline_forcing_ratio=baseline_forcing,restart200_forcing_ratio=checks['linear']['true_forcing_ratio'],forcing_reduced=checks['linear']['true_forcing_ratio']<baseline_forcing)
    emit('arithmetic-audit.json',checks)
    return dict(verdict='saved_outcome_audit_pass' if complete else 'incomplete_saved_records_consistent',producer_status=result['status'],comparison_qualified=complete,producer_error=result.get('error'),raw_bytes=rawbytes,launcher_bytes=launchbytes,checks=checks,scope='saved arithmetic/provenance only; no independent Arnoldi trajectory, A*p or map regeneration; no orbit or continuation acceptance')

def run(out,pins_path,pins_sha,source):
    require(os.getpriority(os.PRIO_PROCESS,0)>=10 and not out.exists(),'nice10 and exclusive output')
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==source,'exact audit HEAD')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean audit checkout')
    sources={}
    for name in AUDIT_FILES:
        b=(ROOT/name).read_bytes();require(b==subprocess.check_output(['git','show',source+':'+name],cwd=ROOT),'committed audit byte');sources[name]=digest(b)
    require(entry(pins_path)['sha256']==pins_sha,'external metadata pin file');pins=read(pins_path);require(set(pins)==PIN_KEYS,'complete metadata pins')
    for k,v in pins.items():
        if k=='raw_manifest_sha256' and v is None:continue
        n=40 if k.endswith('commit') else 64;require(isinstance(v,str) and len(v)==n and all(c in '0123456789abcdef' for c in v),'full pin '+k)
    out.mkdir(parents=True,exist_ok=False)
    def emit(name,value,final=False):
        data=canonical(value);require(sum(p.stat().st_size for p in out.iterdir())+len(data)<=(3584 if final else 3072)*1024,'audit raw cap')
        with (out/name).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    start=time.monotonic();record=dict(status='running',audit_source=source,audit_sources=sources,pins=pins,pins_file=entry(pins_path),started_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),cpu_threads=1,thread_env={k:os.environ[k] for k in THREADS},python=platform.python_version(),numpy=np.__version__,gpu=False,paid=False,model_evaluations=0)
    emit('runtime-start.json',record)
    def stop(*_):raise TimeoutError('15s saved restart audit cap')
    oa=signal.signal(signal.SIGALRM,stop);ot=signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,15)
    try:record.update(audit(pins,emit),status='finished')
    except BaseException as e:record.update(status='failed',verdict='audit_failed',error=type(e).__name__+': '+str(e))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,oa);signal.signal(signal.SIGTERM,ot);record.update(elapsed_seconds=time.monotonic()-start,finished_utc=datetime.now(timezone.utc).isoformat());emit('result.json',record,True);emit('manifest.json',{p.name:entry(p) for p in sorted(out.iterdir())},True)
        for p in out.iterdir():p.chmod(0o444)
        out.chmod(0o555)
    print(json.dumps({k:record.get(k) for k in ('status','verdict','error','elapsed_seconds','producer_status','comparison_qualified')},sort_keys=True));return 2 if record['status']=='failed' else 0
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--pins',type=Path,required=True);p.add_argument('--pins-sha256',required=True);p.add_argument('--source-commit',required=True);a=p.parse_args();raise SystemExit(run(a.out,a.pins,a.pins_sha256,a.source_commit))
