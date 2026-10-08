"""Independent saved-dose audit: scalar-loop analytic gradient, never a map step."""
import hashlib,json,math,subprocess
from pathlib import Path
import numpy as np
from .period2_branch_io import canonical,file_entry,identity,load_record

PRODUCER_SOURCE='075a76ecddddaeeae92f714d8ebbf170812b0500'
PRODUCER_REVIEW='167d55bfe638ff788dae3343b62204ca960a6dbb'
INPUT_REVIEW='42d901c0957221180aec5966b2c1726ea999b11b'
INPUT_MANIFEST='2f962ced4b6f32aca588f1012985591acc670437ff944e8e15375acf4fd2e03f'
STATE_HASHES=('55c3effe673c620fd4d6be8b8b0276dee5d6c8d5564ee8e5ea50c0bf3c35e075','bb49b69dd75af049fc3bad609027b42c6d29821b0912f46e949ea2caad5adae2','cf9e59e755e2626c35a235c3fd7eac0a3cee05846df3d5ad9e52f40c4d668762','1cfb60ebc30e10609137b35114a7276b96621055cfb40b6deead65a4980fad8d')
PRODUCER_FILES=('src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py','src/neural_shrimp/gdmaps.py','src/neural_shrimp/period2_branch.py','src/neural_shrimp/period2_branch_io.py','src/neural_shrimp/utility_dose.py','tools/explore_utility_dose_feasibility.py','tools/launch_utility_dose_feasibility.py','docs/plans/utility-dose-feasibility.md')
TOTAL_BYTES=1048576;STORE_BYTES=786432;LAUNCHER_BYTES=262144
ROOT=Path(__file__).resolve().parents[2]
INPUT=Path('/data2/neural-shrimp-data/audits/utility-inputs/attempt-001')
TARGET=Path('/data2/neural-shrimp-data/explore/utility-dose-feasibility/attempt-001')
TARGET_LAUNCHER=TARGET.with_name('attempt-001-launcher')

def need(condition,message):
    if not condition:raise ValueError(message)
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_bytes())
def digest_string(s):return isinstance(s,str) and len(s)==64 and all(c in '0123456789abcdef' for c in s)
def bit_equal(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes(order='C')==b.tobytes(order='C')
def finite(a,shape=None):
    a=np.asarray(a);need(a.dtype==np.float64 and np.isfinite(a).all() and (shape is None or a.shape==shape),'finite float64 shape');return a

def inventory(folder,pin,cap):
    need(folder.is_dir() and not folder.is_symlink(),'real input directory')
    m=folder/'manifest.json';need(m.is_file() and not m.is_symlink() and digest_string(pin) and file_entry(m)['sha256']==pin,'externally pinned manifest')
    obj=read(m);rows=obj['files'];need(isinstance(rows,dict) and set(p.name for p in folder.iterdir())==set(rows)|{'manifest.json'},'complete input inventory')
    need(sum(p.stat().st_size for p in folder.iterdir())<=cap,'input size cap')
    for name,e in rows.items():
        need(Path(name).name==name and name not in ('.','..'),'safe inventory basename')
        p=folder/name;need(p.is_file() and not p.is_symlink() and file_entry(p)==e,'input member hash '+name)
    return obj

def training_only():
    # Match the frozen input generator's operation order to verify its exact train bytes.
    streams=np.random.SeedSequence(28011).spawn(4);r=np.random.default_rng(streams[0])
    w=r.normal(size=(2,3))/np.sqrt(2);b=r.normal(scale=.3,size=3);v=r.normal(size=3)/np.sqrt(3);c=float(r.normal(scale=.1))
    r=np.random.default_rng(streams[1]);x=r.normal(size=(24,2)).astype(np.float64)
    y=np.tanh(x@w+b)@v+c;y=(y+0.*r.normal(size=24)).astype(np.float64)
    return x,y

def reference_gradient(theta,x,y):
    """Independent sample/coordinate-loop chain rule, including both bias groups."""
    t=finite(theta,(13,));x=finite(x);y=finite(y,(len(x),));need(x.shape==(len(x),2) and len(x)>0,'2D nonempty train data')
    w=t[:6].reshape(2,3);b=t[6:9];v=t[9:12];out=np.zeros(13)
    for row,target in zip(x,y):
        pre=np.array([sum(float(row[k])*float(w[k,j]) for k in range(2))+float(b[j]) for j in range(3)])
        h=np.tanh(pre);error=sum(float(h[j])*float(v[j]) for j in range(3))+float(t[12])-float(target)
        for j in range(3):
            hidden=error*float(v[j])*(1-float(h[j])**2)/len(x)
            for k in range(2):out[k*3+j]+=float(row[k])*hidden
            out[6+j]+=hidden;out[9+j]+=error*float(h[j])/len(x)
        out[12]+=error/len(x)
    return out+.001*t

def dose_oracle(g,rates,hidden=9):
    """Independently form the promised algebra, using no producer prediction code."""
    g=np.asarray(g,dtype=np.float64);rates=finite(np.asarray(rates,dtype=np.float64),(2,))
    need(g.ndim==1 and 0<hidden<len(g) and np.all(rates>0),'dose dimensions/rates')
    ans=dict(status='infeasible',reason=None,hidden_slice=[0,hidden],output_slice=[hidden,len(g)],uniform_fraction=.05,maximum_group_fraction=.25,fractions=None,arm_predictions=None)
    if not np.isfinite(g).all():ans['reason']='nonfinite_gradient';return ans
    with np.errstate(over='ignore',invalid='ignore'):s=np.concatenate((rates[0]*g[:hidden],rates[1]*g[hidden:]))
    ans['scaled_gradient']=s
    if not np.isfinite(s).all():ans['reason']='nonfinite_scaled_gradient';return ans
    hn=math.hypot(*s[:hidden]);on=math.hypot(*s[hidden:]);total=math.hypot(hn,on)
    ans.update(hidden_norm=hn if math.isfinite(hn) else None,output_norm=on if math.isfinite(on) else None,total_norm=total if math.isfinite(total) else None)
    if not all(math.isfinite(v) and v>0 for v in (hn,on,total)):ans['reason']='zero_or_nonfinite_block_norm';return ans
    radius=.05*total;fractions={'uniform':.05,'hidden':radius/hn,'output':radius/on}
    ans.update(radius=radius if math.isfinite(radius) else None,fractions={k:v if math.isfinite(v) else None for k,v in fractions.items()})
    if not math.isfinite(radius) or radius<=0 or not all(math.isfinite(v) and v>0 for v in fractions.values()):ans['reason']='nonfinite_or_zero_dose';return ans
    arms={}
    for name in ('sham','uniform','hidden','output'):
        h=fractions['uniform'] if name=='uniform' else fractions['hidden'] if name=='hidden' else 0.
        o=fractions['uniform'] if name=='uniform' else fractions['output'] if name=='output' else 0.
        delta=np.concatenate((h*s[:hidden],o*s[hidden:]))
        step=np.concatenate((-(1-h)*s[:hidden],-(1-o)*s[hidden:]))
        length=math.hypot(*delta);expected=0. if name=='sham' else radius;error=abs(length-expected);tol=max(1e-14,1e-12*expected)
        arms[name]=dict(reduction_fractions=[h,o],treatment_minus_sham_increment=delta,predicted_update_increment=step,predicted_total_step_length=math.hypot(*step),dose=length,absolute_dose_error=error,dose_tolerance=tol,dose_equality_pass=bool(math.isfinite(length) and error<=tol))
    ans['arm_predictions']=arms;feasible=max(fractions.values())<=.25 and all(v['dose_equality_pass'] for v in arms.values())
    ans.update(status='feasible' if feasible else 'infeasible',reason='fraction_cap' if max(fractions.values())>.25 else 'dose_equality' if not feasible else None)
    return ans

def compare(a,b,atol=1e-11,rtol=1e-9,path='value'):
    """Every entry is compared; labels/booleans/structure remain exact."""
    errors=[]
    if isinstance(a,dict) and isinstance(b,dict):
        need(set(a)==set(b),'exact comparison keys '+path)
        for k in a:errors+=compare(a[k],b[k],atol,rtol,path+'.'+str(k))
    elif isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
        need(len(a)==len(b),'exact comparison length '+path)
        for i,(u,v) in enumerate(zip(a,b)):errors+=compare(u,v,atol,rtol,path+f'[{i}]')
    elif isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
        a=np.asarray(a);b=np.asarray(b);need(a.shape==b.shape and a.dtype==b.dtype,'comparison array shape/dtype '+path)
        need(np.array_equal(np.isfinite(a),np.isfinite(b)),'comparison finite mask '+path)
        if np.isfinite(a).all():
            err=np.abs(a-b);limit=atol+rtol*np.abs(b);need(bool(np.all(err<=limit)),'comparison array values '+path);errors.append(float(np.max(err,initial=0.)))
        else:need(bit_equal(a,b),'nonfinite retained array bits '+path)
    elif a is None or b is None or type(a) is bool or type(b) is bool or isinstance(a,str) or isinstance(b,str):need(type(a)==type(b) and a==b,'exact comparison label '+path)
    elif isinstance(a,(int,float,np.number)) and isinstance(b,(int,float,np.number)):
        need(math.isfinite(float(a)) and math.isfinite(float(b)),'finite comparison scalar '+path)
        error=abs(float(a)-float(b));need(error<=atol+rtol*abs(float(b)),'comparison scalar '+path);errors.append(error)
    else:need(a==b,'comparison value '+path)
    return errors

def historical_sources(execution,inventory_values):
    need(set(inventory_values)==set(PRODUCER_FILES),'complete producer source inventory')
    for name,e in inventory_values.items():
        expected=subprocess.check_output(['git','show',PRODUCER_SOURCE+':'+name],cwd=ROOT)
        executed=subprocess.check_output(['git','show',execution+':'+name],cwd=ROOT)
        need(expected==executed and e=={'bytes':len(expected),'sha256':sha(expected)},'reviewed historical source '+name)

def bound_original_inputs():
    inv=inventory(INPUT,INPUT_MANIFEST,2*1024**2);forks=read(INPUT/'local-forks.json');model=read(INPUT/'model-identity.json')['map']
    need(len(forks)==4 and model['map_id']=='S2-28011' and model['network']==dict(d=2,width=3,activation='tanh',in_scale=1.,out_scale=1.,bias=True,loss_scale=.5,ridge=.001),'fixed input model')
    x,y=training_only()
    for name,a in [('x',x),('y',y)]:need(sha(repr(a.shape).encode()+a.tobytes())==model['data_sha256'][name],'exact training-only data '+name)
    rows=[]
    for i,f in enumerate(forks):
        need(f['point']==i//2 and f['origin']==('near_orbit' if i%2==0 else 'ordinary_initialization') and f['array_index']==[4100,i%2] and f['accepted_updates_at_fork']==8202,'four fixed fork ordering')
        state=finite(np.asarray(f['state_values'],dtype=np.float64),(13,));rates=finite(np.asarray(f['recovery_rate_values'],dtype=np.float64),(2,))
        for a,name in [(state,'state'),(rates,'recovery_rates')]:
            need(identity(a)=={k:f[name][k] for k in ('dtype','shape','order','sha256')},'original array hash '+name)
            need([format(int(v),'016x') for v in a.view('<u8')]==f[name]['binary64_hex'],'original array bits '+name)
        need(f['state']['sha256']==STATE_HASHES[i] and bit_equal(rates,np.array([10.**v for v in f['recovery_log_rates']])),'pinned state/rates')
        rows.append(dict(point=i//2,origin=f['origin'],state=state,rates=rates,state_sha256=STATE_HASHES[i]))
    return x,y,rows,inv

def records(folder,manifest):
    journal=[json.loads(v) for v in (folder/'journal.jsonl').read_text().splitlines()]
    expected={'runtime-start.json','journal.jsonl','checkpoint.json','result.json'};out=[]
    for i,row in enumerate(journal):
        need(row['sequence']==i and row['file']==f"{i:05d}-{row['kind']}.json",'sequential journal')
        meta=read(folder/row['file']);need(meta['sequence']==i and meta['kind']==row['kind'],'record/journal binding');expected.add(row['file'])
        if 'arrays_file' in meta:
            need(meta['arrays_file']==row['file'].removesuffix('.json')+'.npz','record array filename binding');expected.add(meta['arrays_file'])
        out.append(load_record(folder,row['file']))
    need(set(manifest['files'])==expected,'exact consumed raw inventory')
    return out


def validate_envelope(raw,launch,runtime,producer,receipt,execution):
    need(launch['schema']=='utility-dose-launcher-v1' and producer['schema']=='utility-dose-feasibility-v1','producer schemas')
    need(runtime['source_commit']==launch['source_commit']==receipt['source_commit']==execution,'execution binding')
    need(runtime['source_review_commit']==launch['source_review_commit']==receipt['source_review_commit']==PRODUCER_REVIEW,'producer review binding')
    need(runtime['input_review_commit']==INPUT_REVIEW and runtime['input_manifest_sha256']==INPUT_MANIFEST,'original input audit binding')
    need(runtime['sources']==launch['sources'],'source inventory mirror')
    need(launch['output']==str(TARGET) and launch['launcher']==str(TARGET_LAUNCHER),'fixed producer paths')
    command=['timeout','--signal=TERM','--kill-after=5s','10s','nice','-n','10','/data2/neural-shrimp/.venv/bin/python','tools/explore_utility_dose_feasibility.py','--out',str(TARGET),'--review-commit',PRODUCER_REVIEW]
    need(launch['command']==command,'exact producer command')
    need(launch['external_seconds']==10 and launch['kill_grace_seconds']==5 and launch['total_raw_bytes']==TOTAL_BYTES and launch['store_bytes']==STORE_BYTES and launch['launcher_bytes']==LAUNCHER_BYTES,'launcher caps')
    need(runtime['limits']==dict(internal_seconds=5,external_seconds=10,kill_grace_seconds=5,gradients=4,total_bytes=TOTAL_BYTES,store_bytes=STORE_BYTES,launcher_bytes=LAUNCHER_BYTES),'runtime caps')
    for record in (runtime,launch):
        need(record['nice']>=10 and record['thread_env']=={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},'one-thread nice10 execution')
    for record,name in [(runtime,'pid'),(launch,'launcher_pid'),(launch,'child_pid')]:need(type(record[name]) is int and record[name]>0,'positive owned PID')
    need(launch['child_pid']==receipt['child_pid'] and runtime['pid']!=launch['launcher_pid'],'process binding')
    need(receipt['status']=='process_finished' and receipt['process_exit']==0 and receipt['failure'] is None and receipt['log_truncated'] is False and receipt['within_store_cap'] is True,'complete successful producer process')
    need(producer['status'] in ('feasible_for_further_design','fixed_candidate_infeasible') and producer['gradient_attempts']==producer['gradient_returns']==4 and len(producer['forks'])==4,'all four completed gradients')
    for seconds,cap in [(producer['elapsed_seconds'],5),(receipt['elapsed_seconds'],15)]:need(type(seconds) in (int,float) and math.isfinite(seconds) and 0<seconds<=cap,'completed runtime inside fixed cap')
    need(producer['elapsed_seconds']<=receipt['elapsed_seconds'],'inner/outer runtime order')
    need(receipt['raw_file_count']==len(raw) and receipt['raw_bytes']==sum(e['bytes'] for e in raw.values()) and receipt['raw_inventory_sha256']==sha(canonical(raw)),'launcher raw inventory arithmetic')
    need(receipt['raw_manifest']==raw['manifest.json'] and receipt['raw_result']==raw['result.json'],'launcher result/manifest binding')

def audit_saved(raw_pin,launcher_pin,execution,store,check,stats):
    raw_manifest=inventory(TARGET,raw_pin,STORE_BYTES);launch_manifest=inventory(TARGET_LAUNCHER,launcher_pin,LAUNCHER_BYTES)
    need(set(launch_manifest['files'])=={'launcher-source.py','runtime-start.json','stdout.log','receipt.json'},'exact launcher inventory')
    raw_entries={p.name:file_entry(p) for p in TARGET.iterdir()}
    store.emit('input_inventories',dict(raw_manifest=dict(path=str(TARGET/'manifest.json'),**file_entry(TARGET/'manifest.json')),launcher_manifest=dict(path=str(TARGET_LAUNCHER/'manifest.json'),**file_entry(TARGET_LAUNCHER/'manifest.json')),raw_inventory=raw_manifest,launcher_inventory=launch_manifest))
    runtime=read(TARGET/'runtime-start.json');launch=read(TARGET_LAUNCHER/'runtime-start.json');receipt=read(TARGET_LAUNCHER/'receipt.json');producer=read(TARGET/'result.json')
    validate_envelope(raw_entries,launch,runtime,producer,receipt,execution)
    need(launch_manifest['status']==receipt['status'] and raw_manifest['status']==producer['status'],'manifest status mirrors')
    historical_sources(execution,runtime['sources'])
    need(file_entry(TARGET_LAUNCHER/'launcher-source.py')==runtime['sources']['tools/launch_utility_dose_feasibility.py'],'saved launcher source')
    need(receipt['stdout']==file_entry(TARGET_LAUNCHER/'stdout.log') and receipt['stdout']['bytes']<=131072,'stdout binding/cap')
    from datetime import datetime
    begin=datetime.fromisoformat(launch['started_utc']);inner=datetime.fromisoformat(runtime['started_utc']);end=datetime.fromisoformat(receipt['finished_utc'])
    need(begin.tzinfo is not None and inner.tzinfo is not None and end.tzinfo is not None and begin<=inner<=end,'UTC launch ordering')
    x,y,forks,original_inventory=bound_original_inputs();check()
    stages=records(TARGET,raw_manifest)
    need([k for k,v in stages]==['bound_inputs']+['gradient_pending','gradient_return','dose_prediction']*4,'exact four-gradient event order')
    bound=stages[0][1]
    need(bound['model']==dict(d=2,width=3,activation='tanh',in_scale=1.,out_scale=1.,bias=True,loss_scale=.5,ridge=.001),'bound model')
    need(bit_equal(bound['x'],x) and bit_equal(bound['y'],y),'exact returned training data')
    need(bound['packing']==dict(W0=[0,6],b0=[6,9],w1=[9,12],b1=[12,13]) and len(bound['forks'])==4,'bias-inclusive packing/forks')
    projected=1+1000*4*63.64927253895439/1232000
    need(bound['cost_gate']==producer['cost_gate'] and bound['cost_gate']['projected_seconds']==projected<5 and bound['cost_gate']['formula']=='1+1000*4*63.64927253895439/1232000','historical engineering screen')
    compare(bound['forks'],forks,atol=0,rtol=0)
    for observed,expected in zip(bound['forks'],forks):
        need(bit_equal(observed['state'],expected['state']) and bit_equal(observed['rates'],expected['rates']),'exact bound fork state/rates')
    checkpoint=read(TARGET/'checkpoint.json')
    need(checkpoint==dict(status=producer['status'],latest=dict(sequence=12,kind='dose_prediction',file='00012-dose_prediction.json')),'terminal checkpoint binding')
    store.emit('bound_inputs',dict(forks=forks,x=x,y=y,original_input_manifest=INPUT_MANIFEST,original_inventory=original_inventory,producer_source=execution,reviewed_source=PRODUCER_SOURCE,producer_review=PRODUCER_REVIEW,source_files=runtime['sources'],historical_cost_screen=projected))
    rows=[]
    for i,f in enumerate(forks):
        check();pending,returned,prediction=(stages[1+3*i+j][1] for j in range(3))
        for record in (pending,returned,prediction):need(record['index']==i,'fixed gradient index')
        for record in (pending,prediction):need(record['point']==f['point'] and record['origin']==f['origin'],'fixed point/origin')
        need(bit_equal(pending['state'],f['state']) and bit_equal(pending['rates'],f['rates']),'gradient at unchanged native state')
        saved=np.asarray(returned['gradient']);need(saved.shape==(13,) and saved.dtype==np.float64,'original gradient full packing')
        stats['reference_gradient_attempts']+=1
        reference=reference_gradient(f['state'],x,y);stats['reference_gradient_returns']+=1
        store.emit('independent_gradient_return',dict(index=i,state=f['state'],rates=f['rates'],gradient=reference),final=True);check()
        gradient_errors=compare(saved,reference,atol=1e-12,rtol=1e-10,path='gradient')
        replay=dose_oracle(saved,f['rates']);independent=dose_oracle(reference,f['rates'])
        saved_prediction=prediction['prediction']
        replay_errors=compare(saved_prediction,replay,path='saved-gradient dose replay')
        independent_errors=compare(saved_prediction,independent,path='independent-gradient dose')
        need(saved_prediction['uniform_fraction']==.05 and saved_prediction['maximum_group_fraction']==.25,'literal dose/cap constants')
        if saved_prediction['arm_predictions'] is not None:
            for arm,values in saved_prediction['arm_predictions'].items():
                need(values['dose_tolerance']==max(1e-14,1e-12*(0. if arm=='sham' else replay['radius'])),'unrelaxed dose tolerance')
        expected_row=dict(index=i,status=replay['status'],reason=replay['reason'])
        need(producer['forks'][i]==expected_row,'producer fork decision')
        row=dict(index=i,status=replay['status'],reason=replay['reason'],max_gradient_absolute_error=max(gradient_errors,default=0),max_saved_gradient_dose_absolute_error=max(replay_errors,default=0),max_independent_gradient_dose_absolute_error=max(independent_errors,default=0),replay=replay,independent=independent)
        store.emit('fork_comparison',row);rows.append(row)
    check();feasible=all(row['status']=='feasible' for row in rows)
    need(producer['candidate_feasible'] is feasible and producer['status']==('feasible_for_further_design' if feasible else 'fixed_candidate_infeasible'),'complete fixed-candidate decision')
    return dict(status='audit_passed',producer_status=producer['status'],fixed_candidate_feasible=feasible,forks=[{k:v for k,v in row.items() if k not in ('replay','independent')} for row in rows],interpretation='Fixed dose arithmetic independently verified; infeasibility rejects this candidate design, not utility generally. Missing controls/support still bar pulse or geometry claims.',raw_manifest_sha256=raw_pin,launcher_manifest_sha256=launcher_pin)
