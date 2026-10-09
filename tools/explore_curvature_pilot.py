"""Six fixed exploratory references; profile/run are explicit, no default execution."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse
from datetime import datetime,timezone
from fractions import Fraction
import hashlib,json,platform,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from neural_shrimp.period2_branch_io import Store,file_entry,digest,identity,load_record
from neural_shrimp.period2_branch import PilotStop
from neural_shrimp.curvature_mechanism import analyze_curvature,require
from neural_shrimp.curvature_pilot import (LIMITS,Budget,Acquisition,finite,evaluate_slot,aggregate,request_stop)
PLAN='docs/plans/curvature-subspace-input-inventory.json'
CARD='docs/plans/curvature-six-reference-pilot.md'
SOURCES=('src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py',
 'src/neural_shrimp/period2_branch.py','src/neural_shrimp/period2_branch_io.py',
 'src/neural_shrimp/curvature_mechanism.py','src/neural_shrimp/curvature_pilot.py',
 'tools/explore_curvature_pilot.py','tools/launch_curvature_pilot.py',PLAN,CARD,
 'docs/plans/curvature-subspace-mechanism.md','experiments/trainability/EXP-028-plan.json')

def sources():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean checkout required')
    values={}
    for name in SOURCES:
        data=(ROOT/name).read_bytes();require(data==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed source '+name)
        values[name]=dict(bytes=len(data),sha256=digest(data))
    return values

def canonical_hash(x):return digest(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def shape_hash(x):return digest(repr(x.shape).encode()+np.ascontiguousarray(x).tobytes())
def json_read(path):return json.loads(Path(path).read_bytes())
def local_input(path):
    token='/neural-shrimp/'
    return ROOT/path.split(token,1)[1] if token in path else Path(path)

def inputs():
    plan=json_read(ROOT/PLAN);bindings=[]
    for pin in plan['input_inventory']:
        p=local_input(pin['path']);require(not p.is_symlink() and file_entry(p)=={k:pin[k] for k in ('bytes','sha256')},'input identity '+pin['path'])
        bindings.append(dict(original_path=pin['path'],used_path=str(p),**file_entry(p)))
    oldplan=json_read(ROOT/'experiments/trainability/EXP-028-plan.json');models={}
    for item in plan['training_data']:
        family=item['family'];p=Path(item['file']['path'])
        if family.startswith('S2'):
            obj=json_read(p);model=obj['network'].copy()
            for k in ('in_scale','out_scale','loss_scale','ridge'):model[k]=float(Fraction(model[k]))
            require(canonical_hash(obj['x'])==item['x_canonical_json_sha256'] and canonical_hash(obj['y'])==item['y_canonical_json_sha256'],'exact S2 data serialization')
            x=np.array([[float(Fraction(v)) for v in row] for row in obj['x']],dtype=np.float64);y=np.array([float(Fraction(v)) for v in obj['y']],dtype=np.float64)
        else:
            _,obj=load_record(p.parent,p.name);model=obj['model'];x=obj['x'];y=obj['y']
            for k,a in [('x',x),('y',y)]:require(identity(a)=={k:v for k,v in item['descriptors'][k].items() if k!='array'},'S1R data descriptor')
            require(model==item['model'],'S1R model inventory')
        historical=next(v for v in oldplan['maps'] if v['map_id']==family)
        require(not 29000<=historical['seed']<=29999 and model==historical['network'],'model/seed identity')
        require(shape_hash(x)==historical['data_sha256']['x'] and shape_hash(y)==historical['data_sha256']['y'],'original shape plus byte data hashes')
        models[family]=dict(x=finite(x),y=finite(y),model=model,identity=dict(x=identity(x),y=identity(y),historical_x=shape_hash(x),historical_y=shape_hash(y)))
    slots=[]
    for spec in plan['panel']:
        slot=dict(slot=spec['slot'],stratum=spec['slot'].split('-',2)[2],family=spec.get('model_family',spec['slot'].split('-')[0]+'-'+spec['slot'].split('-')[1]))
        try:
            p=Path(spec['file']);obj=json_read(p)
            if 'row_index' in spec and 'rates' in spec:
                row=obj[spec['row_index']];require(canonical_hash(row)==spec['record_canonical_json_sha256'],'certificate row identity')
                cert=row['certificate'];require(cert['verified'] and cert['primitive_period']==2 and cert['attracting'],'historical primitive attraction')
                require(canonical_hash(cert['orbit'])==spec['retained']['orbit_canonical_json_sha256'],'orbit JSON identity')
                pair=np.asarray(cert['orbit'],dtype=np.float64);rates=np.array([row['lr_hidden'],row['lr_output']])
                require(rates.tolist()==[spec['rates']['hidden'],spec['rates']['output']],'rate literals')
                slot['saved_values']=np.array([complex(v['real'],v['imag']) for v in cert['multipliers']])
            elif 'row_index' in spec:
                row=obj['rows'][spec['row_index']];require(canonical_hash(row)==spec['record_canonical_json_sha256'],'criticality row identity')
                require(canonical_hash(row['orbit_point'])==spec['retained']['orbit_point_canonical_json_sha256'],'single phase JSON identity')
                require(row['log10_hidden_last_attracting']==spec['log10_hidden'] and row['log10_output']==spec['log10_output'],'criticality rate literals')
                pair=np.asarray([row['orbit_point']],dtype=np.float64);rates=np.array([10.0 ** float(spec['log10_hidden']),10.0 ** float(spec['log10_output'])],dtype=np.float64);slot['saved_mu']=spec['retained']['mu']
            else:
                _,row=load_record(p.parent,p.name);require(row['id']==spec['node_id'] and row['accepted'],'anchor native node')
                for k in ('pair','params','rates'):require(identity(row[k])=={k:v for k,v in spec['array_descriptors'][k].items() if k!='array'},'anchor '+k)
                pair=row['pair'];rates=row['rates'];slot['saved_matrices']=np.stack([v['matrix'] for v in row['spectra']['phases']]);slot['saved_values']=row['spectra']['phases'][0]['values']
            data=models[slot['family']];m=data['model'];n=m['d']*m['width']+m['width']+m['bias']*(m['width']+1);hidden=m['d']*m['width']+m['bias']*m['width']
            finite(pair,(len(pair),n));require(len(pair) in (1,2) and np.all(np.isfinite(rates)) and np.all(rates>0),'state pair and positive rates')
            slot.update(rate_convention='historical actual rates, or Python float scalar 10.0 ** log10 for S2 crossing',pair=pair,rates=rates,rate_vector=np.r_[np.full(hidden,rates[0]),np.full(n-hidden,rates[1])],input_record=spec)
        except Exception as error:slot['input_error']=type(error).__name__+': '+str(error)
        slots.append(slot)
    require(len(slots)==6 and len({s['slot'] for s in slots})==6,'fixed six unique slots')
    return dict(plan=plan,bindings=bindings,models=models,slots=slots)


def safe(x):
    if isinstance(x,np.ndarray):return x
    if isinstance(x,np.generic):return safe(x.item())
    if isinstance(x,float) and not np.isfinite(x):return {'nonfinite_scalar':repr(x)}
    if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [safe(v) for v in x]
    return x

def synthetic_profile():
    n=72;u=np.eye(n)[0];v=.6*u+.8*np.eye(n)[1]
    return np.zeros((2,n)),np.stack([np.eye(n)-(.5*np.eye(n)-1.2*np.outer(z,z)) for z in (u,v)]),np.ones(n)

def profile_work(data,budget):
    slot=data['slots'][2];require(slot['slot']=='S1R-28002-stable-real' and 'input_error' not in slot,'fixed first S1R profile')
    m=data['models'][slot['family']];acq=Acquisition(budget,slot['slot'],m['x'],m['y'],m['model'],slot['rate_vector']);phase=acq.phase(slot['pair'][0])
    times={k:[] for k in ('eigh','eig','svd')};starts={}
    def observer(kind,row):
        if kind=='decomposition_start':starts[row['id']]=time.monotonic()
        elif kind=='decomposition_return':times[row['kind']].append(time.monotonic()-starts[row['id']])
        acq.observer(kind,row)
    result=analyze_curvature(*synthetic_profile(),observer=observer);budget.emit('synthetic_profile_analysis',result)
    require(result['decomposition_counts']==dict(eigh=2,eig=1,svd=1),'complete synthetic profile decompositions')
    projection=2*(23*phase['elapsed_seconds']+6*(2*max(times['eigh'])+max(times['eig'])+max(times['svd'])))+5
    passed=projection<=60
    budget.emit('profile_return',dict(slot=slot['slot'],phase=phase,model=m['model'],data_identity=m['identity'],rates=slot['rate_vector'],timings=times,projection_seconds=projection,passed=passed))
    return dict(profile_passed=passed,projection_seconds=projection,decomposition_times=times)


def reuse_profile(folder,sha,source_inventory,data):
    require(folder is not None and isinstance(sha,str) and len(sha)==64,'profile folder and hash required')
    require(not folder.is_symlink() and not folder.stat().st_mode&0o222 and all(not p.stat().st_mode&0o222 for p in folder.iterdir()),'immutable profile evidence')
    require(file_entry(folder/'manifest.json')['sha256']==sha,'external profile manifest pin')
    manifest=json_read(folder/'manifest.json');require(set(p.name for p in folder.iterdir())==set(manifest['files'])|{'manifest.json'},'profile inventory')
    for name,e in manifest['files'].items():require(Path(name).name==name and not (folder/name).is_symlink() and file_entry(folder/name)==e,'profile file binding')
    runtime=json_read(folder/'runtime-start.json');result=json_read(folder/'result.json')
    require(runtime['sources']==source_inventory and runtime['numpy']==np.__version__ and runtime['scipy']==scipy.__version__ and runtime['python']==platform.python_version() and runtime['host']==platform.node() and runtime['platform']==platform.platform(),'profile source/runtime/CPU host identity')
    require(result['status']=='profile_complete' and result['profile_passed'],'profile timing gate')
    rows=[json.loads(line) for line in (folder/'journal.jsonl').read_text().splitlines()];selected=[r for r in rows if r['kind']=='profile_return'];require(len(selected)==1,'one profile phase')
    _,r=load_record(folder,selected[0]['file']);s=data['slots'][2];m=data['models'][s['family']]
    require(r['slot']==s['slot'] and r['model']==m['model'] and r['data_identity']==m['identity'],'profile model/data identity')
    require(identity(r['rates'])==identity(s['rate_vector']) and identity(r['phase']['theta'])==identity(s['pair'][0]),'profile exact rates/state')
    return r['phase']


def run(out,review_commit,mode,profile=None,profile_sha256=None):
    inventory=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1' for k in THREADS),'CPU1 nice10')
    require(len(review_commit)==40,'full source review identity');subprocess.run(['git','cat-file','-e',review_commit+'^{commit}'],cwd=ROOT,check=True)
    store=Store(out,LIMITS['store_bytes']);emit=lambda kind,row:store.emit(kind,safe(row));budget=Budget(emit,5. if mode=='profile' else 60.)
    result=dict(schema='curvature-six-reference-v1',status='running',mode=mode,slots=[],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),review_commit=review_commit,sources=inventory,limits=LIMITS)
    try:
        store.json('runtime-start.json',dict(**result,pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),thread_env={k:os.environ[k] for k in THREADS},numpy=np.__version__,scipy=scipy.__version__,python=platform.python_version(),host=platform.node(),platform=platform.platform(),started_utc=datetime.now(timezone.utc).isoformat(),gpu=False,paid=False))
        data=inputs();emit('bound_inputs',data);budget.check()
        if mode=='profile':result.update(profile_work(data,budget));result['status']='profile_complete'
        else:
            phase=reuse_profile(profile,profile_sha256,inventory,data);emit('profile_binding',dict(path=str(profile),manifest_sha256=profile_sha256,source_commit=json_read(profile/'runtime-start.json')['source_commit']))
            for index,slot in enumerate(data['slots']):
                try:
                    budget.check();require('input_error' not in slot,slot.get('input_error','input identity'))
                    m=data['models'][slot['family']];row=evaluate_slot(slot,m['x'],m['y'],m['model'],budget,reuse=phase if index==2 else None)
                except PilotStop:raise
                except Exception as error:row=dict(slot=slot['slot'],stratum=slot['stratum'],qualified=False,reasons=[type(error).__name__+': '+str(error)])
                result['slots'].append(row);emit('slot_complete',row)
            result['aggregate']=aggregate(result['slots']);result['status']='panel_complete';emit('aggregate',result['aggregate'])
    except BaseException as error:
        result.update(status='failed_or_interrupted',error=type(error).__name__+': '+str(error))
        # Keep all six slots visible, including slots that resource/identity failure prevented.
        if mode=='run':
            known={r['slot'] for r in result['slots']}
            for s in json_read(ROOT/PLAN)['panel']:
                if s['slot'] not in known:result['slots'].append(dict(slot=s['slot'],qualified=False,reasons=['not_completed: '+result['error']]))
            result['aggregate']=aggregate(result['slots'])
    finally:
        result.update(operation_attempted=budget.attempted,operation_returned=budget.returned,elapsed_seconds=budget.clock()-budget.start,finished_utc=datetime.now(timezone.utc).isoformat())
        store.finish(safe(result))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--review-commit',required=True);p.add_argument('--mode',choices=['profile','run'],required=True);p.add_argument('--profile',type=Path);p.add_argument('--profile-sha256')
    a=p.parse_args();require(a.mode!='run' or (a.profile is not None and a.profile_sha256 is not None),'profile binding required before output creation');signal.signal(signal.SIGTERM,request_stop)
    r=run(a.out,a.review_commit,a.mode,a.profile,a.profile_sha256);print(json.dumps(dict(status=r['status'],mode=r['mode'],counts=r['operation_returned']),sort_keys=True))
    return 0 if r['status'] in ('profile_complete','panel_complete') else 2
if __name__=='__main__':raise SystemExit(main())
