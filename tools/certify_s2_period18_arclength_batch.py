"""Exclusive streamed16-chart validated arclength batch."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for name in THREADS:os.environ[name]='1'
import argparse,gzip,json,math,platform,resource,signal,subprocess,sys,time
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path
import flint,numpy,scipy
from flint import ctx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from neural_shrimp.rigorous_orbits import canonical_bytes,sha256,check_orbit
from neural_shrimp.rigorous_connection import HiddenRateMap,require
from neural_shrimp.rigorous_arclength_batch import compact_chart,stream_charts
from neural_shrimp.rigorous_folds import FoldTanhMap
from neural_shrimp.rigorous_gradient_enclosure import MeanValueHiddenRateMap
RECEIPT='certificates/s2-period18-receipt.json'
RECEIPT_SHA='f5b8ed6bc4bded4f6c3515fed5a92d6cfc12614e4e68f0169037810b0d317277'
CARD='docs/plans/s2-period18-arclength-batch-pilot.md'
SOURCES=['src/neural_shrimp/__init__.py','src/neural_shrimp/rigorous_orbits.py',
 'src/neural_shrimp/rigorous_connection.py','src/neural_shrimp/rigorous_connection_frames.py',
 'src/neural_shrimp/rigorous_folds.py','src/neural_shrimp/rigorous_flips.py',
 'src/neural_shrimp/controls.py','src/neural_shrimp/models.py','src/neural_shrimp/rigorous_gradient_enclosure.py',
 'src/neural_shrimp/rigorous_quadratic_connection.py','src/neural_shrimp/rigorous_arclength_connection.py','src/neural_shrimp/rigorous_arclength_batch.py',
 'tests/test_rigorous_quadratic_connection.py','tests/test_rigorous_arclength_connection.py','tests/test_rigorous_arclength_batch.py',
 'tools/certify_s2_period18_arclength_batch.py','docs/reviews/s2-period18-quadratic-weights.json','docs/reviews/s2-period18-weighted-outcome.json','docs/reviews/s2-period18-arclength-outcome.json',CARD,RECEIPT]
SETTINGS=dict(precision=256,halfwidth_exponent=-6,radius_exponent=-22,newton_steps=2)
CAP=384*1024**2

class CountedJetMap(FoldTanhMap):
    def __init__(self,inputs,cap):
        super().__init__(inputs);self.attempted=0;self.returned=0;self.cap=cap
    def jet_step(self,theta,rate):
        require(self.attempted<self.cap,'native path-jet cap')
        self.attempted+=1;value=super().jet_step(theta,rate);self.returned+=1;return value
    def counts(self):return {'attempted':self.attempted,'returned':self.returned,'cap':self.cap}


def entry(path):
    b=path.read_bytes();return {'sha256':sha256(b),'bytes':len(b)}
def read(path):
    if path.suffix=='.gz':
        with gzip.open(path,'rb') as f: data=f.read(256*1024**2+1)
        require(len(data)<=256*1024**2,'decompressed artifact cap')
        if path.name.endswith('.json.gz'):
            meta=json.loads(path.with_name(path.name.removesuffix('.json.gz')+'-payload.json').read_bytes())
            require(len(data)==meta['uncompressed_bytes'] and sha256(data)==meta['uncompressed_sha256'],'decompressed payload identity')
        return json.loads(data)
    return json.loads(path.read_bytes())
def now():return datetime.now(timezone.utc).isoformat()
def total(out):return sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
def save(path,obj):
    data=canonical_bytes(obj);root=path.parent
    payload={'uncompressed_bytes':len(data),'uncompressed_sha256':sha256(data)}
    require(len(data)<256*1024**2,'uncompressed artifact cap')
    if path.suffix=='.gz':data=gzip.compress(data,compresslevel=6,mtime=0)
    require(len(data)<60*1024**2,'single stored artifact cap')
    require(total(root)+len(data)<CAP,'aggregate raw cap')
    temp=path.with_name(path.name+'.tmp')
    with temp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
    return payload
def inventory(out):return {str(p.relative_to(out)):entry(p) for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'manifest.json'}
def verify_inventory(out,members):
    require(isinstance(members,dict) and bool(members),'inventory required')
    actual={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p!=out/'manifest.json'}
    require(actual==set(members),'exact inventory')
    for name,e in members.items():
        p=out/name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and not p.is_symlink(),'safe member')
        require(entry(p)==e,'file identity '+name)
def sources():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source required')
    result={}
    for name in SOURCES:
        b=(ROOT/name).read_bytes();require(b==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed source '+name)
        result[name]=sha256(b)
    return result

def original_anchor():
    receipt_path=ROOT/RECEIPT;require(entry(receipt_path)['sha256']==RECEIPT_SHA,'fixed canonical anchor')
    receipt=read(receipt_path);base=Path(receipt['external_root'])
    verify_inventory(base,{**receipt['files'],'receipt.json':entry(receipt_path)})
    envelope=read(base/'certificate.json');anchor=envelope['result'];inp=read(base/'inputs.json')
    require(receipt['passed'] and anchor['passed'] and receipt['statement']==anchor['exact_statement'],'certified anchor statement')
    require(inp['period']==18 and anchor['exact_statement']['dimension']==13,'fixed 18x13 state order')
    require(inp['lr_hidden']=='1254985102288567/17592186044416' and inp['lr_output']=='1848862009810323/2251799813685248','original exact layer rates')
    require(entry(base/'inputs.json')['sha256']=='9c01121b61a195baf20050eaf3bf567ebba6b7a0310d8fbced0aabc67f98248f','exact consumed model/data')
    require(entry(ROOT/'src/neural_shrimp/rigorous_orbits.py')['sha256']==receipt['source_sha256']['src/neural_shrimp/rigorous_orbits.py'],'unchanged anchor kernel')
    phases=anchor['orbit_enclosures'][:18]
    initial=[[str(Fraction(v['mid'][0])*Fraction(2)**v['mid'][1]) for v in row] for row in phases]
    return {'model_inputs':inp,'initial':initial,'rate':inp['lr_hidden'],
            'anchor':{'rate':inp['lr_hidden'],'phase_boxes':phases},
            'provenance':{'receipt_sha256':RECEIPT_SHA,'receipt':RECEIPT,'certificate':entry(base/'certificate.json'),
                          'inputs':entry(base/'inputs.json'),'order':'native phases0..17 of unchanged C1 certificate; no phase permutation',
                          'model_identity':'exact same model_inputs used for every phase and rate; only hidden rate varies',
                          'anchor_source_commit':receipt['source_commit']}}

def synthetic():
    inp={'schema':'neural-shrimp-c1-input-v1','network':{'activation':'tanh','d':2,'width':3,'bias':True,
         'in_scale':'1','out_scale':'1','loss_scale':'1/2','ridge':'1'},
         'x':[[str(Fraction(i-12,48)),str(Fraction((i*7)%23-11,48))] for i in range(24)],
         'y':['0']*24,'lr_hidden':'1/128','lr_output':'1/128'}
    return {'model_inputs':inp,'initial':[['0']*13 for _ in range(18)],'rate':'1/128','anchor':None,
            'provenance':{'synthetic':True,'scientific_claim':False,'purpose':'same18x13 dense shooting algebra and24-sample tanh derivatives; no target inputs'}}

WEIGHTS=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-quadratic-weights-2026-10-10/attempt-001/run')
WEIGHTS_MANIFEST='cbc4c267601135a732268bccf99c6ce91163f7a01171cf0f8c3b5fc1bdd05afa'
WEIGHTS_FILE='d2ea032791154ca47d6c57a58af8c6616ef224a10ca49bf3904adb4d47210e5a'

def prepare_weights():
    require(entry(WEIGHTS/'manifest.json')['sha256']==WEIGHTS_MANIFEST,'fixed weight manifest')
    verify_inventory(WEIGHTS,read(WEIGHTS/'manifest.json')['files'])
    require(entry(WEIGHTS/'chart-0.json')['sha256']==WEIGHTS_FILE,'fixed first-chart weights')
    runtime=read(WEIGHTS/'runtime-start.json');outcome=read(WEIGHTS/'outcome.json');row=read(WEIGHTS/'chart-0.json')
    require(runtime['source_commit']=='b41b465e3aa10ebdf2b7c2cdee4de9fbf43a3f00' and outcome['exit_code']==0,'fixed completed weight proposal')
    require(row['contraction_passed'] and row['inclusion_passed'] and len(row['exact_weights'])==234,'qualified saved-domain proposal')
    return row['exact_weights']

PARENT=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-arclength-connection-2026-10-10/attempt-001-target/run')
PARENT_MANIFEST='0460fd73c44e52b479538908e55df12c61a082e2f3541c3c54186bfd7750f71c'


def prepare():
    original=original_anchor()
    require(entry(PARENT/'manifest.json')['sha256']==PARENT_MANIFEST,'fixed arclength predecessor manifest')
    verify_inventory(PARENT,read(PARENT/'manifest.json')['files'])
    old=read(PARENT/'certificate.json.gz');result=old['result'];runtime=old['runtime']
    require(runtime['source_commit']=='11eb9a2dd76a00f592e1d1d28a52fc54cdcabc21','fixed predecessor execution')
    for name,pin in runtime['source_sha256'].items():
        require(entry(PARENT/'source'/name)['sha256']==pin,'archived predecessor source '+name)
        require(sha256(subprocess.check_output(['git','show',runtime['source_commit']+':'+name],cwd=ROOT))==pin,'committed predecessor source '+name)
    saved=read(PARENT/'inputs.json')
    require(saved['model_inputs']==original['model_inputs'],'identical original model/data')
    require(result['passed'] and len(result['charts'])==2 and all(all(c['gates'].values()) for c in result['charts']),'qualified connected primitive predecessor')
    first,last=result['charts'];require(first['normal']==last['normal']==saved['normal'] and first['origin']==last['origin']==saved['origin'],'same fixed gauge identity')
    require(last['statement']['period']==18 and last['statement']['phase_dimension']==13 and last['settings']==SETTINGS,'fixed predecessor dimensions/settings')
    require(last['anchor']['exact_parameter']==first['exact_parameter_endpoints'][1] and last['anchor']['state_box']==first['endpoint_root_enclosures'][1],'predecessor exact endpoint join')
    return {'model_inputs':original['model_inputs'],'previous':compact_chart(last),
            'provenance':{'parent_manifest_sha256':PARENT_MANIFEST,'parent_certificate':entry(PARENT/'certificate.json.gz'),
              'parent_execution':runtime['source_commit'],'original_anchor':original['provenance'],
              'native_order':'unchanged native phases0..17 followed by hidden rate; fixed original output',
              'gauge_identity':'same exact normal/origin for all batch charts and predecessor; exact_parameter only from that certified predecessor'}}


def profile_input():
    from flint import arb,arb_mat
    from neural_shrimp.rigorous_orbits import ball_json,vector_json,matrix_json,exact
    old=synthetic();old['model_inputs']['lr_hidden']=old['model_inputs']['lr_output']='1/4'
    initial=[arb(0)]*234+[exact('1/4')];delta=exact('1/64')
    previous={'passed':False,'gates':{'uniform_root':True,'primitive_period':False,'anchor_identified':True},
        'statement':{'period':18,'phase_dimension':13},'settings':SETTINGS,'normal':['0']*234+['1'],
        'origin':['0']*234+['1/4'],'center':vector_json(initial),'parameter_center':'0',
        'exact_parameter_endpoints':['-1/64','1/64'],'predictor_slope':matrix_json(arb_mat([[int(i==234)] for i in range(235)])),
        'predictor_curvature':matrix_json(arb_mat(235,1)),'exact_weights':prepare_weights()+['1'],
        'endpoint_root_enclosures':[vector_json(initial[:-1]+[initial[-1]+sign*delta]) for sign in (-1,1)]}
    return {'model_inputs':old['model_inputs'],'previous':previous,'provenance':old['provenance']}


def profile_environment(runtime,result):
    elapsed=result['elapsed_seconds'];rss=result['max_rss_kib']
    require(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<20,'finite nonnegative bounded profile time')
    require(type(rss) in (int,float) and math.isfinite(rss) and 0<=rss<2*1024**2,'finite nonnegative bounded profile RSS')
    require(result['exit_code']==0 and runtime['mode']=='profile','successful profile worker')
    expected={'python':platform.python_version(),'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__,
        'numpy':numpy.__version__,'scipy':scipy.__version__,'internal_seconds':20,'tick_cap':300,'map_cap':144,'unit_derivative_cap':288,'jet_cap':108}
    require(all(runtime.get(k)==v for k,v in expected.items()),'profile library/settings equality')
    require(runtime['threads']==dict.fromkeys(THREADS,'1') and runtime['nice']>=10,'profile threads/nice')


def profile_gate(path,pin):
    require(entry(path/'manifest.json')['sha256']==pin,'approved profile manifest')
    verify_inventory(path,read(path/'manifest.json')['files'])
    result=read(path/'outcome.json');runtime=read(path/'runtime-start.json')
    profile_environment(runtime,result)
    charts=read(path/'batch.json')['charts']
    require(len(charts)==2 and all(c['gates']['uniform_root'] for c in charts),'profile covers both quadratic residual and root stages')
    require(result['adapter_counts']=={'map_attempted':144,'map_returned':144,'unit_attempted':288,'unit_returned':288,'map_cap':144,'unit_cap':288},'complete C1 derivative profile'
    )
    require(result['jet_counts']=={'attempted':108,'returned':108,'cap':108},'complete path-jet profile')
    require(result['status']=='synthetic_profile_completed' and result['elapsed_seconds']<20,'completed profile timing')
    require(runtime['source_sha256']==sources(),'same profiled source bytes')
    require(16*result['elapsed_seconds']+5<180,'conservative profile projection below100 seconds')
    require(max(total(path)*16,16*18*1024**2)<352*1024**2,'storage estimate includes real parent compression cost')
    return {'manifest_sha256':pin,'elapsed_seconds':result['elapsed_seconds'],
            'projected_seconds':16*result['elapsed_seconds']+5,'profile_bytes':total(path),'projected_bytes':max(total(path)*16,16*18*1024**2),
            'scope':'conservative engineering gate, not a proof or exact runtime prediction'}

def execute(out,mode,profile=None,profile_pin=None,bundle=None,bundle_pin=None):
    bound=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10 required')
    limits=20 if mode=='profile' else 180
    run_count=2 if mode=='profile' else 16;map_cap=72*run_count;jet_cap=54*run_count;tick_cap=300 if mode=='profile' else 2200
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();ticks=0;serial=0;model=None;jet_model=None
    runtime={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'source_sha256':bound,'mode':mode,'pid':os.getpid(),'ppid':os.getppid(),'started_utc':now(),
      'threads':{k:os.environ[k] for k in THREADS},'nice':os.getpriority(os.PRIO_PROCESS,0),
      'python':platform.python_version(),'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__,'numpy':numpy.__version__,'scipy':scipy.__version__,
      'internal_seconds':limits,'map_cap':map_cap,'unit_derivative_cap':2*map_cap,'jet_cap':jet_cap,'tick_cap':tick_cap,'raw_cap':CAP,'cpu_only':True,'paid':False}
    save(out/'runtime-start.json',runtime)
    for name in SOURCES:
        dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,limits)
    resource.setrlimit(resource.RLIMIT_FSIZE,(CAP,CAP))
    def tick():
        nonlocal ticks
        ticks+=1
        if ticks>tick_cap or stopped or time.monotonic()-start>limits:raise TimeoutError('tick/wall budget')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>2*1024**2:raise MemoryError('RSS cap')
        if total(out)>352*1024**2:raise OSError('ordinary raw allowance')
    def observe(kind,row):
        nonlocal serial
        name=f'stage-{serial:03d}-{kind}'
        if kind=='chart_completed':
            payload=save(out/(name+'.json.gz'),row);save(out/(name+'-payload.json'),payload)
        else:save(out/(name+'.json'),row)
        serial+=1;tick()
    outcome={'passed':False,'mode':mode,'status':'started'};code=2
    try:
        if mode=='target':save(out/'cost-gate.json',profile_gate(profile,profile_pin))
        inp=profile_input() if mode=='profile' else prepare()
        save(out/'inputs.json',inp)
        def emit_chart(index,row):
            name=f'chart-{index:03d}.json.gz';payload=save(out/name,row)
            save(out/f'chart-{index:03d}-payload.json',payload)
            identity={'path':name,**entry(out/name),'payload':payload}
            outcome['last_durable_chart']={'index':index,'passed':row['passed'],'identity':identity}
            return identity
        with ctx.workprec(SETTINGS['precision']):
            model=MeanValueHiddenRateMap(inp['model_inputs'],max_calls=map_cap);jet_model=CountedJetMap(inp['model_inputs'],jet_cap)
            result=stream_charts(model,jet_model,inp['previous'],count=run_count,synthetic_profile=mode=='profile',tick=tick,observe=observe,emit_chart=emit_chart)
        save(out/'batch.json',result)
        profile_complete=mode=='profile'
        outcome.update(passed=bool(result['passed']) and not profile_complete,
            status='synthetic_profile_completed' if profile_complete else ('joined_intervals_certified' if result['passed'] else 'inconclusive'),
            gates={'connected_primitive_branch':result['passed'],'requested_charts':run_count,'completed_charts':len(result['charts']),'attraction_evaluated':False},scientific_claim=result['passed'] and not profile_complete)
        code=0 if profile_complete or result['passed'] else 2
    except BaseException as exc:
        outcome.update(status='failed_or_interrupted',error={'type':type(exc).__name__,'message':str(exc)});save(out/'failure.json',outcome['error'])
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if model is not None:outcome['adapter_counts']=model.counts()
        if jet_model is not None:outcome['jet_counts']=jet_model.counts()
        outcome.update(elapsed_seconds=time.monotonic()-start,ticks=ticks,stages=serial,finished_utc=now(),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,exit_code=code)
        save(out/'outcome.json',outcome);files=inventory(out)
        save(out/'manifest.json',{'files':files,'total_bytes':sum(v['bytes'] for v in files.values()),'within_limit':total(out)<CAP})
    print(json.dumps(outcome),flush=True);return code


def launch(out,mode,profile=None,profile_pin=None,bundle=None,bundle_pin=None):
    bound=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'launcher nice10 required')
    seconds=25 if mode=='profile' else 195
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();proc=None
    meta={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'source_sha256':bound,'mode':mode,'started_utc':now(),'launcher_pid':os.getpid(),'load_average':os.getloadavg(),
       'meminfo':Path('/proc/meminfo').read_text(),'external_seconds':seconds,'kill_grace_seconds':5,'cpu_only':True}
    cmd=['timeout','--signal=TERM','--kill-after=5s',f'{seconds}s','nice','-n','10',sys.executable,
         str(Path(__file__).resolve()),'--execute','--mode',mode,'--out',str(out/'run')]
    for flag,value in [('--profile',profile),('--profile-manifest',profile_pin),('--bundle',bundle),('--manifest-sha256',bundle_pin)]:
        if value is not None:cmd.extend([flag,str(value)])
    meta['argv']=cmd;save(out/'launcher-start.json',meta)
    with (out/'process.log').open('xb') as log:
        try:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            meta['owned_group_pid']=proc.pid;save(out/'launcher-start.json',meta);code=proc.wait(timeout=seconds+7)
        except BaseException as exc:
            meta['error']={'type':type(exc).__name__,'message':str(exc)};code=124
        finally:
            if proc is not None:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:code=124
    live=[]
    if proc is not None:
        for q in Path('/proc').iterdir():
            if not q.name.isdigit():continue
            try:
                fields=(q/'stat').read_text().rsplit(')',1)[1].split()
                if int(fields[2])==proc.pid and fields[0]!='Z':live.append(int(q.name))
            except (OSError,ValueError):pass
    if live:code=124
    meta.update(exit_code=code,elapsed_seconds=time.monotonic()-start,finished_utc=now(),owned_live_pids=live)
    save(out/'launcher-result.json',meta)
    if (out/'run/manifest.json').exists():verify_inventory(out/'run',read(out/'run/manifest.json')['files'])
    files=inventory(out);require(total(out)<CAP+1024**2,'total launcher plus worker cap')
    save(out/'manifest.json',{'files':files,'total_bytes':sum(v['bytes'] for v in files.values()),'within_limit':total(out)<CAP+1024**2})
    for q in sorted(out.rglob('*'),reverse=True):q.chmod(0o555 if q.is_dir() else 0o444)
    out.chmod(0o555)
    print(json.dumps({'out':str(out),'exit_code':code,'manifest':entry(out/'manifest.json'),'elapsed_seconds':meta['elapsed_seconds']}),flush=True)
    return code


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['profile','target'],required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--profile',type=Path);p.add_argument('--profile-manifest');p.add_argument('--bundle',type=Path);p.add_argument('--manifest-sha256')
    a=p.parse_args();require((a.mode=='target')==(a.profile is not None and a.profile_manifest is not None),'target requires fixed profile pin')
    require(a.bundle is None and a.manifest_sha256 is None,'batch replay uses independent saved witness review')
    return (execute if a.execute else launch)(a.out,a.mode,a.profile,a.profile_manifest,a.bundle,a.manifest_sha256)
if __name__=='__main__':raise SystemExit(main())
