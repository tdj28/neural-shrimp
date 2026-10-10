"""Exclusive continuation with at most one retained reboxing per chart."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for name in THREADS:os.environ[name]='1'
import argparse,gzip,json,math,platform,resource,signal,subprocess,sys,time
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path
import flint,numpy,scipy
from flint import ctx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
from analyze_s2_period18_march_failure import fold_alignments
from neural_shrimp.rigorous_orbits import canonical_bytes,sha256,check_orbit
from neural_shrimp.rigorous_connection import HiddenRateMap,require
from neural_shrimp.rigorous_arclength_batch import compact_chart
from neural_shrimp.rigorous_arclength_rebox import stream_reboxed
from neural_shrimp.rigorous_folds import FoldTanhMap
from neural_shrimp.rigorous_reboxing_gradient import ReboxingMeanValueHiddenRateMap
RECEIPT='certificates/s2-period18-receipt.json'
RECEIPT_SHA='f5b8ed6bc4bded4f6c3515fed5a92d6cfc12614e4e68f0169037810b0d317277'
CARD='docs/plans/s2-period18-arclength-reboxing.md'
SOURCES=['src/neural_shrimp/__init__.py','src/neural_shrimp/rigorous_orbits.py',
 'src/neural_shrimp/rigorous_connection.py','src/neural_shrimp/rigorous_connection_frames.py',
 'src/neural_shrimp/rigorous_folds.py','src/neural_shrimp/rigorous_flips.py',
 'src/neural_shrimp/controls.py','src/neural_shrimp/models.py','src/neural_shrimp/rigorous_gradient_enclosure.py',
 'src/neural_shrimp/rigorous_batched_gradient.py','src/neural_shrimp/rigorous_quadratic_connection.py','src/neural_shrimp/rigorous_arclength_connection.py','src/neural_shrimp/rigorous_arclength_batch.py',
 'tests/test_rigorous_quadratic_connection.py','tests/test_rigorous_arclength_connection.py','tests/test_rigorous_arclength_batch.py',
 'tools/certify_s2_period18_arclength_march.py','tools/certify_s2_period18_arclength_batch.py','tests/test_rigorous_arclength_march.py','docs/reviews/s2-period18-quadratic-weights.json','docs/reviews/s2-period18-weighted-outcome.json','docs/reviews/s2-period18-arclength-outcome.json',CARD,RECEIPT]
SOURCES += ['src/neural_shrimp/rigorous_adaptive_arclength.py','tools/certify_s2_period18_arclength_radii.py','tests/test_rigorous_adaptive_arclength.py','docs/plans/s2-period18-arclength-march.md','tools/analyze_s2_period18_march_failure.py','tools/launch_s2_period18_march_failure.py','tests/test_march_failure_diagnostic.py','docs/plans/s2-period18-march-failure-diagnostic.md']
SOURCES += ['src/neural_shrimp/rigorous_arclength_rebox.py','src/neural_shrimp/rigorous_reboxing_gradient.py','tools/certify_s2_period18_arclength_reboxing.py','tests/test_rigorous_arclength_rebox.py','docs/plans/s2-period18-arclength-radii.md','certificates/s2-period18-fold-receipt.json']
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

INITIAL=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-arclength-connection-2026-10-10/attempt-001-target/run')
INITIAL_MANIFEST='0460fd73c44e52b479538908e55df12c61a082e2f3541c3c54186bfd7750f71c'
PROOF_SOURCES=['src/neural_shrimp/'+name for name in ('rigorous_orbits.py','rigorous_connection.py','rigorous_connection_frames.py',
    'rigorous_folds.py','rigorous_flips.py','rigorous_gradient_enclosure.py','rigorous_arclength_connection.py',
    'rigorous_arclength_batch.py','rigorous_batched_gradient.py','models.py','controls.py','__init__.py')]


DIAG=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-march-diagnosis-2026-10-10/attempt-001/run')
DIAG_MANIFEST='dc4ad29ca305f45b3cc58438f30c1e588616cafb7014586b83995e15ec994235'
PARTIAL_ROOT=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-arclength-radii-2026-10-10/unit-002/run')
PARTIAL_MANIFEST='26cf51439d0d409dbe8b9d03c5b774805d5c93f836d3a3045681f3c065dc3135'
PARTIAL_EXECUTION='f5922aa97383e81ad455d79980b04b668d4e4fe8'


def prepare_radii():
    require(entry(DIAG/'manifest.json')['sha256']==DIAG_MANIFEST,'fixed radii proposal manifest')
    verify_inventory(DIAG,read(DIAG/'manifest.json')['files'])
    rt=read(DIAG/'runtime-start.json');outcome=read(DIAG/'outcome.json')
    require(rt['source_commit']=='f972ba604248bc05b13768e0a69b469d403d6b52' and outcome['exit_code']==0 and outcome['status']=='saved_diagnosis_completed','completed fixed saved diagnosis')
    for name,digest in rt['source_sha256'].items():
        require(entry(ROOT/name)['sha256']==digest,'unchanged diagnosis source '+name)
        require(sha256(subprocess.check_output(['git','show',rt['source_commit']+':'+name],cwd=ROOT))==digest,'committed diagnosis source '+name)
    proposal=read(DIAG/'failure-diagnostic.json')['proposal'];weights=proposal['exact_weights'];radius=proposal['radius_exponent']
    from neural_shrimp.rigorous_orbits import exact
    require(radius==-21 and len(weights)==235,'fixed proposed domain settings')
    values=[exact(v) for v in weights]
    require(all(v.is_exact() and v.is_finite() and 0<v<=1 for v in values),'positive exact proposed weights')
    require(all(str(Fraction(v)*Fraction(2)**radius)==proposal['exact_coordinate_radii'][i] for i,v in enumerate(weights)),'exact proposal normalization')
    return weights,radius


def validate_prefix(previous,rows,loader,*,expected_count,allow_last_failure,expected_settings,expected_weights):
    require(previous['passed'] and all(previous['gates'].values()),'proved inherited predecessor')
    require(len(rows)==expected_count,'fixed complete saved prefix inventory')
    gauge=(previous['normal'],previous['origin']);qualified=0
    for index,row in enumerate(rows):
        require(row['index']==index,'ordered saved charts')
        chart=loader(row)
        if expected_settings is not None:
            require(chart['settings']==expected_settings and chart['exact_weights']==expected_weights,'actual parent domain settings')
        else:
            st=chart['settings'];require(st['precision']==256 and st['halfwidth_exponent']==-6 and st['newton_steps']==2 and -100<=st['radius_exponent']<=-10,'bounded adapted parent settings')
            from neural_shrimp.rigorous_orbits import exact
            ww=[exact(v) for v in chart['exact_weights']];require(len(ww)==235 and all(v.is_exact() and v.is_finite() and 0<v<=1 for v in ww),'positive exact saved weights')
        require(chart['statement']['period']==18 and chart['statement']['phase_dimension']==13,'native18x13 predecessor')
        require((chart['normal'],chart['origin'])==gauge,'fixed gauge')
        require(chart['anchor']['state_box']==previous['endpoint_root_enclosures'][1] and chart['anchor']['exact_parameter']==previous['exact_parameter_endpoints'][1]==chart['exact_parameter_endpoints'][0],'full native exact shared endpoint')
        require(chart['passed']==row['passed'] and chart['gates']==row['gates'],'mirrored saved verdict')
        if not chart['passed']:
            require(allow_last_failure and index==expected_count-1 and chart['gates']=={'uniform_root':False,'primitive_period':True,'anchor_identified':True},'only retained final root failure')
            # Deliberately do not replace previous with this unqualified chart.
            continue
        require(all(chart['gates'].values()) and all(chart['coordinate_inclusions']) and chart['anchor_check']['parameter_inside'] and all(chart['anchor_check']['coordinate_inclusions']),'all predecessor proof gates')
        require([v['divisor'] for v in chart['divisors'] if v['excluded']]==[1,2,3,6,9],'primitive18 exclusions')
        previous=compact_chart(chart);qualified+=1
    return previous,qualified


def prepare(parent,pin,execution):
    import re
    require(parent is not None and re.fullmatch('[0-9a-f]{64}',pin or '') and re.fullmatch('[0-9a-f]{40}',execution or ''),'explicit complete predecessor pins')
    original=original_anchor();weights,radius=prepare_radii()
    require(entry(parent/'manifest.json')['sha256']==pin,'fixed predecessor manifest')
    verify_inventory(parent,read(parent/'manifest.json')['files'])
    runtime=read(parent/'runtime-start.json');outcome=read(parent/'outcome.json');batch=read(parent/'batch.json');saved=read(parent/'inputs.json')
    require(runtime['source_commit']==execution and runtime['mode']=='target','exact predecessor execution')
    for name,value in runtime['source_sha256'].items():
        require(entry(parent/'source'/name)['sha256']==value,'archived predecessor source '+name)
        require(sha256(subprocess.check_output(['git','show',execution+':'+name],cwd=ROOT))==value,'committed predecessor source '+name)
    for name in PROOF_SOURCES:require(runtime['source_sha256'].get(name)==entry(ROOT/name)['sha256'],'unchanged proof source '+name)
    current='tools/certify_s2_period18_arclength_reboxing.py';old='tools/certify_s2_period18_arclength_radii.py'
    if current in runtime['source_sha256']:
        require(runtime['source_sha256'][current]==entry(ROOT/current)['sha256'] and runtime['executed_runner']==current,'unchanged actual radii runner')
        for name in ('src/neural_shrimp/rigorous_arclength_rebox.py','src/neural_shrimp/rigorous_reboxing_gradient.py'):
            require(runtime['source_sha256'].get(name)==entry(ROOT/name)['sha256'],'unchanged reboxing source '+name)
        require(outcome['exit_code']==0 and outcome['passed'] and batch['passed'] and batch['completed_charts']==batch['qualified_charts']==16,'complete later adapted batch')
        count=16;partial=False;settings=None;old_weights=None
    else:
        require(parent==PARTIAL_ROOT and pin==PARTIAL_MANIFEST and execution==PARTIAL_EXECUTION,'only exact recorded partial batch')
        require(runtime['source_sha256'].get(old)==entry(ROOT/old)['sha256'] and runtime['executed_runner']==old,'exact old march runner')
        require(outcome['exit_code']==2 and not outcome['passed'] and outcome['status']=='inconclusive' and not batch['passed'] and batch['completed_charts']==2 and batch['qualified_charts']==1,'explicit retained first failure')
        count=2;partial=True;settings={**SETTINGS,'radius_exponent':radius};old_weights=weights
    require(saved['model_inputs']==original['model_inputs'] and saved['provenance']['original_anchor']==original['provenance'],'identical model/data lineage')
    require(entry(INITIAL/'manifest.json')['sha256']==INITIAL_MANIFEST,'fixed initial gauge manifest');verify_inventory(INITIAL,read(INITIAL/'manifest.json')['files'])
    initial=read(INITIAL/'inputs.json');require(saved['previous']['normal']==initial['normal'] and saved['previous']['origin']==initial['origin'],'original exact gauge')
    def loader(row):
        name=row['artifact']['path'];require(name==f"chart-{row['index']:03d}.json.gz",'exact chart path')
        require(entry(parent/name)=={k:row['artifact'][k] for k in ('sha256','bytes')},'bound chart artifact')
        chart=read(parent/name)
        if 'reboxing' in chart:
            initial=chart['reboxing']['initial_artifact'];iname=f"initial-{row['index']:03d}.json.gz"
            require(initial['path']==iname and entry(parent/iname)=={k:initial[k] for k in ('sha256','bytes')},'retained initial identity')
            trial=read(parent/iname)
            for key in ('center','normal','origin','parameter_center','predictor_slope','predictor_curvature','preconditioner','predictor_residual','center_path_derivatives','whole_path_third_derivative','anchor'):
                require(chart[key]==trial[key],'unchanged reboxed predictor/Taylor/anchor '+key)
            require(not trial['gates']['uniform_root'] and trial['gates']['primitive_period'] and trial['gates']['anchor_identified'],'only one failed root trial adapted')
        return chart
    last,qualified=validate_prefix(saved['previous'],batch['charts'],loader,expected_count=count,allow_last_failure=partial,expected_settings=settings,expected_weights=old_weights)
    require(qualified==(1 if partial else 16),'exact qualified prefix count')
    return {'model_inputs':original['model_inputs'],'previous':last,
      'provenance':{'parent_root':str(parent),'parent_manifest_sha256':pin,'parent_execution':execution,'qualified_parent_charts':qualified,
       'retained_unqualified_tail':1 if partial else 0,'proposal_manifest_sha256':DIAG_MANIFEST,'original_anchor':original['provenance'],
       'native_order':'unchanged phases0..17 and actual hidden rate; failed chart cannot anchor','scope':'same original gauge/output; freshly acquired new error domains'}}


def known_fold(model_inputs):
    receipt=read(ROOT/'certificates/s2-period18-fold-receipt.json');require(receipt['passed'],'qualified canonical fold')
    root=Path(receipt['external_root'])
    for name,identity in receipt['files'].items():require(entry(root/name)==identity,'bound fold input '+name)
    envelope=read(root/'run/certificate.json');fold=envelope['result'];inputs=read(root/'run/inputs.json')
    require(fold['passed'] and all(fold['gates'].values()) and inputs['period']==18,'all local fold gates')
    require(all(inputs['map_inputs'][k]==model_inputs[k] for k in ('schema','network','x','y','lr_output')),'identical fold map/data/output')
    return {'orbit':fold['orbit_enclosures'][:18],'rate':fold['box'][13],
        'identity':{'canonical_receipt':entry(ROOT/'certificates/s2-period18-fold-receipt.json'),
                    'certificate':entry(root/'run/certificate.json'),'inputs':entry(root/'run/inputs.json')}}


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
        'numpy':numpy.__version__,'scipy':scipy.__version__,'internal_seconds':20,'tick_cap':400,'map_cap':180,'unit_derivative_cap':360,'jet_cap':108}
    require(all(runtime.get(k)==v for k,v in expected.items()),'profile library/settings equality')
    require(runtime['threads']==dict.fromkeys(THREADS,'1') and runtime['nice']>=10,'profile threads/nice')


def profile_gate(path,pin):
    require(entry(path/'manifest.json')['sha256']==pin,'approved profile manifest')
    verify_inventory(path,read(path/'manifest.json')['files'])
    result=read(path/'outcome.json');runtime=read(path/'runtime-start.json')
    profile_environment(runtime,result)
    charts=read(path/'batch.json')['charts']
    require(len(charts)==2 and all(c['gates']['uniform_root'] and c['reboxed'] for c in charts),'profile covers both initial and reboxed root stages')
    require(result['adapter_counts']=={'map_attempted':180,'map_returned':180,'unit_attempted':360,'unit_returned':360,'map_cap':180,'unit_cap':360},'complete C1 derivative profile'
    )
    require(result['jet_counts']=={'attempted':108,'returned':108,'cap':108},'complete path-jet profile')
    require(result['status']=='synthetic_profile_completed' and result['elapsed_seconds']<20,'completed profile timing')
    require(runtime['source_sha256']==sources(),'same profiled source bytes')
    require(16*result['elapsed_seconds']+5<180,'conservative profile projection below180 seconds')
    require(max(total(path)*16,16*20*1024**2)<352*1024**2,'storage estimate includes real parent compression cost')
    return {'manifest_sha256':pin,'elapsed_seconds':result['elapsed_seconds'],
            'projected_seconds':16*result['elapsed_seconds']+5,'profile_bytes':total(path),'projected_bytes':max(total(path)*16,16*20*1024**2),
            'scope':'conservative engineering gate, not a proof or exact runtime prediction'}

def execute(out,mode,profile=None,profile_pin=None,parent=None,parent_pin=None,parent_execution=None):
    bound=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10 required')
    limits=20 if mode=='profile' else 180
    run_count=2 if mode=='profile' else 16;map_cap=90*run_count;jet_cap=54*run_count;tick_cap=400 if mode=='profile' else 3200
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();ticks=0;serial=0;model=None;jet_model=None
    runtime={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'source_sha256':bound,'executed_runner':str(Path(__file__).resolve().relative_to(ROOT)),'argv':sys.argv,'mode':mode,'pid':os.getpid(),'ppid':os.getppid(),'started_utc':now(),
      'threads':{k:os.environ[k] for k in THREADS},'nice':os.getpriority(os.PRIO_PROCESS,0),
      'predecessor':{'root':str(parent) if parent is not None else None,'manifest_sha256':parent_pin,'execution':parent_execution},
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
        inp=profile_input() if mode=='profile' else prepare(parent,parent_pin,parent_execution)
        save(out/'inputs.json',inp)
        fold=known_fold(inp['model_inputs']) if mode=='target' else None
        if fold is not None:save(out/'fold-input-identity.json',fold['identity'])
        def emit_chart(index,row):
            if fold is not None:
                row['known_fold_membership']={'identity':fold['identity'],'all_cyclic_alignments':fold_alignments(row,fold['orbit'],fold['rate']),
                  'scope':'full correlated fold-root containment in this qualified chart; no attraction or whole geometry'}
            name=f'chart-{index:03d}.json.gz';payload=save(out/name,row)
            save(out/f'chart-{index:03d}-payload.json',payload)
            identity={'path':name,**entry(out/name),'payload':payload}
            outcome['last_durable_chart']={'index':index,'passed':row['passed'],'identity':identity}
            return identity
        def emit_initial(index,row):
            name=f'initial-{index:03d}.json.gz';payload=save(out/name,row);save(out/f'initial-{index:03d}-payload.json',payload)
            identity={'path':name,**entry(out/name),'payload':payload}
            outcome['last_durable_initial']={'index':index,'identity':identity};return identity
        with ctx.workprec(SETTINGS['precision']):
            model=ReboxingMeanValueHiddenRateMap(inp['model_inputs'],max_calls=map_cap);jet_model=CountedJetMap(inp['model_inputs'],jet_cap)
            result=stream_reboxed(model,jet_model,inp['previous'],count=run_count,synthetic_profile=mode=='profile',force_profile_rebox=mode=='profile',tick=tick,observe=observe,emit_initial=emit_initial,emit_chart=emit_chart)
        save(out/'batch.json',result)
        profile_complete=mode=='profile'
        outcome.update(passed=bool(result['passed']) and not profile_complete,
            status='synthetic_profile_completed' if profile_complete else ('joined_intervals_certified' if result['passed'] else 'inconclusive'),
            gates={'connected_primitive_branch':result['passed'],'requested_charts':run_count,'completed_charts':len(result['charts']),'attraction_evaluated':False,'known_fold_identified':result.get('known_fold_identified',False)},scientific_claim=result['passed'] and not profile_complete)
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


def launch(out,mode,profile=None,profile_pin=None,parent=None,parent_pin=None,parent_execution=None):
    bound=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'launcher nice10 required')
    seconds=25 if mode=='profile' else 195
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();proc=None
    meta={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'source_sha256':bound,'mode':mode,'started_utc':now(),'launcher_pid':os.getpid(),'load_average':os.getloadavg(),
       'meminfo':Path('/proc/meminfo').read_text(),'external_seconds':seconds,'kill_grace_seconds':5,'cpu_only':True}
    cmd=['timeout','--signal=TERM','--kill-after=5s',f'{seconds}s','nice','-n','10',sys.executable,
         str(Path(__file__).resolve()),'--execute','--mode',mode,'--out',str(out/'run')]
    for flag,value in [('--profile',profile),('--profile-manifest',profile_pin),('--parent',parent),('--parent-manifest',parent_pin),('--parent-execution',parent_execution)]:
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
    p.add_argument('--execute',action='store_true');p.add_argument('--profile',type=Path);p.add_argument('--profile-manifest')
    p.add_argument('--parent',type=Path);p.add_argument('--parent-manifest');p.add_argument('--parent-execution')
    a=p.parse_args();require((a.mode=='target')==(a.profile is not None and a.profile_manifest is not None),'target requires fixed profile pin')
    require((a.mode=='target')==(a.parent is not None and a.parent_manifest is not None and a.parent_execution is not None),'target requires exact separate predecessor pins')
    return (execute if a.execute else launch)(a.out,a.mode,a.profile,a.profile_manifest,a.parent,a.parent_manifest,a.parent_execution)
if __name__=='__main__':raise SystemExit(main())
