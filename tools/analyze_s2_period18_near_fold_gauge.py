"""Fixed saved-only near-fold gauge proposals; never an existence or connection test."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,json,resource,signal,subprocess,sys,time
from pathlib import Path
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
import certify_s2_period18_fold_small_march as io
import analyze_s2_period18_gauge as old
from neural_shrimp.rigorous_connection import require,restore_ball
from neural_shrimp.rigorous_connection_frames import fraction_mid,restore_matrix
from neural_shrimp.rigorous_orbits import exact,ball_json,vector_json,midpoint_matrix,norm_inf_upper
CARD='docs/plans/s2-period18-near-fold-gauge.md'
CURRENT='tools/analyze_s2_period18_near_fold_gauge.py'
SOURCES=list(dict.fromkeys(io.SOURCES+old.SOURCES+[CURRENT,'tools/launch_s2_period18_near_fold_gauge.py','tests/test_s2_period18_near_fold_gauge.py',CARD]))
PARENT=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-fold-small-march-2026-10-10/unit-001/run')
PIN='a5399a6b1888006dc6ca2b32e8ae0f7beb367b4281c66d55ab20f33b7f8ec3d2'
EXECUTION='e987c8669efb5fb9275046f401cd4cfa172cfa9b'
for name in ('entry','read','verify_inventory'):globals()[name]=getattr(io,name)
def validate_prefix(inp,rows,load,expected=10):
    """Read all durable native joins; never use the unqualified terminal state."""
    require(len(rows)==expected,'exact qualified prefix count')
    previous=inp['previous'];finals=[];initials=[]
    for index,row in enumerate(rows):
        name=f'chart-{index:03d}.json.gz'
        require(row['index']==index and row['artifact']['path']==name,'ordered durable chart index')
        c=load(name,row['artifact']);io.joined(previous,c)
        require(c['passed']==row['passed'] and c['gates']==row['gates'],'mirrored durable gates')
        require(not row['known_fold_identified'] and not any(a['contained_in_qualified_chart'] for a in c['known_fold_membership']['all_cyclic_alignments']),'not already joined to fold')
        require(row['reboxed']==('reboxing' in c),'mirrored rebox status')
        if row['reboxed']:
            ident=c['reboxing']['initial_artifact'];require(ident['path']==f'initial-{index:03d}.json.gz','exact retained initial name')
            old=load(ident['path'],ident)
            require(old['gates']=={'uniform_root':False,'primitive_period':True,'anchor_identified':True},'root-only initial failure')
            for key in ('center','normal','origin','parameter_center','predictor_slope','predictor_curvature','preconditioner','predictor_residual','center_path_derivatives','whole_path_third_derivative','anchor'):
                require(c[key]==old[key],'unchanged rebox predictor '+key)
            initials.append(ident['path'])
        finals.append(name);previous=io.compact_chart(c)
    return previous,finals,initials


def partial_input():
    require(entry(PARENT/'manifest.json')['sha256']==PIN,'exact incomplete-parent manifest')
    verify_inventory(PARENT,read(PARENT/'manifest.json')['files'])
    runtime=read(PARENT/'runtime-start.json');io.bound_sources(PARENT,runtime,EXECUTION)
    require(runtime['executed_runner']==io.CURRENT and runtime['source_sha256'].get(io.CURRENT)==entry(ROOT/io.CURRENT)['sha256'],'exact parent runner')
    require(not (PARENT/'batch.json').exists(),'preserved absent terminal batch')
    outcome=read(PARENT/'outcome.json')
    require(outcome['exit_code']==2 and not outcome['passed'] and outcome['error']=={'type':'ValueError','message':'nonnegative finite radii proposal'},'exact retained terminal proposal failure')
    inp=read(PARENT/'inputs.json');original,endpoint=io.initial_anchor()
    require(inp['model_inputs']==original['model_inputs'] and inp['previous']==endpoint,'exact initial new-gauge anchor and model')
    require(inp['provenance']['parent_root']==str(io.INITIAL) and inp['provenance']['parent_manifest']==io.INITIAL_PIN and inp['provenance']['parent_execution']==io.INITIAL_EXECUTION,'exact initial dependency')
    rows=[read(q) for q in sorted(PARENT.glob('stage-*-chart_committed.json'))]
    def load(name,identity):
        require(entry(PARENT/name)=={k:identity[k] for k in ('sha256','bytes')},'bound retained artifact '+name)
        return read(PARENT/name)
    previous,finals,initials=validate_prefix(inp,rows,load)
    require(outcome['last_durable_chart']['index']==9 and outcome['last_durable_chart']['identity']==rows[-1]['artifact'],'last durable successful chart')
    terminal=outcome['last_durable_initial'];require(terminal['index']==10 and terminal['identity']['path']=='initial-010.json.gz','terminal failed initial identity')
    failed=load('initial-010.json.gz',terminal['identity'])
    require(not failed['passed'] and failed['gates']=={'uniform_root':False,'primitive_period':True,'anchor_identified':True},'unqualified tail stays excluded')
    require(failed['anchor']['state_box']==previous['endpoint_root_enclosures'][1] and failed['anchor']['exact_parameter']==previous['exact_parameter_endpoints'][1]=='45/128','failed tail references qualified endpoint only')
    require({p.name for p in PARENT.glob('chart-*.json.gz')}==set(finals),'exact final chart set')
    require({p.name for p in PARENT.glob('initial-*.json.gz')}==set(initials+['initial-010.json.gz']),'exact retained initial set')
    return {'model_inputs':inp['model_inputs'],'previous':previous,'provenance':{'parent_root':str(PARENT),'parent_manifest':PIN,'parent_execution':EXECUTION,'qualified_prefix':10,'last_chart':rows[-1]['artifact'],'failed_tail':terminal,'original_anchor':inp['provenance']['original_anchor'],'initial_repaired_manifest':io.INITIAL_PIN}}



def unit_normal(vector):
    require(vector and all(v.is_finite() for v in vector),'finite proposed vector')
    length=sum((v*v for v in vector),arb(0)).sqrt()
    require(length>0,'nonzero proposed vector')
    return [exact(str(fraction_mid(v/length))) for v in vector]


def proposals(chart,foldbox):
    endpoint=[restore_ball(v) for v in chart['endpoint_root_enclosures'][1]]
    require(len(endpoint)==len(foldbox),'complete physical dimensions')
    origin=[v.mid() for v in endpoint]
    tangent=old.gauge_proposal(chart)
    # This is explicitly informed by the unassociated fold. It neither replaces
    # either whole root box nor provides an identity test.
    secant=unit_normal([foldbox[i].mid()-origin[i] for i in range(len(origin))])
    choices=[('inherited',[exact(v) for v in chart['normal']]),
      ('endpoint_taylor',[exact(v) for v in tangent['normal']]),('target_informed_secant',secant)]
    rows=[]
    for label,normal in choices:
        rows.append({'label':label,'normal':[str(fraction_mid(v)) for v in normal],
          'origin':[str(fraction_mid(v)) for v in origin],
          'endpoint_gauge_box':ball_json(old.project(endpoint,normal,origin)),
          'fold_gauge_box':ball_json(old.project(foldbox,normal,origin)),
          'normal_squared_norm':ball_json(sum((v*v for v in normal),arb(0))),
          'hidden_normal_component':ball_json(normal[-1]),
          'scope':'fixed exact dyadic proposal only; full physical endpoint and fold boxes retained'})
    return rows


def condition(chart,normal):
    point=midpoint_matrix(restore_matrix(chart['jacobian_tube']));size=point.nrows()
    require(point.ncols()==size==len(normal),'full bordered dimensions')
    matrix=arb_mat([[point[i,j] if i<size-1 else normal[j] for j in range(size)] for i in range(size)])
    try:
        inverse=matrix.inv()
        return {'invertible_saved_midpoint':True,'condition_inf_upper':ball_json((norm_inf_upper(matrix)*norm_inf_upper(inverse)).upper()),
          'inverse_inf_upper':ball_json(norm_inf_upper(inverse)),
          'scope':'same saved tube midpoint matrix with replaced gauge row; not fresh endpoint derivatives or whole-box conditioning'}
    except Exception as exc:return {'invertible_saved_midpoint':False,'error':str(exc)}


def execute(out):
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source')
    require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    bound={}
    for name in SOURCES:
        data=(ROOT/name).read_bytes();require(data==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed source');bound[name]=io.sha256(data)
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();code=2;stopped=False
    runtime={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':bound,
      'pid':os.getpid(),'ppid':os.getppid(),'started_utc':io.now(),'nice':os.getpriority(os.PRIO_PROCESS,0),'precision':256,
      'python_flint':io.flint.__version__,'flint':io.flint.__FLINT_VERSION__,'numpy':io.numpy.__version__,'scipy':io.scipy.__version__,
      'threads':{k:os.environ[k] for k in io.THREADS},'internal_seconds':30,'rss_cap_kib':1024**2,'raw_cap':3*1024**2,'parent_manifest':PIN}
    io.save(out/'runtime-start.json',runtime)
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,30)
    def tick():
        if stopped or time.monotonic()-start>=30:raise TimeoutError('saved diagnosis time cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=1024**2:raise MemoryError('saved diagnosis RSS cap')
    outcome={'status':'started','scientific_claim':False}
    try:
        with ctx.workprec(256):
            inp=partial_input();io.save(out/'inputs.json',inp);tick()
            last=read(PARENT/'chart-009.json.gz');fold=io.known_fold(inp['model_inputs'])
            foldbox=[restore_ball(v) for row in fold['orbit'] for v in row]+[restore_ball(fold['rate'])]
            io.save(out/'known-fold.json',fold);tick()
            choices=proposals(last,foldbox)
            for row in choices:
                normal=[exact(v) for v in row['normal']];origin=[exact(v) for v in row['origin']]
                tick();row['bordered_diagnostic']=condition(last,normal);tick()
                row['all_cyclic_projections']=old.fold_projections(fold['orbit'],fold['rate'],normal,origin,
                  restore_ball(row['endpoint_gauge_box']),arb((1,-8)))
                io.save(out/(row['label']+'.json'),row)
        tick();code=0;outcome.update(status='saved_gauge_comparison_completed',choices=[r['label'] for r in choices])
    except BaseException as exc:outcome.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)})
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);outcome.update(elapsed_seconds=time.monotonic()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,exit_code=code,finished_utc=io.now())
        io.save(out/'outcome.json',outcome);require(io.total(out)<3*1024**2,'worker raw cap');io.save(out/'manifest.json',{'files':io.inventory(out)})
        for p in out.rglob('*'):p.chmod(0o444 if p.is_file() else 0o555)
        out.chmod(0o555)
    print(json.dumps(outcome));return code

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();raise SystemExit(execute(a.out))
