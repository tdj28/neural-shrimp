"""Two fixed saved-jet norm-cocycle cases; no model/derivative acquisition."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,json,resource,signal,subprocess,sys,time
from fractions import Fraction
from pathlib import Path
from flint import ctx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
import certify_s2_period18_correlated_cocycle as io
from neural_shrimp.rigorous_norm_cocycle import framed_cocycle
from neural_shrimp.rigorous_connection_frames import restore_matrix
from neural_shrimp.rigorous_orbits import exact
from neural_shrimp.rigorous_connection import require
CARD='docs/plans/s2-period18-norm-cocycle.md'
SOURCES=list(dict.fromkeys(io.SOURCES+['src/neural_shrimp/rigorous_norm_cocycle.py','tools/certify_s2_period18_norm_cocycle.py','tools/launch_s2_period18_norm_cocycle.py','tests/test_rigorous_norm_cocycle.py','tests/test_s2_period18_norm_cocycle.py',CARD]))
PARENT=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-correlated-cocycle-2026-10-10/attempt-001-target/run')
PIN='dafbf4da8cf43a60b4c65883aa2d6874b756a788af97925fc531a0c810ea364b'
EXECUTION='4ef38b0530ada42ec8164777d136768627989eed'
io.CAP=32*1024**2

class DurabilityError(RuntimeError):pass

def cases():
    center=Fraction(1439,4096)
    return [{'role':role,'halfwidth':str(delta),'exact_endpoints':[str(center-delta),str(center+delta)]}
        for role,delta in [('full',Fraction(1,4096)),('quarter',Fraction(1,16384))]]


def load():
    rt,old=io.authenticate(PARENT,PIN,EXECUTION)
    require(rt['mode']=='target' and old['completed_cases']==1 and old['completed_products']==18 and old['mixed_counts']['returned']==18,'complete fixed derivative acquisition')
    # Saved point metrics are fixed frame proposals, not assumed proof for
    # the newly transformed interval. Every exact frame is revalidated.
    prt,po=io.authenticate(io.PARENT,io.PIN,io.EXECUTION)
    require(prt['mode']=='target' and po['completed_cases']==2,'original fixed periodic metric input')
    prior=io.read(io.PARENT/'probe-result.json.gz');native=io.read(PARENT/'probe-result.json.gz');inp=io.read(PARENT/'inputs.json.gz')
    require(prior['localization']['exact_endpoints']==native['exact_endpoints']==io.ENDPOINTS,'same closed gauge interval')
    require(native['scientific_claim'] and inp['exact_endpoints']==io.ENDPOINTS and inp['uniform_tangent']['passed'],'authenticated true same-gauge derivative')
    require(prior['stability']['point_passed'] and len(prior['stability']['phases'])==18,'complete old point metrics')
    rows=[];bases=[]
    for j,phase in enumerate(prior['stability']['phases']):
        require(phase['phase']==j and phase['P_positive']['passed'],'native frame phase')
        bases.append(restore_matrix(phase['P_positive']['basis']))
        raw=io.read(PARENT/f'stage-{5+j:03d}-correlated_phase_model.json.gz');require(raw['phase']==j,'native mixed derivative phase')
        rows.append({'phase':j,'center_J':restore_matrix(raw['center_J']),'whole_Jprime':restore_matrix(raw['whole_total_Jprime'])})
    return rows,bases


def execute(out):
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source')
    require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    bound={}
    for name in SOURCES:
        data=(ROOT/name).read_bytes();require(data==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed source');bound[name]=io.sha256(data)
    out.mkdir(parents=True,exist_ok=False);started_at=time.monotonic();code=2;stopped=False;serial=0
    runtime={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':bound,
      'pid':os.getpid(),'ppid':os.getppid(),'started_utc':io.now(),'nice':os.getpriority(os.PRIO_PROCESS,0),'precision':256,
      'python_flint':io.flint.__version__,'flint':io.flint.__FLINT_VERSION__,'numpy':io.numpy.__version__,'scipy':io.scipy.__version__,
      'threads':{k:os.environ[k] for k in io.THREADS},'internal_seconds':20,'rss_cap_kib':1024**2,'raw_cap':32*1024**2,'parent_manifest':PIN,
      'fixed_cases':cases(),'model_calls':0,'mixed_derivative_calls':0}
    io.save(out/'runtime-start.json',runtime)
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,20)
    outcome={'status':'started','cases':[],'completed_phase_attempts':0,'completed_phase_returns':0,'scientific_claim':False}
    def tick():
        if stopped or time.monotonic()-started_at>=20:raise TimeoutError('saved norm-cocycle time cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=1024**2:raise MemoryError('saved norm-cocycle RSS cap')
        if io.total(out)>28*1024**2:raise OSError('ordinary raw cap')
    def emit(role):
        def observe(kind,row):
            nonlocal serial
            try:
                name=f'stage-{serial:03d}-{role}-{kind}';payload=io.save(out/(name+'.json.gz'),row);io.save(out/(name+'-payload.json'),payload);serial+=1
                if kind=='norm_phase_start':outcome['completed_phase_attempts']+=1
                if kind=='norm_phase':outcome['completed_phase_returns']+=1
            except Exception as e:raise DurabilityError('failed durable output: '+str(e))from e
            tick()
        return observe
    try:
        with ctx.workprec(256):
            rows,bases=load();tick()
            for setting in cases():
                role=setting['role'];io.save(out/(role+'-start.json'),setting)
                result=None
                try:
                    result=framed_cocycle(rows,bases,exact(setting['halfwidth']),tick=tick,observe=emit(role))
                    summary={'role':role,'passed':result['passed'],'point_passed':result['metric']['point_D_positive']['passed'],'complete_decrease':result['metric']['complete_D_positive']['passed']}
                except (ValueError,ZeroDivisionError) as e:
                    summary={'role':role,'passed':False,'error':{'type':type(e).__name__,'message':str(e)}}
                # Persistence errors never become a continuable numerical case.
                if result is not None:
                    result['exact_endpoints']=setting['exact_endpoints'];payload=io.save(out/(role+'-result.json.gz'),result);io.save(out/(role+'-result-payload.json'),payload)
                else:io.save(out/(role+'-failure.json'),summary)
                outcome['cases'].append(summary);tick()
        code=0;outcome.update(status='fixed_saved_norm_cocycle_completed',scientific_claim=False,
          scope='Producer sufficient-bound result pending independent review; no fresh derivative or whole-branch claim.')
    except BaseException as exc:outcome.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)})
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);outcome.update(elapsed_seconds=time.monotonic()-started_at,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,exit_code=code,finished_utc=io.now(),stages=serial)
        io.save(out/'outcome.json',outcome);io.save(out/'manifest.json',{'files':io.inventory(out)})
        for p in out.rglob('*'):p.chmod(0o444 if p.is_file() else 0o555)
        out.chmod(0o555)
    print(json.dumps(outcome));return code
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();raise SystemExit(execute(a.out))
