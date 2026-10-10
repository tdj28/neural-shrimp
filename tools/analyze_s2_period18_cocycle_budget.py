"""One pinned saved-cocycle decomposition, with no model or decomposition call."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,json,resource,signal,subprocess,sys,time
from pathlib import Path
from flint import ctx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
import certify_s2_period18_correlated_cocycle as io
from neural_shrimp.correlated_cocycle_budget import decompose
from neural_shrimp.rigorous_connection_frames import restore_matrix
from neural_shrimp.rigorous_orbits import exact
from neural_shrimp.rigorous_connection import require
CARD='docs/plans/s2-period18-cocycle-budget.md'
SOURCES=list(dict.fromkeys(io.SOURCES+['src/neural_shrimp/correlated_cocycle_budget.py','tools/analyze_s2_period18_cocycle_budget.py','tools/launch_s2_period18_cocycle_budget.py','tests/test_correlated_cocycle_budget.py',CARD]))
PARENT=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-correlated-cocycle-2026-10-10/attempt-001-target/run')
PIN='dafbf4da8cf43a60b4c65883aa2d6874b756a788af97925fc531a0c810ea364b'
EXECUTION='4ef38b0530ada42ec8164777d136768627989eed'

def execute(out):
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source')
    require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    bound={}
    for name in SOURCES:
        data=(ROOT/name).read_bytes();require(data==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed source');bound[name]=io.sha256(data)
    out.mkdir(parents=True,exist_ok=False);started_at=time.monotonic();code=2;stopped=False
    runtime={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':bound,
      'pid':os.getpid(),'ppid':os.getppid(),'started_utc':io.now(),'nice':os.getpriority(os.PRIO_PROCESS,0),'precision':256,
      'python_flint':io.flint.__version__,'flint':io.flint.__FLINT_VERSION__,'numpy':io.numpy.__version__,'scipy':io.scipy.__version__,
      'threads':{k:os.environ[k] for k in io.THREADS},'internal_seconds':15,'rss_cap_kib':1024**2,'raw_cap':7*1024**2,'parent_manifest':PIN}
    io.save(out/'runtime-start.json',runtime)
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,15)
    def tick():
        if stopped or time.monotonic()-started_at>=15:raise TimeoutError('saved diagnosis time cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=1024**2:raise MemoryError('saved diagnosis RSS cap')
    outcome={'status':'started','scientific_claim':False}
    try:
        require(io.entry(PARENT/'manifest.json')['sha256']==PIN,'exact failed-parent manifest');io.verify_inventory(PARENT,io.read(PARENT/'manifest.json')['files'])
        rt=io.read(PARENT/'runtime-start.json');old=io.read(PARENT/'outcome.json')
        require(rt['source_commit']==EXECUTION and rt['mode']=='target' and old['exit_code']==0 and old['completed_cases']==1 and old['completed_products']==18,'fixed completed case')
        for name,digest in rt['source_sha256'].items():
            require(io.entry(ROOT/name)['sha256']==digest and io.sha256(subprocess.check_output(['git','show',EXECUTION+':'+name],cwd=ROOT))==digest,'exact acquisition source')
        with ctx.workprec(256):
            result=io.read(PARENT/'probe-result.json.gz');metric=result['metric'];tick()
            require(result['exact_endpoints']==io.ENDPOINTS and metric['P_positive']['passed'] and metric['point_D_positive']['passed'],'fixed qualified point witness')
            rows=[]
            for j in range(18):
                raw=io.read(PARENT/f'stage-{5+j:03d}-correlated_phase_model.json.gz');require(raw['phase']==j,'native phase identity')
                rows.append({'phase':j,'coefficients':[restore_matrix(v) for v in raw['coefficients']],
                  'remainder':restore_matrix(raw['remainder']),'center_J':restore_matrix(raw['center_J']),
                  'scaled_Jprime':restore_matrix(raw['whole_total_Jprime'])*exact('1/4096')})
            value=decompose([restore_matrix(v) for v in result['coefficients']],restore_matrix(result['remainder']),restore_matrix(metric['P']),restore_matrix(metric['point_D_positive']['inverse']),rows,tick)
            io.save(out/'budget.json',value)
        code=0;outcome.update(status='saved_cocycle_budget_completed',phase_count=len(rows),scientific_claim=False)
    except BaseException as exc:outcome.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)})
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);outcome.update(elapsed_seconds=time.monotonic()-started_at,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,exit_code=code,finished_utc=io.now())
        io.save(out/'outcome.json',outcome);require(io.total(out)<7*1024**2,'worker raw cap');io.save(out/'manifest.json',{'files':io.inventory(out)})
        for p in out.rglob('*'):p.chmod(0o444 if p.is_file() else 0o555)
        out.chmod(0o555)
    print(json.dumps(outcome));return code
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();raise SystemExit(execute(a.out))
