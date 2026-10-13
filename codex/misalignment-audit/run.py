"""Single bounded, source/card-bound replay; prepared cards never authorize runs."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))  # explicit own imports when the supervisor uses python -I
ALLOWED=Path('/data2/neural-shrimp-data/codex/llm-misalignment-audit')
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(1024*1024): h.update(b)
    return h.hexdigest()


def write(path,obj):
    with path.open('x') as f:
        json.dump(obj,f,indent=2,allow_nan=False)
        f.write('\n')


def runtime():
    import numpy, scipy, sklearn
    from threadpoolctl import threadpool_info
    return {'python':sys.version,'numpy':numpy.__version__,'scipy':scipy.__version__,
            'sklearn':sklearn.__version__,'pid':os.getpid(),'ppid':os.getppid(),
            'nice':os.getpriority(os.PRIO_PROCESS,0),'thread_env':{k:os.environ.get(k) for k in THREADS},
            'threadpools':threadpool_info(),'rlimit_as':resource.getrlimit(resource.RLIMIT_AS),
            'proc_stat':Path('/proc/self/stat').read_text(),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}


def configure(card):
    if card['status']!='accepted_by_root' or not card.get('acceptance_reference') or not card.get('source_commit'):
        raise ValueError('root acceptance and checkpoint required')
    if card['cpu_threads']!=1 or card['nice']!=10 or card['as_bytes']!=4*1024**3:
        raise ValueError('fixed CPU1/nice10/4GiB envelope')
    if any(os.environ.get(k)!='1' for k in THREADS):
        raise ValueError('all thread env limits must be set before Python starts')
    nice=os.getpriority(os.PRIO_PROCESS,0)
    if nice>10: raise ValueError('exact nice10 required')
    if nice<10: os.nice(10-nice)
    hard=resource.getrlimit(resource.RLIMIT_AS)[1]
    limit=card['as_bytes']
    if hard!=resource.RLIM_INFINITY and hard<limit: raise ValueError('inherited hard AS too small')
    resource.setrlimit(resource.RLIMIT_AS,(limit,limit))
    for rel,expected in card['source_sha256'].items():
        p=HERE/rel
        if p.resolve().parent!=HERE or sha(p)!=expected:
            raise ValueError('source identity mismatch '+rel)
    if not set(('run.py','launch.py','reducers.py','inputs.py','detection.py','descriptive.py',
                'schemas.json','claims.json','plan.md')).issubset(card['source_sha256']):
        raise ValueError('incomplete source closure')


def execute(mode,out,card):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from inputs import inventory,files_for,load_detection,TRAIN,TEST
    from detection import make_table,evaluate_l7,evaluate_l8,fit_pca,fit_logistic,score_model
    from reducers import increment
    import descriptive
    with threadpool_limits(limits=1):
        write(out/'runtime.json',runtime())
        if mode=='profile-descriptive':
            start=time.monotonic(); result=descriptive.calculate(descriptive.synthetic())
            write(out/'profile.json',{'seconds':time.monotonic()-start,'synthetic_only':True,
                'arms':33,'draws':4000,'result_row_count':sum(len(x) for x in list(result.values())[:5])})
            return
        if mode=='profile-detection':
            rng=np.random.default_rng(7261)
            start=time.monotonic(); diagnostics=[]
            # Maximum L7 rows at h=1; held-out set has maximum pooled L8 rows.
            train=rng.normal(size=(19968,896)); y=rng.integers(0,2,len(train))
            pca=fit_pca(train,diagnostics); z=pca.transform(train)
            x=np.column_stack((rng.normal(size=(len(train),6)),z))
            model=fit_logistic(x,y,diagnostics); score_model(model,x)
            fit_seconds=time.monotonic()-start
            del train,z,x
            n=39936; groups=np.repeat(np.arange(1024),39)
            labels=rng.integers(0,2,n); a=rng.normal(size=n); b=a+.1*rng.normal(size=n)
            start=time.monotonic(); increment(labels,groups,a,b,2000)
            write(out/'profile.json',{'fit_seconds':fit_seconds,'bootstrap_seconds':time.monotonic()-start,
                'synthetic_only':True,'train_shape':[19968,896],'test_rows':n,'diagnostics':diagnostics})
            return
        paths=files_for(mode)
        if not card.get('input_manifest'):
            raise ValueError('explicit pre-read input pins required')
        actual=inventory(paths)
        if actual!=card['input_manifest']:
            raise ValueError('input manifest differs')
        write(out/'input-manifest.json',actual)
        if mode=='descriptive':
            result=descriptive.calculate(descriptive.load())
            write(out/'descriptive.json',result)
            return
        train=[load_detection(a) for a in TRAIN]
        test=[load_detection(a) for a in TEST] if mode=='l8' else None
        write(out/'actual-sample-args.json',{a['name']:a['args'] for a in train+(test or [])})
        for event in (1,2):
            for h in (1,3,5):
                tab=make_table(train,h,event)
                key=f'event{event}-h{h}'
                if mode=='l7':
                    rows,scores,fold,diagnostics=evaluate_l7(tab)
                    np.savez_compressed(out/(key+'-scores.npz'),y=tab['y'],group=tab['group'],fold=fold,**scores)
                else:
                    other=make_table(test,h,event)
                    rows,scores,diagnostics=evaluate_l8(tab,other)
                    np.savez_compressed(out/(key+'-scores.npz'),y=other['y'],group=other['group'],arm=other['arm'],
                                        baseline=scores[0],with_state=scores[1])
                write(out/(key+'.json'),{'run_length':event,'horizon':h,'rows':rows,'diagnostics':diagnostics})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--card',required=True,type=Path)
    args=parser.parse_args()
    card=json.loads(args.card.read_text())
    mode=card['mode']
    if mode not in ('descriptive','l7','l8','profile-descriptive','profile-detection'):
        raise ValueError('unlisted finite mode')
    configure(card)
    out=Path(card['output'])
    if out.resolve().parent!=ALLOWED or out.exists(): raise ValueError('exclusive direct-child output required')
    out.mkdir(parents=True,exist_ok=False)
    write(out/'run-card.json',card)
    start=time.monotonic(); status={'completed':False,'mode':mode}
    def stop(signum,frame): raise TimeoutError('bounded replay interrupted: '+str(signum))
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop); signal.signal(signal.SIGALRM,stop)
    signal.setitimer(signal.ITIMER_REAL,card['seconds'])
    try:
        execute(mode,out,card)
        status['completed']=True
    except BaseException as e:
        status.update(error=repr(e),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        elapsed=time.monotonic()-start
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        status.update(seconds=elapsed,peak_rss_bytes=rss,pid=os.getpid())
        if elapsed>card['seconds'] or rss>card['as_bytes']: status['completed']=False
        status['outputs']=[{'name':p.name,'bytes':p.stat().st_size,'sha256':sha(p)}
                           for p in sorted(out.iterdir()) if p.is_file()]
        write(out/'result.json',status)
    print(json.dumps(status,allow_nan=False),flush=True)
    return 0 if status['completed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
