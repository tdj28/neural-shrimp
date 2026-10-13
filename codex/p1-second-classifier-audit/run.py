"""One source/card-bound Task11 run. Prepared cards cannot launch."""
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
sys.path.insert(0,str(HERE))
ROOT=Path('/data2/neural-shrimp-data/codex/p1-second-classifier-audit')
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
SOURCES=('run.py','launch.py','definitions.py','inputs.py','analysis.py','task6_reducers.py',
         'task6_detection.py','reuse-provenance.json','producer-score-manifest.json','producer-score-source.json','plan.md')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()


def write(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)


def configure(card):
    if card['status']!='accepted_by_root' or not card.get('acceptance_reference') or not card.get('source_commit'):
        raise ValueError('reviewed private checkpoint/card required')
    if card['mode'] not in ('profile','descriptive','l8'):raise ValueError('finite mode list')
    if card['cpu_threads']!=1 or card['nice']!=10 or card['as_bytes']!=4*1024**3:raise ValueError('resource policy')
    if any(os.environ.get(k)!='1' for k in THREADS):raise ValueError('preimport thread limits')
    priority=os.getpriority(os.PRIO_PROCESS,0)
    if priority>10:raise ValueError('exact nice10 required')
    if priority<10:os.nice(10-priority)
    hard=resource.getrlimit(resource.RLIMIT_AS)[1]
    if hard!=resource.RLIM_INFINITY and hard<card['as_bytes']:raise ValueError('hard AS too small')
    resource.setrlimit(resource.RLIMIT_AS,(card['as_bytes'],card['as_bytes']))
    if set(card['source_sha256'])!=set(SOURCES):raise ValueError('exact source closure')
    for rel,want in card['source_sha256'].items():
        if sha(HERE/rel)!=want:raise ValueError('source mismatch: '+rel)
    if card['mode']!='profile':verify_profile(card)


def verify_profile(card):
    pins=card.get('profile_authority',{})
    roles={'worker','owner','card','descriptive','detection'}
    if set(pins)!=roles:raise ValueError('complete measured profile authority required')
    obj={}
    for role,pin in pins.items():
        p=Path(pin['path'])
        if not p.is_relative_to(ROOT) or sha(p)!=pin['sha256']:raise ValueError('profile identity mismatch')
        obj[role]=json.loads(p.read_text())
    if not obj['worker']['completed'] or not obj['owner']['completed'] or not obj['owner']['owned_worker_exited']:
        raise ValueError('profile failed or owner unresolved')
    if obj['owner']['worker_exit']!=0 or obj['card']['mode']!='profile':raise ValueError('profile not successful')
    if obj['card']['source_sha256']!=card['source_sha256']:raise ValueError('profile source changed')
    if not obj['descriptive']['synthetic_only'] or not obj['detection']['synthetic_only']:
        raise ValueError('synthetic-only profile required')
    forecast=obj['detection']['descriptive_forecast_seconds' if card['mode']=='descriptive' else 'l8_forecast_seconds']
    if forecast>card['seconds'] or 2*obj['worker']['peak_rss_bytes']+512*1024**2>card['as_bytes']:
        raise ValueError('measured forecast exceeds card')


def runtime():
    import numpy,scipy,sklearn
    from threadpoolctl import threadpool_info
    return {'python':sys.version,'numpy':numpy.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,
            'pid':os.getpid(),'ppid':os.getppid(),'nice':os.getpriority(os.PRIO_PROCESS,0),
            'thread_env':{k:os.environ[k] for k in THREADS},'threadpools':threadpool_info(),
            'rlimit_as':resource.getrlimit(resource.RLIMIT_AS),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}


def synthetic():
    import numpy as np
    from definitions import ALL,replicas,L6,L10
    rng=np.random.default_rng(41742);data={}
    for name in ALL:
        n=replicas(name)
        data[name]={'first':rng.uniform(size=(n,40)),'second':rng.uniform(size=(n,40)),
                    'args':{'synthetic_only':True}}
        if name in L6+L10:
            data[name].update(counter=rng.choice([-.04,0.],size=(n,40)),
                              regenerated=rng.integers(0,2,(n,40)).astype(bool),
                              texts=[['neutral wording']*40 for _ in range(n)])
    return data


def profile(out):
    import numpy as np
    from analysis import descriptive
    from task6_detection import fit_pca,fit_logistic,score_model
    from task6_reducers import increment
    start=time.monotonic();result=descriptive(synthetic());desc=time.monotonic()-start
    write(out/'profile-descriptive.json',{'seconds':desc,'arms':44,'instruments':2,
                                         'conclusion_rows':len(result['conclusion_checks']),'synthetic_only':True})
    del result
    # Same maximum feature dimension, above h3 row counts; random balanced labels
    # give a cost measurement, not a promise of real logistic convergence time.
    rng=np.random.default_rng(7261);diagnostics=[];start=time.monotonic()
    x=rng.normal(size=(19968,896));y=rng.integers(0,2,len(x));pca=fit_pca(x,diagnostics)
    z=pca.transform(x);base=rng.normal(size=(len(x),6));full=np.column_stack((base,z))
    for design in (base,full):score_model(fit_logistic(design,y,diagnostics),design)
    fit=time.monotonic()-start;del x,z,base,full,pca
    n=39936;groups=np.repeat(np.arange(1024),39);labels=rng.integers(0,2,n)
    a=rng.normal(size=n);b=a+.1*rng.normal(size=n);start=time.monotonic()
    increment(labels,groups,a,b,2000);boot=time.monotonic()-start
    write(out/'profile-detection.json',{'synthetic_only':True,'fit_seconds':fit,'bootstrap_seconds':boot,
        'train_shape':[19968,896],'test_rows':n,'diagnostics':diagnostics,
        'descriptive_forecast_seconds':2*desc+30,'l8_forecast_seconds':4*fit+8*boot+30,
        'limitation':'synthetic fits can converge faster; external deadlines remain authoritative'})


def execute(card,out):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from definitions import ALL,TRAIN,TEST
    from inputs import authority,load_arm
    from analysis import descriptive,detection
    with threadpool_limits(limits=1):
        write(out/'runtime.json',runtime())
        if card['mode']=='profile':profile(out);return
        pin=card['input_manifest']
        if not pin or sha(pin['path'])!=pin['sha256']:raise ValueError('exact manifest pin required')
        manifest=authority(json.loads(Path(pin['path']).read_text()))
        write(out/'input-manifest.json',manifest)
        if card['mode']=='descriptive':
            data={a:load_arm(a,manifest) for a in ALL}
            write(out/'descriptive.json',descriptive(data))
        else:
            data={a:load_arm(a,manifest,features=True) for a in TRAIN+TEST}
            write(out/'actual-sample-args.json',{a:data[a]['args'] for a in data})
            outputs={}
            for instrument in ('first','second'):
                result,arrays=detection(data,instrument)
                # Save scores and every fit warning even when the scientific comparison is negative.
                np.savez_compressed(out/(instrument+'-scores.npz'),**arrays)
                write(out/(instrument+'-l8.json'),result);outputs[instrument]=result
            write(out/'comparison.json',{'status':'exploratory classifier robustness',
                'first':outputs['first']['rows'],'second':outputs['second']['rows'],
                'positive_pooled_lower_bound_changed':(outputs['first']['positive_pooled_lower_bound']!=outputs['second']['positive_pooled_lower_bound']) if all(v['admissible'] for v in outputs.values()) else None,
                'admissible':{k:v['admissible'] for k,v in outputs.items()},
                'warning_count':{k:len(v['warnings']) for k,v in outputs.items()},
                'all_logistic_converged':{k:v['all_logistic_converged'] for k,v in outputs.items()}})
        authority(manifest) # same complete44 inventory still intact after calculation


def main():
    p=argparse.ArgumentParser();p.add_argument('--card',type=Path,required=True);args=p.parse_args()
    card=json.loads(args.card.read_text());configure(card)
    out=Path(card['output'])
    if out.resolve().parent!=ROOT or out.exists():raise ValueError('exclusive owned output required')
    out.mkdir(parents=True,exist_ok=False);write(out/'run-card.json',card)
    start=time.monotonic();status={'completed':False,'mode':card['mode']}
    def stop(signum,frame):raise TimeoutError('finite worker interrupted '+str(signum))
    for s in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(s,stop)
    signal.setitimer(signal.ITIMER_REAL,card['seconds'])
    try:execute(card,out);status['completed']=True
    except BaseException as e:status.update(error=repr(e),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        elapsed=time.monotonic()-start;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        status.update(seconds=elapsed,peak_rss_bytes=rss,pid=os.getpid())
        if elapsed>card['seconds'] or rss>card['as_bytes']:status['completed']=False
        status['outputs']=[{'name':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.iterdir()) if p.is_file()]
        write(out/'result.json',status)
    print(json.dumps(status),flush=True);return 0 if status['completed'] else 2


if __name__=='__main__':raise SystemExit(main())
