#!/usr/bin/env python3
"""Synthetic-only full-size cost screen; there is intentionally no target mode."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for key in THREADS:os.environ[key]='1'
import gzip,hashlib,json,math,resource,signal,subprocess,sys,time
from fractions import Fraction
from pathlib import Path
import flint,numpy as np
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from neural_shrimp.rigorous_eigenpair_locus import locus_tube,phase,require
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
from neural_shrimp.rigorous_orbits import exact,matrix_json
SOURCES=['src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py',
 'src/neural_shrimp/rigorous_orbits.py','src/neural_shrimp/rigorous_folds.py','src/neural_shrimp/rigorous_cusps.py',
 'src/neural_shrimp/rigorous_eigenpair_locus.py','tests/test_rigorous_eigenpair_locus.py',
 'tools/profile_eigenpair_locus.py','tools/launch_eigenpair_profile.py','docs/plans/eigenpair-locus-prototype.md']

def entry(p):
    b=p.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

class Synthetic13:
    n=13
    def cusp_step(self,z,h,o):
        x=z[0]+sum((z[i]*arb((1,-i)) for i in range(1,13)),0)
        c=exact('3/8');d=exact('9/8')
        fx=-c+(-1-d)*x+c*x*x+d*x*x*x+(h-1)+(o-1)*x
        yy=[z[i]*exact('1/2') for i in range(1,13)]
        return [fx-sum((yy[i-1]*arb((1,-i)) for i in range(1,13)),0)]+yy

def fixture():
    phase0=['1']+['0']*12+['1']+['0']*12
    phase1=['-1']+['0']*12+['2']+['0']*12
    initial=(phase0+phase1)*9+['1','1']
    return initial,dict(period=18,sigma=1,normal=['0']*469+['1'],origin=initial,parameter='0',
        halfwidth_exponent=-14,radius_exponent=-16,newton_steps=2,precision=256,
        anchor={'box':[exact(v) for v in initial],'period':18,'sigma':1,'dimension':13})

def gd_fixture():
    inp={'schema':'neural-shrimp-c1-input-v1','network':{'activation':'tanh','d':2,'width':3,'bias':True,'in_scale':'3/2','out_scale':'2/3','loss_scale':'1/2','ridge':'1/1000'},
         'x':[[str(Fraction(i-12,48)),str(Fraction((i*7)%23-11,48))] for i in range(24)],
         'y':[str(Fraction((i*3)%17-8,64)) for i in range(24)],'lr_hidden':'1/100','lr_output':'1/100'}
    # Synthetic local whole boxes (not a saved model/state or claimed orbit).
    x=[arb(exact(str(Fraction(i-6,64))),arb((1,-20))) for i in range(13)]
    q=[arb(exact(str(Fraction(i+1,32))),arb((1,-20))) for i in range(13)]
    rates=[arb(exact('1/100'),arb((1,-20)))]*2
    return inp,x,q,rates


def projection(chart_times,phase_times,peak,raw_bytes):
    require(len(chart_times)==2 and len(phase_times)==12,'profile count')
    require(all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in chart_times+phase_times),'positive finite profile times')
    require(type(peak) is int and peak>0 and type(raw_bytes) is int and raw_bytes>0,'profile resources')
    seconds=2*(sum(chart_times)+198*max(phase_times))+10
    rss=2*peak+128*1024
    storage=4*raw_bytes+8*1024**2
    return {'two_chart_seconds':seconds,'rss_kib':rss,'compressed_storage_bytes':storage,
            'resource_admitted':seconds<110 and rss<2*1024**2 and storage<256*1024**2,
            'interpretation':'conservative heuristic cost screen, not a hard prediction or target authorization',
            'caveats':'Synthetic spectrum/conditioning and compression differ from target. Actual fixed caps decide.'}


def main(out):
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source')
    require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stop=[False];ticks=[0];counts={};total=[0];serial=[0]
    def save(name,value,final=False):
        raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        payload=gzip.compress(raw,mtime=0) if name.endswith('.gz') else raw+b'\n'
        require(total[0]+len(payload)<(256 if final else 248)*1024**2,'aggregate profile output cap')
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
        total[0]+=len(payload)
    def tick():
        ticks[0]+=1
        if stop[0] or time.monotonic()-start>=120 or ticks[0]>225:raise TimeoutError('profile time/call cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=1024**2:raise MemoryError('profile monitored1GiB RSS')
    def observe(kind,row):
        counts[kind]=counts.get(kind,0)+1
        save(f'events/{serial[0]:04d}-{kind}.json.gz',row);serial[0]+=1
        # Returned payload has been preserved before cooperative stop checks.
        if stop[0] or time.monotonic()-start>=120:raise TimeoutError('profile deadline after saved return')
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    save('runtime.json',{'commit':commit,'pid':os.getpid(),'mode':'synthetic_only','sources':{name:entry(ROOT/name) for name in SOURCES},'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'precision':256,'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':{k:os.environ[k] for k in THREADS},'time_cap':120,'tick_cap':225,'monitored_rss_kib':1024**2,'hard_virtual_address_bytes':3*1024**3})
    for name in SOURCES:
        p=out/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);b=(ROOT/name).read_bytes();p.write_bytes(b);total[0]+=len(b)
    outcome={'passed':False,'chart_times':[],'phase_times':[],'target_acquired':False}
    try:
        with ctx.workprec(256):
            initial,settings=fixture();save('synthetic-inputs.json',{'initial':initial,'settings':{k:v for k,v in settings.items() if k!='anchor'},'known_root':'repeated exact primitive2; intentionally not primitive18'})
            for slot in range(2):
                before=time.monotonic();result,_=locus_tube(Synthetic13(),initial,**settings,tick=tick,observe=observe)
                outcome['chart_times'].append(time.monotonic()-before)
                save(f'chart-{slot}.json.gz',result)
                require(result['gates']['uniform_normalized_eigenpair_locus'] and not result['gates']['primitive_period'],'synthetic repeated2 control')
            inp,x,q,rates=gd_fixture();save('synthetic-gd-inputs.json',inp);model=TwoRateCuspMap(inp)
            for slot in range(12):
                tick();before=time.monotonic();v,t,e=phase(model,x,q,rates);duration=time.monotonic()-before
                save(f'gd-phase-{slot}.json.gz',{'value':matrix_json(v),'block':matrix_json(t),'rates':matrix_json(e)})
                outcome['phase_times'].append(duration)
            outcome['projection']=projection(outcome['chart_times'],outcome['phase_times'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,total[0])
            outcome['passed']=outcome['projection']['resource_admitted']
    except BaseException as exc:
        outcome['failure']={'type':type(exc).__name__,'message':str(exc)}
    outcome.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks[0],event_counts=counts,stored_bytes_before_final=total[0])
    if outcome['elapsed_seconds']>=120 or outcome['peak_rss_kib']>=1024**2:outcome['passed']=False
    save('outcome.json',outcome,True)
    inv={str(p.relative_to(out)):entry(p) for p in sorted(out.rglob('*')) if p.is_file()};save('manifest.json',{'files':inv},True)
    for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir() else 0o444)
    out.chmod(0o555);print(json.dumps(outcome,sort_keys=True));return 0 if outcome['passed'] else 2
if __name__=='__main__':raise SystemExit(main(Path(sys.argv[1])))
