"""Independent saved-arithmetic, exact replay and nested check of one initial tube."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,gzip,hashlib,importlib.util,json,resource,signal,subprocess,sys,time
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import canonical_bytes,eye,ball_json,norm_inf_upper
from neural_shrimp.rigorous_connection import HiddenRateMap,tube,restore_ball
BUNDLE=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-connection-2026-10-09/attempt-002-target')
PIN='0c2f6f597d84ac2e5009b99bb5ffacd79026e47d57b206874d1b904de770e107'
CORE='2caf7c5bae6785fc6d9470f082ff782c88410bba918cb7c6f6575ee7d666633a'
FILES=['src/neural_shrimp/rigorous_connection.py','src/neural_shrimp/rigorous_orbits.py','tools/certify_s2_period18_connection.py','tools/reviews/review_s2_period18_connection.py','tests/reviews/test_s2_period18_connection_review.py','docs/plans/s2-period18-connection-review.md']
CAP=64*1024**2

def require(v,m):
    if not v:raise ValueError(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_bytes())
def pair(x):
    require(isinstance(x,list) and len(x)==2 and all(type(v)is int for v in x),'exact dyadic pair')
    return Fraction(x[0])*Fraction(2)**x[1]
def interval(x):
    a,r=pair(x['mid']),pair(x['rad']);require(r>=0,'negative radius');return a-r,a+r
def subset(a,b,strict=True):
    aa,bb=interval(a),interval(b)
    return bb[0]<aa[0] and aa[1]<bb[1] if strict else bb[0]<=aa[0] and aa[1]<=bb[1]
def matrix(rows):return arb_mat([[restore_ball(v) for v in row] for row in rows])
def exact_norm(rows):return max(sum(max(abs(a),abs(b)) for a,b in map(interval,row)) for row in rows)
def inventory(p):return {str(f.relative_to(p)):{'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(p.rglob('*')) if f.is_file() and f!=p/'manifest.json'}

def saved_math(r,inp):
    require(r['passed'] and all(r['gates'].values()),'full producer gates')
    n,p=13,18;dim=n*p
    require(r['statement']['phase_dimension']==n and r['statement']['shooting_dimension']==dim,'13 physical,234 shooting dimensions')
    require(exact_norm(r['defect'])<1,'exact saved contraction norm')
    radius=interval(r['correction_radius']);require(radius[0]==radius[1] and radius[0]>0,'point positive radius')
    for row in r['image_correction']:
        a,b=interval(row[0]);require(-radius[0]<a<=b<radius[0],'exact strict error inclusion')
    with ctx.workprec(384):
        C=matrix(r['preconditioner']);A=matrix(r['jacobian_tube']);S=matrix(r['predictor_slope']);B=matrix(r['parameter_partial_tube'])
        require(all(interval(v)[0]==interval(v)[1] for row in r['preconditioner'] for v in row),'exact preconditioner')
        h0=matrix(r['residual_center']);delta=arb((1,r['settings']['halfwidth_exponent']));rr=restore_ball(r['correction_radius'])
        D=eye(dim)-C*A;E=arb_mat([[arb(0,rr)] for _ in range(dim)])
        K=-C*(h0+(A*S+B)*arb(0,delta))+D*E
        require(norm_inf_upper(D)<1 and all(arb(0,rr).contains_interior(K[i,0]) for i in range(dim)),'independent uniform Krawczyk inclusion')
        require(inp['anchor']['rate']==r['parameter_center']==inp['model_inputs']['lr_hidden'],'identical anchor rate')
        count=0
        for j in range(p):
            for i in range(n):
                require(arb(restore_ball(r['center'][j][i]),rr).contains(restore_ball(inp['anchor']['phase_boxes'][j][i])),'independent whole anchor inclusion');count+=1
        require([d['divisor'] for d in r['divisors']]==[1,2,3,6,9],'all divisors')
        for d in (1,2,3,6,9):require(any(not(restore_ball(r['phase_boxes'][d][i])-restore_ball(r['phase_boxes'][0][i])).contains(0) for i in range(n)),'uniform divisor exclusion')
        # Sequential order differs from the producer's balanced parenthesization.
        M=eye(n)
        for block in r['phase_jacobians']:M=matrix(block)*M
        power=1;Q=M
        while power<r['attraction_power']:Q=Q*Q;power*=2
        sequential_bound=norm_inf_upper(Q)
        # A weaker interval parenthesization is only a diagnostic, not a reason
        # to reject a valid bound obtained with the recorded balanced tree.
        nodes=[matrix(block) for block in r['phase_jacobians']]
        while len(nodes)>1:
            nxt=[]
            for index in range(0,len(nodes),2):
                nxt.append(nodes[index] if index+1==len(nodes) else nodes[index+1]*nodes[index])
            nodes=nxt
        Q=nodes[0];power=1
        while power<r['attraction_power']:Q=Q*Q;power*=2
        bound=norm_inf_upper(Q);require(bound<1,'reconstructed balanced monodromy power bound')
        return {'strict_uniform_inclusion':True,'anchor_coordinates':count,'contraction':ball_json(norm_inf_upper(D)),'attraction_power':power,'balanced_power_bound':ball_json(bound),'sequential_power_bound':ball_json(sequential_bound),'sequential_diagnostic_passed':bool(sequential_bound<1),'scope':'Independent reduction of saved enclosures with recorded balanced parenthesization; weaker sequential product retained as diagnostic. Objective AD and Arb remain shared.'}

def run(out):
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean committed source')
    sources={f:sha(ROOT/f) for f in FILES};require(sources[FILES[0]]==CORE,'unchanged reviewed tube core')
    for f in FILES:require((ROOT/f).read_bytes()==subprocess.check_output(['git','show','HEAD:'+f],cwd=ROOT),'committed source bytes')
    require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();ticks=0;stages=[];result={'passed':False}
    def save(name,obj):
        data=canonical_bytes(obj)
        if name.endswith('.gz'):data=gzip.compress(data,mtime=0)
        require(sum(f.stat().st_size for f in out.rglob('*') if f.is_file())+len(data)<CAP,'audit storage cap')
        with (out/name).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    def tick():
        nonlocal ticks
        ticks+=1
        if ticks>240 or time.monotonic()-start>100:raise TimeoutError('audit wall/tick cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>2*1024**2:raise MemoryError('audit RSS cap')
    def interrupted(*_):raise TimeoutError('external/internal audit deadline')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGALRM,interrupted);signal.setitimer(signal.ITIMER_REAL,100)
    resource.setrlimit(resource.RLIMIT_FSIZE,(CAP,CAP))
    save('runtime.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':sources,'pid':os.getpid(),'ppid':os.getppid(),'nice':os.getpriority(os.PRIO_PROCESS,0),'started_utc':datetime.now(timezone.utc).isoformat(),'internal_seconds':100,'external_seconds':120,'threads':1,'cpu_only':True})
    for name in FILES:
        q=out/'source'/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/name).read_bytes())
    try:
        require(sha(BUNDLE/'manifest.json')==PIN,'fixed whole producer manifest')
        require(inventory(BUNDLE)==load(BUNDLE/'manifest.json')['files'],'whole producer inventory')
        spec=importlib.util.spec_from_file_location('tube_producer',ROOT/'tools/certify_s2_period18_connection.py');driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
        inp=driver.prepare();require(canonical_bytes(inp)==canonical_bytes(load(BUNDLE/'run/inputs.json')),'independent exact input reconstruction')
        certificate=driver.read(BUNDLE/'run/certificate.json.gz');old=certificate['result'];runtime=certificate['runtime']
        require(certificate['inputs_sha256']==sha(BUNDLE/'run/inputs.json'),'certificate input binding')
        for name,h in runtime['source_sha256'].items():
            require(sha(ROOT/name)==h and sha(BUNDLE/'run/source'/name)==h,'producer source equality')
            require(hashlib.sha256(subprocess.check_output(['git','show',runtime['source_commit']+':'+name],cwd=ROOT)).hexdigest()==h,'historical source commit')
        save('inputs.json',inp);save('saved-math.json',saved_math(old,inp));stages.append('saved_math')
        for label,changes in [('replay',{}),('nested',{'precision':384,'radius_exponent':-29})]:
            settings={**old['settings'],**changes}
            with ctx.workprec(settings['precision']):r=tube(HiddenRateMap(inp['model_inputs']),inp['initial'],inp['rate'],anchor=inp['anchor'],**settings,tick=tick)
            save(label+'.json.gz',r);require(r['passed'],'full '+label+' gates')
            if label=='replay':require(canonical_bytes(r)==canonical_bytes(old),'byte-identical mathematical replay')
            else:require(all(subset(a,b) for aa,bb in zip(r['phase_boxes'],old['phase_boxes']) for a,b in zip(aa,bb)),'strict nested full phase boxes')
            save(label+'-summary.json',{'passed':True,'settings':settings,'gates':r['gates'],'contraction':r['contraction'],'power_bounds':r['power_bounds'],'attraction_power':r['attraction_power']});stages.append(label)
        result.update(passed=True,scope='One center-anchored interval, shared objective/tube implementation and Arb. No boundary connection, wider interval, basin size or utility.')
    except BaseException as exc:result.update(error={'type':type(exc).__name__,'message':str(exc)})
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(stages=stages,elapsed_seconds=time.monotonic()-start,ticks=ticks,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,finished_utc=datetime.now(timezone.utc).isoformat());save('outcome.json',result);save('manifest.json',{'files':inventory(out)})
        for f in sorted(out.rglob('*'),reverse=True):f.chmod(0o555 if f.is_dir() else 0o444)
        out.chmod(0o555)
    print(json.dumps(result),flush=True);return 0 if result['passed'] else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();raise SystemExit(run(a.out))
