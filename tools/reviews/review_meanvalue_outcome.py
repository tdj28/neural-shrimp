"""Saved-arithmetic review of one mean-value wider-root exploratory attempt.

No map/gradient/Hessian, floating solver or stability metric is evaluated.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,gzip,hashlib,json,math as pymath,resource,signal,subprocess,sys,time
from pathlib import Path
from fractions import Fraction
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).resolve().parent)]
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import canonical_bytes,ball_json
import review_joined_period18_math as math
SOURCE='15a2ba1f7b390dc3d23168c4a3e77de832a0689a'
FILES=['src/neural_shrimp/__init__.py','src/neural_shrimp/models.py','src/neural_shrimp/controls.py','src/neural_shrimp/rigorous_orbits.py','tools/reviews/review_joined_period18_math.py','tools/reviews/review_meanvalue_outcome.py','docs/reviews/meanvalue-gradient-source-review.json','docs/plans/meanvalue-outcome-review.md','tools/reviews/launch_meanvalue_outcome_review.py','tests/reviews/test_meanvalue_outcome_review.py']
SETTINGS={'precision':256,'halfwidth_exponent':-10,'radius_exponent':-18,'newton_steps':2,'max_power':4096}
OLD_INPUT=Path('/data2/neural-shrimp-data/audits/s2-period18-joined-branch/attempt-001/run/inputs.json')
OLD_MANIFEST='6b3ca6e8d518da0f3cde74dd330625af5a244dcc69d455102a82f0ad27363289'
CAP=8*1024**2

def need(v,msg):
    if not v:raise ValueError(msg)
def ent(p):return {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p):
    if p.suffix=='.gz':
        with gzip.open(p,'rb') as f:data=f.read(256*1024**2+1)
        need(len(data)<=256*1024**2,'decompression cap');meta=json.loads(p.with_name(p.name.removesuffix('.json.gz')+'-payload.json').read_bytes());need(meta=={'uncompressed_bytes':len(data),'uncompressed_sha256':hashlib.sha256(data).hexdigest()},'payload identity');return json.loads(data)
    return json.loads(p.read_bytes())
def inventory(root):return {str(p.relative_to(root)):ent(p) for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'manifest.json'}
def verify(root,pin):
    need(ent(root/'manifest.json')['sha256']==pin,'fixed inventory pin');need(not any(p.is_symlink() for p in root.rglob('*')),'no symlinks');m=read(root/'manifest.json');need(inventory(root)==m['files'],'exact full inventory');need(sum(e['bytes'] for e in m['files'].values())==m['total_bytes'],'byte total');return m

def bounds(chart):
    # Rebuild native derivative locations; derivative entries remain bound to
    # reviewed mean-value source. No model derivative is reevaluated here.
    need(chart['settings']==SETTINGS,'fixed settings');N=234;n=13
    with ctx.workprec(384):
        A=arb_mat(N,N)
        for phase,J in enumerate(map(math.mat,chart['phase_jacobians'])):
            need(J.nrows()==J.ncols()==n,'native derivative shape')
            for i in range(n):
                for j in range(n):A[phase*n+i,phase*n+j]+=J[i,j]
                A[phase*n+i,((phase+1)%18)*n+i]-=1
        C=math.mat(chart['preconditioner']);S=math.mat(chart['predictor_slope']);B=math.mat(chart['parameter_partial_tube']);H=math.mat(chart['residual_center'])
        for row in chart['preconditioner']:
            for value in row:math.point(value)
        rho=arb((1,-18));delta=arb(0,arb((1,-10)));D=math.ident(N)-C*A
        K=-C*(H+(A*S+B)*delta)+D*arb_mat([[arb(0,rho)] for _ in range(N)])
        q=math.norm(D);absolute=[abs(K[i,0]).upper() for i in range(N)];included=[bool(x<rho) for x in absolute];passed=bool(all(included) and q<1)
        return {'passed_root_bound':passed,'matches_producer_root_gate':passed==chart['gates']['uniform_root'],'contraction':ball_json(q),'image_norm':ball_json(max(absolute)),'radius':ball_json(rho),'failed_coordinates':[i for i,ok in enumerate(included) if not ok],'coordinate_image_bounds':[ball_json(x) for x in absolute]}

def execute(out,bundle,pin,execution):
    need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source');need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');source={}
    for name in FILES:
        b=(ROOT/name).read_bytes();need(b==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed review source');source[name]=hashlib.sha256(b).hexdigest()
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stopped=False;result={'audit_passed':False,'status':'started'}
    def save(name,obj):
        b=canonical_bytes(obj);need(sum(p.stat().st_size for p in out.rglob('*') if p.is_file())+len(b)<CAP,'output cap')
        with(out/name).open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
    def stop(*_):
        nonlocal stopped
        stopped=True
    def tick():
        need(not stopped and time.monotonic()-start<30,'internal30s deadline');need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2*1024**2,'monitored2GiB')
    runtime={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':source,'pid':os.getpid(),'ppid':os.getppid(),'bundle':str(bundle),'bundle_manifest':pin,'producer_execution_commit':execution,'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':{k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
    save('runtime-start.json',runtime)
    for name in FILES:p=out/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,30)
    try:
        verify(bundle,pin);need(all(not(p.stat().st_mode&0o222) for p in bundle.rglob('*')),'immutable producer tree');r=bundle/'run';verify(r,ent(r/'manifest.json')['sha256']);rt=read(r/'runtime-start.json');need(rt['source_commit']==execution and rt['mode']=='target','exact target execution')
        expected=read(ROOT/'docs/reviews/meanvalue-gradient-source-review.json')['source_sha256'];need(rt['source_sha256']==expected,'exact reviewed15-file source')
        for name,h in expected.items():need(ent(r/'source'/name)['sha256']==h and hashlib.sha256(subprocess.check_output(['git','show',execution+':'+name],cwd=ROOT)).hexdigest()==h,'retained/executed source')
        prior=read(OLD_INPUT.parent/'manifest.json');need(ent(OLD_INPUT.parent/'manifest.json')['sha256']==OLD_MANIFEST and ent(OLD_INPUT)==prior['files']['inputs.json'],'independently reviewed original input pin')
        inp=read(r/'inputs.json');need(inp==read(OLD_INPUT),'same exact original model/data/native anchor')
        envelope=read(r/'certificate.json.gz');need(envelope['runtime']==rt and envelope['inputs_sha256']==ent(r/'inputs.json')['sha256'],'certificate runtime/input envelope')
        proof=envelope['result'];outcome=read(r/'outcome.json');launcher=read(bundle/'launcher-result.json');need(outcome['gates']==proof['gates'] and outcome['adapter_counts']==proof['adapter_counts'],'terminal mathematical labels/counts')
        need(launcher['source_commit']==execution and launcher['source_sha256']==expected and launcher['exit_code']==outcome['exit_code'] and launcher['owned_live_pids']==[],'producer terminal ownership/status')
        for value,upper in ((outcome['elapsed_seconds'],120),(launcher['elapsed_seconds'],142),(outcome['max_rss_kib'],2*1024**2)):
            need(type(value)in(int,float) and pymath.isfinite(value) and 0<=value<upper,'finite producer resource bounds')
        need(rt['threads']==runtime['threads'] and rt['nice']>=10 and rt['cpu_only'] is True and rt['paid'] is False,'producer local one-thread contract')
        need(outcome['status'] in ('joined_intervals_certified','inconclusive') and 'error' not in launcher,'complete attempted unit')
        counts=proof['adapter_counts'];need(counts['map_attempted']==counts['map_returned']<=180 and counts['unit_attempted']==counts['unit_returned']==2*counts['map_returned']<=360,'complete bounded adapter evaluations')
        tick();rows=[]
        for chart in proof['charts']:
            rows.append(bounds(chart));save(f'chart-{len(rows)-1}-bounds.json',rows[-1]);tick()
        need(all(x['matches_producer_root_gate'] for x in rows),'independent root-bound verdicts agree')
        result.update(audit_passed=True,root_bound_checks=rows,connected_primitive_branch=proof['gates']['connected_primitive_branch'],producer_uniform_attraction=proof['gates']['uniform_attraction'],stability_independently_reviewed=False)
        if proof['gates']['connected_primitive_branch']:
            joined=math.audit_join(proof['charts'],inp,SETTINGS);save('independent-join.json',joined);need(joined['exact_interval']==proof['connected_parameter_endpoints'],'exact branch endpoints');result.update(status='root_primitive_anchor_join_review_passed',exact_interval=joined['exact_interval'])
        else:result.update(status='retained_failed_root_bound_confirmed',exact_interval=None)
        tick()
    except BaseException as exc:result.update(audit_passed=False,status='review_failed_or_incomplete',error={'type':type(exc).__name__,'message':str(exc)})
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result);files=inventory(out);save('manifest.json',{'files':files,'total_bytes':sum(e['bytes'] for e in files.values())})
        for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir() else 0o444)
        out.chmod(0o555)
    print(json.dumps(result),flush=True);return 0 if result['audit_passed'] else 2

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--manifest',required=True);p.add_argument('--execution',required=True);a=p.parse_args();raise SystemExit(execute(a.out,a.bundle,a.manifest,a.execution))
