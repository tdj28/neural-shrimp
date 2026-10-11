"""Fixed saved error decomposition; hypothetical removals are not certificates."""
import gc,json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import probe_native705_polynomial_complement as P
from neural_shrimp.rigorous_norm_cocycle import advance,frobenius_upper,scalar_remainder_metric
from neural_shrimp.rigorous_orbits import midpoint_matrix,eye,ball_json
from flint import arb,arb_mat,ctx
OUT=P.OUT.parent.with_name('polynomial001-remainder-diagnosis')/'run'
PARENT=P.OUT.parent;PIN='fd297b95b22708c5c33b0284fca27f55c095a2fd2d3d0cfc223c42a9e840e031';EXEC='2d932da23fddf602a90ee747da88dce9085a5ba9'
FILES=sorted(set(P.FILES+['tools/diagnose_native705_remainder.py','tools/launch_native705_remainder_diagnosis.py','tests/test_native705_remainder_diagnosis.py','docs/plans/native705-remainder-diagnosis.md','docs/reviews/native705-polynomial-complement-attempt001.json']))
need=P.need;read=P.read


def reduce_phase(center,derivative,h):
    a=midpoint_matrix(center);hd=derivative*h;b=midpoint_matrix(hd)
    return [a,b,arb_mat(a.nrows(),a.ncols())],center-a,hd-b


def main():
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in P.O.THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=0;out=dict(passed=False,native_calls=0)
    def tick():need(time.monotonic()-start<30 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1024**2,'30s1GiB cap')
    def stop(*_):raise TimeoutError('owned diagnosis deadline')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,30)
    def save(name,value,closing=False):
        nonlocal used
        raw=P.D.canonical_bytes(value);need(used+len(raw)<(8 if closing else 6)*1024**2,'8MiB with2MiB reserve');q=OUT/name
        with q.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
        used+=len(raw)
    try:
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',dict(source_commit=source,pid=os.getpid(),precision=480,native_calls=0,sources={n:P.O.entry(ROOT/n)for n in FILES},python=P.D.platform.python_version(),python_flint=P.D.flint.__version__,flint=P.D.flint.__FLINT_VERSION__,numpy=P.D.np.__version__))
        for n in FILES:
            b=(ROOT/n).read_bytes();need(b==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'literal committed source');q=OUT/'source'/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b);used+=len(b)
        P.O.verify(PARENT,PIN);P.D.B.archived(PARENT/'run',EXEC,'runtime.json','source');old=read(PARENT/'run/outcome.json');la=read(PARENT/'launcher-result.json');need(not old['passed']and old['status']=='completed_correlated_bound_unresolved'and old['native_calls']==216 and old['target_native_calls']==108 and la['exit_code']==2 and not la['owned_live_pids'],'exact complete negative metric unit')
        phasepaths=sorted((PARENT/'run/events').glob('*-phase.json'));need(len(phasepaths)==18,'all18 original phases');scenarios=[('original',arb(1),True,True),('remove_center_uncertainty',arb(1),False,True),('remove_derivative_uncertainty',arb(1),True,False),('center_error_floor',arb(0),True,False),('same_center_halfwidth_1_2',arb(1)/2,True,True),('same_center_halfwidth_1_4',arb(1)/4,True,True),('same_center_halfwidth_1_8',arb(1)/8,True,True)];answers=[]
        with ctx.workprec(480):
            phases=[]
            for j,q in enumerate(phasepaths):
                r=read(q);need(r['phase']==j and r['operator_bound']['passed'],'literal qualified phase');Q=P.U.R.matrix(r['center_quotient']);Qp=P.U.R.matrix(r['whole_quotient_derivative']);gamma=P.U.R.original_ball(r['operator_bound']['gamma']);coeff,e0,e1=reduce_phase(Q,Qp,P.U.R.exact(P.HALF));phases.append((Q,Qp,gamma))
                if j==0:decomp=[]
                decomp.append(dict(phase=j,center=frobenius_upper(e0),derivative=frobenius_upper(e1),combined=frobenius_upper(e0+e1*arb(0,1))))
            save('phase-components.json',dict(rows=[{k:ball_json(v)if k!='phase'else v for k,v in row.items()}for row in decomp]))
            for name,scale,center_on,deriv_on in scenarios:
                acc=([eye(11),arb_mat(11,11),arb_mat(11,11)],arb(0),arb(1))
                for Q,Qp,gamma in phases:
                    coeff,e0,e1=reduce_phase(Q,Qp,P.U.R.exact(P.HALF)*scale);err=(e0 if center_on else arb_mat(11,11))+(e1*arb(0,1)if deriv_on else arb_mat(11,11));acc,_=advance(acc,(coeff,frobenius_upper(err),gamma));tick()
                m=scalar_remainder_metric(*acc,tick=tick);rec=dict(name=name,relative_halfwidth=ball_json(scale),metric_passed=m['passed'],remainder=ball_json(acc[1]),penalty=m['remainder_penalty'],polynomial_norm=m['polynomial_norm_upper'],center_uncertainty_included=center_on,derivative_uncertainty_included=deriv_on,hypothetical=not(center_on and deriv_on));save(name+'.json',rec);answers.append(rec)
            need(not answers[0]['metric_passed'],'original failure reproduced');save('input-binding.json',dict(parent=str(PARENT),outer=PIN,source=EXEC,phases={q.name:P.O.entry(q)for q in phasepaths}));out.update(passed=True,rows=answers,scope='Saved error-budget diagnosis only. Removed-error scenarios are hypothetical. Smaller centered widths cover only their own strict subset and do not cover the original quarter endpoints; no cells, instability or complementary theorem promoted here.')
    except BaseException as e:out.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);out.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',out,True);save('manifest.json',dict(files={str(q.relative_to(OUT)):P.O.entry(q)for q in OUT.rglob('*')if q.is_file()}),True)
    print(json.dumps({k:v for k,v in out.items()if k!='rows'}));return 0 if out['passed']else 2
if __name__=='__main__':raise SystemExit(main())
