"""Saved-only positive-weight proposal and Arb reductions; no model acquisition."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,json,resource,signal,subprocess,sys,time
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.linalg import eigvals,solve
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
import certify_s2_period18_quadratic_connection as io
from neural_shrimp.rigorous_orbits import exact,ball_json,matrix_json,norm_inf_upper,canonical_bytes
from neural_shrimp.rigorous_connection import require,restore_ball
from neural_shrimp.rigorous_connection_frames import restore_matrix
CARD='docs/plans/s2-period18-positive-weights.md'
SOURCES=list(dict.fromkeys(io.SOURCES+['tools/analyze_s2_period18_quadratic_weights.py','tools/launch_s2_period18_quadratic_weights.py',CARD,'tests/test_quadratic_weight_analysis.py']))
PARENT=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-quadratic-connection-2026-10-10/attempt-001-target/run')
PIN='60bddca8270784a7264bd0d9d6650e5efd46ce5eca18b9ff1f30f515c863fbc3'


def weights_from_defect(defect):
    n=defect.nrows();bound=np.array([[float(abs(defect[i,j]).upper()) for j in range(n)] for i in range(n)])
    require(np.isfinite(bound).all(),'finite proposal matrix')
    spectrum=eigvals(bound,check_finite=True);rho=float(np.max(np.abs(spectrum)))
    target=min(.95,max(.05,2*rho));require(np.isfinite(rho) and rho<target<1,'proposal Perron target')
    candidate=solve(np.eye(n)-bound/target,np.ones(n),assume_a='gen',check_finite=True)
    require(np.isfinite(candidate).all() and np.min(candidate)>0,'positive finite weight proposal')
    candidate/=np.max(candidate)
    require(np.min(candidate)>=1e-6,'weight condition cap')
    rationals=[str(Fraction(float(v))) for v in candidate]
    return rationals,{'spectral_radius_float_proposal_only':rho,'target_float_proposal_only':target,
        'weight_min_float':float(np.min(candidate)),'rule':'w=(I-B/q)^-1 1, q=min(.95,max(.05,2*rho_float(B))); normalize max(w)=1; all floats are proposals only'}


def reduce_chart(chart,weights=None):
    require(chart['passed'] and chart['gates']['uniform_root'],'proved parent chart required')
    pre=restore_matrix(chart['preconditioner']);residual=restore_matrix(chart['predictor_residual']);defect=restore_matrix(chart['defect'])
    radius=restore_ball(chart['correction_radius']);n=defect.nrows()
    require(defect.ncols()==pre.nrows()==pre.ncols()==residual.nrows()==n and residual.ncols()==1,'saved dimensions')
    if weights is None:weights,proposal=weights_from_defect(defect)
    else:proposal={'synthetic_supplied_weights':True}
    require(len(weights)==n,'weight dimensions');w=[exact(v) for v in weights]
    require(all(v.is_exact() and v.is_finite() and v>0 and v<=1 for v in w),'strict positive exact normalized weights')
    require(radius.is_exact() and radius>0,'exact parent radius')
    forcing=-pre*residual
    unweighted_error=arb_mat([[arb(0,radius)] for _ in range(n)])
    weighted_error=arb_mat([[arb(0,radius*v)] for v in w])
    propagated=defect*weighted_error;image=forcing+propagated
    norm=max((sum((abs(defect[i,j]).upper()*w[j] for j in range(n)),arb(0))/w[i]).upper() for i in range(n))
    inclusion=[bool(abs(image[i,0]).upper()<radius*w[i]) for i in range(n)]
    forcing_ratio=max((abs(forcing[i,0]).upper()/(radius*w[i])).upper() for i in range(n))
    propagated_ratio=max((abs(propagated[i,0]).upper()/(radius*w[i])).upper() for i in range(n))
    return {'proposal':proposal,'exact_weights':weights,'weighted_norm_upper':ball_json(norm),
        'weighted_forcing_ratio_upper':ball_json(forcing_ratio),'weighted_propagated_ratio_upper':ball_json(propagated_ratio),
        'unweighted_forcing_norm':ball_json(norm_inf_upper(forcing)),
        'unweighted_propagated_norm':ball_json(norm_inf_upper(defect*unweighted_error)),
        'weighted_image':matrix_json(image),'weighted_forcing':matrix_json(forcing),'weighted_propagation':matrix_json(propagated),
        'coordinate_inclusions':inclusion,'contraction_passed':bool(norm.is_finite() and norm<1),
        'inclusion_passed':all(inclusion),'subdomain_of_parent':True,
        'scope':'saved-domain weighted Krawczyk reduction only; no wider-domain Jacobian or branch claim'}


def execute(out):
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source')
    require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    bound={}
    for name in SOURCES:
        data=(ROOT/name).read_bytes();require(data==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed source')
        bound[name]=io.sha256(data)
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();code=2;stopped=False;rows=[]
    runtime={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':bound,
        'pid':os.getpid(),'ppid':os.getppid(),'started_utc':io.now(),'nice':os.getpriority(os.PRIO_PROCESS,0),
        'python_flint':io.flint.__version__,'flint':io.flint.__FLINT_VERSION__,'numpy':np.__version__,'scipy':io.scipy.__version__,
        'threads':{k:os.environ[k] for k in io.THREADS},'internal_seconds':15,'rss_cap_kib':1024**2,'raw_cap':1024**2,'precision':384,'parent_manifest':PIN}
    io.save(out/'runtime-start.json',runtime)
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,15)
    def tick():
        if stopped or time.monotonic()-start>=15:raise TimeoutError('saved analysis cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=1024**2:raise MemoryError('saved analysis RSS')
    outcome={'status':'started'}
    try:
        require(io.entry(PARENT/'manifest.json')['sha256']==PIN,'fixed parent manifest');io.verify_inventory(PARENT,io.read(PARENT/'manifest.json')['files'])
        parent=io.read(PARENT/'certificate.json.gz');require(parent['runtime']['source_commit']=='02ab12123a3a3f5bedba6284f3c984f3e3f85c90','fixed execution')
        require(parent['result']['passed'] and len(parent['result']['charts'])==2,'two proved charts')
        for name,digest in parent['runtime']['source_sha256'].items():require(io.entry(ROOT/name)['sha256']==digest,'unchanged parent source')
        with ctx.workprec(384):
            for index,chart in enumerate(parent['result']['charts']):
                tick();row=reduce_chart(chart);rows.append(row);io.save(out/f'chart-{index}.json',row);tick()
        code=0;outcome.update(status='saved_analysis_completed',scope='no model/derivative/root acquisition; exact saved-domain reductions plus floating proposal only')
    except BaseException as exc:outcome.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)})
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        outcome.update(elapsed_seconds=time.monotonic()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,exit_code=code,completed_charts=len(rows),finished_utc=io.now())
        io.save(out/'outcome.json',outcome);files=io.inventory(out);require(io.total(out)<1024**2,'raw cap');io.save(out/'manifest.json',{'files':files})
        for p in out.rglob('*'):p.chmod(0o444 if p.is_file() else 0o555)
        out.chmod(0o555)
    print(json.dumps(outcome));return code


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();raise SystemExit(execute(a.out))
