"""Saved-only anisotropic radii proposal for a correlated first anchor."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,json,math,resource,signal,subprocess,sys,time
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.linalg import solve
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
import certify_s2_period18_reanchor as io
from neural_shrimp.rigorous_orbits import exact,ball_json,vector_json
from neural_shrimp.rigorous_connection import require,restore_ball
from neural_shrimp.rigorous_connection_frames import restore_matrix
CARD='docs/plans/s2-period18-correlated-radii-diagnostic.md'
SOURCES=list(dict.fromkeys(io.SOURCES+['tools/analyze_s2_period18_correlated_radii.py','tools/launch_s2_period18_correlated_radii.py','tests/test_correlated_radii_diagnostic.py',CARD,'certificates/s2-period18-fold-receipt.json']))
PARENT=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-reanchor-2026-10-10/attempt-001-target/run')
PIN='a4d46f5d06e52b7c54251031baf847c4164f3f171f00faae96ace24849569885'
EXECUTION='169af5843e79b672e0af4832fd5ba9073ed459aa'
FOLD_RECEIPT='certificates/s2-period18-fold-receipt.json'

def diagnose(chart,observe=lambda row:None):
    pre=restore_matrix(chart['preconditioner']);rr=restore_matrix(chart['predictor_residual']);d=restore_matrix(chart['defect'])
    n=d.nrows();radii=[restore_ball(v) for v in chart['coordinate_radii']]
    require(d.ncols()==pre.nrows()==pre.ncols()==rr.nrows()==len(radii)==n and rr.ncols()==1,'saved dimensions')
    require(all(v.is_exact() and v>0 for v in radii),'exact positive old radii')
    forcing=-pre*rr;prop=d*arb_mat([[arb(0,v)] for v in radii]);image=forcing+prop
    center=[restore_ball(v) for v in chart['center']];s=restore_matrix(chart['predictor_slope']);t=restore_matrix(chart['predictor_curvature'])
    require('exact_parameter' not in chart['anchor'],'correlated anchor required')
    box=[restore_ball(v) for v in chart['anchor']['state_box']]
    normal=[exact(v) for v in chart['normal']];origin=[exact(v) for v in chart['origin']]
    require(len(box)==len(normal)==len(origin)==n,'full correlated anchor')
    projection=sum((normal[i]*(box[i]-origin[i]) for i in range(n)),arb(0))
    dd=projection-exact(chart['parameter_center'])
    anchor=[abs(restore_ball(v)-(center[i]+s[i,0]*dd+t[i,0]*dd*dd/2)).upper() for i,v in enumerate(chart['anchor']['state_box'])]
    rows=[{'index':i,'forcing_upper':ball_json(abs(forcing[i,0]).upper()),'propagated_upper':ball_json(abs(prop[i,0]).upper()),
        'image_ratio_upper':ball_json((abs(image[i,0]).upper()/radii[i]).upper()),'forcing_ratio_upper':ball_json((abs(forcing[i,0]).upper()/radii[i]).upper()),
        'propagated_ratio_upper':ball_json((abs(prop[i,0]).upper()/radii[i]).upper()),'anchor_required_radius':ball_json(anchor[i]),
        'inclusion':bool(abs(image[i,0]).upper()<radii[i])} for i in range(n)]
    observe({'rows':rows,'full_correlated_projection':ball_json(projection),'failed_indices':[r['index'] for r in rows if not r['inclusion']]})
    # Floats suggest a future box only. They never establish a gate or even a
    # valid bound on the changed-domain Jacobian.
    b=np.array([[float(abs(d[i,j]).upper()) for j in range(n)] for i in range(n)])
    a=np.array([float(abs(forcing[i,0]).upper()) for i in range(n)])
    require(np.isfinite(b).all() and np.isfinite(a).all(),'finite saved proposal arrays')
    sol=solve(np.eye(n)-b,a,check_finite=True)
    require(np.isfinite(sol).all() and np.min(sol)>=0,'nonnegative finite radii solution')
    floor=min(float(v) for v in radii)/16
    proposed=np.nextafter(np.maximum(np.maximum(2*sol,2*np.array([float(v) for v in anchor])),floor),np.inf)
    require(np.isfinite(proposed).all() and np.min(proposed)>0,'positive finite proposal')
    # frexp avoids log2 rounding a nextafter-above-power value back to that power.
    mantissa,binary_exponent=math.frexp(float(np.max(proposed)))
    exponent=binary_exponent-1 if mantissa==0.5 else binary_exponent
    scale=Fraction(2)**exponent
    exact_radii=[Fraction(float(v)) for v in proposed];weights=[v/scale for v in exact_radii]
    require(-100<=exponent<=-10 and all(0<v<=1 for v in weights),'normalized exact proposal range')
    exact_weights=[exact(str(v)) for v in weights]
    old_domain_norm=max((sum((abs(d[i,j]).upper()*exact_weights[j] for j in range(n)),arb(0))/exact_weights[i]).upper() for i in range(n))
    return {'full_correlated_projection':ball_json(projection),'old_domain_proposed_weight_norm':ball_json(old_domain_norm),'rows':rows,'original_inclusions':chart['coordinate_inclusions'],'old_exact_weights':chart['exact_weights'],
        'proposal':{'radius_exponent':exponent,'exact_weights':[str(v) for v in weights],'exact_coordinate_radii':[str(v) for v in exact_radii],
        'linear_residual_float':float(np.max(np.abs((np.eye(n)-b)@sol-a))),
        'expanded_old_coordinates':[i for i,v in enumerate(exact_radii) if exact(str(v))>radii[i]],
        'rule':'e=max(2*(I-B)^-1 a,2*old-anchor-requirement,min(old radii)/16), next binary64 upward, then exact dyadic normalized weights',
        'scope':'proposal only; reacquire B on final proposed box and require strict inclusion, weighted contraction and full predecessor containment'},
        'failed_indices':[v['index'] for v in rows if not v['inclusion']]}

def execute(out):
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source')
    require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    bound={}
    for name in SOURCES:
        data=(ROOT/name).read_bytes();require(data==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed source');bound[name]=io.sha256(data)
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();code=2;stopped=False
    runtime={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':bound,
      'pid':os.getpid(),'ppid':os.getppid(),'started_utc':io.now(),'nice':os.getpriority(os.PRIO_PROCESS,0),'precision':256,
      'python_flint':io.flint.__version__,'flint':io.flint.__FLINT_VERSION__,'numpy':np.__version__,'scipy':io.scipy.__version__,
      'threads':{k:os.environ[k] for k in io.THREADS},'internal_seconds':20,'rss_cap_kib':1024**2,'raw_cap':2*1024**2,'parent_manifest':PIN}
    io.save(out/'runtime-start.json',runtime)
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,20)
    def tick():
        if stopped or time.monotonic()-start>=20:raise TimeoutError('saved diagnosis time cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=1024**2:raise MemoryError('saved diagnosis RSS cap')
    outcome={'status':'started'}
    try:
        require(io.entry(PARENT/'manifest.json')['sha256']==PIN,'exact failed parent manifest');io.verify_inventory(PARENT,io.read(PARENT/'manifest.json')['files'])
        rt=io.read(PARENT/'runtime-start.json');outcome_parent=io.read(PARENT/'outcome.json')
        require(rt['source_commit']==EXECUTION and outcome_parent['exit_code']==2 and not outcome_parent['passed'],'fixed retained parent failure')
        for name,digest in rt['source_sha256'].items():
            require(io.entry(ROOT/name)['sha256']==digest and io.entry(PARENT/'source'/name)['sha256']==digest,'unchanged parent source')
            require(io.sha256(subprocess.check_output(['git','show',EXECUTION+':'+name],cwd=ROOT))==digest,'committed parent source')
        tick();certificate=io.read(PARENT/'certificate.json.gz');require(len(certificate['charts'])==1,'one retained first chart')
        failed=certificate['charts'][0]
        require(failed['gates']=={'uniform_root':False,'primitive_period':True,'anchor_identified':True} and not failed['passed'],'exact root-bound failure')
        require(len(failed['center'])==235 and failed['statement']['period']==18 and failed['statement']['phase_dimension']==13,'native235D chart')
        with ctx.workprec(256):
            diagnostic=diagnose(failed,lambda row:io.save(out/'forcing-anchor.json',row));io.save(out/'radii-proposal.json',diagnostic)
        tick();code=0;outcome.update(status='saved_correlated_radii_proposed',failed_indices=diagnostic['failed_indices'],new_domain_evaluated=False)

    except BaseException as exc:outcome.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)})
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);outcome.update(elapsed_seconds=time.monotonic()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,exit_code=code,finished_utc=io.now())
        io.save(out/'outcome.json',outcome);require(io.total(out)<2*1024**2,'worker raw cap');io.save(out/'manifest.json',{'files':io.inventory(out)})
        for p in out.rglob('*'):p.chmod(0o444 if p.is_file() else 0o555)
        out.chmod(0o555)
    print(json.dumps(outcome));return code

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();raise SystemExit(execute(a.out))
