"""Saved-only root-width, metric-error and implicit-tangent budgets."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,json,resource,signal,subprocess,sys,time
from pathlib import Path
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
import certify_s2_period18_arclength_slices as io
from neural_shrimp.rigorous_connection import require,restore_ball
from neural_shrimp.rigorous_connection_frames import restore_matrix,fraction_mid
from neural_shrimp.rigorous_orbits import exact,ball_json,vector_json,matrix_json,eye,midpoint_matrix,norm_inf_upper
CARD='docs/plans/s2-period18-slice-budget-diagnostic.md'
CURRENT='tools/analyze_s2_period18_slice_budget.py'
SOURCES=list(dict.fromkeys(io.SOURCES+[CURRENT,'tools/launch_s2_period18_slice_budget.py','tests/test_s2_period18_slice_budget.py',CARD]))
PARENT=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-arclength-slices-2026-10-10/attempt-001-target/run')
PIN='ccb8f1c8d1cd08389fdcd9ce31d07debb077e6ca7f8d000056d35cba67f84f3a'
EXECUTION='1b3df14e08b5b1ae89a4e834f57295cc22207834'


def tangent_bound(pre,jac,weights):
    n=jac.nrows();require(jac.ncols()==pre.nrows()==pre.ncols()==len(weights)==n,'square implicit system')
    require(all(w.is_exact() and w>0 for w in weights),'exact positive derivative weights')
    d=eye(n)-pre*jac
    def vn(x):return max((abs(x[i,0]).upper()/weights[i]).upper() for i in range(n))
    q=max((sum((abs(d[i,j]).upper()*weights[j] for j in range(n)),arb(0))/weights[i]).upper() for i in range(n))
    result={'weighted_neumann_bound':ball_json(q),'passed':bool(q.is_finite() and q<1)}
    if not result['passed']:return result
    v=pre*arb_mat([[int(i==n-1)] for i in range(n)])
    radius=(vn(d*v)/(1-q)).upper()
    y=arb_mat([[arb(v[i,0],weights[i]*radius)] for i in range(n)])
    result.update(preconditioned_rhs=matrix_json(v),initial_derivative_box=matrix_json(y),weighted_error_bound=ball_json(radius))
    rows=[]
    for _ in range(3):
        image=v+d*y;updated=arb_mat([[y[i,0].intersection(image[i,0])] for i in range(n)])
        rows.append({'image':matrix_json(image),'retained':matrix_json(updated)});y=updated
    result.update(refinements=rows,derivative_box=matrix_json(y),
        scope='conditional uniform bound on true Uprime=A(root)^-1 e_last using the saved final localization Jacobian domain; no new model derivative')
    return result


def phase_budget(block,p,pnext,inverse):
    mid=midpoint_matrix(block);err=block-mid
    pieces={'point_D':p-mid.transpose()*pnext*mid,
            'left_linear':-mid.transpose()*pnext*err,
            'right_linear':-err.transpose()*pnext*mid,
            'quadratic':-err.transpose()*pnext*err}
    return {'J_radius_inf':ball_json(norm_inf_upper(err)),
      'P_inf':ball_json(norm_inf_upper(p)),
      'unscaled_piece_inf':{k:ball_json(norm_inf_upper(v)) for k,v in pieces.items()},
      'point_D_congruence_piece_inf':{k:ball_json(norm_inf_upper(inverse*v*inverse.transpose())) for k,v in pieces.items()}}


def diagnose(result,start,final,chart,tick=lambda:None):
    from fractions import Fraction
    lo,hi=map(Fraction,result['localization']['exact_endpoints']);delta=exact(str((hi-lo)/2))
    slope=restore_matrix(start['slope']);curve=restore_matrix(start['curvature']);error=restore_matrix(final['retained_error'])
    dimensions=[]
    for i in range(slope.nrows()):
        dimensions.append({'index':i,'linear_radius':ball_json((abs(slope[i,0]).upper()*delta).upper()),
          'quadratic_bound':ball_json((abs(curve[i,0]).upper()*delta*delta/2).upper()),
          'retained_error_radius':ball_json(error[i,0].rad().upper())})
    witness=result['stability'];phases=witness['phases'];rows=[]
    for j,phase in enumerate(phases):
        tick();block=restore_matrix(witness['interval_J'][j]);p=restore_matrix(phase['P']);pp=restore_matrix(phases[(j+1)%len(phases)]['P'])
        inv=restore_matrix(phase['point_D_positive']['inverse'])
        rows.append({'phase':j,**phase_budget(block,p,pp,inv),'saved_point_passed':phase['point_passed'],
          'saved_interval_passed':phase.get('interval_D_positive',{}).get('passed',False)})
    tick();tangent=tangent_bound(restore_matrix(start['preconditioner']),restore_matrix(final['jacobian']),[exact(v) for v in chart['exact_weights']])
    return {'exact_endpoints':[str(lo),str(hi)],'coordinate_width_components':dimensions,'phase_error_budgets':rows,
      'implicit_tangent':tangent,'scope':'saved bounds and conditional implicit derivative only; public Arb decode may widen stored balls; no replay of model or independent stability proof'}

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
      'threads':{k:os.environ[k] for k in io.THREADS},'internal_seconds':20,'rss_cap_kib':1024**2,'raw_cap':7*1024**2,'parent_manifest':PIN}
    io.save(out/'runtime-start.json',runtime)
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,20)
    def tick():
        if stopped or time.monotonic()-started_at>=20:raise TimeoutError('saved diagnosis time cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=1024**2:raise MemoryError('saved diagnosis RSS cap')
    outcome={'status':'started','scientific_claim':False}
    try:
        require(io.entry(PARENT/'manifest.json')['sha256']==PIN,'exact failed-parent manifest');io.verify_inventory(PARENT,io.read(PARENT/'manifest.json')['files'])
        rt=io.read(PARENT/'runtime-start.json');outcome0=io.read(PARENT/'outcome.json');inp=io.read(PARENT/'inputs.json.gz')
        require(rt['source_commit']==EXECUTION and rt['mode']=='target' and outcome0['exit_code']==0 and outcome0['completed_cases']==2,'fixed completed two-case pilot')
        for name,digest in rt['source_sha256'].items():
            require(io.entry(ROOT/name)['sha256']==digest and io.sha256(subprocess.check_output(['git','show',EXECUTION+':'+name],cwd=ROOT))==digest,'exact acquisition source')
        summaries=[]
        with ctx.workprec(256):
            for role,offset in [('floor',0),('probe',7)]:
                result=io.read(PARENT/(role+'-result.json.gz'))
                start=io.read(PARENT/f'stage-{offset:03d}-slice_start.json.gz')
                final=io.read(PARENT/f'stage-{offset+3:03d}-slice_intersection.json.gz')
                require(result['role']==role and result['stage_indices']==list(range(offset,offset+7)),'fixed complete stage join')
                tick();value=diagnose(result,start,final,inp['chart'],tick)
                io.save(out/(role+'-budget.json'),value)
                summaries.append({'role':role,'implicit_tangent_bound_passed':value['implicit_tangent']['passed']})
        code=0;outcome.update(status='saved_slice_budgets_completed',cases=summaries,scientific_claim=False)

    except BaseException as exc:outcome.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)})
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);outcome.update(elapsed_seconds=time.monotonic()-started_at,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,exit_code=code,finished_utc=io.now())
        io.save(out/'outcome.json',outcome);require(io.total(out)<7*1024**2,'worker raw cap');io.save(out/'manifest.json',{'files':io.inventory(out)})
        for p in out.rglob('*'):p.chmod(0o444 if p.is_file() else 0o555)
        out.chmod(0o555)
    print(json.dumps(outcome));return code

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();raise SystemExit(execute(a.out))
