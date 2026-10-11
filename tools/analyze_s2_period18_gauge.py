"""Saved-only tangent/gauge diagnosis at the127-chart endpoint. No model calls."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,json,math,resource,signal,subprocess,sys,time
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.linalg import solve
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
import certify_s2_period18_arclength_reboxing as io
from neural_shrimp.rigorous_orbits import exact,ball_json,vector_json,midpoint_matrix,norm_inf_upper
from neural_shrimp.rigorous_connection import require,restore_ball
from neural_shrimp.rigorous_connection_frames import restore_matrix,fraction_mid
CARD='docs/plans/s2-period18-arclength-gauge-diagnostic.md'
SOURCES=list(dict.fromkeys(io.SOURCES+['tools/analyze_s2_period18_gauge.py','tools/launch_s2_period18_gauge.py','tests/test_arclength_gauge_diagnostic.py',CARD,'certificates/s2-period18-fold-receipt.json']))
PARENT=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-arclength-reboxing-2026-10-10/unit-001/run')
PIN='dd0bae85fe2ada8f90d0b50aa1d0284e71a87961a8925026a3f74fe8aad0d7dd'
EXECUTION='0e48ce6bc3f87b8e2fac12da14fc7e28648e50f3'
FOLD_RECEIPT='certificates/s2-period18-fold-receipt.json'

def project(box,normal,origin):
    require(len(box)==len(normal)==len(origin),'full projection dimensions')
    return sum((normal[i]*(box[i]-origin[i]) for i in range(len(box))),arb(0))


def gauge_proposal(chart):
    require(chart['passed'] and all(chart['gates'].values()),'qualified parent')
    endpoint=[restore_ball(v) for v in chart['endpoint_root_enclosures'][1]]
    slope=restore_matrix(chart['predictor_slope']);curvature=restore_matrix(chart['predictor_curvature'])
    size=len(endpoint);delta=arb((1,chart['settings']['halfwidth_exponent']))
    require(slope.nrows()==curvature.nrows()==size and slope.ncols()==curvature.ncols()==1,'tangent dimensions')
    tangent=[slope[i,0]+curvature[i,0]*delta for i in range(size)]
    length=sum((v*v for v in tangent),arb(0)).sqrt()
    require(length.is_finite() and length>0,'positive finite proposed tangent length')
    normal=[exact(str(fraction_mid(v/length))) for v in tangent]
    origin=[v.mid() for v in endpoint]
    require(all(v.is_finite() and v.is_exact() for v in normal+origin),'fixed exact proposed gauge')
    require(sum((normal[i]*tangent[i] for i in range(size)),arb(0))>0,'forward orientation')
    projection=project(endpoint,normal,origin)
    # Deliberately conservative full-box diagnostic. No root is replaced by
    # its midpoint, nor is this approximate tangent treated as its derivative.
    linear_direction=[v/length for v in tangent]
    anchor_need=[abs(endpoint[i]-origin[i]-linear_direction[i]*projection).upper() for i in range(size)]
    return {'normal':[str(fraction_mid(v)) for v in normal],'origin':[str(fraction_mid(v)) for v in origin],
      'endpoint_gauge_box':ball_json(projection),'endpoint_box':vector_json(endpoint),
      'proposed_tangent_length':ball_json(length),'proposed_normal_squared_norm':ball_json(sum((v*v for v in normal),arb(0))),
      'endpoint_taylor_tangent':vector_json(tangent),'linearized_anchor_required_radii':vector_json(anchor_need),
      'max_linearized_anchor_radius':ball_json(max(anchor_need)),
      'scope':'exact dyadic gauge proposal from endpoint Taylor tangent, not a validated root tangent; full-box projection retained'}


def bordered_diagnostics(chart,normal,tick=lambda:None):
    whole=restore_matrix(chart['jacobian_tube']);point=midpoint_matrix(whole);size=point.nrows()
    require(point.ncols()==size==len(normal),'bordered diagnostic dimensions')
    rows=[]
    for label,lastrow in [('old',[exact(v) for v in chart['normal']]),('proposed',normal)]:
        tick();matrix=arb_mat([[point[i,j] if i<size-1 else lastrow[j] for j in range(size)] for i in range(size)])
        try:
            inverse=matrix.inv();bound=(norm_inf_upper(matrix)*norm_inf_upper(inverse)).upper()
            rows.append({'gauge':label,'invertible_saved_midpoint_matrix':bool(bound.is_finite()),'condition_inf_upper':ball_json(bound),
              'inverse_inf_upper':ball_json(norm_inf_upper(inverse))})
        except Exception as exc:rows.append({'gauge':label,'invertible_saved_midpoint_matrix':False,'error':str(exc)})
    return {'diagnostics':rows,'scope':'exact midpoint of saved tube matrix, with row replacement; not an acquired endpoint Jacobian or whole-domain condition bound'}


def fold_projections(orbit,rate,normal,origin,endpoint_projection,delta):
    p=len(orbit);n=len(orbit[0]);require(p>0 and all(len(row)==n for row in orbit) and len(normal)==p*n+1,'full fold dimensions')
    rows=[]
    for phase in range(p):
        box=[restore_ball(v) for j in range(p) for v in orbit[(j+phase)%p]]+[restore_ball(rate)]
        projection=project(box,normal,origin);gap=projection-endpoint_projection
        rows.append({'cyclic_shift':phase,'projected_full_fold_box':ball_json(projection),'projected_gap_box':ball_json(gap),
          'nominal_chart_count_from_projected_midpoint':math.ceil(abs(float(gap.mid()))/(2*float(delta))),
          'midpoint_distance_inf':ball_json(max(abs(box[i].mid()-origin[i]).upper() for i in range(len(box))))})
    return {'alignments':rows,'scope':'all native cyclic projections; count is a heuristic using fixed gauge width, no connection or inclusion claim'}


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
        require(io.entry(PARENT/'manifest.json')['sha256']==PIN,'exact qualified parent manifest');io.verify_inventory(PARENT,io.read(PARENT/'manifest.json')['files'])
        rt=io.read(PARENT/'runtime-start.json');batch=io.read(PARENT/'batch.json');inp=io.read(PARENT/'inputs.json')
        require(rt['source_commit']==EXECUTION and batch['qualified_charts']==16 and batch['completed_charts']==16 and batch['passed'],'fixed complete16-chart parent')
        for name,digest in rt['source_sha256'].items():require(io.entry(ROOT/name)['sha256']==digest,'unchanged parent source')
        tick();last=io.read(PARENT/'chart-015.json.gz')
        require(last['passed'] and all(last['gates'].values()) and last['exact_parameter_endpoints'][1]=='253/64','last qualified native endpoint')
        require(last['statement']['period']==18 and last['statement']['phase_dimension']==13 and last['statement']['shooting_dimension']==235,'native235D parent')
        foldrec=io.read(ROOT/FOLD_RECEIPT);foldroot=Path(foldrec['external_root']);require(foldrec['passed'],'proved fold receipt')
        for name,identity in foldrec['files'].items():require(io.entry(foldroot/name)==identity,'fold artifact identity')
        foldenv=io.read(foldroot/'run/certificate.json');fold=foldenv['result'];foldinp=io.read(foldroot/'run/inputs.json')
        require(fold['passed'] and all(fold['gates'].values()) and foldinp['period']==18,'qualified native fold')
        fields=('schema','network','x','y','lr_output')
        require(all(inp['model_inputs'][k]==foldinp['map_inputs'][k] for k in fields),'exact common map/data/output')
        with ctx.workprec(256):
            proposal=gauge_proposal(last);io.save(out/'gauge-proposal.json',proposal);tick()
            normal=[exact(v) for v in proposal['normal']];origin=[exact(v) for v in proposal['origin']]
            diagnostic=bordered_diagnostics(last,normal,tick);io.save(out/'bordered-diagnostic.json',diagnostic);tick()
            projections=fold_projections(fold['orbit_enclosures'][:18],fold['box'][13],normal,origin,
                restore_ball(proposal['endpoint_gauge_box']),arb((1,last['settings']['halfwidth_exponent'])))
            io.save(out/'fold-projections.json',projections)
        tick();code=0;outcome.update(status='saved_gauge_diagnosis_completed',new_root_or_connection_claim=False)

    except BaseException as exc:outcome.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)})
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);outcome.update(elapsed_seconds=time.monotonic()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,exit_code=code,finished_utc=io.now())
        io.save(out/'outcome.json',outcome);require(io.total(out)<2*1024**2,'worker raw cap');io.save(out/'manifest.json',{'files':io.inventory(out)})
        for p in out.rglob('*'):p.chmod(0o444 if p.is_file() else 0o555)
        out.chmod(0o555)
    print(json.dumps(outcome));return code

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();raise SystemExit(execute(a.out))
