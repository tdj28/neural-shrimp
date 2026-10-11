"""Fixed saved-only near-fold gauge proposals; never an existence or connection test."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,json,resource,signal,subprocess,sys,time
from pathlib import Path
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
import certify_s2_period18_fold_reverse as io
from neural_shrimp.rigorous_connection import require,restore_ball
from neural_shrimp.rigorous_connection_frames import restore_matrix,fraction_mid
from neural_shrimp.rigorous_orbits import exact,ball_json,vector_json
CARD='docs/plans/s2-period18-fold-reverse-restriction-diagnostic.md'
CURRENT='tools/analyze_s2_period18_fold_reverse_restriction.py'
SOURCES=list(dict.fromkeys(io.SOURCES+[CURRENT,'tools/launch_s2_period18_fold_reverse_restriction.py','tests/test_s2_period18_fold_reverse_restriction_diagnostic.py',CARD]))
PARENT=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-fold-reverse-2026-10-10/attempt-001-target/run')
PIN='9aee1856145ff51a5f76ab809118a406ef40d478344217dddef0aa5472432f71'
EXECUTION='136b77b4ef7102a49951d44007610d67fcaf596e'


def diagnose(chart):
    require(chart['gates']=={'uniform_root':False,'primitive_period':True,'anchor_identified':True},'retained root-only failure')
    require(chart['settings']['halfwidth_exponent']==-7 and chart['parameter_center']=='0','fixed failed width and center')
    c=restore_matrix(chart['preconditioner']);terms={k:restore_matrix(v) for k,v in chart['center_path_derivatives'].items()}
    h3=restore_matrix(chart['whole_path_third_derivative']);weights=[exact(v) for v in chart['exact_weights']];n=len(weights)
    require(c.nrows()==c.ncols()==n and all(v>0 for v in weights),'positive fixed weights')
    def wn(v):return max((abs(v[i,0]).upper()/weights[i]).upper() for i in range(n))
    delta=arb((1,-10));d=arb(0,delta)
    parts=[terms['f'],terms['d']*d,terms['d2']*(d*d/2),h3*(d*d*d/6)]
    residual=sum(parts[1:],parts[0]);forcing=c*residual;bound=wn(forcing)
    # One exact dyadic radius, never an interval scan. Its scale is only a
    # proposal; the old D enclosure is not reused as a new-domain proof gate.
    proposed=8*bound.upper();require(proposed.is_finite() and proposed>0,'positive finite forcing proposal')
    ratio=fraction_mid(proposed);exponent=ratio.numerator.bit_length()-ratio.denominator.bit_length()
    from fractions import Fraction
    if Fraction(2)**exponent<ratio:exponent+=1
    require(-100<=exponent<=-10,'bounded one radius proposal')
    old=restore_matrix(chart['defect']);radii=[restore_ball(v) for v in chart['coordinate_radii']]
    propagated=old*arb_mat([[arb(0,v)] for v in radii])
    return {'new_halfwidth_exponent':-10,'radius_exponent':exponent,'exact_weights':chart['exact_weights'],
      'weighted_forcing_upper':ball_json(bound),'weighted_taylor_part_upper':[ball_json(wn(c*v)) for v in parts],
      'old_weighted_forcing_upper':ball_json(wn(c*restore_matrix(chart['predictor_residual']))),
      'old_weighted_propagation_upper':ball_json(wn(propagated)),
      'rule':'one radius power of two at least8 times smaller-domain weighted forcing; old broad H3 retained, all center defects retained',
      'scope':'saved proposal only; fresh narrower predictor plus new correction-domain Jacobian/strict gates required'}

def execute(out):
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source')
    require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    bound={}
    for name in SOURCES:
        data=(ROOT/name).read_bytes();require(data==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed source');bound[name]=io.sha256(data)
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();code=2;stopped=False
    runtime={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':bound,
      'pid':os.getpid(),'ppid':os.getppid(),'started_utc':io.now(),'nice':os.getpriority(os.PRIO_PROCESS,0),'precision':256,
      'python_flint':io.flint.__version__,'flint':io.flint.__FLINT_VERSION__,'numpy':io.numpy.__version__,'scipy':io.scipy.__version__,
      'threads':{k:os.environ[k] for k in io.THREADS},'internal_seconds':30,'rss_cap_kib':1024**2,'raw_cap':3*1024**2,'parent_manifest':PIN}
    io.save(out/'runtime-start.json',runtime)
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,30)
    def tick():
        if stopped or time.monotonic()-start>=30:raise TimeoutError('saved diagnosis time cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=1024**2:raise MemoryError('saved diagnosis RSS cap')
    outcome={'status':'started','scientific_claim':False}
    try:
        require(io.entry(PARENT/'manifest.json')['sha256']==PIN,'exact failed-parent manifest');io.verify_inventory(PARENT,io.read(PARENT/'manifest.json')['files'])
        rt=io.read(PARENT/'runtime-start.json');outcome0=io.read(PARENT/'outcome.json');inp=io.read(PARENT/'inputs.json')
        require(rt['source_commit']==EXECUTION and rt['mode']=='target' and outcome0['completed_charts']==1 and not outcome0['passed'],'fixed failed first chart')
        for name,digest in rt['source_sha256'].items():
            require(io.entry(ROOT/name)['sha256']==digest and io.sha256(subprocess.check_output(['git','show',EXECUTION+':'+name],cwd=ROOT))==digest,'exact failed source')
        chart=io.read(PARENT/'certificate.json.gz')['charts'][0]
        require(chart['anchor']==inp['anchor'] and chart['normal']==inp['normal'] and chart['origin']==inp['origin'],'exact correlated fold anchor and gauge')
        require(inp['provenance']['localized_manifest']==io.PIN,'bound refined156th target')
        with ctx.workprec(256):
            tick();proposal=diagnose(chart);io.save(out/'proposal.json',proposal);tick()
        code=0;outcome.update(status='saved_restriction_proposed',scientific_claim=False)

    except BaseException as exc:outcome.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)})
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);outcome.update(elapsed_seconds=time.monotonic()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,exit_code=code,finished_utc=io.now())
        io.save(out/'outcome.json',outcome);require(io.total(out)<3*1024**2,'worker raw cap');io.save(out/'manifest.json',{'files':io.inventory(out)})
        for p in out.rglob('*'):p.chmod(0o444 if p.is_file() else 0o555)
        out.chmod(0o555)
    print(json.dumps(outcome));return code

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();raise SystemExit(execute(a.out))
