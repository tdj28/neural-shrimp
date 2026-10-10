"""Saved-only true branch tangent variation on the failed doubled slice."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,json,resource,signal,subprocess,sys,time
from fractions import Fraction
from pathlib import Path
from flint import ctx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
import certify_s2_period18_stability_slice as acquisition
io=acquisition.io
from neural_shrimp.rigorous_connection import require,restore_ball
from neural_shrimp.rigorous_connection_frames import restore_matrix
from neural_shrimp.rigorous_orbits import exact,vector_json
from neural_shrimp.rigorous_tangent_variation import refine_tangent
CARD='docs/plans/s2-period18-tangent-variation.md'
SOURCES=list(dict.fromkeys(acquisition.SOURCES+['src/neural_shrimp/rigorous_tangent_variation.py','tools/analyze_s2_period18_tangent_variation.py','tools/launch_s2_period18_tangent_variation.py','tests/test_rigorous_tangent_variation.py','tests/test_s2_period18_tangent_variation.py',CARD]))
PARENT=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-doubled-stability-slice-2026-10-10/attempt-001-target/run')
PIN='ac7c9e89a602e4e0f8d120ece947b2827a95fc854a358a0c3046771d10b3bf77'
EXECUTION='b837c59318aeebd3de17673fa74ca5c1f2c2d4e3'
io.CAP=8*1024**2


def joined(inp,result,start,whole,center,physical,phases):
    chart=inp['chart'];ends=acquisition.ENDPOINTS;net=inp['model_inputs']['network']
    require(chart['passed'] and all(chart['gates'].values()) and chart['statement']['period']==18 and chart['statement']['phase_dimension']==13,'same qualified native parent')
    require(inp['exact_endpoints']==result['exact_endpoints']==physical['exact_endpoints']==start['exact_endpoints']==ends,'same closed doubled slice')
    require(start['normal']==chart['normal'] and start['origin']==chart['origin'] and start['preconditioner']==chart['preconditioner'] and start['inherited_image']==chart['image_correction'],'whole same predictor/gauge/root')
    require(whole['iteration']==center['iteration']==2 and whole['retained_error']==result['whole_localization']['final_error'] and center['retained_error']==result['center_localization']['final_error'],'exact completed third intersections')
    lo,hi=map(Fraction,ends);sc=(lo+hi)/2
    require(result['center_localization']['exact_endpoints']==[str(sc),str(sc)] and physical['center_parameter']==str(sc),'actual same-gauge center root')
    velocity=restore_matrix(result['uniform_tangent']['derivative_box'])
    require(result['uniform_tangent']['passed'] and vector_json([velocity[i,0] for i in range(235)])==physical['actual_uniform_tangent'],'exact same uniform derivative enclosure')
    require(vector_json([restore_ball(v) for v in result['whole_localization']['final_state_box']])==physical['actual_whole_box'],'actual whole root-box join')
    require(vector_json([restore_ball(v) for v in result['center_localization']['final_state_box']])==physical['actual_center_box'],'actual center root-box join')
    require(len(phases)==18 and [r['phase'] for r in phases]==list(range(18)),'full native phase order')
    require(type(net['bias']) is bool and type(net['d']) is int and type(net['width']) is int,'exact packing metadata')
    hidden=(net['d']+int(net['bias']))*net['width'];require(hidden+net['width']+int(net['bias'])==13,'exact full state packing')
    return {'pre':restore_matrix(start['preconditioner']),'whole_jac':restore_matrix(whole['jacobian']),'center_jac':restore_matrix(center['jacobian']),
      'phase_j':[restore_matrix(r['whole_J_mixed_diagnostic']) for r in phases],
      'phase_jprime':[restore_matrix(r['whole_total_Jprime']) for r in phases],
      'velocity':velocity,'rate':restore_ball(physical['actual_whole_box'][-1]),'hidden':hidden,
      'weights':[exact(v) for v in chart['exact_weights']],'halfwidth':exact(str((hi-lo)/2))}


def execute(out):
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source');require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    bound={}
    for name in SOURCES:
        data=(ROOT/name).read_bytes();require(data==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed source');bound[name]=io.sha256(data)
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic();code=2;stopped=False
    runtime={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':bound,'pid':os.getpid(),'ppid':os.getppid(),
      'started_utc':io.now(),'nice':os.getpriority(os.PRIO_PROCESS,0),'precision':256,'threads':dict.fromkeys(acquisition.THREADS,'1'),
      'python_flint':io.flint.__version__,'flint':io.flint.__FLINT_VERSION__,'numpy':io.numpy.__version__,'scipy':io.scipy.__version__,
      'internal_seconds':20,'rss_cap_kib':1024**2,'raw_cap':8*1024**2,'parent_manifest':PIN,'model_calls':0,'derivative_calls':0}
    io.save(out/'runtime-start.json',runtime)
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,20)
    def tick():
        if stopped or time.monotonic()-started>=20:raise TimeoutError('saved tangent time cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=1024**2:raise MemoryError('saved tangent RSS cap')
        if io.total(out)>7*1024**2:raise OSError('ordinary raw cap')
    outcome={'status':'started','scientific_claim':False}
    try:
        rt,prior=io.authenticate(PARENT,PIN,EXECUTION)
        require(rt['mode']=='target' and prior['result_status']=='complete_slice_bounds' and prior['stages']==85 and prior['completed_intersections']==6 and prior['completed_physical_phases']==18,'complete fixed doubled acquisition')
        with ctx.workprec(256):
            inp=io.read(PARENT/'inputs.json.gz');result=io.read(PARENT/'slice-result.json.gz')
            args=joined(inp,result,io.read(PARENT/'stage-000-whole_slice_start.json.gz'),io.read(PARENT/'stage-003-whole_slice_intersection.json.gz'),
              io.read(PARENT/'stage-008-center_slice_intersection.json.gz'),io.read(PARENT/'stage-009-physical_inputs.json.gz'),
              [io.read(PARENT/f'stage-{10+j:03d}-phase_derivatives.json.gz') for j in range(18)])
            tick();refined=refine_tangent(**args);tick();payload=io.save(out/'tangent-result.json.gz',refined);io.save(out/'tangent-result-payload.json',payload)
            if refined['passed']:
                tight=restore_matrix(refined['refined_derivative_box']);old=args['velocity']
                diagnostic=[{'coordinate':i,'old_radius':float(old[i,0].rad()),'refined_radius':float(tight[i,0].rad())} for i in range(235)]
                io.save(out/'proposal-widths.json',{'scope':'floating size comparison only; no proof gate','coordinates':diagnostic})
        code=0;outcome.update(status='saved_tangent_variation_completed',conditional_bound_passed=refined['passed'],scope='Derivative enclosure only; new mixed derivatives would be needed to use a narrower tangent in the cocycle.')
    except BaseException as exc:outcome.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)})
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);outcome.update(elapsed_seconds=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,exit_code=code,finished_utc=io.now())
        io.save(out/'outcome.json',outcome);io.save(out/'manifest.json',{'files':io.inventory(out)})
        for path in out.rglob('*'):path.chmod(0o444 if path.is_file() else 0o555)
        out.chmod(0o555)
    print(json.dumps(outcome));return code
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();raise SystemExit(execute(a.out))
