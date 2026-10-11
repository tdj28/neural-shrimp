from pathlib import Path
import datetime, hashlib, json, os, platform, resource, subprocess, sys, time
import numpy, scipy
ROOT=Path('/data2/neural-shrimp')
OUT=Path(__file__).resolve().parent
ENV_KEYS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')
TESTS=['tests/test_cycle_reference_selection.py','tests/test_matched_cycle_reference_acquisition.py','tests/test_matched_cycle_reference_evaluation.py','tests/test_matched_cycle_reference_integration.py']
FILES=['tools/cycle_reference_selection.py','tools/matched_cycle_reference_acquisition.py','tools/matched_cycle_reference_evaluation.py',*TESTS]
def binding(path):
    data=path.read_bytes()
    return {'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
before=[binding(ROOT/f) for f in FILES]
env=dict(os.environ,**{k:'1' for k in ENV_KEYS})
argv=[str(ROOT/'.venv/bin/python'),'-m','pytest','-q',*TESTS]
t0=time.monotonic()
with (OUT/'stdout.txt').open('xb') as stdout,(OUT/'stderr.txt').open('xb') as stderr:
    result=subprocess.run(argv,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
after=[binding(ROOT/f) for f in FILES]
receipt={'schema':'exp022-engineering-adapter-validation-v1','started_at':start,'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cwd':str(ROOT),'argv':argv,'thread_environment':{k:env[k] for k in ENV_KEYS},'exit_code':result.returncode,'wall_seconds':time.monotonic()-t0,'source_before':before,'source_after':after,'source_stable':before==after,'python':platform.python_version(),'numpy':numpy.__version__,'scipy':scipy.__version__,'host':platform.node(),'pid':os.getpid(),'proc_stat':Path('/proc/self/stat').read_text(),'child_peak_rss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'outputs':[binding(OUT/'stdout.txt'),binding(OUT/'stderr.txt')],'validation_script':binding(Path(__file__).resolve()),'scope':'Synthetic engineering validation after EXP022 historical metadata found missing core bins. No target neural integration, candidate replacement, or new reference qualification. Includes exact analytic nine-maximum API fixtures and synthetic injected solvers; short real-SciPy fixtures are synthetic oscillators only.'}
with (OUT/'receipt.json').open('x') as handle:json.dump(receipt,handle,indent=2,sort_keys=True,allow_nan=False);handle.write('\n')
print((OUT/'stdout.txt').read_text(),end='')
print(json.dumps({'exit_code':result.returncode,'source_stable':before==after,'receipt':binding(OUT/'receipt.json')},sort_keys=True))
sys.exit(result.returncode)
