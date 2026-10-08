"""One bounded scalar NS proof attempt or offline, fail-closed receipt replay."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import flint

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from neural_shrimp.rigorous_orbits import canonical_bytes,dyadic,sha256
from neural_shrimp.rigorous_scalar_ns import check_scalar_ns

SOURCES=('src/neural_shrimp/rigorous_orbits.py','src/neural_shrimp/rigorous_scalar_ns.py','tools/certify_scalar_ns.py')
FILES={'inputs.json','certificate.json','runtime-start.json','runtime-end.json'}
SETTINGS=dict(precision=256,radius_exponent=-40,newton_steps=5)
SEED_SHA='cbef5ea46ee1a0253dccbe93fbc4e36523a47f9363533861c8bce09e4381215e'
SEED_PATH=Path('/data2/neural-shrimp-data/explore/scalar-mechanism/attempt-002-pilot/case-15/result.json')


def sources():return {name:sha256((ROOT/name).read_bytes()) for name in SOURCES}


def inputs_from_seed(path):
    raw=path.read_bytes()
    if sha256(raw)!=SEED_SHA:raise ValueError('selected seed receipt hash mismatch')
    seed=json.loads(raw)
    if [seed[k] for k in ('target','ridge','ratio')]!=[1.2,.3,4.]:raise ValueError('selected target changed')
    return dict(schema='scalar-ns-input-v1',target=dyadic(seed['target']),ridge=dyadic(seed['ridge']),
        ratio=dyadic(seed['ratio']),approximation=[dyadic(v) for v in seed['ns_root']['point']+[seed['ns_root']['eta']]],
        provenance=dict(seed_path=str(path.resolve()),seed_sha256=SEED_SHA,case=15,
          selection='disclosed outcome-informed scalar pilot case15; exact integer ratio4, nonresonant proposed trace and nonzero negative coefficient',
          semantics='binary64 coefficients and proposals exported as exact dyadics; eta* is an unknown exact real'))


def compute(inputs):
    try:return check_scalar_ns(inputs,**SETTINGS)
    except Exception as error:
        return dict(schema='scalar-ns-result-v1',passed=False,evidence_rung=0,settings=SETTINGS,
                    research_status='exploratory inconclusive attempt; independent review pending',
                    error=type(error).__name__+': '+str(error))


def summary(result):
    values={k:result[k] for k in ('passed','evidence_rung','research_status')}
    values.update(gates=result.get('gates'),statement=result.get('statement'),error=result.get('error'))
    if 'inclusion' in result:
        values['eta_box']=result['inclusion']['box'][2]
        values['contraction_upper']=result['inclusion']['contraction_upper']
    if 'local' in result:
        values['modulus_slope_at_root']=result['local']['modulus_slope_at_root']
        values['coefficient']=result['local'].get('coefficient')
    return values


def write(path,obj):
    with path.open('xb') as handle:
        handle.write(canonical_bytes(obj));handle.flush();os.fsync(handle.fileno())


def run(seed_path,out):
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    metadata=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       source_sha256=sources(),python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,
       platform=platform.platform(),cpu_threads=1,nice=os.getpriority(os.PRIO_PROCESS,0),pid=os.getpid(),
       gpu=False,paid_compute=False,start_utc=datetime.now(timezone.utc).isoformat())
    write(out/'runtime-start.json',metadata)
    # Source and target selection are fixed before a target outcome is computed.
    subprocess.run(['git','ls-files','--error-unmatch',*SOURCES],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','diff','--quiet','HEAD','--',*SOURCES],cwd=ROOT,check=True)
    inputs=inputs_from_seed(seed_path);write(out/'inputs.json',inputs)
    result=compute(inputs)
    certificate=dict(schema='scalar-ns-certificate-v1',inputs_sha256=sha256(canonical_bytes(inputs)),runtime=metadata,result=result)
    write(out/'certificate.json',certificate)
    ending={**metadata,'elapsed_seconds':time.monotonic()-started,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write(out/'runtime-end.json',ending)
    receipt=dict(schema='scalar-ns-receipt-v1',external_root=str(out.resolve()),
       source_commit=metadata['source_commit'],source_sha256=metadata['source_sha256'],
       library_versions={k:metadata[k] for k in ('python_flint','flint')},
       files={name:dict(bytes=(out/name).stat().st_size,sha256=sha256((out/name).read_bytes())) for name in sorted(FILES)},
       summary=summary(result),elapsed_seconds=ending['elapsed_seconds'],max_rss_kib=ending['max_rss_kib'])
    write(out/'receipt.json',receipt)
    print(json.dumps(receipt,indent=2))
    return 0 if result['passed'] else 2


def verify(path):
    receipt=json.loads(path.read_bytes())
    if receipt.get('schema')!='scalar-ns-receipt-v1':raise ValueError('unknown receipt schema')
    if set(receipt['files'])!=FILES:raise ValueError('file inventory mismatch')
    if set(receipt['source_sha256'])!=set(SOURCES):raise ValueError('source inventory mismatch')
    if receipt['source_sha256']!=sources():raise ValueError('source hash mismatch')
    if receipt['library_versions']!={'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__}:
        raise ValueError('library version mismatch')
    raw=Path(receipt['external_root'])
    if {p.name for p in raw.iterdir()}!=FILES|{'receipt.json'}:raise ValueError('raw inventory mismatch')
    for name,entry in receipt['files'].items():
        data=(raw/name).read_bytes()
        if sha256(data)!=entry['sha256'] or len(data)!=entry['bytes']:raise ValueError('artifact hash mismatch: '+name)
    if json.loads((raw/'receipt.json').read_bytes())!=receipt:raise ValueError('receipt mirror mismatch')
    certificate=json.loads((raw/'certificate.json').read_bytes());inputs=json.loads((raw/'inputs.json').read_bytes())
    if certificate.get('schema')!='scalar-ns-certificate-v1':raise ValueError('unknown certificate schema')
    if certificate['inputs_sha256']!=sha256(canonical_bytes(inputs)):raise ValueError('input binding mismatch')
    start=json.loads((raw/'runtime-start.json').read_bytes());end=json.loads((raw/'runtime-end.json').read_bytes())
    if certificate['runtime']!=start or start['source_sha256']!=receipt['source_sha256'] or start['source_commit']!=receipt['source_commit']:
        raise ValueError('runtime/source binding mismatch')
    if any(end[k]!=v for k,v in start.items()):raise ValueError('runtime end mismatch')
    if any(receipt[k]!=end[k] for k in ('elapsed_seconds','max_rss_kib')):raise ValueError('quoted runtime mismatch')
    result=compute(inputs)
    if canonical_bytes(result)!=canonical_bytes(certificate['result']):raise ValueError('regenerated enclosures differ')
    if summary(result)!=receipt['summary']:raise ValueError('quoted summary mismatch')
    if not result['passed']:raise ValueError('proof gates did not all pass')
    return dict(passed=True,evidence_rung=3,gates=result['gates'],statement=result['statement'],
                certificate_sha256=sha256((raw/'certificate.json').read_bytes()),independent_review='pending')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('run');p.add_argument('--seed-file',type=Path,default=SEED_PATH);p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('verify');p.add_argument('receipt',type=Path)
    args=parser.parse_args()
    if args.action=='run':return run(args.seed_file,args.out)
    try:print(json.dumps(verify(args.receipt),indent=2))
    except Exception as error:
        print(json.dumps(dict(passed=False,error=type(error).__name__+': '+str(error))),file=sys.stderr);return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
