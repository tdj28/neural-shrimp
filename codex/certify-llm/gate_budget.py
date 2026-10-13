"""Synthetic full-size Arb gate resource pilot; never evaluates a target map.

Only the seed's A/B numerical proposal arrays are decoded. Every centre, image
and derivative input is synthetic. Perturbed A also makes the prefix arithmetic
high-mantissa; this engineering stress workload is neither a measured upper
bound nor an exact runtime prediction for a future target certificate.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

STARTED=time.monotonic()
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
OWNED=Path('/data2/neural-shrimp-data/codex/llm-p8-certificate')
PERIOD,DIMENSION,PRECISION=8,896,1024
SCHEMA='codex-synthetic-final-gate-budget-plan-v1'
ENVELOPE=dict(cpu_threads=1,nice=10,memory_gib=18,timeout_seconds=1800)
RECIPE=dict(phase_center='all coordinates equal exact integer 4*phase',
    images='exact next synthetic phase center',nominal_radius_power=-800,
    proposal_perturbation_power=-1000,proposal_perturbation_sign='(-1)^(phase+row+column)',
    jacobian_center='exact perturbed A_i',jacobian_radius_power=-1100,
    inverse_proposal='exact binary64 seed B',bound_method='factored',powers=[1,2])
# These are all transitive local imports, including the strict inventory reader.
SOURCE_PATHS=tuple('codex/certify-llm/'+name+'.py'for name in
    ('gate_budget','derivative_evidence','derivative_shard','streamed_head','ball_jet','ball_qwen','ball_linear',
     'cache_io','phase_domain','derivative_coverage','shooting_gates','proof_gates','shooting_certificate'))+('codex/crosscheck/qwen_map.py',)
SEED_FIELDS={'attempt','receipt_sha256','source_commit','preconditioner_sha256','driver_sha256','map_source_sha256'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def valid_hash(value,length=64):
    return type(value)is str and len(value)==length and all(x in '0123456789abcdef'for x in value)


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write_json(path,value):
    """Stream encoding; retain the .partial file if writing is interrupted."""
    path=Path(path);temporary=path.with_suffix(path.suffix+'.partial')
    with temporary.open('w')as f:
        for chunk in json.JSONEncoder(separators=(',',':'),allow_nan=False).iterencode(value):f.write(chunk)
        f.write('\n')
    temporary.replace(path)


def validate_plan(plan,*,owned=None):
    owned=OWNED.resolve()if owned is None else Path(owned).resolve()
    fields={'schema','period','dimension','precision','synthetic_only','seed','recipe','source_sha256','envelope'}
    if type(plan)is not dict or not fields<=set(plan)or set(plan)-fields-{'purpose','basis'}or plan['schema']!=SCHEMA:
        raise ValueError('complete synthetic gate-budget plan required')
    for key,value in [('period',PERIOD),('dimension',DIMENSION),('precision',PRECISION)]:
        if type(plan[key])is not int or plan[key]!=value:raise ValueError('fixed synthetic scope required: '+key)
    if plan['synthetic_only']is not True:raise ValueError('synthetic-only declaration required')
    if type(plan['recipe'])is not dict or canonical(plan['recipe'])!=canonical(RECIPE):raise ValueError('exact deterministic synthetic recipe required')
    if (type(plan['envelope'])is not dict or plan['envelope']!=ENVELOPE
            or any(type(v)is not int for v in plan['envelope'].values())):
        raise ValueError('exact CPU1/nice10/18GiB/1800-second envelope required')
    pins=plan['source_sha256']
    if type(pins)is not dict or set(pins)!=set(SOURCE_PATHS)or any(not valid_hash(v)for v in pins.values()):
        raise ValueError('all reviewed transitive source dependencies must be pinned')
    seed=plan['seed']
    if type(seed)is not dict or set(seed)!=SEED_FIELDS:raise ValueError('complete external seed pins required')
    for key in SEED_FIELDS-{'attempt'}:
        if not valid_hash(seed[key],40 if key=='source_commit'else 64):raise ValueError('exact seed hash required: '+key)
    if type(seed['attempt'])is not str:raise ValueError('owned seed attempt required')
    path=Path(seed['attempt']).resolve()
    if path.parent!=owned or not path.is_dir():raise ValueError('existing direct-child owned seed required')
    return path


def load_proposals(plan):
    """Authenticate all bytes, but decode/access ONLY A and B NPZ members."""
    import numpy as np
    import derivative_evidence as evidence
    path=validate_plan(plan)
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('reviewed runtime source differs: '+name)
    _,receipt,files,manifest=evidence.authenticate_inventory(path,plan['seed']['receipt_sha256'],successful=True)
    if receipt['source_commit']!=plan['seed']['source_commit']:raise ValueError('seed source commit differs')
    for name,key in [('codex/certify-llm/center_seed.py','driver_sha256'),('codex/crosscheck/qwen_map.py','map_source_sha256')]:
        if sha(files['source/'+name])!=plan['seed'][key]:raise ValueError('seed source bytes differ')
    preconditioner=files['run/preconditioner.npz']
    if sha(preconditioner)!=plan['seed']['preconditioner_sha256']:raise ValueError('seed NPZ bytes differ')
    status=json.loads(files['run/status.json'].read_text())
    if (status.get('status')!='proposal_completed'or status.get('certificate_passed')is not False
            or status.get('preconditioner_sha256')!=plan['seed']['preconditioner_sha256']):
        raise ValueError('complete numerical proposal status required')
    with np.load(preconditioner,allow_pickle=False)as saved:
        if len(saved.files)!=7 or set(saved.files)!={'A','B','centers','parameters','product','left_inverse_defect','right_inverse_defect'}:
            raise ValueError('original seven-member seed archive required')
        A=saved['A'].copy();B=saved['B'].copy()
    for array,shape in [(A,(PERIOD,DIMENSION,DIMENSION)),(B,(DIMENSION,DIMENSION))]:
        if array.dtype!=np.float64 or array.shape!=shape or not np.isfinite(array).all():
            raise ValueError('finite full binary64 A/B proposal arrays required')
        array.flags.writeable=False
    return A,B,dict(seed_attempt=str(path),seed_receipt_sha256=plan['seed']['receipt_sha256'],
        preconditioner_sha256=plan['seed']['preconditioner_sha256'],source_commit=receipt['source_commit'],
        artifacts_checked=len(files),captured_source_files_checked=len(manifest['source']),
        decoded_npz_members=['A','B'],target_centers_or_images_decoded=False,git_commit_blob_verification_performed=False,
        scope='Arbitrary seed-derived numerical proposal constants only; no target evaluation or derivative evidence.')


def perturbed_point(value,sign):
    """Exact dyadic sum, admitted only if its mantissa fits the frozen precision."""
    from fractions import Fraction
    from flint import arb
    import phase_domain as domain
    if sign not in (-1,1)or type(sign)is not int:raise ValueError('exact sign required')
    value=float(value)
    if not __import__('math').isfinite(value):raise ValueError('finite seed entry required')
    exact=Fraction(*value.as_integer_ratio())+sign*Fraction(1,2**(-RECIPE['proposal_perturbation_power']))
    pair=domain._pair(exact)
    if abs(pair[0]).bit_length()>PRECISION:raise ValueError('perturbed proposal requires more than 1024 exact mantissa bits')
    point=arb(tuple(pair))
    if not point.is_exact()or domain._record_parts(domain.pack_ball(point),exact=True)[0]!=exact:
        raise ValueError('perturbed proposal lost exactness')
    return point


def build_synthetic(A,B,out,event):
    import numpy as np
    from flint import arb,arb_mat,ctx
    import shooting_gates as shooting
    import proof_gates as proof
    if ctx.prec!=PRECISION or ctx.threads!=1:raise ValueError('frozen arithmetic context required')
    if (A.dtype!=np.float64 or A.shape!=(PERIOD,DIMENSION,DIMENSION)or B.dtype!=np.float64
            or B.shape!=(DIMENSION,DIMENSION)or not np.isfinite(A).all()or not np.isfinite(B).all()):
        raise ValueError('finite complete binary64 proposals required')
    centers=[[arb(4*i)]*DIMENSION for i in range(PERIOD)]
    images=[list(centers[(i+1)%PERIOD])for i in range(PERIOD)]
    radii=[arb((1,RECIPE['nominal_radius_power']))]*PERIOD
    derivatives=[];jacobians=[];phase_records=[];radius=arb((1,RECIPE['jacobian_radius_power']))
    for phase in range(PERIOD):
        began=time.monotonic();event('construction_phase_started',phase_index=phase)
        a=arb_mat(DIMENSION,DIMENSION);j=arb_mat(DIMENSION,DIMENSION);max_bits=0;min_nonzero_bits=None
        for row in range(DIMENSION):
            for column in range(DIMENSION):
                x=perturbed_point(A[phase,row,column],1 if (phase+row+column)%2==0 else -1)
                a[row,column]=x
                if not a[row,column].is_exact()or a[row,column]!=x:raise ValueError('matrix assignment changed exact proposal')
                j[row,column]=arb(x,radius)
                if not j[row,column].is_finite()or not j[row,column].contains(x)or j[row,column].rad()<radius:
                    raise ValueError('synthetic derivative enclosure narrowed')
                bits=abs(int(x.mid().man_exp()[0])).bit_length();max_bits=max(bits,max_bits)
                if bits:min_nonzero_bits=bits if min_nonzero_bits is None else min(bits,min_nonzero_bits)
        construction_seconds=time.monotonic()-began;event('construction_phase_finished',phase_index=phase,seconds=construction_seconds)
        hashing=time.monotonic()
        record=dict(phase_index=phase,synthetic_only=True,certificate_passed=False,
            proposal_sha256=shooting._matrix_sha256(a),jacobian_sha256=shooting._matrix_sha256(j),
            construction_seconds=construction_seconds,hash_seconds=time.monotonic()-hashing,
            minimum_nonzero_proposal_mantissa_bits=min_nonzero_bits,maximum_proposal_mantissa_bits=max_bits,
            derivative_radius_recipe=RECIPE['jacobian_radius_power'])
        write_json(out/f'synthetic-phase-{phase:02d}.json',record);phase_records.append(record)
        derivatives.append(a);jacobians.append(j);event('construction_phase_saved',phase_index=phase,sha256=sha(out/f'synthetic-phase-{phase:02d}.json'))
    inverse=arb_mat(DIMENSION,DIMENSION)
    for i in range(DIMENSION):
        for k in range(DIMENSION):inverse[i,k]=arb(float(B[i,k]))
    proof._matrix(inverse,'exact seed-derived B',n=DIMENSION,exact=True)
    scope=dict(schema='codex-synthetic-affine-phase-map-v1',period=PERIOD,dimension=DIMENSION,recipe=RECIPE,
        local_map='F(x)=c_(i+1)+A_i(x-c_i) on disjoint neighborhoods of each synthetic box; never Qwen F',
        proposal_hashes=[v['proposal_sha256']for v in phase_records],inverse_sha256=shooting._matrix_sha256(inverse))
    contract=shooting.ShootingEnclosureContract(canonical(scope),sha(ROOT/'codex/certify-llm/gate_budget.py'),
        canonical(phase_records),PERIOD,shooting.domain_sha256(centers,radii),
        shooting.enclosures_sha256(images,jacobians),shooting.proposals_sha256(derivatives,inverse),shooting.SHOOTING_DECLARATION)
    write_json(out/'synthetic-inputs.json',dict(synthetic_only=True,certificate_passed=False,scope=scope,
        contract=__import__('dataclasses').asdict(contract),centers=[proof.vector_record(v)for v in centers],
        images=[proof.vector_record(v)for v in images],radii=proof.vector_record(radii),phase_records=phase_records,
        construction='Original base A/B and deterministic exact recipe reconstruct all synthetic inputs. No original seed centers/images are decoded.'))
    return (centers,radii,images,jacobians,derivatives,inverse),contract


def measure_gate(args,contract,out,event):
    from flint import ctx
    from shooting_certificate import check_shooting_certificate
    event('full_conditional_gate_started',precision=ctx.prec,synthetic_only=True)
    began=time.monotonic()
    result=check_shooting_certificate(*args,contract=contract,bound_method='factored',powers=(1,2))
    gate_seconds=time.monotonic()-began
    summary=dict(synthetic_only=True,certificate_passed=False,gate_seconds=gate_seconds,
        synthetic_conditional_gates=result['conditional_gates'],
        synthetic_conditional_gates_passed=result['conditional_gates_passed'],
        scope='Engineering workload only; no target map was evaluated and these are not target enclosures.')
    write_json(out/'gate-summary.json',summary);event('full_conditional_gate_finished',**summary)
    event('full_report_serialization_started');began=time.monotonic()
    write_json(out/'synthetic-gates.json',dict(synthetic_only=True,certificate_passed=False,
        scope=summary['scope'],conditional_synthetic_result=result))
    serialization_seconds=time.monotonic()-began;size=(out/'synthetic-gates.json').stat().st_size
    event('full_report_serialization_finished',seconds=serialization_seconds,bytes=size)
    hashing=time.monotonic();digest=sha(out/'synthetic-gates.json');hash_seconds=time.monotonic()-hashing
    return dict(**summary,serialization_seconds=serialization_seconds,report_bytes=size,
        report_sha256=digest,report_hash_seconds=hash_seconds)


def run(plan,out,event):
    import numpy as np
    import flint
    from flint import ctx
    ctx.prec=PRECISION;ctx.threads=1
    write_json(out/'arithmetic.json',dict(synthetic_only=True,certificate_passed=False,precision=ctx.prec,threads=ctx.threads,
        versions=dict(python=sys.version,python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,numpy=np.__version__)))
    event('seed_admission_started');began=time.monotonic();A,B,admission=load_proposals(plan)
    write_json(out/'seed-admission.json',dict(**admission,synthetic_only=True,certificate_passed=False,seconds=time.monotonic()-began))
    event('seed_admission_finished',seconds=time.monotonic()-began)
    event('base_proposals_copy_started');began=time.monotonic()
    temporary=out/'base-proposals.npz.partial'
    with temporary.open('wb')as f:np.savez(f,A=A,B=B)
    temporary.replace(out/'base-proposals.npz')
    write_json(out/'base-proposals.json',dict(synthetic_only=True,certificate_passed=False,
        sha256=sha(out/'base-proposals.npz'),keys=['A','B'],shape_A=list(A.shape),shape_B=list(B.shape)))
    event('base_proposals_copy_finished',seconds=time.monotonic()-began)
    event('synthetic_inputs_started');began=time.monotonic();args,contract=build_synthetic(A,B,out,event)
    construction_seconds=time.monotonic()-began;event('synthetic_inputs_finished',seconds=construction_seconds)
    measurements=measure_gate(args,contract,out,event)
    return dict(status='synthetic_gate_budget_completed',**measurements,synthetic_inputs_seconds=construction_seconds,
        precision=ctx.prec,threads=ctx.threads,period=PERIOD,dimension=DIMENSION,
        versions=dict(python=sys.version,python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,numpy=np.__version__),
        measured_upper_bound_for_target=False,model_evaluations=0,target_centers_or_images_decoded=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path);parser.add_argument('--plan',required=True,type=Path)
    args=parser.parse_args();out=args.output.resolve()
    if out.name!='run'or out.parent.parent!=OWNED.resolve():raise ValueError('new direct-child owned attempt/run required')
    out.mkdir(parents=True,exist_ok=False);stage='admission'
    def stopped(signum,frame):raise TimeoutError('owned launcher sent SIGTERM')
    old_handler=signal.signal(signal.SIGTERM,stopped)
    def event(name,**data):
        nonlocal stage
        stage=name
        value=dict(data)
        value.update(event=name,utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-STARTED,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,synthetic_only=True,certificate_passed=False)
        with (out/'events.jsonl').open('a')as f:f.write(json.dumps(value,allow_nan=False)+'\n')
    try:
        if args.plan.resolve().parent!=HERE or args.plan.suffix!='.json':raise ValueError('captured committed plan required')
        source=os.environ.get('CERT_SOURCE_COMMIT','')
        if not valid_hash(source,40):raise ValueError('source commit from bounded launcher required')
        plan=json.loads(args.plan.read_text());validate_plan(plan)
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
        os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
        if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
        cap=18*1024**3;soft,hard=resource.getrlimit(resource.RLIMIT_AS)
        cap=min([cap]+[v for v in (soft,hard)if v!=resource.RLIM_INFINITY]);resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
        write_json(out/'inputs.json',dict(plan=plan,plan_sha256=sha(args.plan),source_commit=source,driver_sha256=sha(__file__),
            synthetic_only=True,certificate_passed=False,pid=os.getpid(),effective_address_space_cap=cap,
            statement='No target center/image/derivative evaluation; exact synthetic stress workload, not a forecast or an upper runtime bound.'))
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):result=run(plan,out,event)
        result.update(elapsed_seconds=time.monotonic()-STARTED,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(out/'status.json',result);event('completed')
    except BaseException as error:
        write_json(out/'status.json',dict(status='synthetic_gate_budget_failed',synthetic_only=True,certificate_passed=False,
            phase=stage,error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-STARTED,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        raise
    finally:
        signal.signal(signal.SIGTERM,old_handler)


if __name__=='__main__':main()
