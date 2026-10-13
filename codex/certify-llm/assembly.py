"""Authenticate final shooting evidence and run conditional gates, never certify.

No transformer, optimizer or derivative evaluator runs here. Exact target
constants are read as literals from the externally pinned independent source;
all mathematical enclosures must come from complete authenticated attempts.
Final independent result review/finalization is a separate unimplemented step.
"""
from __future__ import annotations
import time
STARTED=time.monotonic()
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import sys
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
OWNED=Path('/data2/neural-shrimp-data/codex/llm-p8-certificate')
PERIOD,DIMENSION,COLUMNS,PRECISION=8,896,64,1024
SCHEMA='codex-final-shooting-assembly-plan-v1'
ENVELOPE=dict(cpu_threads=1,nice=10,memory_gib=18,timeout_seconds=3600)
REVIEW_SCOPES=('point_evaluator','derivative_evaluator','map_regularity','gate_mathematics')
# Loaded lazily, after CLI resource controls. No model constructor is imported
# or invoked as part of the actual evidence/gate path.
def modules():
    import point_evidence,derivative_evidence,shooting_certificate,shooting_gates
    return point_evidence,derivative_evidence,shooting_certificate,shooting_gates


def sources():
    point,_,_,_=modules()
    return tuple(sorted(set(point.SOURCES)|{'codex/certify-llm/assembly.py','codex/certify-llm/shooting_certificate.py'}))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def same(left,right,label):
    if canonical(left)!=canonical(right):raise ValueError(label+' differs')


def hash_value(value,label):
    if type(value)is not str or len(value)!=64 or any(c not in '0123456789abcdef'for c in value):
        raise ValueError('external lowercase SHA256 required: '+label)


def write_json(path,value):
    """Atomic completion, retaining interrupted streaming output for diagnosis."""
    path=Path(path);partial=path.with_suffix(path.suffix+'.partial')
    with partial.open('w')as f:
        for piece in json.JSONEncoder(sort_keys=True,separators=(',',':'),allow_nan=False).iterencode(value):f.write(piece)
        f.write('\n');f.flush();os.fsync(f.fileno())
    partial.replace(path)
    directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)


def _file_pin(record,label,*,owned=None):
    if type(record)is not dict or set(record)!={'path','sha256'}or type(record['path'])is not str:
        raise ValueError('complete original file pin required: '+label)
    hash_value(record['sha256'],label);path=Path(record['path'])
    owned=OWNED.resolve()if owned is None else Path(owned).resolve()
    if not path.is_absolute()or path.resolve()!=path or not path.is_relative_to(owned)or not path.is_file()or path.is_symlink():
        raise ValueError('original owned file required: '+label)
    return path


def original_json(record,label):
    path=_file_pin(record,label);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=record['sha256']:raise ValueError('original bytes differ: '+label)
    return parse_original(raw),raw


def parse_original(raw):
    def unique(pairs):
        out={}
        for key,value in pairs:
            if key in out:raise ValueError('duplicate original JSON key: '+key)
            out[key]=value
        return out
    return json.loads(raw,object_pairs_hook=unique)


def _attempt(path):
    if type(path)is not str:raise ValueError('owned original attempt required')
    p=Path(path)
    if not p.is_absolute()or p.resolve()!=p or p.parent!=OWNED.resolve()or not p.is_dir()or p.is_symlink():
        raise ValueError('original direct-child attempt required')
    return p


def validate_plan(plan):
    point,_,_,gates=modules()
    fields={'schema','period','dimension','precision','definition','point_admission','source_sha256',
        'map','map_sha256','point_backend','derivative_backend','assembly_backend','derivative_prefix',
        'seed','shards','reviews','gate','envelope','enclosure_declaration'}
    if type(plan)is not dict or not fields<=set(plan)or set(plan)-fields-{'purpose','basis'}or plan['schema']!=SCHEMA:
        raise ValueError('complete known final assembly plan required')
    for key,expected in [('period',PERIOD),('dimension',DIMENSION),('precision',PRECISION)]:
        if type(plan[key])is not int or plan[key]!=expected:raise ValueError('fixed assembly scope required: '+key)
    if (point.base.PERIOD,point.base.DIMENSION,point.derivative.PERIOD,point.derivative.DIMENSION,point.derivative.COLUMNS)!=(PERIOD,DIMENSION,PERIOD,DIMENSION,COLUMNS):
        raise ValueError('point/derivative/full assembly dimensions disagree')
    if DIMENSION%COLUMNS:raise ValueError('complete aligned derivative coverage required')
    _file_pin(plan['definition'],'definition');_file_pin(plan['point_admission'],'point admission')
    pins=plan['source_sha256']
    if type(pins)is not dict or set(pins)!=set(sources()):raise ValueError('all wrapper/transitive sources required')
    for name,value in pins.items():hash_value(value,name)
    if type(plan['map'])is not dict:raise ValueError('independent exact map recipe required')
    hash_value(plan['map_sha256'],'map')
    if canonical(plan['map'])!=plan['map_sha256']:raise ValueError('map identity differs')
    for key,precision in [('point_backend',2048),('derivative_backend',1024),('assembly_backend',PRECISION)]:
        backend=plan[key]
        if type(backend)is not dict or set(backend)!={'python','python_flint','flint','numpy','torch','precision','threads'}:
            raise ValueError('complete arithmetic backend required: '+key)
        if type(backend['precision'])is not int or backend['precision']!=precision or type(backend['threads'])is not int or backend['threads']!=1:
            raise ValueError('fixed arithmetic context required: '+key)
    prefix=plan['derivative_prefix']
    if type(prefix)is not dict or set(prefix)!=point.evidence.PREFIX_FIELDS:raise ValueError('separate derivative prefix contract required')
    _attempt(prefix['attempt'])
    for name in('receipt_sha256','map_source_sha256','cache_sha256'):hash_value(prefix[name],name)
    if prefix['completion']not in('retained_prefix_before_box_failure','completed_prefix_point'):
        raise ValueError('reviewed prefix-stage outcome required')
    seed=plan['seed'];seed_fields={'attempt','receipt_sha256','source_commit','preconditioner_sha256','driver_sha256','map_source_sha256'}
    if type(seed)is not dict or set(seed)!=seed_fields:raise ValueError('complete original numerical seed pins required')
    _attempt(seed['attempt'])
    for name in seed_fields-{'attempt','source_commit'}:hash_value(seed[name],name)
    if type(seed['source_commit'])is not str or len(seed['source_commit'])!=40 or any(c not in '0123456789abcdef'for c in seed['source_commit']):
        raise ValueError('original seed commit required')
    refs=plan['shards'];expected=[(i,j)for i in range(PERIOD)for j in range(0,DIMENSION,COLUMNS)]
    if type(refs)is not list or len(refs)!=len(expected):raise ValueError('all assigned final derivative shards required')
    paths=set()
    for ref,assignment in zip(refs,expected):
        if type(ref)is not dict or set(ref)!={'phase_index','start_column','attempt','receipt_sha256'}:
            raise ValueError('complete assigned shard reference required')
        if type(ref['phase_index'])is not int or type(ref['start_column'])is not int or (ref['phase_index'],ref['start_column'])!=assignment:
            raise ValueError('canonical complete phase/column assignment required')
        path=_attempt(ref['attempt']);hash_value(ref['receipt_sha256'],'shard receipt')
        if path in paths:raise ValueError('duplicate shard attempt')
        paths.add(path)
    reviews=plan['reviews']
    if type(reviews)is not list or len(reviews)!=len(REVIEW_SCOPES):raise ValueError('all external hypothesis reviews required')
    for record,scope in zip(reviews,REVIEW_SCOPES):
        if type(record)is not dict or set(record)!={'scope','path','sha256'}or record['scope']!=scope:
            raise ValueError('explicit independently reviewed obligation scope required')
        path=record['path'];hash_value(record['sha256'],'review')
        if type(path)is not str or Path(path).is_absolute()or '..'in Path(path).parts or not path.startswith('codex/certify-llm/'):
            raise ValueError('source-bound local review path required')
    same(plan['gate'],dict(bound_method='factored',powers=[1,2],stability_weights=None),'frozen conditional gates')
    if type(plan['envelope'])is not dict or any(type(v)is not int for v in plan['envelope'].values()):raise ValueError('exact resource envelope required')
    same(plan['envelope'],ENVELOPE,'provisional bounded envelope')
    if plan['enclosure_declaration']!=gates.SHOOTING_DECLARATION:raise ValueError('explicit external theorem hypotheses required')
    return plan


def torch_version_metadata():
    """Read installed version literals without importing/executing Torch.

    Distribution metadata can omit a local build suffix (e.g. +cu130); the
    producer records torch.__version__, whose literal lives in version.py.
    """
    distribution=importlib.metadata.distribution('torch')
    path=Path(distribution.locate_file('torch/version.py')).resolve()
    raw=path.read_bytes();tree=ast.parse(raw);values=[]
    for node in tree.body:
        if isinstance(node,ast.Assign)and any(isinstance(target,ast.Name)and target.id=='__version__'for target in node.targets):
            values.append(ast.literal_eval(node.value))
        elif isinstance(node,ast.AnnAssign)and isinstance(node.target,ast.Name)and node.target.id=='__version__':
            values.append(ast.literal_eval(node.value))
    if len(values)!=1 or type(values[0])is not str or not values[0]:raise ValueError('unambiguous literal producer Torch version required')
    return dict(distribution_version=distribution.version,producer_version=values[0],
        version_file=str(path),version_file_sha256=hashlib.sha256(raw).hexdigest(),
        torch_imported_by_reader=False,metadata_file_executed=False)


def expected_constants():
    """Read only reviewed literal assignments; never import torch/model code."""
    tree=ast.parse((ROOT/'codex/crosscheck/qwen_map.py').read_text())
    constants={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name)and target.id in('PINNED_SHA256','PREFIX_IDS'):
                    if target.id in constants:raise ValueError('ambiguous pinned constant assignment')
                    constants[target.id]=ast.literal_eval(node.value)
    if set(constants)!={'PINNED_SHA256','PREFIX_IDS'}:raise ValueError('independent map constants missing')
    return constants['PINNED_SHA256'],list(constants['PREFIX_IDS']),torch_version_metadata()['producer_version']


def source_and_reviews(plan):
    for name,pin in plan['source_sha256'].items():
        if sha(ROOT/name)!=pin:raise ValueError('externally reviewed source changed: '+name)
    for review in plan['reviews']:
        if sha(ROOT/review['path'])!=review['sha256']:raise ValueError('external obligation review bytes changed')


def verify_exact_proposals(original_A,original_B,A,B):
    """Compare every native value with an independently decoded binary64 dyadic.

    A different exact proposal is mathematically permitted by the theorem, but
    cannot be attributed to the byte-pinned original seed. Signed zero denotes
    the same exact real number; the original NPZ still preserves its bit pattern.
    """
    from flint import arb
    if len(A)!=PERIOD:raise ValueError('all converted original A phases required')
    count=0
    for original,native in list(zip(original_A,A))+[(original_B,B)]:
        if native.nrows()!=DIMENSION or native.ncols()!=DIMENSION or original.shape!=(DIMENSION,DIMENSION):
            raise ValueError('original/native proposal shapes differ')
        for row in range(DIMENSION):
            for column in range(DIMENSION):
                numerator,denominator=float(original[row,column]).as_integer_ratio()
                if denominator<=0 or denominator&(denominator-1):raise ValueError('binary64 dyadic denominator required')
                expected=arb((numerator,-(denominator.bit_length()-1)))
                actual=native[row,column]
                if not actual.is_finite()or not actual.is_exact()or actual!=expected:
                    raise ValueError('native proposal differs from original binary64 exact value')
                count+=1
    return dict(native_conversion_exact=True,original_binary64_values_compared=count,
        comparison='Independent float.as_integer_ratio to exact dyadic tuple, every A_i and B entry; no proposal arithmetic.')


def cross_bind(plan,definition,point_report,derivative_report,centers,radii,images,jacobians,A,B):
    point,_,_,gates=modules();domain=point.domain
    original_c=[[domain.pack_ball(v)for v in row]for row in centers]
    original_r=[domain.pack_ball(v)for v in radii]
    same(original_c,definition['centers'],'native original exact centers');same(original_r,definition['radii'],'native exact nominal radii')
    _,_,mixed,shooting=point.coverage._nominal(original_c,original_r,PERIOD,DIMENSION)
    same(point_report['phase_domain_sha256'],mixed,'point nominal-domain identity')
    same(point_report['shooting_domain_sha256'],shooting,'point shooting-domain identity')
    coverage=derivative_report['coverage']
    for key,value in [('phase_domain_sha256',mixed),('shooting_domain_sha256',shooting)]:same(coverage[key],value,'derivative '+key)
    for key,value in [('coverage_complete',True),('phase_column_counts',[DIMENSION]*PERIOD),('total_columns',PERIOD*DIMENSION)]:same(coverage[key],value,'full '+key)
    matrices=[gates._matrix_sha256(j)for j in jacobians]
    same(matrices,coverage['phase_matrix_sha256'],'actual native derivative matrices')
    # Both original enclosures contain F(c_i) if their external contracts hold.
    # A contradiction cannot be repaired by wider assembly-time decoding.
    intersections=coverage['original_primal_intersections']
    for phase in range(PERIOD):
        for coordinate in range(DIMENSION):
            original=point_report['original_images'][phase][coordinate]
            mid,rad=domain._record_parts(original)
            interval=intersections[phase][coordinate]
            lower=domain._fraction(interval['lower']);upper=domain._fraction(interval['upper'])
            if max(mid-rad,lower)>min(mid+rad,upper):
                raise ValueError('original point and whole-box primal enclosures are disjoint')
    same(canonical([[domain.pack_ball(v)for v in row]for row in images]),point_report['actual_native_images_sha256'],'actual native images')
    same(point_report['derivative_map'],plan['map'],'point bridge exact map')
    same(derivative_report['map_sha256'],plan['map_sha256'],'derivative exact map')
    same(point_report['definition_sha256'],plan['definition']['sha256'],'point frozen definition')
    same(derivative_report['design_sha256'],plan['definition']['sha256'],'derivative frozen definition')
    same(point_report['point_backend'],plan['point_backend'],'point backend')
    same(point_report['derivative_backend'],plan['derivative_backend'],'derivative backend bridge')
    same(derivative_report['backend_sha256'],canonical(plan['derivative_backend']),'derivative backend digest')
    same(point_report['assembly_precision'],PRECISION,'point assembly precision')
    same(derivative_report['assembly_precision'],PRECISION,'derivative assembly precision')
    same(point_report['assembly_threads'],1,'point assembly threads')
    expected_refs=plan['shards'];audits=derivative_report['authenticated_attempts']
    if len(audits)!=len(expected_refs):raise ValueError('every assigned shard must be authenticated')
    for ref,audit in zip(expected_refs,audits):
        for key in('attempt','receipt_sha256','phase_index'):same(audit[key],ref[key],'assigned original shard '+key)
        same(audit['direction_indices'],list(range(ref['start_column'],ref['start_column']+COLUMNS)),'assigned derivative block')
    return dict(phase_domain_sha256=mixed,shooting_domain_sha256=shooting,
        enclosures_sha256=gates.enclosures_sha256(images,jacobians),proposals_sha256=gates.proposals_sha256(A,B),
        actual_native_images_sha256=point_report['actual_native_images_sha256'],phase_matrix_sha256=matrices,
        original_centers_sha256=canonical(original_c),original_radii_sha256=canonical(original_r))


def assemble(plan,out,event,*,frozen_plan_path,frozen_plan_sha256,expected_assets,prefix_ids,torch_version):
    """Pure evidence/gate path; independent expectations are not artifact-selected."""
    from flint import arb,ctx
    point,evidence,certificate,gates=modules();base=point.base
    hash_value(frozen_plan_sha256,'original assembly plan')
    frozen_bytes=Path(frozen_plan_path).read_bytes()
    if hashlib.sha256(frozen_bytes).hexdigest()!=frozen_plan_sha256:raise ValueError('original assembly plan bytes differ')
    same(parse_original(frozen_bytes),plan,'original assembly plan object')
    validate_plan(plan);source_and_reviews(plan)
    if (ctx.prec,ctx.threads)!=(PRECISION,1):raise ValueError('frozen assembly context required')
    mapping,derivative_map=point.map_bridge(expected_assets,prefix_ids)
    same(plan['map'],derivative_map,'independent exact map')
    current_backend=point.worker.backend(torch_version)
    same(plan['assembly_backend'],current_backend,'assembly arithmetic backend')
    same(plan['point_backend'],dict(current_backend,precision=2048),'point arithmetic backend')
    same(plan['derivative_backend'],dict(current_backend,precision=1024),'derivative arithmetic backend')
    definition,definition_bytes=original_json(plan['definition'],'definition')
    point_plan,point_plan_bytes=original_json(plan['point_admission'],'point admission')
    # These external original byte copies survive later failures.
    (out/'candidate-definition.json').write_bytes(definition_bytes)
    (out/'point-admission-plan.json').write_bytes(point_plan_bytes)
    seed=plan['seed'];bp=point_plan['base_plan']
    for key,other in [('attempt','center_seed_attempt'),('receipt_sha256','center_seed_receipt_sha256'),
            ('preconditioner_sha256','preconditioner_sha256'),('driver_sha256','center_seed_driver_sha256'),('map_source_sha256','center_seed_map_source_sha256')]:
        same(seed[key],bp[other],'same original numerical seed '+key)
    if Path(bp['prefix_attempt'])==Path(plan['derivative_prefix']['attempt']):raise ValueError('distinct precision-compatible prefix attempts required')
    event('point_admission_started');began=time.monotonic()
    centers,images,point_report=point.authenticate_points(plan['point_admission']['path'],plan['point_admission']['sha256'],
        plan['definition']['path'],plan['definition']['sha256'],expected_assets=expected_assets,prefix_ids=prefix_ids,
        torch_version=torch_version,expected_source_sha256={n:plan['source_sha256'][n]for n in point.SOURCES},
        expected_derivative_backend=plan['derivative_backend'],expected_derivative_prefix=plan['derivative_prefix'])
    write_json(out/'point-admission.json',point_report);event('point_admission_completed',seconds=time.monotonic()-began,sha256=sha(out/'point-admission.json'))
    event('derivative_admission_started');began=time.monotonic()
    refs=({key:ref[key]for key in('attempt','receipt_sha256')}for ref in plan['shards'])
    jacobians,derivative_report=evidence.authenticate_and_collect(plan['definition']['path'],plan['definition']['sha256'],
        expected_map=plan['map'],expected_backend=plan['derivative_backend'],
        expected_source_sha256={n:plan['source_sha256'][n]for n in point.derivative.SOURCE_PATHS},expected_prefix=plan['derivative_prefix'],attempts=refs)
    write_json(out/'derivative-admission.json',derivative_report);event('derivative_admission_completed',seconds=time.monotonic()-began,sha256=sha(out/'derivative-admission.json'))
    event('proposal_admission_started');began=time.monotonic();arrays,binding=base.load_bound_inputs(bp,expected_assets,prefix_ids)
    same(binding['seed_source'],seed['source_commit'],'original numerical seed source')
    A=[base.exact_matrix(array)for array in arrays['A']];B=base.exact_matrix(arrays['B'])
    conversion=verify_exact_proposals(arrays['A'],arrays['B'],A,B);del arrays
    # Exactness/shape admission is also repeated by proposal hashing.
    proposal_record=dict(certificate_passed=False,original_seed=seed,source_binding=binding,
        native_proposals_sha256=gates.proposals_sha256(A,B),proposal_only=True,optimizer_actions=0,
        unused_seed_centers_not_substituted=True,conversion=conversion)
    write_json(out/'proposal-admission.json',proposal_record);event('proposal_admission_completed',seconds=time.monotonic()-began,sha256=sha(out/'proposal-admission.json'))
    radii=[arb(tuple(record['mid']))for record in definition['radii']]
    event('native_binding_started');began=time.monotonic()
    bindings=cross_bind(plan,definition,point_report,derivative_report,centers,radii,images,jacobians,A,B)
    same(bindings['proposals_sha256'],proposal_record['native_proposals_sha256'],'actual converted proposals')
    evaluator_bundle=dict(schema='codex-combined-point-derivative-evaluator-source-v1',
        source_sha256={n:plan['source_sha256'][n]for n in point.SOURCES},
        point_map=mapping,derivative_map=derivative_map,reviews=plan['reviews'])
    admission=dict(schema='codex-final-shooting-admission-v1',certificate_passed=False,
        artifact_authentication_completed=True,independent_final_result_review_pending=True,
        original_plan_sha256=frozen_plan_sha256,canonical_plan_sha256=canonical(plan),definition=plan['definition'],point_admission_plan=plan['point_admission'],
        point_report_sha256=sha(out/'point-admission.json'),derivative_report_sha256=sha(out/'derivative-admission.json'),
        proposal_report_sha256=sha(out/'proposal-admission.json'),original_seed=seed,shards=plan['shards'],
        evaluator_bundle=evaluator_bundle,evaluator_source_sha256=canonical(evaluator_bundle),
        source_sha256=plan['source_sha256'],map_sha256=plan['map_sha256'],
        point_backend=plan['point_backend'],derivative_backend=plan['derivative_backend'],assembly_backend=current_backend,
        bindings=bindings,original_centers=definition['centers'],nominal_radii=definition['radii'],
        native_images=[[point.domain.pack_ball(v)for v in row]for row in images],
        native_matrix_retention='Reassemble from all original authenticated shard records at the recorded context; compare every retained native matrix digest. No midpoint projection.',
        external_enclosure_declaration=plan['enclosure_declaration'],external_reviews=plan['reviews'],
        review_bytes_authentication_does_not_prove_hypotheses=True,git_commit_blob_verification_performed=False,
        model_evaluations=0,optimizer_actions=0,
        scope='Evidence admission and actual native input binding only; neither evaluator truth nor an orbit is established by metadata.')
    write_json(out/'admission-manifest.json',admission)
    contract=gates.ShootingEnclosureContract(plan['map_sha256'],admission['evaluator_source_sha256'],sha(out/'admission-manifest.json'),PERIOD,
        bindings['shooting_domain_sha256'],bindings['enclosures_sha256'],bindings['proposals_sha256'],gates.SHOOTING_DECLARATION)
    write_json(out/'contract.json',__import__('dataclasses').asdict(contract))
    event('native_binding_completed',seconds=time.monotonic()-began,admission_sha256=sha(out/'admission-manifest.json'))
    event('conditional_gates_started');began=time.monotonic()
    result=certificate.check_shooting_certificate(centers,radii,images,jacobians,A,B,contract=contract,bound_method='factored',powers=(1,2))
    seconds=time.monotonic()-began
    if (ctx.prec,ctx.threads)!=(PRECISION,1):raise ValueError('arithmetic context changed')
    for key,value in [('domain_sha256',bindings['shooting_domain_sha256']),('enclosures_sha256',bindings['enclosures_sha256']),('proposals_sha256',bindings['proposals_sha256'])]:
        same(result[key],value,'gate actual input '+key)
    if result['certificate_passed']is not False:raise ValueError('conditional module must never issue a certificate')
    summary=dict(certificate_passed=False,conditional_gates_passed=result['conditional_gates_passed'],
        conditional_gates=result['conditional_gates'],independent_final_result_review_pending=True,gate_seconds=seconds,
        admission_manifest_sha256=sha(out/'admission-manifest.json'),bindings=bindings,
        scope='Conditional exact-real period-8 root/primitive/attraction gates only; external enclosure hypotheses and independent final audit remain required.')
    write_json(out/'gate-summary.json',summary);event('conditional_gates_completed',**summary)
    event('full_report_serialization_started');began=time.monotonic()
    write_json(out/'conditional-gates.json',dict(certificate_passed=False,independent_final_result_review_pending=True,
        scope=summary['scope'],conditional_result=result))
    serialization=time.monotonic()-began
    event('full_report_serialization_completed',seconds=serialization,bytes=(out/'conditional-gates.json').stat().st_size)
    # Final immutable checks catch changed bindings; failed final checks preserve
    # all reports but cannot produce a completed admission status.
    source_and_reviews(plan)
    if sha(frozen_plan_path)!=frozen_plan_sha256:raise ValueError('original assembly plan changed during run')
    for record in(plan['definition'],plan['point_admission']):
        if sha(record['path'])!=record['sha256']:raise ValueError('original pinned input changed during assembly')
    return dict(status='conditional_assembly_completed',**summary,serialization_seconds=serialization,
        report_sha256=sha(out/'conditional-gates.json'),report_bytes=(out/'conditional-gates.json').stat().st_size,
        period=PERIOD,dimension=DIMENSION,precision=PRECISION,threads=1,model_evaluations=0,optimizer_actions=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path);parser.add_argument('--plan',required=True,type=Path)
    args=parser.parse_args();out=args.output.resolve()
    if out.name!='run'or out.parent.parent!=OWNED.resolve()or(out.parent/'receipt.json').exists():raise ValueError('fresh owned assembly attempt/run required')
    out.mkdir(parents=True,exist_ok=False);stage='admission'
    def stopped(signum,frame):raise TimeoutError('bounded assembly launcher sent SIGTERM')
    old=signal.signal(signal.SIGTERM,stopped)
    def event(name,**data):
        nonlocal stage
        stage=name;row=dict(data,event=name,utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic()-STARTED,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        row['certificate_passed']=False
        with(out/'events.jsonl').open('a')as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
    try:
        if args.plan.resolve().parent!=HERE or args.plan.suffix!='.json' or args.plan.is_symlink():raise ValueError('captured committed assembly plan required')
        source=os.environ.get('CERT_SOURCE_COMMIT','')
        if len(source)!=40 or any(c not in'0123456789abcdef'for c in source):raise ValueError('bounded source snapshot identity required')
        for name in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
        os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
        if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
        cap=min([18*1024**3]+[v for v in resource.getrlimit(resource.RLIMIT_AS)if v!=resource.RLIM_INFINITY]);resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
        raw=args.plan.read_bytes();plan=parse_original(raw)
        write_json(out/'inputs.json',dict(source_commit=source,plan=plan,plan_sha256=hashlib.sha256(raw).hexdigest(),
            driver_sha256=sha(__file__),certificate_passed=False,effective_address_space_cap=cap,pid=os.getpid(),
            independent_final_result_review_pending=True))
        from flint import ctx
        from threadpoolctl import threadpool_limits
        ctx.prec=PRECISION;ctx.threads=1
        validate_plan(plan);source_and_reviews(plan)
        assets,ids,version=expected_constants()
        version_metadata=torch_version_metadata()
        if version_metadata['producer_version']!=version:raise ValueError('Torch version metadata changed during admission')
        write_json(out/'backend-version-metadata.json',dict(version_metadata,certificate_passed=False))
        with threadpool_limits(limits=1):result=assemble(plan,out,event,frozen_plan_path=args.plan,frozen_plan_sha256=hashlib.sha256(raw).hexdigest(),expected_assets=assets,prefix_ids=ids,torch_version=version)
        if sha(args.plan)!=hashlib.sha256(raw).hexdigest():raise ValueError('frozen assembly plan changed during run')
        result.update(elapsed_seconds=time.monotonic()-STARTED,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(out/'status.json',result);event('completed')
    except BaseException as exc:
        write_json(out/'status.json',dict(status='conditional_assembly_failed',certificate_passed=False,
            independent_final_result_review_pending=True,stage=stage,error=repr(exc),traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic()-STARTED,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        raise
    finally:signal.signal(signal.SIGTERM,old)

if __name__=='__main__':main()
