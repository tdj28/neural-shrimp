"""Authenticate completed point images for an externally proposed final domain.

No model, optimizer, secant replay or matrix action occurs. Original capsule
centres are compared with the byte-pinned definition before any numerical
centre decoding. Legacy admission intentionally rehashes the complete immediate
parent but trusts its externally audited capsule for older archives.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from flint import ctx
import center_continue as continuation
import derivative_evidence as evidence
import derivative_shard as derivative
import derivative_coverage as coverage
import phase_domain as domain
import proof_gates as proof

parent=continuation.parent
base=continuation.base
worker=continuation.worker
ROOT=Path(__file__).resolve().parents[2]
SOURCES=tuple(sorted(set(continuation.SOURCES)|set(derivative.SOURCE_PATHS)|{
    'codex/certify-llm/point_evidence.py','codex/certify-llm/derivative_evidence.py'}))


def pinned_json(path,expected):
    coverage._hash(expected,'external original file')
    path=Path(path).resolve();raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('original external file bytes differ')
    return json.loads(raw),raw


def original_nominal(definition):
    """Only exact integer/Fraction checks here; no Arb decoder may run yet."""
    fields={'schema','role','period','dimension','centers','radii','map','map_sha256','selection_provenance'}
    if (type(definition)is not dict or set(definition)!=fields or definition['schema']!=derivative.DEFINITION_SCHEMA
            or definition['role']!=coverage.ROLE):raise ValueError('proposed final-domain definition required; no prototype')
    p,n=base.PERIOD,base.DIMENSION
    if (type(definition['period'])is not int or definition['period']!=p or type(definition['dimension'])is not int
            or definition['dimension']!=n or (derivative.PERIOD,derivative.DIMENSION)!=(p,n)):
        raise ValueError('same full phase/state dimensions required')
    domain._nominal_domain(definition['centers'],definition['radii'],p,n)
    for row in definition['centers']:coverage._records(row,n,'canonical original nominal centers',exact=True)
    coverage._records(definition['radii'],p,'canonical exact nominal radii',exact=True)


def map_bridge(assets,prefix_ids):
    """Reconstruct both reviewed schemas from independent identical constants."""
    point=worker.map_spec(assets,prefix_ids);jacobian=derivative.map_definition(assets,prefix_ids)
    for key in ('dimension','layers','pinned_assets','prefix_ids','steering_layer','alpha_ratio',
                'log10_temperature_ratio','temperature','rotary'):
        evidence.equal(point[key],jacobian[key],'independent point/derivative map '+key)
    evidence.equal(point['full_vocabulary'],jacobian['vocabulary'],'full vocabulary')
    if point['period']!=base.PERIOD:raise ValueError('point period differs')
    return point,jacobian


def point_prefix(plan,assets,prefix_ids,torch_version):
    bp=plan['base_plan']
    _,receipt,files,manifest=evidence.authenticate_inventory(bp['prefix_attempt'],bp['prefix_receipt_sha256'],successful=True)
    inputs=json.loads(files['run/inputs.json'].read_text());status=json.loads(files['run/status.json'].read_text())
    mapping=worker.map_spec(assets,prefix_ids)
    for key in ('pinned_assets','prefix_ids','alpha_ratio','log10_temperature_ratio','temperature','rotary'):
        evidence.equal(inputs[key],mapping[key],'2048-bit point prefix '+key)
    evidence.equal(inputs['versions'],parent.backend_versions(torch_version),'point prefix backend')
    evidence.equal(inputs['arb_context'],dict(precision=2048,threads=1),'point prefix context')
    if (type(inputs['precision'])is not int or inputs['precision']!=2048 or inputs.get('skip_box')is not True
            or inputs['source_commit']!=receipt['source_commit'] or status.get('status')!='point_setup_completed'
            or status.get('certificate_passed')is not False):raise ValueError('completed 2048-bit point prefix required')
    if evidence.digest(files['source/codex/certify-llm/ball_qwen.py'])!=bp['prefix_map_source_sha256']:
        raise ValueError('externally reviewed point-prefix graph differs')
    return dict(attempt=bp['prefix_attempt'],receipt_sha256=bp['prefix_receipt_sha256'],
        source_commit=receipt['source_commit'],map_source_sha256=bp['prefix_map_source_sha256'],
        cache_sha256=evidence.digest(files['run/prefix-cache.json.gz']),precision=2048,
        artifacts_checked=len(files),captured_source_files_checked=len(manifest['source']),
        prefix_recomputed=False,cache_installed_in_new_model=False,
        scope='Complete saved prefix/point receipt and metadata authenticated; correctness of cache construction and original receiving-array checks relies on the reviewed producer and unchanged capsule admission.')


def authenticate_points(admission_plan_path,admission_plan_sha256,definition_path,definition_sha256,*,
        expected_assets,prefix_ids,torch_version,expected_source_sha256,
        expected_derivative_backend,expected_derivative_prefix):
    """Return (exact centers, outward image balls, provenance) at caller context.

    External pins come from a reviewed frozen design. Inputs/source files remain
    immutable throughout this read. A source-file hash is not a correctness
    proof or a Git blob audit. No capsule-chosen source or map trust is inferred.
    """
    plan,plan_bytes=pinned_json(admission_plan_path,admission_plan_sha256)
    definition,definition_bytes=pinned_json(definition_path,definition_sha256)
    if type(expected_source_sha256)is not dict or set(expected_source_sha256)!=set(SOURCES):
        raise ValueError('all explicit wrapper and reader dependencies must be externally pinned')
    for name,value in expected_source_sha256.items():
        coverage._hash(value,name)
        if evidence.digest(ROOT/name)!=value:raise ValueError('externally reviewed source differs: '+name)
    continuation.validate_plan(plan)
    if plan['parent_kind']!='center_continue':raise ValueError('completed independently audited continuation capsule required')
    for name in continuation.SOURCES:
        evidence.equal(plan['source_sha256'][name],expected_source_sha256[name],'externally pinned point-reader source')
        evidence.equal(plan['parent_source_sha256'][name],expected_source_sha256[name],'externally pinned immediate point source')
    continuation.verify_current_sources(plan)
    original_nominal(definition)
    capsule_path=Path(plan['parent_attempt'])/'run/continuation-capsule.json'
    cap,_=pinned_json(capsule_path,plan['parent_capsule_sha256'])
    original_centers=cap['after']['centers']
    if type(original_centers)is not list or len(original_centers)!=base.PERIOD:
        raise ValueError('all original capsule phase centers required')
    for row in original_centers:coverage._records(row,base.DIMENSION,'original capsule centers',exact=True)
    # This is before load_capsule/read_tuple, trust_anchors, or any Arb decoder.
    evidence.equal(original_centers,definition['centers'],'original exact capsule/definition centers')
    point_map,derivative_map=map_bridge(expected_assets,prefix_ids)
    evidence.equal(definition['map'],derivative_map,'independently reconstructed derivative map')
    evidence.equal(definition['map_sha256'],domain.canonical_hash(derivative_map),'derivative map digest')
    caller_context=ctx.prec,ctx.threads
    try:
        ctx.prec=2048;ctx.threads=1
        point_prefix_record=point_prefix(plan,expected_assets,prefix_ids,torch_version)
        arrays,binding=base.load_bound_inputs(plan['base_plan'],expected_assets,prefix_ids)
        del arrays  # The unchanged reader's unused seed arrays never enter arithmetic.
        admitted=parent.load_capsule(plan,expected_assets,prefix_ids,torch_version,binding)
        evidence.equal(admitted['after']['record']['centers'],original_centers,'admitted original centers')
        evidence.equal(admitted['after']['record']['original_images'],cap['after']['original_images'],'admitted raw original images')
        expected_derivative_sources={name:expected_source_sha256[name]for name in derivative.SOURCE_PATHS}
        anchor=evidence.trust_anchors(definition_path,definition_sha256,derivative_map,expected_derivative_backend,
            expected_derivative_sources,expected_derivative_prefix)
        derivative_prefix_record=evidence.authenticate_prefix(anchor)
        point_backend=worker.backend(torch_version)
        for key in ('python','python_flint','flint','numpy','torch'):
            evidence.equal(point_backend[key],expected_derivative_backend[key],'point/derivative backend '+key)
        original_images=admitted['after']['record']['original_images']
        for row in original_images:coverage._records(row,base.DIMENSION,'original full point images')
        if len(original_images)!=base.PERIOD:raise ValueError('all original point images required')
        provenance=admitted['after']['record']['phase_provenance']
        admission=admitted['admission']
    finally:
        ctx.prec,ctx.threads=caller_context
    # Decode after restoring the requested assembly context. Constructor/pack
    # endpoint checks must hold even if this context has lower precision.
    centers=[[coverage._decode(v)for v in row]for row in original_centers]
    if [[domain.pack_ball(v)for v in row]for row in centers]!=original_centers:
        raise ValueError('assembly centre decode lost canonical exact data')
    images=[[coverage._decode(v)for v in row]for row in original_images]
    native_images=[[domain.pack_ball(v)for v in row]for row in images]
    for path,expected in [(admission_plan_path,admission_plan_sha256),(definition_path,definition_sha256),
                          (capsule_path,plan['parent_capsule_sha256'])]:
        if evidence.digest(path)!=expected:raise ValueError('immutable external input changed during admission')
    report=dict(schema='codex-authenticated-final-point-images-v1',certificate_passed=False,
        artifact_authentication_completed=True,evaluator_correctness_proved_by_metadata=False,
        model_evaluations=0,optimizer_actions=0,historical_dense_arithmetic_replayed=False,
        source_sha256=expected_source_sha256,plan_sha256=admission_plan_sha256,definition_sha256=definition_sha256,
        parent_attempt=plan['parent_attempt'],parent_receipt_sha256=plan['parent_receipt_sha256'],
        parent_capsule_sha256=plan['parent_capsule_sha256'],external_parent_audit=plan['parent_audit'],
        point_admission=admission,original_phase_provenance=provenance,
        original_centers=original_centers,original_images=original_images,nominal_radii=definition['radii'],
        original_centers_sha256=domain.canonical_hash(original_centers),
        original_images_sha256=domain.canonical_hash(original_images),
        actual_native_images_sha256=domain.canonical_hash(native_images),
        phase_domain_sha256=anchor['phase_domain_sha256'],shooting_domain_sha256=anchor['shooting_domain_sha256'],
        point_map=point_map,point_map_sha256=domain.canonical_hash(point_map),
        derivative_map=derivative_map,derivative_map_sha256=domain.canonical_hash(derivative_map),
        map_equivalence_scope='Distinct reviewed scalar and streamed-Jet evaluators of identical declared exact-real constants and full-vocabulary map; metadata does not prove implementation correctness.',
        point_backend=point_backend,derivative_backend=expected_derivative_backend,
        point_precision=2048,derivative_precision=1024,assembly_precision=caller_context[0],assembly_threads=caller_context[1],
        point_prefix=point_prefix_record,derivative_prefix=derivative_prefix_record,
        original_endpoint_enclosures_preserved=True,older_ancestry_rehashed=False,git_commit_blob_verification_performed=False,
        scope='Authenticated exact-center point images for a proposed final domain only. No residual correction, full-box derivative, root, primitive-period, attraction or orbit certificate is computed.')
    return centers,images,report
