"""Stream authenticated final-domain derivative artifacts into pure coverage.

This adapter verifies artifact provenance and exact byte/domain bindings. It
neither runs a model nor proves the correctness of the externally reviewed
Arb evaluator. Successful derivative attempts are mandatory; the separately
pinned Task34 prefix must be successfully closed at its original 2048-bit
precision. No prototype derivative is admissible.
"""
from __future__ import annotations
import gzip
import hashlib
import json
import math
from pathlib import Path

from flint import ctx
import importlib.util,sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'codex/certify-llm'))
spec=importlib.util.spec_from_file_location('task34_derivative_producer',HERE/'derivative_shard.py')
producer=importlib.util.module_from_spec(spec);spec.loader.exec_module(producer)
import derivative_coverage as coverage
import phase_domain as domain

ROOT=Path(__file__).resolve().parents[2]
OWNED=producer.OWNED
SCHEMA='codex-authenticated-derivative-evidence-v1'
PREFIX_FIELDS={'attempt','receipt_sha256','map_source_sha256','cache_sha256','source_commit','receipt_path','admission_path','admission_sha256'}


def digest(path):return producer.sha(path)
def canonical(value):return domain.canonical_hash(value)
def equal(a,b,label):
    if canonical(a)!=canonical(b):raise ValueError(label+' differs')


def json_bytes(path):
    raw=Path(path).read_bytes()
    return json.loads(raw),raw


def owned_attempt(value):
    if not isinstance(value,(str,Path)):raise ValueError('owned attempt path required')
    path=Path(value).resolve()
    if path.parent!=OWNED.resolve() or not path.is_dir():raise ValueError('existing direct-child owned attempt required')
    return path


def authenticate_inventory(attempt,receipt_sha256,*,successful):
    """Check exact whole inventory and captured manifest, not just chosen files."""
    attempt=owned_attempt(attempt);coverage._hash(receipt_sha256,'external receipt')
    receipt,raw=json_bytes(attempt/'receipt.json')
    if hashlib.sha256(raw).hexdigest()!=receipt_sha256:raise ValueError('external receipt bytes differ')
    if (type(receipt.get('exit_code'))is not int or type(receipt.get('timed_out'))is not bool
            or successful and (receipt['exit_code']!=0 or receipt['timed_out'])):
        raise ValueError('successful untimed-out derivative attempt required')
    source=receipt.get('source_commit')
    if type(source)is not str or len(source)!=40 or any(c not in '0123456789abcdef' for c in source):
        raise ValueError('explicit source commit required')
    rows=receipt.get('artifacts');files={}
    if type(rows)is not list or not rows:raise ValueError('complete receipt inventory required')
    for row in rows:
        if type(row)is not dict or set(row)!={'path','bytes','sha256'}:raise ValueError('complete artifact row required')
        name=row['path'];coverage._hash(row['sha256'],'artifact')
        if (type(name)is not str or Path(name).is_absolute() or '..' in Path(name).parts
                or name in files or name=='receipt.json' or type(row['bytes'])is not int or row['bytes']<0):
            raise ValueError('unique relative artifact with exact size required')
        path=attempt/name
        if not path.is_file() or path.is_symlink() or path.resolve()!=path:
            raise ValueError('original regular artifact path required')
        if path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:
            raise ValueError('artifact bytes differ: '+name)
        files[name]=path
    observed=set()
    for path in attempt.rglob('*'):
        if path.is_symlink():raise ValueError('symlink in evidence attempt refused')
        if path.is_file():observed.add(str(path.relative_to(attempt)))
    if observed!=set(files)|{'receipt.json'}:raise ValueError('whole original inventory missing or has extra files')
    manifest,_=json_bytes(files['manifest.json'])
    if manifest.get('source_commit')!=source:raise ValueError('captured manifest source differs')
    entries=manifest.get('source')
    if type(entries)is not list or not entries or len({r.get('path') for r in entries})!=len(entries):
        raise ValueError('complete unique captured source manifest required')
    if {'source/'+r['path'] for r in entries}!={name for name in files if name.startswith('source/')}:
        raise ValueError('captured source manifest incomplete')
    for row in entries:
        name=row['path'];path=files['source/'+name]
        if type(row.get('bytes'))is not int or path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:
            raise ValueError('captured source manifest bytes differ')
    return attempt,receipt,files,manifest


def trust_anchors(definition_path,definition_sha256,expected_map,expected_backend,expected_source_sha256,expected_prefix):
    coverage._hash(definition_sha256,'external definition')
    if (type(expected_source_sha256)is not dict or set(expected_source_sha256)!=set(producer.SOURCE_PATHS)):
        raise ValueError('all externally reviewed evaluator dependencies required')
    for name,sha in expected_source_sha256.items():
        coverage._hash(sha,name)
        if digest(ROOT/name)!=sha:raise ValueError('adapter parser requires the reviewed immutable evaluator source: '+name)
    if type(expected_map)is not dict or type(expected_backend)is not dict:
        raise ValueError('independent complete map/backend objects required')
    # Reconstruct the exact map recipe from external assets/prefix, rather than
    # accepting a self-consistent arbitrary map hash in a producer's metadata.
    equal(expected_map,producer.map_definition(expected_map['pinned_assets'],expected_map['prefix_ids']),'reviewed map recipe')
    equal(expected_backend,dict(producer.backend_definition(expected_backend['torch']),precision=1024,threads=1),'reviewed backend')
    if type(expected_prefix)is not dict or set(expected_prefix)!=PREFIX_FIELDS:
        raise ValueError('explicit original prefix receipt/admission/cache/source required')
    for key in ('receipt_sha256','map_source_sha256','cache_sha256'):coverage._hash(expected_prefix[key],key)
    owned_attempt(expected_prefix['attempt'])
    path=Path(definition_path).resolve()
    if not path.is_relative_to(OWNED.resolve()) or not path.is_file():raise ValueError('owned externally frozen definition required')
    definition,raw=json_bytes(path)
    if hashlib.sha256(raw).hexdigest()!=definition_sha256:raise ValueError('frozen definition bytes differ')
    fields={'schema','role','period','dimension','centers','radii','map','map_sha256','selection_provenance'}
    if (type(definition)is not dict or set(definition)!=fields
            or definition['schema']!=producer.DEFINITION_SCHEMA or definition['role']!=coverage.ROLE):
        raise ValueError('prototype/non-final candidate definition refused')
    if type(definition['period'])is not int or definition['period']!=3:raise ValueError('Task34 cascade period3 required')
    for name,value in [('dimension',producer.DIMENSION)]:
        if type(definition[name])is not int or definition[name]!=value:raise ValueError('complete target dimensions required')
    equal(definition['map'],expected_map,'candidate map')
    if definition['map_sha256']!=canonical(expected_map):raise ValueError('candidate map hash differs')
    if type(definition['selection_provenance'])is not dict or not definition['selection_provenance']:
        raise ValueError('selection provenance must be retained, not used to authenticate derivatives')
    _,_,mixed,shooting=coverage._nominal(definition['centers'],definition['radii'],definition['period'],producer.DIMENSION)
    bundle=dict(schema='codex-derivative-evaluator-source-bundle-v1',source_sha256=expected_source_sha256)
    return dict(definition=definition,definition_path=path,definition_bytes=raw,
        design_sha256=definition_sha256,map=expected_map,map_sha256=canonical(expected_map),
        backend=expected_backend,backend_sha256=canonical(expected_backend),source_bundle=bundle,
        evaluator_sha256=canonical(bundle),phase_domain_sha256=mixed,shooting_domain_sha256=shooting,
        prefix=dict(expected_prefix))


def authenticate_prefix(anchor):
    """Original Task34 supervisor authority, never a relabelled P8 receipt."""
    e=anchor['prefix'];path=Path(e['receipt_path'])
    if digest(path)!=e['receipt_sha256']:raise ValueError('original prefix receipt pin')
    receipt=producer.read_json(path)
    if receipt['source_commit']!=e['source_commit'] or receipt['returncode']!=0 or receipt['failure']is not None or not receipt['worker_reaped'] or not receipt['worker_group_session_absent']:raise ValueError('original prefix not closed successfully')
    files={row['path']:row for row in receipt['output_files']}
    if len(files)!=len(receipt['output_files']):raise ValueError('duplicate prefix inventory')
    def checked(path):
        path=Path(path);row=files[str(path)]
        if path.is_symlink() or path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:raise ValueError('original prefix artifact')
        return path
    for name in files:checked(name)
    ap=Path(e['admission_path'])
    if digest(ap)!=e['admission_sha256']:raise ValueError('prefix admission pin')
    adm=producer.read_json(ap)
    if adm['source_commit']!=e['source_commit']:raise ValueError('prefix admission source')
    pins={v['relative_path']:v['sha256']for v in adm['source_blobs']}
    if pins.get('codex/certify-llm/ball_qwen.py')!=e['map_source_sha256']:raise ValueError('prefix evaluator source')
    root=Path(e['attempt']);meta=producer.read_json(checked(root/'inputs.json'))
    equal(meta.get('cell'),'cascade-p3','prefix named exact cell')
    for key in ('pinned_assets','prefix_ids','steering_layer','alpha_ratio','log10_temperature_ratio','temperature','rotary'):equal(meta[key],anchor['map'][key],'prefix map '+key)
    equal(meta['source_commit'],e['source_commit'],'prefix metadata source')
    equal(meta['precision'],2048,'prefix stored precision')
    equal(meta['versions'],{k:anchor['backend'][k]for k in ('python_flint','flint')},'prefix backend')
    cache=checked(root/'prefix-cache.json.gz')
    if digest(cache)!=e['cache_sha256']:raise ValueError('prefix cache pin')
    return dict(e,stored_precision=2048,receiver_precision=1024,artifacts_checked=len(files),model_constructed=False)


def _head_diagnostic(record,anchor,phase,actual_hash,expected_passes):
    for key,expected in [('design_sha256',anchor['design_sha256']),('map_sha256',anchor['map_sha256']),
                         ('phase_index',phase),('actual_inputs_sha256',actual_hash),('certificate_passed',False)]:
        equal(record[key],expected,'head binding '+key)
    equal(record['head_passes'],[list(row)for row in expected_passes],'all-token head passes')
    d=record['diagnostic'];v=producer.VOCABULARY
    if (type(d.get('vocabulary_count'))is not int or d['vocabulary_count']!=v
            or type(d.get('competitor_count'))is not int or d['competitor_count']!=v-1
            or type(d.get('candidate_index'))is not int or not 0<=d['candidate_index']<v
            or type(d.get('strict_margin_established'))is not bool or d.get('certificate_passed')is not False):
        raise ValueError('complete all-token diagnostic required')
    lower,lr=domain._record_parts(d['candidate_lower']);upper,ur=domain._record_parts(d['largest_competitor_upper'])
    margin,radius=domain._record_parts(d['strict_margin'])
    for key in ('candidate_lower','largest_competitor_upper','strict_margin'):
        coverage._records([d[key]],1,'head diagnostic');coverage._decode(d[key])
    if lr or ur or not margin-radius<=lower-upper<=margin+radius or d['strict_margin_established']!=(margin-radius>0):
        raise ValueError('original exact label margin/flag disagrees')
    return dict(candidate_index=d['candidate_index'],strict_margin_established=d['strict_margin_established'],
        scope='Label diagnostic on this receiver box only, never an orbit gate.')


def _receiver_witnesses(anchor,phase,receiver):
    """Recreate saved endpoint witnesses from ORIGINAL records, never decode."""
    definition=anchor['definition'];radius=domain._record_parts(definition['radii'][phase],exact=True)[0]
    pairs=coverage._records(receiver['actual_inputs'],producer.DIMENSION,'original live inputs')
    result=[]
    for j,(mid,rad) in enumerate(pairs):
        c=domain._record_parts(definition['centers'][phase][j],exact=True)[0]
        nl,nu=c-radius,c+radius;al,au=mid-rad,mid+rad;error=abs(c-mid)
        lower,upper,rm=nl-al,au-nu,rad-radius-error
        if min(lower,upper,rm)<0:raise ValueError('original live receiver loses nominal endpoints')
        # producer.receiver_record checks one selected phase at local index 0.
        result.append(dict(phase=0,coordinate=j,nominal_lower=domain._pair(nl),nominal_upper=domain._pair(nu),
            actual_lower=domain._pair(al),actual_upper=domain._pair(au),center_rounding_error=domain._pair(error),
            nominal_radius=domain._pair(radius),actual_radius=domain._pair(rad),lower_margin=domain._pair(lower),
            upper_margin=domain._pair(upper),radius_margin=domain._pair(rm)))
    equal(receiver['selected_phase_endpoint_witnesses'],result,'original selected-phase endpoint witnesses')
    return canonical(result)


def authenticate_shard(reference,anchor,prefix):
    if type(reference)is not dict or set(reference)!={'attempt','receipt_sha256','source_commit'}:raise ValueError('whole attempt and external receipt pin required')
    attempt,receipt,files,manifest=authenticate_inventory(reference['attempt'],reference['receipt_sha256'],successful=True)
    expected_source=reference['source_commit']
    if type(expected_source)is not str or len(expected_source)!=40 or any(c not in '0123456789abcdef'for c in expected_source):raise ValueError('external campaign source required')
    equal(receipt['source_commit'],expected_source,'assigned campaign source')
    if attempt==Path(anchor['prefix']['attempt']).resolve():raise ValueError('prefix attempt cannot be used as derivative evidence')
    pins=anchor['source_bundle']['source_sha256']
    for name,sha in pins.items():
        if digest(files['source/'+name])!=sha:raise ValueError('captured evaluator dependency differs: '+name)
    inputs,_=json_bytes(files['run/inputs.json']);status,_=json_bytes(files['run/status.json'])
    plan=inputs['plan'];producer.validate_plan(plan,owned=OWNED)
    required=dict(cell='cascade-p3',definition_sha256=anchor['design_sha256'],map_sha256=anchor['map_sha256'],backend_sha256=anchor['backend_sha256'],
        source_sha256=pins,evaluator_sha256=anchor['evaluator_sha256'],prefix_attempt=str(Path(anchor['prefix']['attempt']).resolve()),
        prefix_receipt_sha256=anchor['prefix']['receipt_sha256'],prefix_map_source_sha256=anchor['prefix']['map_source_sha256'],
        prefix_cache_sha256=anchor['prefix']['cache_sha256'],period=anchor['definition']['period'],prefix_source_commit=anchor['prefix']['source_commit'],prefix_output=anchor['prefix']['attempt'],prefix_receipt=anchor['prefix']['receipt_path'],prefix_admission_path=anchor['prefix']['admission_path'],prefix_admission_sha256=anchor['prefix']['admission_sha256'])
    for name,value in required.items():equal(plan[name],value,'independent frozen plan '+name)
    if Path(plan['definition_path']).resolve()!=anchor['definition_path']:raise ValueError('different frozen candidate file')
    if files['run/candidate-definition.json'].read_bytes()!=anchor['definition_bytes']:
        raise ValueError('producer original candidate bytes differ')
    # Recovery snapshots may retain several byte-identical plan names. The
    # original authenticated command, not global hash uniqueness, selects one.
    command=manifest.get('command')
    if type(command)is not list or len(command)!=9 or any(type(v)is not str for v in command):
        raise ValueError('captured command requires nine literal string arguments')
    plan_key='source/codex/certificate-ladder-task34/'+Path(command[8]).name
    plan_file=files.get(plan_key)
    if (not plan_key.endswith('.json') or plan_file is None
            or command[7]!='--plan' or command[8]!=str(attempt/plan_key)
            or plan_file!=attempt/plan_key):
        raise ValueError('command-bound captured JSON plan must be literal and inventoried')
    if digest(plan_file)!=inputs['plan_sha256']:
        raise ValueError('command-bound captured plan bytes differ from original inputs')
    equal(producer.read_json(plan_file),plan,'captured plan')
    expected_binding={key:anchor[key]for key in ('design_sha256','map_sha256','evaluator_sha256','backend_sha256',
        'phase_domain_sha256','shooting_domain_sha256','source_bundle','backend','map')}
    equal(inputs['binding'],expected_binding,'producer independent input binding')
    for name,value in [('source_commit',receipt['source_commit']),('driver_sha256',pins['codex/certificate-ladder-task34/derivative_shard.py']),
                       ('precision',1024),('threads',1),('certificate_passed',False)]:
        equal(inputs[name],value,'producer input '+name)
    for name,value in [('threads',1),('memory_gib',18),('timeout_seconds',3600)]:equal(manifest[name],value,'launcher envelope '+name)
    if (type(command)is not list or len(command)!=9 or command[1:4]!=['-I','-B','-u']
            or Path(command[4]).resolve()!=files['source/codex/certificate-ladder-task34/derivative_shard.py']
            or command[5]!='--output' or Path(command[6]).resolve()!=attempt/'run'
            or command[7]!='--plan' or Path(command[8]).resolve()!=plan_file):
        raise ValueError('captured command is not the isolated reviewed shard driver')
    phase,start=plan['phase_index'],plan['start_column'];columns=list(range(start,start+producer.COLUMNS))
    shard,_=json_bytes(files['run/derivative-shard.json'])
    for name,value in [('status','final_derivative_shard_completed'),('certificate_passed',False),('phase_index',phase),
        ('direction_indices',columns),('columns_completed',producer.COLUMNS),('full_state_dimension',producer.DIMENSION),
        ('full_vocabulary',producer.VOCABULARY),('precision',1024),('shard_sha256',digest(files['run/derivative-shard.json']))]:
        equal(status[name],value,'completed status '+name)
    for name,value in [('schema',producer.SHARD_SCHEMA),('role',coverage.ROLE),('certificate_passed',False),
        ('phase_index',phase),('direction_indices',columns),('precision',1024),('full_vocabulary',producer.VOCABULARY),('token_chunk',1024)]:
        equal(shard[name],value,'original final shard '+name)
    for name in ('design_sha256','map_sha256','evaluator_sha256','backend_sha256','phase_domain_sha256','shooting_domain_sha256'):
        equal(shard[name],anchor[name],'original shard anchor '+name)
    for name in ('forward_seconds','elapsed_seconds'):
        if type(status[name])not in (float,int)or not math.isfinite(status[name])or status[name]<0:raise ValueError('finite nonnegative measured time required')
    if status['forward_seconds']!=shard['forward_seconds']or status['forward_seconds']>status['elapsed_seconds']:
        raise ValueError('retained forward timing inconsistent')
    receiver=producer.read_json(files['run/actual-inputs.json']);direction=producer.read_json(files['run/direction-inputs.json'])
    if shard['actual_inputs_file_sha256']!=digest(files['run/actual-inputs.json'])or shard['direction_inputs_file_sha256']!=digest(files['run/direction-inputs.json']):
        raise ValueError('original receiver/direction file binding differs')
    equal(receiver['phase_index'],phase,'receiver phase');equal(receiver['actual_inputs'],shard['actual_inputs'],'original live receiver')
    actual_hash=canonical(shard['actual_inputs'])
    if (receiver['actual_inputs_sha256']!=actual_hash or shard['actual_inputs_sha256']!=actual_hash
            or status['actual_inputs_sha256']!=actual_hash or receiver['full_nominal_phase_box_contained']is not True):
        raise ValueError('original live receiver digest differs')
    endpoint_hash=_receiver_witnesses(anchor,phase,receiver)
    equal(direction['direction_indices'],columns,'original directions coverage');equal(direction['directions'],shard['directions'],'original live directions')
    if direction['directions_sha256']!=canonical(direction['directions']):raise ValueError('original direction digest differs')
    # Original endpoint containment, exact basis, finite full derivatives and
    # all-coverage checks are performed by the unchanged pure collector. Check
    # saved domain construction separately using its externally inventoried bytes.
    artifact=producer.read_json(files['run/domain-construction.json'])
    if artifact['helper_source_sha256']!=pins['codex/certify-llm/phase_domain.py']:
        raise ValueError('domain helper source differs')
    body={k:v for k,v in artifact.items()if k!='artifact_sha256'}
    trusted_body_hash=canonical(body)  # Body came from an authenticated full receipt.
    domain.decode_verified_domains(artifact,anchor['definition']['centers'],anchor['definition']['radii'],
        period=anchor['definition']['period'],dimension=producer.DIMENSION,target_precision=1024,expected_artifact_sha256=trusted_body_hash)
    prefix_record=producer.read_json(files['run/prefix-binding.json'])
    equal(prefix_record,dict(original_receipt_sha256=prefix['receipt_sha256'],source_commit=prefix['source_commit'],stored_precision=2048,receiver_precision=1024,radii_preserved=True),'actual receiving prefix')
    expected_passes=[(kind,first,min(first+1024,producer.VOCABULARY))for kind in ('head_logits','head_derivatives')for first in range(0,producer.VOCABULARY,1024)]
    head=producer.read_json(files['run/head-diagnostics.json'])
    if shard['head_diagnostics_file_sha256']!=digest(files['run/head-diagnostics.json']):raise ValueError('full head diagnostics hash differs')
    label=_head_diagnostic(head,anchor,phase,actual_hash,expected_passes)
    events=[json.loads(line)for line in files['run/events.jsonl'].read_text().splitlines()]
    stages=[(r['event'],r.get('layer'))for r in events if r['event']in ('step_start','step_end')]
    equal(stages,[(kind,i)for i in range(producer.LAYERS)for kind in ('step_start','step_end')],'all ordered decoder stages')
    head_events=[(r['event'],r.get('row_start'),r.get('row_stop'))for r in events if r['event']in ('head_logits','head_derivatives')]
    equal(head_events,expected_passes,'both full all-token passes')
    selected=[r for r in events if r['event']=='live_input_saved']
    if len(selected)!=1 or selected[0]['sha256']!=shard['actual_inputs_file_sha256']:raise ValueError('original live-input event hash differs')
    names=[r['event']for r in events]
    if not names or names[-1]!='completed'or names.count('completed')!=1:raise ValueError('completed event required')
    first_head=next(i for i,r in enumerate(events)if r['event']=='head_logits')
    last_layer=max(i for i,r in enumerate(events)if r['event']=='step_end')
    if first_head<=last_layer:raise ValueError('head must follow the complete decoder')
    if names.index('live_input_saved')>=names.index('step_start'):raise ValueError('original inputs must be saved before decoder evaluation')
    return shard,dict(attempt=str(attempt),receipt_sha256=reference['receipt_sha256'],source_commit=receipt['source_commit'],
        artifacts_checked=len(files),captured_source_files_checked=len(manifest['source']),
        evaluator_dependencies_checked=len(pins),phase_index=phase,direction_indices=columns,
        original_shard_file_sha256=digest(files['run/derivative-shard.json']),original_live_inputs_sha256=actual_hash,
        original_endpoint_witnesses_sha256=endpoint_hash,head_diagnostic=label,original_payload_forwarded=True)


def authenticate_and_collect(definition_path,definition_sha256,*,expected_map,expected_backend,
        expected_source_sha256,expected_prefix,attempts):
    """Yield one ORIGINAL shard at a time; do not accumulate 112 JSON payloads.

    Expected anchors must be supplied independently by a frozen reviewed design,
    not copied from the first untrusted shard. Inputs remain immutable throughout
    this read. Arbitrary centre-selection provenance is never evaluator evidence.
    """
    anchor=trust_anchors(definition_path,definition_sha256,expected_map,expected_backend,expected_source_sha256,expected_prefix)
    prefix=authenticate_prefix(anchor);audits=[];seen=set();assembly_precision=ctx.prec
    maximum=anchor['definition']['period']*(producer.DIMENSION//producer.COLUMNS)
    def records():
        for index,reference in enumerate(attempts):
            if index>=maximum:raise ValueError('too many full derivative attempts')
            if type(reference)is not dict:raise ValueError('whole attempt reference required')
            path=owned_attempt(reference.get('attempt'))
            if path in seen:raise ValueError('duplicate derivative attempt')
            seen.add(path)
            shard,audit=authenticate_shard(reference,anchor,prefix)
            audits.append(audit)
            yield shard
        if digest(anchor['definition_path'])!=anchor['design_sha256']:
            raise ValueError('frozen candidate changed during collection')
    matrices,conditional=coverage.collect_derivatives(anchor['definition']['centers'],anchor['definition']['radii'],records(),
        period=anchor['definition']['period'],dimension=producer.DIMENSION,expected_precision=1024,
        **{'expected_'+key:anchor[key]for key in ('design_sha256','map_sha256','evaluator_sha256','backend_sha256')})
    report=dict(schema=SCHEMA,adapter_source_sha256=digest(__file__),certificate_passed=False,artifact_authentication_completed=True,
        evaluator_correctness_independently_proved_by_adapter=False,center_selection_lineage_imported=False,
        git_commit_blob_verification_performed=False,
        design_sha256=anchor['design_sha256'],map_sha256=anchor['map_sha256'],evaluator_sha256=anchor['evaluator_sha256'],
        backend_sha256=anchor['backend_sha256'],source_precision=1024,assembly_precision=assembly_precision,
        trusted_prefix=prefix,authenticated_attempts=audits,coverage=conditional,
        statement='Complete source-bound artifact authentication plus conditional whole-box derivative coverage; correctness depends on the independently reviewed evaluator/backend. No root, primitive-period, attraction or certificate gate is evaluated.')
    return matrices,report
