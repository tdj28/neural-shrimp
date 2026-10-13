"""Complete secant-parent admission with an explicitly audited capsule boundary.

Initial parents use the unchanged full-history reader. Later capsules are
externally audited trust anchors: their full immediate attempt is authenticated,
while older archive bytes are deliberately NOT rehashed or decoded here.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
from flint import ctx
import center_secant as initial
import secant_restore as restore
import phase_point as worker
saved=initial.resume
base=saved.base
ROOT=Path(__file__).resolve().parents[2]
CAPS=restore.CAPS
CAPSULE_SCHEMA='codex-complete-secant-capsule-v1'
IMMUTABLE=tuple(saved.COMMON_SOURCES)+('codex/certify-llm/center_resume.py',
    'codex/certify-llm/center_secant.py','codex/certify-llm/inverse_secant.py')


def canonical(value):return worker.canonical(value)


def same_json(a,b,label):
    if canonical(a)!=canonical(b):raise ValueError(label+' differs')


def authenticated_attempt(attempt, receipt_sha256):
    receipt,files=base.checked_attempt(attempt,receipt_sha256)
    root=Path(attempt).resolve()
    if type(receipt.get('exit_code'))is not int or receipt['exit_code']!=0:
        raise ValueError('successful explicit parent exit required')
    observed=set()
    for path in root.rglob('*'):
        if path.is_symlink():raise ValueError('parent inventory cannot contain symlinks')
        if path.is_file():observed.add(str(path.relative_to(root)))
    if observed!=set(files)|{'receipt.json'}:raise ValueError('complete exact immediate parent inventory required')
    manifest=saved.read_json(files['manifest.json'])
    if manifest['source_commit']!=receipt['source_commit']:raise ValueError('source manifest identity differs')
    source_rows=manifest.get('source')
    if type(source_rows)is not list or len(source_rows)!=len({r['path'] for r in source_rows}):
        raise ValueError('unique complete source manifest required')
    if {'source/'+r['path'] for r in source_rows}!={p for p in files if p.startswith('source/')}:
        raise ValueError('source inventory incomplete')
    for row in source_rows:
        path=files['source/'+row['path']]
        if path.stat().st_size!=row['bytes'] or base.digest(path)!=row['sha256']:
            raise ValueError('source manifest bytes differ')
    return receipt,files


def backend_versions(torch_version):
    import flint,numpy as np
    return dict(python=sys.version,python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,
                numpy=np.__version__,torch=str(torch_version))


def common_parent(plan,expected_assets,prefix_ids,torch_version,binding):
    receipt,files=authenticated_attempt(plan['parent_attempt'],plan['parent_receipt_sha256'])
    for relative,sha in plan['parent_source_sha256'].items():
        if base.digest(files['source/'+relative])!=sha or base.digest(ROOT/relative)!=sha:
            raise ValueError('same-source parent dependency differs: '+relative)
    inputs=saved.read_json(files['run/inputs.json']);status=saved.read_json(files['run/status.json'])
    if (inputs['source_commit']!=receipt['source_commit'] or inputs['precision']!=2048
            or type(inputs['precision'])is not int or inputs['threads']!=1 or type(inputs['threads'])is not int
            or inputs.get('certificate_passed')is not False or status.get('certificate_passed')is not False):
        raise ValueError('parent source/precision/setup scope differs')
    same_json(inputs['pinned_assets'],expected_assets,'assets')
    same_json(inputs['prefix_ids'],list(prefix_ids),'prefix IDs')
    same_json(inputs['versions'],backend_versions(torch_version),'arithmetic backend')
    matches=[p for n,p in files.items() if n.startswith('source/') and n.endswith('.json')
             and base.digest(p)==inputs['plan_sha256']]
    if len(matches)!=1:raise ValueError('one exact saved committed parent plan required')
    same_json(saved.read_json(matches[0]),inputs['plan'],'source plan')
    driver='codex/certify-llm/'+plan['parent_kind']+'.py'
    if inputs['driver_sha256']!=plan['parent_source_sha256'][driver]:raise ValueError('parent driver identity differs')
    for key in ('prefix_source','seed_source'):
        if inputs['source_binding'][key]!=binding[key]:raise ValueError('fixed prefix/seed lineage differs')
    return receipt,files,inputs,status


def tuple_from_records(centers,images,original_images,*,index,provenance):
    if type(index)is not int or index<0:raise ValueError('chronological tuple index required')
    points=saved.decoded_centers(centers,base.PERIOD,base.DIMENSION)
    if saved.center_records(points)!=centers:raise ValueError('canonical exact centre records required')
    if len(images)!=base.PERIOD or len(original_images)!=base.PERIOD:
        raise ValueError('all full phase image records required')
    values=[saved.decoded_vector(row,base.DIMENSION) for row in images]
    for row in original_images:saved.decoded_vector(row,base.DIMENSION)
    saved._encloses_images(images,original_images)
    if type(provenance)is not list or len(provenance)!=base.PERIOD:
        raise ValueError('all original phase evidence bindings required')
    if [p.get('phase_index') for p in provenance]!=list(range(base.PERIOD)):
        raise ValueError('canonical complete phase provenance required')
    for row in provenance:
        for key in ('input_sha256','point_sha256'):
            if not base.valid_hash(row.get(key)):raise ValueError('full phase artifact hashes required')
    upper,lower,summary=base.residual_summary(points,values)
    residuals=[[values[i][j]-points[(i+1)%base.PERIOD][j] for j in range(base.DIMENSION)]
               for i in range(base.PERIOD)]
    return dict(centers=points,images=values,residuals=residuals,upper=upper,lower=lower,
        record=dict(sequence_index=index,centers=centers,images=images,original_images=original_images,
                    phase_provenance=provenance,residual=summary))


def read_tuple(record):
    expected={'sequence_index','centers','images','original_images','phase_provenance','residual'}
    if type(record)is not dict or set(record)!=expected:raise ValueError('complete capsule tuple schema required')
    state=tuple_from_records(record['centers'],record['images'],record['original_images'],
        index=record['sequence_index'],provenance=record['phase_provenance'])
    saved._residual_record(record['residual'],state['centers'],record['images'])
    return state


def factor_records(history,references):
    if len(history)!=len(references):raise ValueError('every factor requires its original witness reference')
    return [dict(u=[base.packed(v) for v in u],w=[base.packed(v) for v in w],witness=ref)
            for (u,w),ref in zip(history,references)]


def read_factors(records):
    if type(records)is not list or len(records)>4:raise ValueError('at most four retained dyadic factors required')
    result=[];references=[]
    for record in records:
        if type(record)is not dict or set(record)!={'u','w','witness'}:raise ValueError('complete factor and witness reference required')
        ref=record['witness']
        if (type(ref)is not dict or set(ref)!={'attempt','source_commit','path','sha256'}
                or not base.valid_hash(ref['source_commit'],40) or not base.valid_hash(ref['sha256'])
                or type(ref['attempt'])is not str or Path(ref['attempt']).resolve().parent!=base.OWNED.resolve()
                or type(ref['path'])is not str or Path(ref['path']).is_absolute() or '..' in Path(ref['path']).parts):
            raise ValueError('hash-bound immutable original factor witness required')
        u=restore.exact_vector(record['u'],base.PERIOD*base.DIMENSION)
        w=restore.exact_vector(record['w'],base.PERIOD*base.DIMENSION)
        # These arbitrary exact proposals need no certified inverse claim.
        bound=max(abs(restore.point_fraction(v)) for v in record['u'])*sum(abs(restore.point_fraction(v)) for v in record['w'])
        if bound>CAPS['update_norm_limit']:raise ValueError('restored factor exceeds fixed update cap')
        result.append((u,w));references.append(ref)
    return result,references


def pending_identity(before,after):
    if after['record']['sequence_index']!=before['record']['sequence_index']+1:
        raise ValueError('one consecutive pending pair required')
    return canonical(dict(old_tuple=before['record']['sequence_index'],new_tuple=after['record']['sequence_index'],
        old_centers=before['record']['centers'],new_centers=after['record']['centers']))


def load_initial(plan,expected_assets,prefix_ids,torch_version,binding):
    receipt,files,inputs,status=common_parent(plan,expected_assets,prefix_ids,torch_version,binding)
    old_plan=inputs['plan'];initial.validate_plan(old_plan)
    same_json(old_plan['parent_admission_plan']['base_plan'],plan['base_plan'],'unchanged base plan')
    if (status['status']!='proposal_secant_completed' or status.get('candidates')!=1
            or status.get('new_phase_evaluations')!=base.PERIOD):
        raise ValueError('only a complete single-candidate initial secant parent is admissible')
    states,inherited=initial.load_history(old_plan,expected_assets,prefix_ids,binding=binding)
    records=saved.read_json(files['run/history-tuples.json'])
    same_json(records,[s['record'] for s in states],'complete initial chronological history')
    if status['history_tuples']!=len(states) or status['cumulative_phase_evaluations']!=inherited['cumulative_phase_evaluations']+base.PERIOD:
        raise ValueError('initial cumulative tuple/phase counts differ')
    old=states[-1];before_centers=saved.read_json(files['run/centers-before.json'])
    saved.same_centers(before_centers,old['centers'])
    before_images=saved.read_json(files['run/images-before.json'])
    same_json(before_images,records[-1]['decoded_images'],'saved inherited full images')
    before=tuple_from_records(before_centers,before_images,[v['stored_image'] for v in records[-1]['phase_artifacts']],
        index=len(states)-1,provenance=records[-1]['phase_artifacts'])
    after_centers=saved.read_json(files['run/centers-after.json'])
    points=saved.decoded_centers(after_centers,base.PERIOD,base.DIMENSION)
    images,rows,reuse=saved.inspect_pass(files,'run','after',points)
    if saved.read_json(files['run/phase-outcomes.json'])!=rows or status['phase_outcomes']!=rows:
        raise ValueError('all and only eight completed initial phase rows required')
    after=tuple_from_records(after_centers,saved.center_records(images),[v['stored_image'] for v in reuse],
        index=len(states),provenance=reuse)
    outcomes=saved.read_json(files['run/secant-outcomes.json'])
    if outcomes!=status['secant_outcomes'] or len(outcomes)!=len(states)-1:
        raise ValueError('complete ordered initial secant outcomes required')
    decisions=[];refs=[]
    for index,row in enumerate(outcomes,1):
        name=f'secant-{index:02d}.json'
        if row['status']!='completed' or row['secant_index']!=index or row['decision_path']!=name:
            raise ValueError('one chronological initial decision required')
        path=files['run/'+name]
        if base.digest(path)!=row['decision_sha256']:raise ValueError('initial decision hash differs')
        decision=saved.read_json(path)
        expected=dict(old_tuple=index-1,new_tuple=index,history_file_sha256=base.digest(files['run/history-tuples.json']),
            old_receipt_sha256=records[index-1]['receipt_sha256'],new_receipt_sha256=records[index]['receipt_sha256'])
        same_json(decision['evidence'],expected,'initial decision pair evidence')
        same_json(decision['evidence'],row['evidence'],'initial outcome evidence')
        for name2,which in [('old',index-1),('new',index)]:
            same_json(decision[name2+'_centers'],[v for r in records[which]['exact_centers'] for v in r],'decision exact centres')
            same_json(decision[name2+'_residuals'],[v for r in records[which]['recomputed_residual']['residuals'] for v in r],'decision complete residuals')
        if decision['accepted'] is not row['accepted'] or decision['reason']!=row['reason'] or decision['active_history']!=row['active_history']:
            raise ValueError('decision/outcome status differs')
        decisions.append(decision)
        if decision['accepted']:refs.append(dict(attempt=str(Path(plan['parent_attempt']).resolve()),source_commit=receipt['source_commit'],path='run/'+name,sha256=base.digest(path)))
        else:refs=[]
    proposal=saved.read_json(files['run/proposal.json'])
    same_json(proposal['centers'],[v for r in before_centers for v in r],'proposal input centres')
    same_json(proposal['next_centers'],[v for r in after_centers for v in r],'proposal output centres')
    before_summary=saved.read_json(files['run/residual-before.json']);after_summary=saved.read_json(files['run/residual-after.json'])
    same_json(proposal['residuals'],[v for r in before_summary['residuals'] for v in r],'proposal residuals')
    saved._residual_record(before_summary,before['centers'],before_images)
    saved._residual_record(after_summary,after['centers'],after['record']['original_images'])
    comparison=saved.read_json(files['run/residual-comparison.json'])
    saved._comparison_before_after(comparison,before_summary,after_summary)
    same_json(comparison,status['comparison'],'completed comparison')
    history,audit=restore.restore_factors(decisions,proposal,period=base.PERIOD,dimension=base.DIMENSION)
    if not history:refs=[]
    if len(refs)!=len(history):raise ValueError('post-proposal factor provenance differs')
    bp=plan['base_plan'];prefix=saved.read_json(files['run/prefix-binding.json'])
    if (prefix['receipt_sha256']!=bp['prefix_receipt_sha256'] or prefix['graph_sha256']!=bp['prefix_map_source_sha256']
            or prefix['source_commit']!=binding['prefix_source'] or prefix['precision']!=2048
            or prefix['receiver_construction_checked']is not True or prefix['steering_layer']!=12
            or prefix['prefix_sha256']!=base.digest(Path(bp['prefix_attempt'])/'run/prefix-cache.json.gz')):
        raise ValueError('initial prefix receiver identity differs')
    return dict(before=before,after=after,history=history,factor_references=refs,
        pending_pair_sha256=pending_identity(before,after),archive_links=[],
        prior_phase_evaluations=status['cumulative_phase_evaluations'],
        admission=dict(kind='full_initial_history',immediate_artifacts_checked=len(files),
            historical_dense_arithmetic_replayed=False,older_ancestry_rehashed=True,
            restoration=audit,parent_source_commit=receipt['source_commit']))


def decode_capsule(cap,plan,status,*,expected_map,expected_backend):
    """Pure structural admission AFTER the external receipt/audit/hash gates."""
    fields={'schema','certificate_passed','caps','base_plan','map','backend','before','after',
        'post_proposal_factors','active_history_after_proposal','pending_pair_sha256',
        'archive_links','cumulative_phase_evaluations','trust_boundary'}
    if (type(cap)is not dict or set(cap)!=fields or cap.get('schema')!=CAPSULE_SCHEMA
            or cap.get('certificate_passed')is not False
            or status.get('status')!='proposal_continuation_completed' or status.get('candidates')!=1
            or status.get('new_phase_evaluations')!=base.PERIOD
            or status.get('capsule_sha256')!=plan['parent_capsule_sha256']):
        raise ValueError('complete externally audited continuation capsule required')
    same_json(cap['caps'],CAPS,'unchanged secant caps')
    same_json(cap['base_plan'],plan['base_plan'],'unchanged base plan')
    same_json(cap['map'],expected_map,'exact capsule map')
    same_json(cap['backend'],expected_backend,'capsule backend')
    before=read_tuple(cap['before']);after=read_tuple(cap['after'])
    if cap['pending_pair_sha256']!=pending_identity(before,after):
        raise ValueError('pending pair was changed or consumed twice')
    history,refs=read_factors(cap['post_proposal_factors'])
    if type(cap['active_history_after_proposal'])is not int or cap['active_history_after_proposal']!=len(history):
        raise ValueError('capsule must store post-proposal active history')
    links=cap['archive_links']
    if type(links)is not list or not links:raise ValueError('retained archive trust chain required')
    for link in links:
        if (type(link)is not dict or set(link)!={'attempt','receipt_sha256','external_audit','capsule_sha256'}
                or type(link['attempt'])is not str or Path(link['attempt']).resolve().parent!=base.OWNED.resolve()
                or not base.valid_hash(link['receipt_sha256'])
                or link['capsule_sha256']is not None and not base.valid_hash(link['capsule_sha256'])):
            raise ValueError('complete immutable archive link required')
        audit=link['external_audit']
        if (type(audit)is not dict or set(audit)!={'path','sha256'} or not base.valid_hash(audit['sha256'])
                or type(audit['path'])is not str or Path(audit['path']).is_absolute() or '..' in Path(audit['path']).parts):
            raise ValueError('archive external audit link required')
    return before,after,history,refs


def load_capsule(plan,expected_assets,prefix_ids,torch_version,binding):
    receipt,files,inputs,status=common_parent(plan,expected_assets,prefix_ids,torch_version,binding)
    path=files['run/continuation-capsule.json']
    if base.digest(path)!=plan['parent_capsule_sha256']:raise ValueError('external admitted capsule pin differs')
    cap=saved.read_json(path)
    before,after,history,refs=decode_capsule(cap,plan,status,
        expected_map=worker.map_spec(expected_assets,prefix_ids),expected_backend=worker.backend(torch_version))
    same_json(inputs['plan']['base_plan'],plan['base_plan'],'immediate parent base plan')
    immediate=inputs['plan']
    same_json(cap['archive_links'][-1],dict(attempt=immediate['parent_attempt'],
        receipt_sha256=immediate['parent_receipt_sha256'],external_audit=immediate['parent_audit'],
        capsule_sha256=immediate['parent_capsule_sha256']),'immediate archived parent link')
    proposal=saved.read_json(files['run/proposal.json'])
    same_json(proposal['next_centers'],[v for r in after['record']['centers'] for v in r],'capsule completed candidate')
    same_json(proposal['centers'],[v for r in before['record']['centers'] for v in r],'capsule proposal source')
    if proposal['active_history']!=len(history):raise ValueError('capsule proposal clearing differs')
    factors=saved.read_json(files['run/factor-state.json'])
    same_json(factors['active_factors'],cap['post_proposal_factors'],'post-proposal factor snapshot')
    if factors['history_after_proposal']!=len(history):raise ValueError('final factor count differs')
    # Immediate raw images are independently checked again. Archived older
    # before images and factor witnesses are trusted only through the external
    # audit/capsule pin, never through a self-reported validation boolean.
    candidate=Path(plan['parent_attempt'])/'run/candidate-01'
    request=saved.read_json(candidate/'request.json')
    worker.admit(request,expected_assets,prefix_ids,torch_version,receipt['source_commit'])
    binding_record=saved.read_json(files['run/request-binding.json'])
    if (binding_record['request_sha256']!=base.digest(candidate/'request.json')
            or binding_record['centers_sha256']!=request['centers_sha256']):
        raise ValueError('externally inventoried request binding differs')
    expected=expected_worker_identity(expected_assets,prefix_ids,torch_version,receipt['source_commit'],plan['base_plan'])
    raw_images,provenance=collect_parallel_images(candidate,after['centers'],expected_identity=expected)
    same_json(raw_images,after['record']['original_images'],'capsule original phase outputs')
    same_json(provenance,after['record']['phase_provenance'],'capsule immediate phase evidence')
    same_json(saved.read_json(files['run/centers-before.json']),before['record']['centers'],'capsule before file')
    same_json(saved.read_json(files['run/centers-after.json']),after['record']['centers'],'capsule after file')
    before_summary=saved.read_json(files['run/residual-before.json']);after_summary=saved.read_json(files['run/residual-after.json'])
    same_json(proposal['residuals'],[v for row in before_summary['residuals'] for v in row],'capsule proposal full residuals')
    validate_proposal_binding(proposal,before,after,len(history))
    saved._residual_record(before_summary,before['centers'],before['record']['images'])
    saved._residual_record(after_summary,after['centers'],raw_images)
    comparison=saved.read_json(files['run/residual-comparison.json'])
    saved._comparison_before_after(comparison,before_summary,after_summary)
    same_json(comparison,status['comparison'],'capsule comparison')
    if (type(cap['cumulative_phase_evaluations'])is not int
            or cap['cumulative_phase_evaluations']!=status['cumulative_phase_evaluations']
            or cap['cumulative_phase_evaluations']!=(after['record']['sequence_index']+1)*base.PERIOD):
        raise ValueError('capsule chronological evaluation count differs')
    if type(cap['archive_links'])is not list or not cap['archive_links']:
        raise ValueError('retained archive trust chain required')
    return dict(before=before,after=after,history=history,factor_references=refs,
        pending_pair_sha256=cap['pending_pair_sha256'],archive_links=cap['archive_links'],
        prior_phase_evaluations=cap['cumulative_phase_evaluations'],
        admission=dict(kind='audited_capsule',immediate_artifacts_checked=len(files),
            historical_dense_arithmetic_replayed=False,older_ancestry_rehashed=False,
            parent_source_commit=receipt['source_commit'],external_audit=plan['parent_audit'],
            limitation='Unread older archive bytes were not rehashed; the external audited capsule is the explicit trust anchor.'))


def validate_proposal_binding(proposal,before,after,active):
    if (proposal.get('kind')!='proposal' or proposal.get('certificate_passed')is not False
            or type(proposal.get('prior_history'))is not int or not 0<=proposal['prior_history']<=4
            or type(proposal.get('active_history'))is not int or proposal['active_history']!=active
            or restore.point_fraction(proposal['step_ratio_limit'])!=CAPS['step_ratio_limit']):
        raise ValueError('exact retained proposal kind/count/cap required')
    reason=proposal['reason'];prior=proposal['prior_history']
    if reason=='accelerated':
        if not 0<active==prior:raise ValueError('accelerated history cannot clear')
        selected='accelerated_correction'
    elif reason=='baseline_no_history':
        if active!=0 or prior!=0:raise ValueError('baseline requires no incoming history')
        selected='baseline_correction'
    elif reason in restore.FALLBACKS:
        if active!=0 or prior==0:raise ValueError('fallback must clear incoming factors')
        selected='baseline_correction'
    else:raise ValueError('unknown capsule proposal reason')
    same_json(proposal['chosen_correction'],proposal[selected],'chosen capsule correction')
    size=base.PERIOD*base.DIMENSION
    chosen=saved.decoded_vector(proposal['chosen_correction'],size)
    old=[v for row in before['centers'] for v in row];new=[v for row in after['centers'] for v in row]
    if any((c+d).mid()!=n for c,d,n in zip(old,chosen,new)):
        raise ValueError('capsule candidate differs from saved midpoint construction')
    if reason in ('accelerated','step_cap_baseline_fallback'):
        saved.decoded_vector(proposal['accelerated_correction'],size)
        saved.decoded_vector(proposal['baseline_correction'],size)
        upper=restore.point_fraction(proposal['accelerated_step_upper'])
        lower=restore.point_fraction(proposal['baseline_step_lower'])
        actual_upper=max(abs(saved.dyadic(v['mid']))+saved.dyadic(v['rad']) for v in proposal['accelerated_correction'])
        actual_lower=max(max(0,abs(saved.dyadic(v['mid']))-saved.dyadic(v['rad'])) for v in proposal['baseline_correction'])
        if upper<actual_upper or not 0<=lower<=actual_lower or (upper<=CAPS['step_ratio_limit']*lower)!=(reason=='accelerated'):
            raise ValueError('capsule proposal step decision differs')


def expected_worker_identity(expected_assets,prefix_ids,torch_version,source_commit,bp):
    mapping=worker.map_spec(expected_assets,prefix_ids);backend=worker.backend(torch_version)
    return dict(source_commit=source_commit,source_sha256={p:base.digest(ROOT/p) for p in worker.SOURCES},
        map=mapping,map_sha256=canonical(mapping),backend=backend,backend_sha256=canonical(backend),
        prefix_attempt=bp['prefix_attempt'],prefix_receipt_sha256=bp['prefix_receipt_sha256'],
        prefix_map_source_sha256=bp['prefix_map_source_sha256'],
        prefix_cache_sha256=base.digest(Path(bp['prefix_attempt'])/'run/prefix-cache.json.gz'),precision=2048)


def collect_parallel_images(candidate,centers,*,expected_identity):
    import parallel_phase_pass as scheduling
    candidate=Path(candidate);request_path=candidate/'request.json';request=saved.read_json(request_path)
    request_hash=base.digest(request_path)
    for key,value in expected_identity.items():same_json(request[key],value,'independently expected worker '+key)
    required={'source_commit','source_sha256','map','map_sha256','backend','backend_sha256','prefix_attempt',
        'prefix_receipt_sha256','prefix_map_source_sha256','prefix_cache_sha256','precision'}
    if set(expected_identity)!=required:raise ValueError('complete independent worker identity required')
    canonical_centers=saved.center_records(centers)
    same_json(request['centers'],canonical_centers,'exact frozen request centres')
    if request['centers_sha256']!=canonical(canonical_centers):raise ValueError('request centre digest differs')
    report=saved.read_json(candidate/'parallel-pass.json')
    envelope=dict(max_workers=4,worker_timeout_seconds=600,reserve_seconds=600,
        aggregate_rss_bytes=32*1024**3,minimum_available_bytes=12*1024**3)
    for key,value in envelope.items():
        if type(report.get(key))not in (int,float) or report[key]!=value:
            raise ValueError('worker pass frozen envelope differs')
    if (report.get('scheduler_sha256')!=base.digest(Path(scheduling.__file__))
            or report.get('sampled_rss_is_hard_cap')is not False
            or report.get('scientific_endpoint_audit_performed')is not False):
        raise ValueError('reviewed scheduler identity/scope differs')
    if (report['status']!='completed' or report['request_sha256']!=request_hash or report['unstarted_phases']
            or [r['phase_index'] for r in report['phase_receipts']]!=list(range(base.PERIOD))
            or any(r['status']!='completed' or type(r.get('phase_index'))is not int for r in report['phase_receipts'])
            or report.get('certificate_passed')is not False
            or report.get('source_commit')!=expected_identity['source_commit']
            or report.get('worker_sha256')!=expected_identity['source_sha256'][scheduling.WORKER_KEY]
            or [r.get('phase_index') for r in report['termination_verification']]!=list(range(base.PERIOD))
            or any(type(r.get('phase_index'))is not int or r['verified_termination']is not True
                   or type(r.get('leader_exit_code'))is not int or r['leader_exit_code']!=0
                   or r.get('remaining_observed_owned_processes')!=[] for r in report['termination_verification'])):
        raise ValueError('all eight completed and terminated workers required')
    images=[];provenance=[]
    for index in range(base.PERIOD):
        directory=candidate/f'phase-{index:02d}'
        receipt_path=candidate/'parallel-workers'/f'phase-{index:02d}.receipt.json'
        if base.digest(receipt_path)!=report['phase_receipts'][index]['receipt_sha256']:
            raise ValueError('phase receipt hash mismatch')
        receipt=saved.read_json(receipt_path)
        if (receipt['status']!='completed' or type(receipt['exit_code'])is not int or receipt['exit_code']!=0
                or receipt['verified_exit']is not True or type(receipt.get('phase_index'))is not int
                or receipt['phase_index']!=index or receipt.get('request_sha256')!=request_hash
                or receipt.get('worker_sha256')!=expected_identity['source_sha256'][scheduling.WORKER_KEY]
                or receipt.get('certificate_passed')is not False):
            raise ValueError('successful phase worker exit required')
        if scheduling.inventory(directory)!=receipt['artifacts']:raise ValueError('full worker output inventory differs')
        status=scheduling.validate_success(directory,index,request,request_hash)
        inputs=saved.read_json(directory/'input.json');live=saved.read_json(directory/'live-input.json')
        same_json(inputs['center'],canonical_centers[index],'original phase input centre')
        same_json(inputs['next_center'],canonical_centers[(index+1)%base.PERIOD],'original successor centre')
        same_json(live['center'],canonical_centers[index],'original exact live receiver')
        if (type(inputs.get('phase_index'))is not int or inputs['phase_index']!=index
                or type(live.get('phase_index'))is not int or live['phase_index']!=index):
            raise ValueError('original global phase input indices differ')
        if any(v!=0 for v in saved.decoded_vector(live['tangent'],base.DIMENSION,exact=True)):
            raise ValueError('live input tangent is not exact zero')
        point=saved.read_json(directory/'point.json')
        saved.decoded_vector(point['value'],base.DIMENSION)
        if any(v!=0 for v in saved.decoded_vector(point['tangent'],base.DIMENSION,exact=True)):
            raise ValueError('point output tangent is not exact zero')
        prefix=saved.read_json(directory/'prefix-binding.json')
        if (prefix['receipt_sha256']!=request['prefix_receipt_sha256']
                or prefix['graph_sha256']!=request['prefix_map_source_sha256']
                or prefix['prefix_sha256']!=request['prefix_cache_sha256']
                or prefix['precision']!=2048 or prefix['receiver_construction_checked']is not True):
            raise ValueError('worker actual prefix binding differs')
        events=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
        layers=[(e['event'],e['layer']) for e in events if e['event'] in ('step_start','step_end')]
        if layers!=[(name,i) for i in range(worker.LAYERS) for name in ('step_start','step_end')]:
            raise ValueError('complete ordered decoder events required')
        images.append(point['value'])
        provenance.append(dict(phase_index=index,input_path=f'run/candidate-01/phase-{index:02d}/input.json',
            input_sha256=status['input_sha256'],point_path=f'run/candidate-01/phase-{index:02d}/point.json',
            point_sha256=status['point_sha256'],live_input_sha256=status['live_input_sha256'],
            worker_receipt_sha256=base.digest(receipt_path)))
    return images,provenance
