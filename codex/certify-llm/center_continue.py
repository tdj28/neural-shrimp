"""One complete secant-parent continuation, one frozen parallel point pass.

No model is loaded by this controller. An externally audited capsule is an
explicit trust anchor for older archived evidence; every new phase is freshly
evaluated by the unchanged rigorous worker. This is proposal setup, not proof.
"""
from __future__ import annotations
import time
PROCESS_STARTED=time.monotonic()  # Before non-stdlib imports and all admission.
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import resource
import sys
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from flint import ctx
import secant_parent as parent
import secant_restore as restore
import inverse_secant_fast
import phase_point as worker
import parallel_phase_pass as scheduling
saved=parent.saved
base=parent.base
OWNED=base.OWNED
SCHEMA='codex-llm-center-continue-v1'
SOURCES=tuple(sorted(set(parent.IMMUTABLE)|set(worker.SOURCES)|{
    'codex/certify-llm/'+name+'.py' for name in
    ('center_continue','secant_parent','secant_restore','inverse_secant_fast','parallel_phase_pass')}))
ENVELOPE=dict(cpu_threads=1,workers=4,nice=10,memory_gib=18,worker_memory_gib=18,
    aggregate_rss_gib=32,minimum_available_gib=12,timeout_seconds=1800,
    worker_timeout_seconds=600,worker_reserve_seconds=600)


def validate_plan(plan,*,owned=None):
    owned=OWNED.resolve() if owned is None else Path(owned).resolve()
    fields={'schema','parent_kind','parent_attempt','parent_receipt_sha256','parent_source_sha256',
        'parent_audit','parent_capsule_sha256','base_plan','source_sha256','precision','max_candidates','caps','envelope'}
    if type(plan)is not dict or not fields<=set(plan) or set(plan)-fields-{'purpose','basis'} or plan['schema']!=SCHEMA:
        raise ValueError('complete known one-candidate continuation plan required')
    if plan['parent_kind']not in ('center_secant','center_continue'):raise ValueError('complete secant/continuation parent required')
    for key,value in [('precision',2048),('max_candidates',1)]:
        if type(plan[key])is not int or plan[key]!=value:raise ValueError('one 2048-bit candidate required')
    for key,value in [('caps',restore.CAPS),('envelope',ENVELOPE)]:
        if type(plan[key])is not dict or plan[key]!=value or any(type(v)is not int for v in plan[key].values()):
            raise ValueError('exact fixed '+key+' required')
    if not base.valid_hash(plan['parent_receipt_sha256']):raise ValueError('external complete-parent receipt pin required')
    if type(plan['parent_attempt'])is not str:raise ValueError('owned completed parent path required')
    attempt=Path(plan['parent_attempt']).resolve()
    if attempt.parent!=owned or not attempt.is_dir():raise ValueError('existing direct-child parent required')
    base.validate_plan(plan['base_plan'],owned=owned)
    if attempt in [Path(plan['base_plan'][k]).resolve() for k in ('prefix_attempt','center_seed_attempt')]:
        raise ValueError('parent must be separate from immutable prefix and seed attempts')
    required=set(parent.IMMUTABLE) if plan['parent_kind']=='center_secant' else set(SOURCES)
    for key,names in [('parent_source_sha256',required),('source_sha256',set(SOURCES))]:
        pins=plan[key]
        if type(pins)is not dict or set(pins)!=names or any(not base.valid_hash(v) for v in pins.values()):
            raise ValueError('complete explicit source bundle required: '+key)
    audit=plan['parent_audit']
    if (type(audit)is not dict or set(audit)!={'path','sha256'} or type(audit['path'])is not str
            or Path(audit['path']).is_absolute() or '..' in Path(audit['path']).parts
            or not audit['path'].startswith('codex/certify-llm/') or not base.valid_hash(audit['sha256'])):
        raise ValueError('external committed independent parent-audit anchor required')
    capsule=plan['parent_capsule_sha256']
    if (plan['parent_kind']=='center_secant' and capsule is not None
            or plan['parent_kind']=='center_continue' and not base.valid_hash(capsule)):
        raise ValueError('explicit externally audited capsule pin for continuation parents required')
    return attempt


def verify_current_sources(plan):
    for path,sha in plan['source_sha256'].items():
        if base.digest(ROOT/path)!=sha:raise ValueError('current runtime source differs: '+path)
    audit=plan['parent_audit']
    if base.digest(ROOT/audit['path'])!=audit['sha256']:raise ValueError('external committed parent-audit bytes differ')


def prepare_candidate(admitted,derivatives,inverse,out,event,*,source_commit,attempt):
    """Consume exactly one pending pair, preserve all clears, propose exactly once."""
    before,after=admitted['before'],admitted['after']
    if admitted['pending_pair_sha256']!=parent.pending_identity(before,after):
        raise ValueError('the one pending chronological pair changed')
    base.write_json(out/'admitted-parent.json',dict(before=before['record'],after=after['record'],
        pending_pair_sha256=admitted['pending_pair_sha256'],admission=admitted['admission'],
        post_proposal_factors=parent.factor_records(admitted['history'],admitted['factor_references'])))
    kernel=restore.install_factors(derivatives,inverse,admitted['history'])
    if not isinstance(kernel,inverse_secant_fast.InverseSecant):raise ValueError('reviewed fast kernel must be selected explicitly')
    def guard():
        if len(kernel.A)!=len(derivatives) or any(a is b or a!=b for a,b in zip(derivatives,kernel.A)) or inverse is kernel.B or inverse!=kernel.B:
            raise ValueError('fixed exact A/B proposal matrices changed')
    guard();references=list(admitted['factor_references']);prior=len(kernel.history)
    evidence=dict(pending_pair_sha256=admitted['pending_pair_sha256'],
        old_tuple=before['record']['sequence_index'],new_tuple=after['record']['sequence_index'],
        admitted_parent_sha256=base.digest(out/'admitted-parent.json'))
    event('single_pending_secant_started',**evidence)
    decision=kernel.update(before['centers'],before['residuals'],after['centers'],after['residuals'],evidence=evidence)
    base.write_json(out/'pending-secant.json',decision);guard()
    if decision['accepted']:
        references.append(dict(attempt=str(Path(attempt).resolve()),source_commit=source_commit,
            path='run/pending-secant.json',sha256=base.digest(out/'pending-secant.json')))
    else:references=[]
    after_update=len(kernel.history)
    base.write_json(out/'centers-before.json',after['record']['centers'])
    base.write_json(out/'images-before.json',after['record']['images'])
    base.write_json(out/'residual-before.json',after['record']['residual'])
    proposal=kernel.propose(after['centers'],after['residuals'],evidence=dict(
        final_tuple=after['record']['sequence_index'],pending_secant_sha256=base.digest(out/'pending-secant.json')))
    base.write_json(out/'proposal.json',proposal['record']);guard()
    if not kernel.history:references=[]
    if len(kernel.history)!=len(references):raise ValueError('post-proposal factor/witness count differs')
    base.write_json(out/'factor-state.json',dict(history_before_update=prior,history_after_update=after_update,
        history_after_proposal=len(kernel.history),pending_pair_consumed_once=admitted['pending_pair_sha256'],
        active_factors=parent.factor_records(kernel.history,references),certificate_passed=False))
    base.write_json(out/'preconditioner-guard.json',dict(independent_kernel_copies=True,
        original_exact_matrices_unchanged=True,checked_after_update_and_proposal=True,
        checked_matrix_count=len(derivatives)+1))
    base.write_json(out/'centers-after.json',saved.center_records(proposal['centers']))
    event('single_candidate_saved',reason=proposal['record']['reason'],active_history=len(kernel.history))
    return proposal,kernel.history,references


def freeze_request(plan,centers,candidate,expected_assets,prefix_ids,torch_version,source_commit):
    if candidate.exists():raise ValueError('fresh candidate directory required')
    candidate.mkdir()
    bp=plan['base_plan'];mapping=worker.map_spec(expected_assets,prefix_ids);backend=worker.backend(torch_version)
    request=dict(schema=worker.SCHEMA,source_commit=source_commit,
        source_sha256={p:base.digest(ROOT/p) for p in worker.SOURCES},map=mapping,map_sha256=parent.canonical(mapping),
        backend=backend,backend_sha256=parent.canonical(backend),centers=saved.center_records(centers),
        centers_sha256=parent.canonical(saved.center_records(centers)),prefix_attempt=bp['prefix_attempt'],
        prefix_receipt_sha256=bp['prefix_receipt_sha256'],prefix_map_source_sha256=bp['prefix_map_source_sha256'],
        prefix_cache_sha256=base.digest(Path(bp['prefix_attempt'])/'run/prefix-cache.json.gz'),precision=2048,
        selection_provenance=dict(plan_schema=plan['schema'],parent_attempt=plan['parent_attempt'],
            parent_receipt_sha256=plan['parent_receipt_sha256'],proposal_sha256=base.digest(candidate.parent/'proposal.json'),
            statement='One chronological numerical candidate; selection provenance is not a proof.'))
    worker.admit(request,expected_assets,prefix_ids,torch_version,source_commit)
    base.write_json(candidate/'request.json',request)
    return request,base.digest(candidate/'request.json')


def finish_candidate(admitted,proposal,history,references,out,plan,expected_assets,prefix_ids,torch_version,event):
    expected=parent.expected_worker_identity(expected_assets,prefix_ids,torch_version,
        os.environ['CERT_SOURCE_COMMIT'],plan['base_plan'])
    images,provenance=parent.collect_parallel_images(out/'candidate-01',proposal['centers'],expected_identity=expected)
    previous=admitted['after']
    after=parent.tuple_from_records(saved.center_records(proposal['centers']),images,images,
        index=previous['record']['sequence_index']+1,provenance=provenance)
    base.write_json(out/'residual-after.json',after['record']['residual'])
    comparison=dict(before_inf_upper=base.packed(previous['upper']),before_inf_lower=base.packed(previous['lower']),
        after_inf_upper=base.packed(after['upper']),after_inf_lower=base.packed(after['lower']),
        upper_bound_decreased=bool(after['upper']<previous['upper']),
        strict_residual_norm_decrease_established=bool(after['upper']<previous['lower']),
        note='Complete improvement or regression retained; no retry, selection, or orbit certificate.')
    base.write_json(out/'residual-comparison.json',comparison)
    links=admitted['archive_links']+[dict(attempt=plan['parent_attempt'],receipt_sha256=plan['parent_receipt_sha256'],
        external_audit=plan['parent_audit'],capsule_sha256=plan['parent_capsule_sha256'])]
    capsule=dict(schema=parent.CAPSULE_SCHEMA,certificate_passed=False,caps=restore.CAPS,base_plan=plan['base_plan'],
        map=worker.map_spec(expected_assets,prefix_ids),backend=worker.backend(torch_version),
        before=previous['record'],after=after['record'],post_proposal_factors=parent.factor_records(history,references),
        active_history_after_proposal=len(history),pending_pair_sha256=parent.pending_identity(previous,after),
        archive_links=links,cumulative_phase_evaluations=admitted['prior_phase_evaluations']+base.PERIOD,
        trust_boundary='Requires an externally committed independent audit before reuse. Complete immediate inventory is checked; older archive bytes need not be rehashed. Factors are arbitrary exact numerical proposals.')
    base.write_json(out/'continuation-capsule.json',capsule);event('capsule_saved',pending_pair_sha256=capsule['pending_pair_sha256'])
    return comparison,capsule


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path);parser.add_argument('--plan',required=True,type=Path)
    args=parser.parse_args();out=args.output.resolve()
    if out.name!='run' or out.parent.parent!=OWNED.resolve() or (out.parent/'receipt.json').exists():
        raise ValueError('fresh direct-child owned attempt/run required')
    if args.plan.resolve().parent!=HERE or args.plan.suffix!='.json':raise ValueError('committed source-snapshot plan required')
    plan=saved.read_json(args.plan);completed_parent=validate_plan(plan)
    if out.is_relative_to(completed_parent):raise ValueError('completed parent is immutable')
    out.mkdir(parents=True,exist_ok=False);stage='admission';deadline=PROCESS_STARTED+1800
    def event(name,**fields):
        with (out/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(event=name,stage=stage,elapsed_seconds=time.monotonic()-PROCESS_STARTED,
                utc=datetime.now(timezone.utc).isoformat(),**fields),allow_nan=False)+'\n')
    try:
        source=os.environ.get('CERT_SOURCE_COMMIT','')
        if not base.valid_hash(source,40):raise ValueError('bounded-launcher source commit required')
        if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
        os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
        import torch
        from threadpoolctl import threadpool_limits
        sys.path.insert(0,str(ROOT/'codex/crosscheck'))
        from qwen_map import PINNED_SHA256,PREFIX_IDS  # Constants only; no model load.
        torch.set_num_threads(1);torch.set_num_interop_threads(1);ctx.prec=2048;ctx.threads=1
        verify_current_sources(plan)
        with threadpool_limits(limits=1):
            inputs=dict(plan=plan,plan_sha256=base.digest(args.plan),source_commit=source,driver_sha256=base.digest(__file__),
                source_sha256=plan['source_sha256'],pinned_assets=PINNED_SHA256,prefix_ids=list(PREFIX_IDS),precision=2048,threads=1,
                versions=parent.backend_versions(torch.__version__),pid=os.getpid(),certificate_passed=False,
                deadline_monotonic=deadline,statement='One pending secant and one fresh eight-phase pass; numerical setup only.')
            base.write_json(out/'inputs.json',inputs)
            stage='bound_input_admission';arrays,binding=base.load_bound_inputs(plan['base_plan'],PINNED_SHA256,PREFIX_IDS)
            inputs['source_binding']=binding;base.write_json(out/'inputs.json',inputs)
            stage='complete_parent_admission'
            loader=parent.load_initial if plan['parent_kind']=='center_secant' else parent.load_capsule
            admitted=loader(plan,PINNED_SHA256,PREFIX_IDS,torch.__version__,binding)
            base.write_json(out/'parent-admission.json',admitted['admission'])
            if time.monotonic()>=deadline-scheduling.RESERVE_SECONDS:raise ValueError('insufficient frozen worker reserve after admission')
            stage='one_pending_secant'
            arrays['A'].flags.writeable=False;arrays['B'].flags.writeable=False
            derivatives=[base.exact_matrix(a) for a in arrays['A']];inverse=base.exact_matrix(arrays['B'])
            proposal,history,references=prepare_candidate(admitted,derivatives,inverse,out,event,
                source_commit=source,attempt=out.parent)
            stage='freeze_candidate_request'
            request,request_sha=freeze_request(plan,proposal['centers'],out/'candidate-01',PINNED_SHA256,PREFIX_IDS,torch.__version__,source)
            base.write_json(out/'request-binding.json',dict(request_sha256=request_sha,centers_sha256=request['centers_sha256']))
            stage='parallel_fresh_phase_pass'
            scheduling.run_parallel_pass(out/'candidate-01/request.json',request_sha,out/'candidate-01',deadline=deadline,
                max_workers=4,per_worker_timeout=600,aggregate_rss_bytes=32*1024**3,minimum_available_bytes=12*1024**3)
            stage='complete_output_admission'
            comparison,capsule=finish_candidate(admitted,proposal,history,references,out,plan,PINNED_SHA256,PREFIX_IDS,torch.__version__,event)
            verify_current_sources(plan)
        base.write_json(out/'status.json',dict(status='proposal_continuation_completed',certificate_passed=False,
            candidates=1,new_phase_evaluations=base.PERIOD,cumulative_phase_evaluations=capsule['cumulative_phase_evaluations'],
            comparison=comparison,capsule_sha256=base.digest(out/'continuation-capsule.json'),
            elapsed_seconds=time.monotonic()-PROCESS_STARTED,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            reason='Full point residual enclosures for one candidate; no root, attraction or primitive-period proof.'))
        event('completed')
    except BaseException as error:
        base.write_json(out/'status.json',dict(status='proposal_continuation_failed',certificate_passed=False,
            stage=stage,error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-PROCESS_STARTED,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        raise


if __name__=='__main__':main()
