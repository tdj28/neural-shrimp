"""One bounded inverse-secant centre proposal from complete authenticated history.

Only completed fixed-R center_refine/center_resume ancestors are admitted. The
existing strict importer authenticates their entire lineage. This driver then
reads every genuinely evaluated tuple in chronological order, recomputes its
full residuals, replays all consecutive secants, and evaluates exactly one new
candidate at all phases. No secant parent, partial-parent recovery, retry, model
change, certificate, or history selection by improvement is supported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import flint
from flint import arb_mat, ctx
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import center_resume as resume
from inverse_secant import InverseSecant
base = resume.base

OWNED = base.OWNED
SCHEMA = 'codex-llm-center-secant-v1'
METHOD_SOURCES = ('codex/certify-llm/center_resume.py', 'codex/certify-llm/inverse_secant.py')
CAPS = dict(max_history=4, signal_ratio=128, update_norm_limit=1024, step_ratio_limit=4)
ENVELOPE = dict(cpu_threads=1, nice=10, memory_gib=18, timeout_seconds=1800)


def _fixed_integers(value, expected, label):
    if (type(value) is not dict or value != expected
            or any(type(v) is not int for v in value.values())):
        raise ValueError('exact fixed integer '+label+' required')


def validate_plan(plan, *, owned=None):
    owned = OWNED if owned is None else Path(owned).resolve()
    fields = {'schema','parent_admission_plan','method_source_sha256','precision',
              'history_tuple_count','max_candidates','caps','envelope'}
    if (type(plan) is not dict or not fields <= set(plan)
            or set(plan)-fields-{'purpose','basis'} or plan['schema'] != SCHEMA):
        raise ValueError('complete known one-candidate secant plan required')
    for key, expected in [('precision',2048),('max_candidates',1)]:
        if type(plan[key]) is not int or plan[key] != expected:
            raise ValueError('fixed one-candidate 2048-bit scope required')
    if type(plan['history_tuple_count']) is not int or not 2 <= plan['history_tuple_count'] <= 16:
        raise ValueError('explicit complete history count from two through sixteen required')
    _fixed_integers(plan['caps'], CAPS, 'secant caps')
    _fixed_integers(plan['envelope'], ENVELOPE, 'CPU1/18GiB/1800-second envelope')
    # The nested object is an existing importer admission contract only. Its
    # one-hour envelope never authorizes work by this new half-hour driver.
    parent = resume.validate_plan(plan['parent_admission_plan'], owned=owned)
    pins = plan['method_source_sha256']
    if (type(pins) is not dict or set(pins) != set(METHOD_SOURCES)
            or any(not base.valid_hash(v) for v in pins.values())):
        raise ValueError('explicit current importer and inverse-secant source hashes required')
    return parent


def _tuple_from_files(files, entry, directory, label, index):
    path = directory+'/centers-'+label+'.json'
    centers = resume.decoded_centers(resume.read_json(files[path]), base.PERIOD, base.DIMENSION)
    images, rows, reuse = resume.inspect_pass(files, directory, label, centers)
    # Do not reuse a saved residual: the successor belongs to this complete
    # tuple, and imported image radii may have widened conservatively on load.
    upper, lower, residual_record = base.residual_summary(centers, images)
    residuals = [[images[i][j]-centers[(i+1)%base.PERIOD][j]
                  for j in range(base.DIMENSION)] for i in range(base.PERIOD)]
    record = dict(sequence_index=index, attempt=entry['attempt'], receipt_sha256=entry['receipt_sha256'],
        source_commit=entry['source_commit'], kind=entry['kind'], directory=directory, evaluation_pass=label,
        centers_path=path, centers_sha256=base.digest(files[path]),
        exact_centers=resume.center_records(centers), decoded_images=resume.center_records(images),
        phase_artifacts=reuse, original_phase_outcomes=rows, recomputed_residual=residual_record,
        residual_recomputed=True, no_image_radius_discarded=True)
    return dict(centers=centers, images=images, residuals=residuals,
                upper=upper, lower=lower, record=record)


def load_history(plan, expected_assets, prefix_ids, *, binding):
    """Authenticate with the unchanged importer, then extract every full tuple."""
    validate_plan(plan)
    for name, expected in plan['method_source_sha256'].items():
        if base.digest(ROOT/name) != expected:
            raise ValueError('reviewed importer/kernel source mismatch: '+name)
    inherited = resume.load_completed_parent(plan['parent_admission_plan'], expected_assets,
                                             prefix_ids, binding=binding)
    states = []
    for entry_index, entry in enumerate(inherited['lineage']):
        receipt, files = base.checked_attempt(entry['attempt'], entry['receipt_sha256'])
        if receipt['source_commit'] != entry['source_commit']:
            raise ValueError('history source changed after complete-parent admission')
        if entry_index == 0:
            if entry['kind'] != 'center_refine':
                raise ValueError('history must begin with complete original refinement')
            states.append(_tuple_from_files(files, entry, 'run', 'before', len(states)))
            states.append(_tuple_from_files(files, entry, 'run', 'after', len(states)))
        else:
            if entry['kind'] != 'center_resume':
                raise ValueError('only completed fixed-R resume ancestors are supported')
            iterations = resume.read_json(files['run/iterations.json'])
            for iteration in iterations:
                number = iteration['correction_index']
                directory = f'run/correction-{number:02d}'
                # The imported before tuple is an alias, not a fresh evaluation.
                resume.same_centers(resume.read_json(files[directory+'/centers-before.json']),
                                    states[-1]['centers'])
                states.append(_tuple_from_files(files, entry, directory, 'after', len(states)))
    if len(states) != plan['history_tuple_count']:
        raise ValueError('all-history tuple count differs from the frozen plan')
    if inherited['cumulative_phase_evaluations'] != len(states)*base.PERIOD:
        raise ValueError('history omits or duplicates actual complete phase evaluations')
    resume.same_centers(states[-1]['record']['exact_centers'], inherited['centers'])
    if states[-1]['record']['phase_artifacts'] != inherited['reuse']:
        raise ValueError('final history image provenance differs from admitted parent')
    return states, inherited


def prepare_candidate(states, derivatives, inverse, out, event):
    """Replay all chronological secants, persist each decision, propose once."""
    if not isinstance(states,list) or not 2 <= len(states) <= 16:
        raise ValueError('complete bounded chronological history required')
    records = [state['record'] for state in states]
    if [record['sequence_index'] for record in records] != list(range(len(states))):
        raise ValueError('history must be in complete chronological order')
    base.write_json(out/'history-tuples.json', records)
    # Give the mutable Arb kernel independent matrices. Native exact equality
    # checks after every action detect changes before any model evaluation.
    kernel = InverseSecant([arb_mat(a) for a in derivatives], arb_mat(inverse), **CAPS)
    def check_preconditioner():
        if (any(saved is active or saved != active for saved,active in zip(derivatives,kernel.A))
                or inverse is kernel.B or inverse != kernel.B):
            raise ValueError('fixed exact preconditioner changed during secant replay')
    check_preconditioner()
    attempts = []
    for number in range(1,len(states)):
        old, new = states[number-1], states[number]
        metadata = dict(old_tuple=number-1,new_tuple=number,
            history_file_sha256=base.digest(out/'history-tuples.json'),
            old_receipt_sha256=old['record']['receipt_sha256'],
            new_receipt_sha256=new['record']['receipt_sha256'])
        attempt = dict(secant_index=number, status='running', evidence=metadata)
        attempts.append(attempt); base.write_json(out/'secant-outcomes.json', attempts)
        event('secant_started', secant_index=number)
        try:
            decision = kernel.update(old['centers'],old['residuals'],new['centers'],new['residuals'],evidence=metadata)
            path = out/f'secant-{number:02d}.json';base.write_json(path,decision)
            check_preconditioner()
        except Exception as error:
            attempt.update(status='failed',error=repr(error));base.write_json(out/'secant-outcomes.json',attempts)
            raise
        attempt.update(status='completed',accepted=decision['accepted'],reason=decision['reason'],
                       decision_path=path.name,decision_sha256=base.digest(path),active_history=decision['active_history'])
        base.write_json(out/'secant-outcomes.json',attempts)
        event('secant_completed', **attempt)
    final = states[-1]
    base.write_json(out/'centers-before.json', resume.center_records(final['centers']))
    base.write_json(out/'images-before.json', resume.center_records(final['images']))
    base.write_json(out/'residual-before.json', final['record']['recomputed_residual'])
    # Inputs and all history are already on disk if proposal arithmetic fails.
    proposal = kernel.propose(final['centers'],final['residuals'],
        evidence=dict(final_tuple=len(states)-1,history_file_sha256=base.digest(out/'history-tuples.json'),
                      secant_outcomes_sha256=base.digest(out/'secant-outcomes.json')))
    base.write_json(out/'proposal.json',proposal['record'])
    check_preconditioner()
    base.write_json(out/'preconditioner-guard.json',dict(independent_kernel_copies=True,
        original_exact_matrices_unchanged=True,checked_after_each_secant_and_proposal=True,
        checked_matrix_count=len(derivatives)+1))
    base.write_json(out/'centers-after.json',resume.center_records(proposal['centers']))
    event('single_candidate_saved',reason=proposal['record']['reason'])
    return proposal,attempts


def evaluate_candidate(model, proposal, previous, out, event, outcomes):
    """Exactly one fresh complete phase pass, including a worsened residual."""
    images = base.evaluate_pass(model,proposal['centers'],out,'after',event,outcomes)
    upper,lower,record = base.residual_summary(proposal['centers'],images)
    base.write_json(out/'residual-after.json',record)
    comparison = dict(before_inf_upper=base.packed(previous['upper']),before_inf_lower=base.packed(previous['lower']),
        after_inf_upper=base.packed(upper),after_inf_lower=base.packed(lower),
        upper_bound_decreased=bool(upper<previous['upper']),
        strict_residual_norm_decrease_established=bool(upper<previous['lower']),
        note='One candidate only; retained improvement or regression is a setup diagnostic, never a certificate.')
    base.write_json(out/'residual-comparison.json',comparison)
    return comparison


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out.name!='run' or out.parent.parent!=OWNED.resolve():
        raise ValueError('fresh direct-child Task1 attempt/run output required')
    if args.plan.resolve().parent!=HERE or args.plan.suffix!='.json':
        raise ValueError('plan must belong to the committed source snapshot')
    plan=resume.read_json(args.plan);parent=validate_plan(plan)
    if out.is_relative_to(parent):raise ValueError('completed parents are immutable')
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();phase='admission';outcomes=[];attempts=[]
    def event(name,**fields):
        row=dict(event=name,phase=phase,elapsed_seconds=time.monotonic()-started,
                 utc=datetime.now(timezone.utc).isoformat(),**fields)
        with (out/'events.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
    try:
        source=os.environ.get('CERT_SOURCE_COMMIT','')
        if not base.valid_hash(source,40):raise ValueError('bounded-launcher source commit required')
        if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
        os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
        import torch
        from threadpoolctl import threadpool_limits
        sys.path.insert(0,str(ROOT/'codex/crosscheck'))
        from qwen_map import ManualQwenSoftMap,PINNED_SHA256,PREFIX_IDS
        torch.set_num_threads(1);torch.set_num_interop_threads(1);ctx.prec=2048;ctx.threads=1
        bp=plan['parent_admission_plan']['base_plan']
        with threadpool_limits(limits=1):
            inputs=dict(plan=plan,plan_sha256=base.digest(args.plan),source_commit=source,
                driver_sha256=base.digest(__file__),method_source_sha256=plan['method_source_sha256'],
                pinned_assets=PINNED_SHA256,prefix_ids=list(PREFIX_IDS),precision=ctx.prec,threads=1,pid=os.getpid(),
                certificate_passed=False,versions=dict(python=sys.version,python_flint=flint.__version__,
                    flint=flint.__FLINT_VERSION__,numpy=np.__version__,torch=str(torch.__version__)),
                statement='All authenticated fixed-R history, one inverse-secant midpoint candidate, one fresh full phase pass; no proof.')
            base.write_json(out/'inputs.json',inputs)
            phase='bound_input_admission';arrays,binding=base.load_bound_inputs(bp,PINNED_SHA256,PREFIX_IDS)
            inputs['source_binding']=binding;base.write_json(out/'inputs.json',inputs)
            phase='completed_history_admission';states,inherited=load_history(plan,PINNED_SHA256,PREFIX_IDS,binding=binding)
            base.write_json(out/'history-admission.json',dict(lineage=inherited['lineage'],
                history_tuple_count=len(states),cumulative_prior_phase_evaluations=inherited['cumulative_phase_evaluations'],
                cumulative_prior_elapsed_seconds=inherited['cumulative_elapsed_seconds'],
                all_complete_tuples_included=True,residuals_recomputed=True,no_image_radius_discarded=True))
            phase='secant_replay'
            arrays['A'].flags.writeable=False;arrays['B'].flags.writeable=False
            derivatives=[base.exact_matrix(a) for a in arrays['A']];inverse=base.exact_matrix(arrays['B'])
            proposal,attempts=prepare_candidate(states,derivatives,inverse,out,event)
            phase='model_load';model=ManualQwenSoftMap.from_pinned_files()
            weights={name:tensor.numpy() for name,tensor in model.weights.items()}
            for weight in weights.values():weight.flags.writeable=False
            vector=model.vector.numpy();vector.flags.writeable=False
            ball=base.BallQwen(model.config,weights,PREFIX_IDS,vector,alpha=base.ALPHA,log10_temperature=base.LOG10_TEMPERATURE,
                linear_options=dict(chunk_rows=64,chunk_cols=896,cache=base.DyadicWeightCache(max_elements=262144)),
                progress=lambda kind,index,values:event(kind,layer=index))
            phase='prefix_restore'
            prefix_binding=base.load_prefix(bp['prefix_attempt'],ball,expected_receipt_sha256=bp['prefix_receipt_sha256'],
                expected_map_source_sha256=bp['prefix_map_source_sha256'],expected_assets=PINNED_SHA256,
                expected_config=model.config,expected_weights=weights,expected_vector=vector,expected_steering_layer=model.steering_layer)
            base.write_json(out/'prefix-binding.json',prefix_binding)
            phase='single_fresh_phase_pass'
            comparison=evaluate_candidate(ball,proposal,states[-1],out,event,outcomes)
            if len(outcomes)!=base.PERIOD or any(v['status']!='completed' for v in outcomes):
                raise ValueError('one complete fresh phase tuple required')
        elapsed=time.monotonic()-started
        base.write_json(out/'status.json',dict(status='proposal_secant_completed',certificate_passed=False,
            candidates=1,history_tuples=len(states),secant_outcomes=attempts,phase_outcomes=outcomes,comparison=comparison,
            new_phase_evaluations=len(outcomes),cumulative_phase_evaluations=inherited['cumulative_phase_evaluations']+len(outcomes),
            prior_elapsed_seconds=inherited['cumulative_elapsed_seconds'],elapsed_seconds=elapsed,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            reason='One exact midpoint proposal and fresh residual enclosures only; no root, primitive-period or attraction proof.'))
        event('completed')
    except Exception as error:
        for row in outcomes:
            if row['status']=='running':row.update(status='failed',error=repr(error))
        base.write_json(out/'phase-outcomes.json',outcomes)
        base.write_json(out/'status.json',dict(status='proposal_secant_failed',certificate_passed=False,
            phase=phase,phase_outcomes=outcomes,error=repr(error),traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        raise


if __name__=='__main__':main()
