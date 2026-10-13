"""Bounded 64-column streamed-head pilot on the original phase-zero box.

A completed, externally pinned 32-column box baseline is mandatory. This
module does not invent or discover receipt hashes and performs no target work
on import. Prototype-box labels and derivatives are not orbit certificates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import flint
from flint import arb, arb_mat, ctx
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(HERE))
import box_derivative_pilot as base
from streamed_head import StreamedHeadQwenJet

OWNED = base.OWNED
DIMENSION, VOCABULARY, DIRECTIONS, BASE_COLUMNS = 896, 151936, 64, 32
SCHEMA = 'codex-llm-streamed-pilot-v1'
BASE_SOURCES = ('codex/certify-llm/box_derivative_pilot.py',
    'codex/certify-llm/ball_qwen.py','codex/certify-llm/ball_jet.py',
    'codex/certify-llm/ball_linear.py','codex/certify-llm/cache_io.py',
    'codex/certify-llm/phase_domain.py','codex/crosscheck/qwen_map.py')
HASH_KEYS = ('baseline_receipt_sha256','baseline_step_sha256',
             'baseline_actual_inputs_sha256','baseline_actual_balls_sha256')


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_plan(plan, *, owned=None):
    owned = OWNED if owned is None else Path(owned).resolve()
    required = {'schema','baseline_plan','baseline_attempt','baseline_source_sha256',
                'precision','direction_indices','token_chunk','envelope',*HASH_KEYS}
    if not isinstance(plan,dict) or not required<=set(plan) or set(plan)-required-{'purpose','basis'}:
        raise ValueError('complete known streamed-pilot plan required')
    if plan['schema'] != SCHEMA:
        raise ValueError('unsupported streamed-pilot schema')
    base.validate_plan(plan['baseline_plan'],owned=owned)
    if type(plan['precision']) is not int or plan['precision'] != 1024:
        raise ValueError('1024-bit prototype-box scope required')
    if type(plan['token_chunk']) is not int or plan['token_chunk'] != 1024:
        raise ValueError('exactly 1024-token head chunks required')
    indices = plan['direction_indices']
    if not isinstance(indices,list) or indices != list(range(64)) or any(type(i) is not int for i in indices):
        raise ValueError('exactly signed directions e0 through e63 required')
    expected = dict(cpu_threads=1,nice=10,memory_gib=18,timeout_seconds=3600)
    if type(plan['envelope']) is not dict or plan['envelope'] != expected or any(type(v) is not int for v in plan['envelope'].values()):
        raise ValueError('fixed CPU1/18GiB/one-hour envelope required')
    if any(not base.valid_hash(plan[k]) for k in HASH_KEYS):
        raise ValueError('externally supplied full baseline hashes required')
    hashes = plan['baseline_source_sha256']
    if type(hashes) is not dict or set(hashes) != set(BASE_SOURCES) or any(not base.valid_hash(v) for v in hashes.values()):
        raise ValueError('all explicit baseline evaluator/source hashes required')
    if not isinstance(plan['baseline_attempt'],str):
        raise ValueError('owned completed baseline path required')
    baseline = Path(plan['baseline_attempt']).resolve()
    if baseline != owned/'box-derivative-001' or not baseline.is_dir():
        raise ValueError('the completed box-derivative-001 baseline is required')
    return baseline


def basis_record(dimension, count):
    matrix = arb_mat(dimension,count)
    if count > dimension: raise ValueError('coordinate directions exceed full state dimension')
    for k in range(count): matrix[k,k] = 1
    columns = [[base.pack_ball(matrix[i,k]) for i in range(dimension)] for k in range(count)]
    return matrix,dict(direction_indices=list(range(count)),columns=columns,
                       direction_data_sha256=base.canonical_hash(columns))


def load_baseline(plan, expected_assets, prefix_ids):
    """Authenticate every baseline row and reproduce its exact actual domain."""
    baseline = validate_plan(plan)
    if ctx.prec != 1024 or ctx.threads != 1:
        raise ValueError('baseline admission requires the planned 1024-bit CPU1 context')
    receipt, files = base.authenticate_attempt(baseline,plan['baseline_receipt_sha256'])
    if receipt.get('exit_code') != 0 or receipt.get('timed_out') is not False:
        raise ValueError('baseline must have completed successfully without timeout')
    for relative, expected in plan['baseline_source_sha256'].items():
        if base.sha(files['source/'+relative]) != expected or base.sha(ROOT/relative) != expected:
            raise ValueError('baseline/common source mismatch: '+relative)
    inputs,status = read_json(files['run/inputs.json']),read_json(files['run/status.json'])
    bp = plan['baseline_plan']
    if (inputs['plan'] != bp or inputs['source_commit'] != receipt['source_commit']
            or inputs['driver_sha256'] != plan['baseline_source_sha256']['codex/certify-llm/box_derivative_pilot.py']
            or inputs['pinned_assets'] != expected_assets or tuple(inputs['prefix_ids']) != tuple(prefix_ids)
            or inputs['precision'] != 1024 or inputs['threads'] != 1 or inputs['certificate_passed'] is not False):
        raise ValueError('baseline source/plan/map/input scope mismatch')
    plans = [path for name,path in files.items() if name.startswith('source/') and name.endswith('.json')
             and base.sha(path) == inputs['plan_sha256']]
    if len(plans) != 1 or read_json(plans[0]) != bp:
        raise ValueError('baseline plan must bind exactly one captured source file')
    import torch  # Runtime version only, no model construction.
    versions = inputs['versions']
    if (versions['python'] != sys.version or versions['python_flint'] != flint.__version__
            or versions['flint'] != flint.__FLINT_VERSION__ or versions['numpy'] != np.__version__
            or versions['torch'] != str(torch.__version__)):
        raise ValueError('baseline arithmetic backend mismatch')
    expected_status = dict(status='boxed_derivative_setup_completed',certificate_passed=False,
        columns_completed=32,omitted_columns=896-32,comparison_rows_checked=8,
        full_state_dimension=896,full_vocabulary=151936,phase_index=0,precision=1024)
    if any(status.get(k) != v for k,v in expected_status.items()) or status.get('certificate_passed') is not False:
        raise ValueError('complete full-scope 32-column baseline required')
    for k in ('arb_seconds','elapsed_seconds'):
        if type(status.get(k)) not in (float,int) or not math.isfinite(status[k]) or status[k]<0:
            raise ValueError('finite retained baseline timings required')
    if base.sha(files['run/boxed-step.json']) != plan['baseline_step_sha256']:
        raise ValueError('baseline full step hash mismatch')
    if base.sha(files['run/actual-inputs.json']) != plan['baseline_actual_inputs_sha256']:
        raise ValueError('baseline actual input file hash mismatch')
    # Reauthenticate original prefix/proposal/signed point, then regenerate the
    # independently checked actual input balls from the original exact center.
    points, prior8, original_binding = base.load_inputs(bp,expected_assets,prefix_ids)
    artifact,state,domain = base.domain_inputs(points[0])
    saved_domain = read_json(files['run/actual-inputs.json'])
    if (saved_domain != domain or domain['actual_balls_sha256'] != plan['baseline_actual_balls_sha256']
            or read_json(files['run/phase-domain.json']) != artifact):
        raise ValueError('baseline and new actual phase domains must match exactly')
    origin = read_json(files['run/domain-origin.json'])
    if (origin['proposal_sha256'] != base.PINS['proposal_sha256']
            or origin['plan_sha256'] != inputs['plan_sha256']
            or origin['driver_source_sha256'] != inputs['driver_sha256']
            or origin['constructor_source_sha256'] != artifact['helper_source_sha256']
            or origin['generated_artifact_sha256'] != artifact['artifact_sha256']
            or origin['artifact_file_sha256'] != base.sha(files['run/phase-domain.json'])
            or origin['source_orbit_period'] != 8 or origin['source_phase_index'] != 0
            or origin['selected_phase_count'] != 1):
        raise ValueError('baseline domain origin mismatch')
    _,expected_directions = basis_record(DIMENSION,BASE_COLUMNS)
    if read_json(files['run/direction-inputs.json']) != expected_directions:
        raise ValueError('baseline must contain exactly the declared coordinate directions')
    step = read_json(files['run/boxed-step.json'])
    base.validate_saved_step(step,dimension=DIMENSION,count=BASE_COLUMNS)
    if (step['phase_index'] != 0 or step['actual_balls_sha256'] != domain['actual_balls_sha256']
            or step['actual_inputs_file_sha256'] != plan['baseline_actual_inputs_sha256']
            or step['direction_inputs_file_sha256'] != base.sha(files['run/direction-inputs.json'])):
        raise ValueError('baseline step/input/direction binding mismatch')
    comparison = base.compare_saved_steps(step,prior8,dimension=DIMENSION,count=BASE_COLUMNS,compared=8)
    comparison.update(comparison_receipt_sha256=base.PINS['comparison_receipt_sha256'],
                      comparison_step_sha256=base.PINS['comparison_step_sha256'])
    if not comparison['comparison_passed'] or read_json(files['run/comparison.json']) != comparison:
        raise ValueError('baseline comparison did not pass every original point row')
    if inputs['upstream_binding'] != original_binding:
        raise ValueError('baseline upstream lineage mismatch')
    prefix_binding = read_json(files['run/prefix-binding.json'])
    if (prefix_binding['receipt_sha256'] != base.PINS['prefix_receipt_sha256']
            or prefix_binding['graph_sha256'] != base.PINS['prefix_map_source_sha256']
            or prefix_binding['source_commit'] != original_binding['prefix_source_commit']
            or prefix_binding['precision'] != 1024 or prefix_binding['receiver_construction_checked'] is not True
            or prefix_binding['steering_layer'] != 12
            or prefix_binding['prefix_sha256'] != base.sha(Path(bp['prefix_attempt'])/'run/prefix-cache.json.gz')):
        raise ValueError('baseline prefix receiver binding mismatch')
    binding = dict(baseline_source_commit=receipt['source_commit'],baseline_artifacts_checked=len(files),
        original_upstream_binding=original_binding,baseline_rows_rechecked=32,baseline_point_rows_rechecked=8,
        baseline_actual_balls_sha256=domain['actual_balls_sha256'],baseline_arb_seconds=status['arb_seconds'])
    return points,step,binding,artifact,state,domain


def compare_steps(step, baseline):
    comparison = base.compare_saved_steps(step,baseline,dimension=DIMENSION,count=DIRECTIONS,compared=BASE_COLUMNS)
    for row in comparison['columns']:
        row['compared_to_baseline_box'] = row.pop('compared_to_point')
    comparison['statement'] = 'Exact overlap on the identical prototype box; 64 of 896 columns, not an orbit or full-Jacobian proof.'
    return comparison


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out.name!='run' or out.parent.parent!=OWNED.resolve():raise ValueError('fresh owned attempt/run output required')
    if args.plan.resolve().parent!=HERE or args.plan.suffix!='.json':raise ValueError('plan must belong to committed source snapshot')
    plan=read_json(args.plan);baseline=validate_plan(plan)
    protected=[baseline,*base.validate_plan(plan['baseline_plan'],owned=OWNED)]
    if any(out.is_relative_to(p) for p in protected):raise ValueError('upstream attempts are immutable')
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();phase='admission';times={}
    def event(name,**fields):
        row=dict(event=name,phase=phase,elapsed_seconds=time.monotonic()-started,utc=datetime.now(timezone.utc).isoformat(),**fields)
        with (out/'events.jsonl').open('a') as handle:handle.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
    def progress(kind,index,values):
        if kind=='step_start' and 'decoder_start' not in times:times['decoder_start']=time.monotonic()
        if kind=='step_end':times['decoder_end']=time.monotonic()
        event(kind,layer=index)
    def head_progress(kind,first,last):event(kind,row_start=first,row_stop=last)
    try:
        source=os.environ.get('CERT_SOURCE_COMMIT','')
        if not base.valid_hash(source,40):raise ValueError('bounded-launcher source commit required')
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
        os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
        if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
        import torch
        from threadpoolctl import threadpool_limits
        sys.path.insert(0,str(ROOT/'codex/crosscheck'))
        from qwen_map import ManualQwenSoftMap,PINNED_SHA256,PREFIX_IDS
        torch.set_num_threads(1);torch.set_num_interop_threads(1);ctx.prec=1024;ctx.threads=1
        with threadpool_limits(limits=1):
            points,oldstep,binding,artifact,state,domain=load_baseline(plan,PINNED_SHA256,PREFIX_IDS)
            base.write_json(out/'inputs.json',dict(plan=plan,plan_sha256=base.sha(args.plan),source_commit=source,
                driver_sha256=base.sha(__file__),streamed_head_sha256=base.sha(HERE/'streamed_head.py'),
                upstream_binding=binding,pinned_assets=PINNED_SHA256,prefix_ids=PREFIX_IDS,precision=1024,threads=1,pid=os.getpid(),
                versions=dict(python=sys.version,python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,numpy=np.__version__,torch=torch.__version__),
                certificate_passed=False,statement='64-column engineering pilot on original tiny prototype box; no orbit inclusion.'))
            phase='input_domain'
            base.write_json(out/'phase-domain.json',artifact);base.write_json(out/'actual-inputs.json',domain)
            base.write_json(out/'domain-origin.json',dict(proposal_sha256=base.PINS['proposal_sha256'],plan_sha256=base.sha(args.plan),
                baseline_receipt_sha256=plan['baseline_receipt_sha256'],baseline_actual_balls_sha256=plan['baseline_actual_balls_sha256'],
                actual_balls_sha256=domain['actual_balls_sha256'],same_actual_domain=True,
                source_orbit_period=8,source_phase_index=0,selected_phase_count=1,
                statement='Recreated from pinned prototype and checked against the complete baseline actual domain.'))
            directions,records=basis_record(DIMENSION,DIRECTIONS);base.write_json(out/'direction-inputs.json',records)
            phase='model_load';event('load')
            model=ManualQwenSoftMap.from_pinned_files()
            if model.d!=DIMENSION or model.config['vocab_size']!=VOCABULARY:raise ValueError('full assigned state/vocabulary required')
            weights={name:value.numpy() for name,value in model.weights.items()}
            for weight in weights.values():weight.flags.writeable=False
            vector=model.vector.numpy();vector.flags.writeable=False
            ball=StreamedHeadQwenJet(model.config,weights,PREFIX_IDS,vector,direction_count=DIRECTIONS,
                token_chunk=plan['token_chunk'],head_progress=head_progress,alpha=base.ALPHA,log10_temperature=base.LOG10_TEMPERATURE,
                linear_options=dict(chunk_rows=64,chunk_cols=896,cache=base.DyadicWeightCache(max_elements=262144)),progress=progress)
            phase='prefix_restore';bp=plan['baseline_plan']
            prefix_binding=base.load_prefix(bp['prefix_attempt'],ball,expected_receipt_sha256=base.PINS['prefix_receipt_sha256'],
                expected_map_source_sha256=base.PINS['prefix_map_source_sha256'],expected_assets=PINNED_SHA256,
                expected_config=model.config,expected_weights=weights,expected_vector=vector,expected_steering_layer=model.steering_layer)
            base.write_json(out/'prefix-binding.json',prefix_binding);event('prefix_restored')
            phase='streamed_box_jvp';began=time.monotonic()
            values,derivatives=ball.forward(state,directions);finished=time.monotonic()
            step=base.capture_step(values,derivatives,dimension=DIMENSION,count=DIRECTIONS)
            step.update(seconds=finished-began,phase_index=0,actual_balls_sha256=domain['actual_balls_sha256'],
                actual_inputs_file_sha256=base.sha(out/'actual-inputs.json'),direction_inputs_file_sha256=base.sha(out/'direction-inputs.json'))
            base.write_json(out/'streamed-step.json',step);event('streamed_step_saved',sha256=base.sha(out/'streamed-step.json'))
            label=ball.last_head_diagnostics
            if (not isinstance(label,dict) or label.get('vocabulary_count')!=VOCABULARY
                    or label.get('competitor_count')!=VOCABULARY-1 or label.get('certificate_passed') is not False):
                raise ValueError('complete full-vocabulary local label diagnostic required')
            base.write_json(out/'head-diagnostics.json',dict(argmax=label,actual_balls_sha256=domain['actual_balls_sha256'],
                scope='Original prototype input box only; no claim about the unknown periodic orbit.'))
            if not ('decoder_start' in times and 'decoder_end' in times and began<=times['decoder_start']<=times['decoder_end']<=finished):
                raise ValueError('complete decoder timing markers required')
            timing=dict(forward_seconds=finished-began,decoder_seconds=times['decoder_end']-times['decoder_start'],
                post_decoder_seconds=finished-times['decoder_end'],
                post_decoder_scope='Final RMSNorm plus streamed readout and Python return overhead; not isolated head-only time.')
            base.write_json(out/'timing.json',timing)
            phase='baseline_comparison';comparison=compare_steps(step,oldstep)
            comparison.update(baseline_receipt_sha256=plan['baseline_receipt_sha256'],baseline_step_sha256=plan['baseline_step_sha256'],
                              same_actual_domain=True,actual_balls_sha256=domain['actual_balls_sha256'])
            base.write_json(out/'comparison.json',comparison)
            if not comparison['comparison_passed']:raise ValueError('one or more full baseline-box overlap rows failed; all output retained')
            base.write_json(out/'status.json',dict(status='streamed_derivative_setup_completed',certificate_passed=False,
                columns_completed=DIRECTIONS,omitted_columns=DIMENSION-DIRECTIONS,comparison_rows_checked=BASE_COLUMNS,
                phase_index=0,full_state_dimension=DIMENSION,full_vocabulary=VOCABULARY,precision=1024,token_chunk=1024,
                timing=timing,elapsed_seconds=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                reason='Prototype-box derivative and optional local-label measurement only; no orbit, inclusion, full Jacobian, or attraction certificate.'))
            event('completed')
    except Exception as error:
        base.write_json(out/'status.json',dict(status='streamed_derivative_setup_failed',certificate_passed=False,phase=phase,
            error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        raise


if __name__=='__main__':main()
