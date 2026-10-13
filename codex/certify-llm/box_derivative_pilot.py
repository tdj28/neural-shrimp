"""Bounded phase-zero, 32-column whole-box derivative engineering pilot.

The nominal radius 2^-800 is far below the measured prototype residual and
is not proposed as an isolating orbit box. This is never a root gate.
Only the inherited BallQwenJet architecture evaluates the map. No checkpoint
or model evaluation occurs on import; actual execution needs a pushed plan.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np
import flint
from flint import arb, arb_mat, ctx

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ball_jet import BallQwenJet
from ball_linear import DyadicWeightCache
from ball_qwen import exact
from cache_io import load_prefix, unpack
from phase_domain import (build_phase_domains, decode_verified_domains,
                          containment_witnesses, canonical_hash, pack_ball)

OWNED = Path('/data2/neural-shrimp-data/codex/llm-p8-certificate')
DIMENSION, VOCABULARY, PERIOD, DIRECTIONS = 896, 151936, 8, 32
ALPHA, LOG10_TEMPERATURE = .071674, -.115
PINS = dict(
    prefix_receipt_sha256='565ecc98b37b592bb1da16a58ad4109e1dd2ab5f2eac71682f2af860531f33b9',
    prefix_map_source_sha256='3250d8b262e4a940b83de84005ad6f3155bc70635bfc623b0a000804cfc28c3e',
    proposal_sha256='d8c57bff1d8a986c7c6a0f7d083caa77b4f6373c560d1f60ce0c57461caf668b',
    comparison_receipt_sha256='32b73defdf9672e2593ad33370a37fce2535ccb7c9ddc04769d2c16cd70add7d',
    comparison_step_sha256='6bf1bb80e2a662adeb5f87880091ae27a75ff299319f82ef0578d0d4d0f4d666')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def valid_hash(value, length=64):
    return isinstance(value,str) and len(value)==length and all(v in '0123456789abcdef' for v in value)


def validate_plan(plan, *, owned=None):
    owned = OWNED if owned is None else Path(owned).resolve()
    required = {'schema','prefix_attempt','comparison_attempt','precision','alpha','log10_temperature',
                'period','dimension','vocabulary','phase_index','nominal_radius_power','direction_indices',
                'envelope',*PINS}
    if not isinstance(plan,dict) or not required<=set(plan) or set(plan)-required-{'purpose','basis'}:
        raise ValueError('complete known boxed-derivative plan fields required')
    if plan['schema']!='codex-llm-box-derivative-pilot-v1':
        raise ValueError('unsupported boxed-derivative schema')
    for key,value in dict(precision=1024,period=8,dimension=896,vocabulary=151936,
                          phase_index=0,nominal_radius_power=-800).items():
        if type(plan[key]) is not int or plan[key]!=value:
            raise ValueError('different fixed engineering scope: '+key)
    for key,value in [('alpha',ALPHA),('log10_temperature',LOG10_TEMPERATURE)]:
        if type(plan[key]) is not float or plan[key]!=value:
            raise ValueError('different assigned parameter: '+key)
    indices=plan['direction_indices']
    if not isinstance(indices,list) or any(type(v) is not int for v in indices) or indices!=list(range(32)):
        raise ValueError('exactly directions e0 through e31 required')
    envelope=plan['envelope'];expected=dict(cpu_threads=1,nice=10,memory_gib=18,timeout_seconds=3600)
    if not isinstance(envelope,dict) or envelope!=expected or any(type(v) is not int for v in envelope.values()):
        raise ValueError('fixed CPU1/18GiB/one-hour envelope required')
    if any(plan[key]!=value for key,value in PINS.items()):
        raise ValueError('different immutable upstream evidence pin')
    result=[]
    for field,name in [('prefix_attempt','arb-pilot-002'),('comparison_attempt','signed-pilot-001')]:
        if not isinstance(plan[field],str):raise ValueError('owned upstream attempt path required')
        path=Path(plan[field]).resolve()
        if path!=owned/name or not path.is_dir():raise ValueError('wrong fixed upstream attempt: '+field)
        result.append(path)
    return tuple(result)


def authenticate_attempt(attempt, expected_receipt_sha256):
    """Check every receipt artifact and source-manifest entry; no model load.

    Outcome classification is deliberately separate: the pinned prefix came
    from a failed *later box* test, whose complete prefix remains admissible.
    """
    attempt=Path(attempt).resolve();receipt_path=attempt/'receipt.json'
    if not valid_hash(expected_receipt_sha256) or sha(receipt_path)!=expected_receipt_sha256:
        raise ValueError('upstream receipt hash mismatch')
    receipt=json.loads(receipt_path.read_text());rows=receipt.get('artifacts')
    if not valid_hash(receipt.get('source_commit'),40) or not isinstance(rows,list) or not rows:
        raise ValueError('complete source-bound receipt required')
    files={}
    for row in rows:
        name=row['path']
        if not isinstance(name,str) or name in files:raise ValueError('duplicate/invalid inventory path')
        path=(attempt/name).resolve()
        if not path.is_relative_to(attempt) or not path.is_file():raise ValueError('artifact escapes owned attempt')
        if type(row['bytes']) is not int or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise ValueError('artifact size/hash mismatch: '+name)
        files[name]=path
    actual={str(path.relative_to(attempt)) for path in attempt.rglob('*') if path.is_file()}
    if actual!=set(files)|{'receipt.json'}:raise ValueError('unaccounted/missing upstream files')
    manifest=json.loads(files['manifest.json'].read_text())
    if manifest['source_commit']!=receipt['source_commit'] or not isinstance(manifest['source'],list):
        raise ValueError('upstream source manifest mismatch')
    source_names=set()
    for row in manifest['source']:
        name='source/'+row['path']
        if name in source_names or name not in files:raise ValueError('missing/duplicate captured source')
        source_names.add(name)
        if files[name].stat().st_size!=row['bytes'] or sha(files[name])!=row['sha256']:
            raise ValueError('captured source manifest mismatch')
    if source_names!={name for name in files if name.startswith('source/')}:
        raise ValueError('source inventory differs from manifest')
    return receipt,files


def _point_record(record):
    """Exact saved endpoints, independent of outward Arb decoding changes."""
    if not isinstance(record,dict) or set(record)!={'mid','rad'}:
        raise ValueError('packed exact midpoint/radius record required')
    values=[]
    for key in ('mid','rad'):
        pair=record[key]
        if not isinstance(pair,list) or len(pair)!=2 or any(type(v) is not int for v in pair):
            raise ValueError('dyadic integer pairs required')
        m,e=pair;values.append(Fraction(m<<e) if e>=0 else Fraction(m,1<<(-e)))
    mid,rad=values
    if rad<0:raise ValueError('negative enclosure radius')
    decoded=unpack(record)
    packed=pack_ball(decoded)
    # Constructors may enlarge radius; never compare using falsely narrowed data.
    dm,de=packed['mid'];rm,re=packed['rad']
    actual_mid=Fraction(dm<<de) if de>=0 else Fraction(dm,1<<(-de))
    actual_rad=Fraction(rm<<re) if re>=0 else Fraction(rm,1<<(-re))
    if actual_mid-actual_rad>mid-rad or actual_mid+actual_rad<mid+rad:
        raise ValueError('decoded enclosure lost saved uncertainty')
    return mid-rad,mid+rad


def validate_saved_step(step, *, dimension, count):
    if type(dimension) is not int or dimension<1 or type(count) is not int or count<1:
        raise ValueError('positive integer full dimensions required')
    if not isinstance(step,dict) or step.get('direction_indices')!=list(range(count)):
        raise ValueError('saved direction order mismatch')
    if any(type(v) is not int for v in step['direction_indices']):raise ValueError('integer directions required')
    if not isinstance(step.get('value'),list) or len(step['value'])!=dimension:
        raise ValueError('full finite primal dimension required')
    cols=step.get('columns')
    if not isinstance(cols,list) or len(cols)!=count or any(not isinstance(v,list) or len(v)!=dimension for v in cols):
        raise ValueError('every full derivative column required')
    for value in step['value']:_point_record(value)
    for col in cols:
        for value in col:_point_record(value)
    return step


def compare_saved_steps(current, prior, *, dimension, count=32, compared=8):
    """Check every compared entry via exact rational interval overlap.

    Both evaluations contain the same true point derivative if their reviewed
    evaluator/domain contracts hold. Containment of one interval in the other
    is not required. Remaining columns get complete finite/radius summaries.
    """
    if type(compared) is not int or type(count) is not int or not 1<=compared<=count:raise ValueError('invalid comparison count')
    validate_saved_step(current,dimension=dimension,count=count)
    validate_saved_step(prior,dimension=dimension,count=compared)
    def overlap(a,b):
        lo,hi=_point_record(a);plo,phi=_point_record(b)
        return lo<=phi and plo<=hi
    primal_fail=[i for i,(a,b) in enumerate(zip(current['value'],prior['value'])) if not overlap(a,b)]
    rows=[]
    for k,col in enumerate(current['columns']):
        radius=max((arb(tuple(v['rad'])) for v in col))
        row=dict(direction=k,finite_entries=dimension,maximum_radius=pack_ball(radius),compared_to_point=k<compared)
        if k<compared:
            failed=[i for i,(a,b) in enumerate(zip(col,prior['columns'][k])) if not overlap(a,b)]
            row.update(overlap_passed=not failed,overlap_count=dimension-len(failed),nonoverlap_coordinates=failed)
        rows.append(row)
    passed=not primal_fail and all(v['overlap_passed'] for v in rows[:compared])
    return dict(comparison_passed=bool(passed),primal_overlap_count=dimension-len(primal_fail),
        primal_nonoverlap_coordinates=primal_fail,columns=rows,compared_columns=compared,
        comparison_method='exact saved dyadic endpoints; overlap only, no floating map replay',
        statement='Compatibility diagnostic only; neither overlap nor 32 columns certify the omitted directions or an orbit.')


def load_inputs(plan, expected_assets, prefix_ids):
    prefix,comparison=validate_plan(plan)
    pr,pf=authenticate_attempt(prefix,plan['prefix_receipt_sha256'])
    cr,cf=authenticate_attempt(comparison,plan['comparison_receipt_sha256'])
    pi=json.loads(pf['run/inputs.json'].read_text());ci=json.loads(cf['run/inputs.json'].read_text())
    ps=json.loads(pf['run/status.json'].read_text());cs=json.loads(cf['run/status.json'].read_text())
    if (pr['exit_code']==0 or pr['timed_out'] is not False or ps['status']!='setup_failed'
            or ps['phase']!='rigorous_box_jvp' or ps['certificate_passed'] is not False):
        raise ValueError('unexpected classification of fixed prefix attempt')
    if cr['exit_code']!=0 or cr['timed_out'] is not False or cs['status']!='setup_completed' or cs['certificate_passed'] is not False:
        raise ValueError('signed comparison attempt did not complete')
    for inp,rec in [(pi,pr),(ci,cr)]:
        if inp['source_commit']!=rec['source_commit'] or inp['pinned_assets']!=expected_assets or inp['precision']!=1024:
            raise ValueError('upstream source/assets/precision mismatch')
    if (tuple(pi['prefix_ids'])!=tuple(prefix_ids) or pi['alpha_ratio']!=list(ALPHA.as_integer_ratio())
            or pi['log10_temperature_ratio']!=list(LOG10_TEMPERATURE.as_integer_ratio())
            or pi['arb_context']!={'precision':1024,'threads':1}):
        raise ValueError('prefix exact-map contract mismatch')
    cp=ci['plan']
    if (cp['alpha']!=ALPHA or cp['log10_temperature']!=LOG10_TEMPERATURE
            or cp['phase_index']!=0 or cp['direction_indices']!=list(range(8))
            or cp['precision']!=1024 or cp['receipt_sha256']!=PINS['prefix_receipt_sha256']
            or cp['map_source_sha256']!=PINS['prefix_map_source_sha256'] or cp['proposal_sha256']!=PINS['proposal_sha256']):
        raise ValueError('comparison point/map/direction binding mismatch')
    if sha(pf['source/codex/certify-llm/ball_qwen.py'])!=PINS['prefix_map_source_sha256']:
        raise ValueError('prefix source graph mismatch')
    if sha(pf['run/proposal.npz'])!=PINS['proposal_sha256'] or sha(cf['run/signed-step.json'])!=PINS['comparison_step_sha256']:
        raise ValueError('proposal/comparison raw hash mismatch')
    with np.load(pf['run/proposal.npz'],allow_pickle=False) as archive:
        points=archive['points'].copy()
    if points.shape!=(9,DIMENSION) or points.dtype!=np.float64 or not np.isfinite(points).all():
        raise ValueError('finite full binary64 proposal required')
    oldstep=json.loads(cf['run/signed-step.json'].read_text())
    validate_saved_step(oldstep,dimension=DIMENSION,count=8)
    return points,oldstep,dict(prefix_artifacts_checked=len(pf),comparison_artifacts_checked=len(cf),
        prefix_source_commit=pr['source_commit'],comparison_source_commit=cr['source_commit'])


def domain_inputs(center, *, precision=1024, radius_power=-800):
    """Build, independently verify, then reconstruct and recheck actual balls.

    The expected artifact hash below is computed from a newly generated trusted
    construction, NOT adopted from an untrusted incoming artifact's self-hash.
    Callers bind this construction to the committed plan and pinned proposal.
    """
    nominal=[[pack_ball(exact(v)) for v in center]]
    radii=[pack_ball(arb((1,radius_power)))]
    artifact=build_phase_domains(nominal,radii,period=1,dimension=len(center),target_precision=precision)
    expected=canonical_hash({k:v for k,v in artifact.items() if k!='artifact_sha256'})
    decoded,_=decode_verified_domains(artifact,nominal,radii,period=1,dimension=len(center),
        target_precision=precision,expected_artifact_sha256=expected)
    # Exercise the real receiver reconstruction, retaining any extra radius ulp.
    actual=[[unpack(pack_ball(v)) for v in row] for row in decoded]
    checked=containment_witnesses(nominal,radii,actual,period=1,dimension=len(center))
    return artifact,actual[0],checked


def capture_step(values, derivatives, *, dimension, count):
    if not isinstance(values,list) or len(values)!=dimension:
        raise ValueError('full primal vector required')
    if not isinstance(derivatives,arb_mat) or (derivatives.nrows(),derivatives.ncols())!=(dimension,count):
        raise ValueError('full declared derivative matrix required')
    result=dict(value=[pack_ball(v) for v in values],
        columns=[[pack_ball(derivatives[i,k]) for i in range(dimension)] for k in range(count)],
        direction_indices=list(range(count)))
    return validate_saved_step(result,dimension=dimension,count=count)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if not out.is_relative_to(OWNED) or out==OWNED:raise ValueError('exclusive owned output required')
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();phase='admission'
    def event(name,**fields):
        row=dict(event=name,phase=phase,elapsed_seconds=time.monotonic()-started,
            utc=datetime.now(timezone.utc).isoformat(),**fields)
        with (out/'events.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
    def progress(kind,index,values):
        event(kind,layer=index)
    try:
        source=os.environ.get('CERT_SOURCE_COMMIT','')
        if not valid_hash(source,40):raise ValueError('exact source commit from bounded launcher required')
        if args.plan.resolve().parent!=HERE or args.plan.suffix!='.json':raise ValueError('plan must be in committed snapshot')
        plan=json.loads(args.plan.read_text());prefix,_=validate_plan(plan)
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
        os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
        if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
        import torch
        from threadpoolctl import threadpool_limits
        sys.path.insert(0,str(ROOT/'codex/crosscheck'))
        from qwen_map import ManualQwenSoftMap,PINNED_SHA256,PREFIX_IDS
        torch.set_num_threads(1);torch.set_num_interop_threads(1);ctx.prec=1024;ctx.threads=1
        with threadpool_limits(limits=1):
            points,oldstep,binding=load_inputs(plan,PINNED_SHA256,PREFIX_IDS)
            write_json(out/'inputs.json',dict(plan=plan,plan_sha256=sha(args.plan),source_commit=source,
                driver_sha256=sha(__file__),upstream_binding=binding,pinned_assets=PINNED_SHA256,
                prefix_ids=PREFIX_IDS,precision=ctx.prec,threads=1,pid=os.getpid(),
                versions=dict(python=sys.version,python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,numpy=np.__version__,torch=torch.__version__),
                certificate_passed=False,statement='one tiny phase box, 32 of 896 derivative columns; no root inclusion gate; radius is not proposed as an isolating orbit box'))
            phase='input_domain';artifact,state,domain=domain_inputs(points[0])
            write_json(out/'phase-domain.json',artifact)
            write_json(out/'domain-origin.json',dict(constructor_source_sha256=artifact['helper_source_sha256'],
                generated_artifact_sha256=artifact['artifact_sha256'],artifact_file_sha256=sha(out/'phase-domain.json'),
                plan_sha256=sha(args.plan),driver_source_sha256=sha(__file__),proposal_sha256=PINS['proposal_sha256'],
                source_orbit_period=8,source_phase_index=0,selected_phase_count=1,
                statement='Fresh domain construction from pinned proposal and committed plan; exact endpoint verification precedes evaluation.'))
            write_json(out/'actual-inputs.json',domain)
            directions=arb_mat(DIMENSION,DIRECTIONS)
            for k in range(DIRECTIONS):directions[k,k]=1
            direction_records=[[pack_ball(directions[i,k]) for i in range(DIMENSION)] for k in range(DIRECTIONS)]
            write_json(out/'direction-inputs.json',dict(direction_indices=list(range(DIRECTIONS)),columns=direction_records,
                direction_data_sha256=canonical_hash(direction_records)))
            phase='model_load';event('load')
            model=ManualQwenSoftMap.from_pinned_files()
            if model.d!=DIMENSION or model.config['vocab_size']!=VOCABULARY:raise ValueError('assigned full state/vocabulary required')
            weights={name:tensor.numpy() for name,tensor in model.weights.items()}
            for weight in weights.values():weight.flags.writeable=False
            vector=model.vector.numpy();vector.flags.writeable=False
            ball=BallQwenJet(model.config,weights,PREFIX_IDS,vector,direction_count=DIRECTIONS,
                alpha=ALPHA,log10_temperature=LOG10_TEMPERATURE,
                linear_options=dict(chunk_rows=64,chunk_cols=896,cache=DyadicWeightCache(max_elements=262144)),progress=progress)
            phase='prefix_restore'
            cache_binding=load_prefix(prefix,ball,expected_receipt_sha256=PINS['prefix_receipt_sha256'],
                expected_map_source_sha256=PINS['prefix_map_source_sha256'],expected_assets=PINNED_SHA256,
                expected_config=model.config,expected_weights=weights,expected_vector=vector,expected_steering_layer=model.steering_layer)
            write_json(out/'prefix-binding.json',cache_binding);event('prefix_restored')
            phase='boxed_signed_jvp';began=time.monotonic()
            values,derivatives=ball.forward(state,directions)
            elapsed=time.monotonic()-began
            # Full finite primal/32 columns retained before any old/new comparison.
            step=capture_step(values,derivatives,dimension=DIMENSION,count=DIRECTIONS)
            step.update(seconds=elapsed,phase_index=0,actual_balls_sha256=domain['actual_balls_sha256'],
                actual_inputs_file_sha256=sha(out/'actual-inputs.json'),direction_inputs_file_sha256=sha(out/'direction-inputs.json'))
            write_json(out/'boxed-step.json',step);event('boxed_step_saved',seconds=elapsed,sha256=sha(out/'boxed-step.json'))
            phase='exact_point_comparison'
            comparison=compare_saved_steps(step,oldstep,dimension=DIMENSION,count=DIRECTIONS,compared=8)
            comparison.update(comparison_receipt_sha256=PINS['comparison_receipt_sha256'],comparison_step_sha256=PINS['comparison_step_sha256'])
            write_json(out/'comparison.json',comparison)
            if not comparison['comparison_passed']:raise ValueError('one or more exact point/box compatibility rows failed; all output retained')
            write_json(out/'status.json',dict(status='boxed_derivative_setup_completed',certificate_passed=False,
                columns_completed=DIRECTIONS,omitted_columns=DIMENSION-DIRECTIONS,comparison_rows_checked=8,
                full_state_dimension=DIMENSION,full_vocabulary=VOCABULARY,phase_index=0,precision=1024,
                arb_seconds=elapsed,elapsed_seconds=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                reason='Whole-box directional engineering measurement only; 864 columns, other phases, root inclusion, primitive period, token labels and attraction absent. Radius 2^-800 is not proposed as an isolating orbit box.'))
            event('completed')
    except Exception as error:
        write_json(out/'status.json',dict(status='boxed_derivative_setup_failed',certificate_passed=False,
            phase=phase,error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        raise


if __name__=='__main__':main()
