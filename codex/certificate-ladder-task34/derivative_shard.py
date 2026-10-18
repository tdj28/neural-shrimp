"""One final-domain, 64-column full-vocabulary derivative shard.

The exact candidate definition is an external, hash-pinned mathematical input;
its centre-selection provenance does not authenticate any derivative. No
refinement lineage is imported. This driver preserves original LIVE receiver
boxes before the first decoder operation and issues no certificate. Receipt
collection/authentication for assembling multiple shards remains separate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT/'codex/certify-llm')]
import phase_domain as domain
import derivative_coverage as coverage
from prefix_import import install_prefix
from ball_linear import DyadicWeightCache
from streamed_head import StreamedHeadQwenJet

OWNED=Path('/data2/neural-shrimp-data/codex/certificate-ladder')
DIMENSION,VOCABULARY,COLUMNS,LAYERS=896,151936,64,24
ALPHA,LOG10_TEMPERATURE=.0650,-.14798
SCHEMA='codex-llm-final-derivative-shard-plan-v1'
DEFINITION_SCHEMA='codex-llm-candidate-final-box-v1'
SHARD_SCHEMA='codex-llm-final-derivative-shard-v1'
ENVELOPE=dict(cpu_threads=1,nice=10,memory_gib=18,timeout_seconds=3600)
SOURCE_PATHS=tuple('codex/certify-llm/'+name+'.py' for name in
    ('streamed_head','ball_jet','ball_qwen','ball_linear','cache_io',
     'phase_domain','derivative_coverage','shooting_gates','proof_gates'))+('codex/crosscheck/qwen_map.py','codex/certificate-ladder-task34/derivative_shard.py','codex/certificate-ladder-task34/prefix_import.py')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def read_json(path):return json.loads(Path(path).read_text())


def map_definition(assets,prefix_ids):
    """Current reviewed exact-real map metadata, independent of candidate data."""
    return dict(schema='codex-qwen-exact-soft-map-v1',dimension=DIMENSION,vocabulary=VOCABULARY,
        layers=LAYERS,pinned_assets=assets,prefix_ids=list(prefix_ids),steering_layer=12,
        alpha_ratio=list(ALPHA.as_integer_ratio()),log10_temperature_ratio=list(LOG10_TEMPERATURE.as_integer_ratio()),
        temperature='exp(log(10)*exact_dyadic_log10T)',rotary='real base^(-2j/head_dim), real sin/cos',
        state='one appended full-dimensional soft embedding after the fixed prefix',
        readout='full-vocabulary expected embedding with the checkpoint embedding matrix',
        checkpoint_constants='stored binary constants interpreted as exact dyadics')


def backend_definition(torch_version):
    return dict(python=sys.version,python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,
                numpy=np.__version__,torch=str(torch_version),precision=ctx.prec,threads=ctx.threads)


def source_bundle():
    return dict(schema='codex-derivative-evaluator-source-bundle-v1',
                source_sha256={name:sha(ROOT/name) for name in SOURCE_PATHS})


def validate_plan(plan,*,owned=None):
    owned=OWNED if owned is None else Path(owned).resolve()
    fields={'schema','definition_path','definition_sha256','map_sha256','backend_sha256',
            'source_sha256','evaluator_sha256','prefix_attempt','prefix_receipt_sha256',
            'prefix_map_source_sha256','prefix_cache_sha256','phase_index','start_column',
            'precision','token_chunk','envelope','period','prefix_source_commit','prefix_output','prefix_receipt','prefix_admission_path','prefix_admission_sha256','cell'}
    if type(plan)is not dict or not fields<=set(plan) or set(plan)-fields-{'purpose','basis'} or plan['schema']!=SCHEMA:
        raise ValueError('complete final derivative-shard plan required')
    for name in ('definition_sha256','map_sha256','backend_sha256','evaluator_sha256',
                 'prefix_receipt_sha256','prefix_map_source_sha256','prefix_cache_sha256','prefix_admission_sha256'):
        coverage._hash(plan[name],name)
    if type(plan['period']) is not int or plan['period']!=3 or plan['cell']!='cascade-p3':raise ValueError('exact cascade-p3 allocation required')
    for key,value in [('precision',1024),('token_chunk',1024)]:
        if type(plan[key])is not int or plan[key]!=value:raise ValueError('fixed 1024-bit/token-chunk scope required')
    if type(plan['phase_index'])is not int or not 0<=plan['phase_index']<plan['period']:
        raise ValueError('one assigned global phase index required')
    start=plan['start_column']
    if type(start)is not int or start<0 or start+COLUMNS>DIMENSION or start%COLUMNS:
        raise ValueError('one aligned contiguous 64-column block required')
    if (type(plan['envelope'])is not dict or plan['envelope']!=ENVELOPE
            or any(type(v)is not int for v in plan['envelope'].values())):
        raise ValueError('CPU1/nice10/18GiB/3600-second envelope required')
    pins=plan['source_sha256']
    if type(pins)is not dict or set(pins)!=set(SOURCE_PATHS):raise ValueError('complete evaluator source bundle required')
    for name,value in pins.items():coverage._hash(value,name)
    if type(plan['definition_path'])is not str or type(plan['prefix_attempt'])is not str:
        raise ValueError('owned definition and prefix paths required')
    definition=Path(plan['definition_path']).resolve();prefix=Path(plan['prefix_attempt']).resolve()
    if not definition.is_relative_to(owned) or not definition.is_file():raise ValueError('existing owned candidate definition required')
    if prefix.parent!=owned or not prefix.is_dir():raise ValueError('existing owned prefix attempt required')
    if Path(plan['prefix_output']).resolve()!=prefix:raise ValueError('prefix output/attempt mismatch')
    if len(plan['prefix_source_commit'])!=40 or any(c not in '0123456789abcdef' for c in plan['prefix_source_commit']):raise ValueError('exact prefix source required')
    return definition,prefix


def admit_definition(plan,assets,prefix_ids,torch_version):
    """Authenticate independent input/trust anchors without constructing a model."""
    path,_=validate_plan(plan)
    if ctx.prec!=1024 or ctx.threads!=1:raise ValueError('planned 1024-bit CPU1 context required')
    bundle=source_bundle()
    if bundle['source_sha256']!=plan['source_sha256'] or domain.canonical_hash(bundle)!=plan['evaluator_sha256']:
        raise ValueError('current evaluator source bundle differs from frozen plan')
    backend=backend_definition(torch_version);backend_hash=domain.canonical_hash(backend)
    if backend_hash!=plan['backend_sha256']:raise ValueError('current arithmetic backend differs from plan')
    current_map=map_definition(assets,prefix_ids);map_hash=domain.canonical_hash(current_map)
    if map_hash!=plan['map_sha256']:raise ValueError('current exact map differs from plan')
    raw_definition=path.read_bytes()
    if hashlib.sha256(raw_definition).hexdigest()!=plan['definition_sha256']:
        raise ValueError('candidate definition file hash mismatch')
    definition=json.loads(raw_definition)
    fields={'schema','role','period','dimension','centers','radii','map','map_sha256','selection_provenance'}
    if (type(definition)is not dict or set(definition)!=fields or definition['schema']!=DEFINITION_SCHEMA
            or definition['role']!=coverage.ROLE):raise ValueError('prototype/non-final candidate definition refused')
    for key,value in [('period',plan['period']),('dimension',DIMENSION)]:
        if type(definition[key])is not int or definition[key]!=value:raise ValueError('complete global candidate dimensions required')
    if definition['map']!=current_map or definition['map_sha256']!=map_hash:
        raise ValueError('candidate exact parameters/assets/prefix/map mismatch')
    if type(definition['selection_provenance'])is not dict or not definition['selection_provenance']:
        raise ValueError('recorded centre-selection provenance required; it is not derivative evidence')
    _,_,mixed_hash,shooting_hash=coverage._nominal(definition['centers'],definition['radii'],plan['period'],DIMENSION)
    return definition,dict(design_sha256=plan['definition_sha256'],map_sha256=map_hash,
        evaluator_sha256=domain.canonical_hash(bundle),backend_sha256=backend_hash,
        phase_domain_sha256=mixed_hash,shooting_domain_sha256=shooting_hash,
        source_bundle=bundle,backend=backend,map=current_map),raw_definition


def construct_inputs(definition,phase,start):
    artifact=domain.build_phase_domains(definition['centers'],definition['radii'],
        period=definition['period'],dimension=DIMENSION,target_precision=1024)
    expected=domain.canonical_hash({k:v for k,v in artifact.items() if k!='artifact_sha256'})
    boxes,_=domain.decode_verified_domains(artifact,definition['centers'],definition['radii'],
        period=definition['period'],dimension=DIMENSION,target_precision=1024,expected_artifact_sha256=expected)
    directions=arb_mat(DIMENSION,COLUMNS)
    for k in range(COLUMNS):directions[start+k,k]=1
    records=[[domain.pack_ball(directions[i,k]) for k in range(COLUMNS)] for i in range(DIMENSION)]
    return artifact,boxes[phase],directions,records


def receiver_record(definition,phase,values):
    """Bind actual live scalar receiver balls, not an earlier encoded precursor."""
    checked=domain.containment_witnesses([definition['centers'][phase]],[definition['radii'][phase]],
                                       [values],period=1,dimension=DIMENSION)
    actual=checked['actual_balls'][0]
    return dict(phase_index=phase,actual_inputs=actual,actual_inputs_sha256=domain.canonical_hash(actual),
        selected_phase_endpoint_witnesses=checked['witnesses'][0],full_nominal_phase_box_contained=True)


def capture_outputs(values,derivatives):
    if not isinstance(values,list) or len(values)!=DIMENSION:
        raise ValueError('complete full-state primal required')
    if not isinstance(derivatives,arb_mat) or (derivatives.nrows(),derivatives.ncols())!=(DIMENSION,COLUMNS):
        raise ValueError('complete full-state 64-column derivative matrix required')
    # pack_ball rejects any nonfinite value; no omitted failed column or phase.
    return [domain.pack_ball(v) for v in values],[[domain.pack_ball(derivatives[i,k]) for k in range(COLUMNS)] for i in range(DIMENSION)]


def evaluate_shard(ball,definition,binding,plan,state,directions,records,out,event):
    """Record the first live decoder inputs before computing the first block."""
    phase=plan['phase_index'];actual=None;started_layers=[];ended_layers=[];head_rows=[]
    def progress(kind,index,hidden):
        nonlocal actual
        if kind=='step_start':
            if index!=len(started_layers) or len(ended_layers)!=len(started_layers):
                raise ValueError('decoder start order mismatch')
            if index==0:
                if actual is not None or len(hidden)!=1 or len(hidden[0])!=DIMENSION:
                    raise ValueError('one complete initial live receiver state required')
                live=hidden[0]
                actual=receiver_record(definition,phase,[v.value for v in live])
                live_directions=[[domain.pack_ball(v) for v in q.tangent] for q in live]
                if live_directions!=records:raise ValueError('live receiver directions differ from exact signed basis')
                write_json(out/'actual-inputs.json',actual)
                event('live_input_saved',sha256=sha(out/'actual-inputs.json'))
            started_layers.append(index)
        elif kind=='step_end':
            if index!=len(ended_layers) or len(started_layers)!=len(ended_layers)+1:
                raise ValueError('decoder end order mismatch')
            ended_layers.append(index)
        event(kind,layer=index)
    expected_head=[(kind,first,min(first+plan['token_chunk'],VOCABULARY))
        for kind in ('head_logits','head_derivatives') for first in range(0,VOCABULARY,plan['token_chunk'])]
    def head_progress(kind,first,last):
        row=(kind,first,last)
        if (type(first)is not int or type(last)is not int or len(head_rows)>=len(expected_head)
                or row!=expected_head[len(head_rows)] or ended_layers!=list(range(LAYERS))):
            raise ValueError('complete ordered full-vocabulary head passes required')
        head_rows.append(row);event(kind,row_start=first,row_stop=last)
    ball.progress=progress;ball.head_progress=head_progress
    began=time.monotonic();values,derivatives=ball.forward(state,directions);seconds=time.monotonic()-began
    if actual is None or started_layers!=list(range(LAYERS)) or ended_layers!=list(range(LAYERS)):
        raise ValueError('all decoder stages and original live inputs required')
    primal,columns=capture_outputs(values,derivatives)
    shard=dict(schema=SHARD_SCHEMA,role=coverage.ROLE,certificate_passed=False,
        **{key:binding[key] for key in ('design_sha256','map_sha256','evaluator_sha256','backend_sha256',
                                       'phase_domain_sha256','shooting_domain_sha256')},
        precision=1024,phase_index=phase,direction_indices=list(range(plan['start_column'],plan['start_column']+COLUMNS)),
        actual_inputs=actual['actual_inputs'],actual_inputs_sha256=actual['actual_inputs_sha256'],
        directions=records,derivatives=columns,primal=primal,
        actual_inputs_file_sha256=sha(out/'actual-inputs.json'),
        direction_inputs_file_sha256=sha(out/'direction-inputs.json'),
        forward_seconds=seconds,full_vocabulary=VOCABULARY,token_chunk=1024,
        statement='One complete signed derivative shard over the declared nominal phase box; external receipt authentication and all proof gates remain separate.')
    # Retain the full numerical enclosures even if a subsequent bookkeeping gate
    # fails. A reader must authenticate the final successful status/receipt too.
    write_json(out/'derivative-shard.json',shard)
    if head_rows!=expected_head:raise ValueError('both complete full-vocabulary head passes required')
    diagnostics=ball.last_head_diagnostics
    if (type(diagnostics)is not dict or type(diagnostics.get('vocabulary_count'))is not int
            or diagnostics['vocabulary_count']!=VOCABULARY or type(diagnostics.get('competitor_count'))is not int
            or diagnostics['competitor_count']!=VOCABULARY-1
            or type(diagnostics.get('candidate_index'))is not int
            or not 0<=diagnostics['candidate_index']<VOCABULARY
            or type(diagnostics.get('strict_margin_established'))is not bool
            or diagnostics.get('certificate_passed')is not False):
        raise ValueError('full-vocabulary label diagnostic required; no orbit claim')
    # Readout gaps remain separate from existence, primitivity and attraction.
    for name in ('candidate_lower','largest_competitor_upper','strict_margin'):
        domain._record_parts(diagnostics[name])
    lower,lower_radius=domain._record_parts(diagnostics['candidate_lower'])
    upper,upper_radius=domain._record_parts(diagnostics['largest_competitor_upper'])
    margin,radius=domain._record_parts(diagnostics['strict_margin'])
    if lower_radius or upper_radius or not margin-radius<=lower-upper<=margin+radius:
        raise ValueError('readout margin must enclose the original exact extremum difference')
    if diagnostics['strict_margin_established']!=(margin-radius>0):
        raise ValueError('readout strict-gap flag disagrees with original exact margin')
    write_json(out/'head-diagnostics.json',dict(design_sha256=binding['design_sha256'],
        map_sha256=binding['map_sha256'],phase_index=phase,
        actual_inputs_sha256=actual['actual_inputs_sha256'],head_passes=head_rows,
        diagnostic=diagnostics,certificate_passed=False,
        scope='Full-vocabulary label witness on this actual receiver box only; no orbit proof.'))
    shard['head_diagnostics_file_sha256']=sha(out/'head-diagnostics.json')
    write_json(out/'derivative-shard.json',shard)
    return shard


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out.name!='run' or out.parent.parent!=OWNED.resolve():raise ValueError('fresh owned attempt/run required')
    if args.plan.resolve().parent!=HERE or args.plan.suffix!='.json':raise ValueError('committed source-snapshot plan required')
    plan=read_json(args.plan);definition_path,prefix=validate_plan(plan)
    if out.is_relative_to(prefix) or definition_path.is_relative_to(out.parent):raise ValueError('upstream inputs are immutable')
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic();phase='admission'
    def event(name,**fields):
        row=dict(event=name,phase=phase,elapsed_seconds=time.monotonic()-started,utc=datetime.now(timezone.utc).isoformat(),**fields)
        with (out/'events.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
    try:
        source=os.environ.get('CERT_SOURCE_COMMIT','')
        if len(source)!=40 or any(c not in '0123456789abcdef' for c in source):
            raise ValueError('bounded-launcher source commit required')
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
        os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
        if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
        import torch
        from threadpoolctl import threadpool_limits
        sys.path.insert(0,str(ROOT/'codex/crosscheck'))
        from qwen_map import ManualQwenSoftMap,PINNED_SHA256,PREFIX_IDS
        torch.set_num_threads(1);torch.set_num_interop_threads(1);ctx.prec=1024;ctx.threads=1
        with threadpool_limits(limits=1):
            definition,binding,raw_definition=admit_definition(plan,PINNED_SHA256,PREFIX_IDS,torch.__version__)
            (out/'candidate-definition.json').write_bytes(raw_definition)
            write_json(out/'inputs.json',dict(plan=plan,plan_sha256=sha(args.plan),source_commit=source,
                driver_sha256=sha(__file__),binding=binding,precision=1024,threads=1,pid=os.getpid(),certificate_passed=False))
            phase='domain_construction'
            artifact,state,directions,records=construct_inputs(definition,plan['phase_index'],plan['start_column'])
            write_json(out/'domain-construction.json',artifact)
            write_json(out/'direction-inputs.json',dict(direction_indices=list(range(plan['start_column'],plan['start_column']+COLUMNS)),
                directions=records,directions_sha256=domain.canonical_hash(records)))
            phase='model_load';model=ManualQwenSoftMap.from_pinned_files()
            if model.d!=DIMENSION or model.layers!=LAYERS or model.config['vocab_size']!=VOCABULARY or model.steering_layer!=12:
                raise ValueError('full assigned model/state/vocabulary/steering required')
            weights={name:value.numpy() for name,value in model.weights.items()}
            for weight in weights.values():weight.flags.writeable=False
            vector=model.vector.numpy();vector.flags.writeable=False
            ball=StreamedHeadQwenJet(model.config,weights,PREFIX_IDS,vector,direction_count=COLUMNS,token_chunk=1024,
                head_progress=lambda kind,first,last:event(kind,row_start=first,row_stop=last),
                alpha=ALPHA,log10_temperature=LOG10_TEMPERATURE,
                linear_options=dict(chunk_rows=64,chunk_cols=896,cache=DyadicWeightCache(max_elements=262144)))
            phase='prefix_restore'
            from cache_io import unpack
            install_prefix(plan,ball,PINNED_SHA256,sha,unpack)
            if sha(Path(plan['prefix_output'])/'prefix-cache.json.gz')!=plan['prefix_cache_sha256']:
                raise ValueError('prefix cache bytes differ from frozen plan')
            prefix_binding=dict(original_receipt_sha256=plan['prefix_receipt_sha256'],source_commit=plan['prefix_source_commit'],stored_precision=2048,receiver_precision=1024,radii_preserved=True)
            write_json(out/'prefix-binding.json',prefix_binding)
            phase='whole_phase_box_derivatives'
            shard=evaluate_shard(ball,definition,binding,plan,state,directions,records,out,event)
            write_json(out/'status.json',dict(status='final_derivative_shard_completed',certificate_passed=False,
                phase_index=plan['phase_index'],direction_indices=shard['direction_indices'],columns_completed=COLUMNS,
                full_state_dimension=DIMENSION,full_vocabulary=VOCABULARY,precision=1024,
                shard_sha256=sha(out/'derivative-shard.json'),actual_inputs_sha256=shard['actual_inputs_sha256'],
                forward_seconds=shard['forward_seconds'],elapsed_seconds=time.monotonic()-started,
                max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                reason='One full-state derivative shard; complete multi-phase coverage, external receipt authentication and all orbit gates remain required.'))
            event('completed')
    except Exception as error:
        write_json(out/'status.json',dict(status='final_derivative_shard_failed',certificate_passed=False,phase=phase,
            error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        raise


if __name__=='__main__':main()
