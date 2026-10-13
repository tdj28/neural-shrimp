"""One independently persisted exact-real phase image for a bounded controller.

This worker has no optimizer, candidate selection or process-spawning logic.
A reviewed outer controller must own its watchdog, process group, aggregate
memory cap and final artifact receipt. Only one pinned global phase is evaluated.
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
from flint import ctx
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import center_resume as saved
base=saved.base
PERIOD,DIMENSION,LAYERS=8,896,24
OWNED=base.OWNED
SCHEMA='codex-llm-parallel-point-request-v1'
SOURCES=tuple('codex/certify-llm/'+name+'.py' for name in
    ('phase_point','center_resume','center_refine','ball_qwen','ball_linear',
     'cache_io','shooting_correction','proof_gates'))+('codex/crosscheck/qwen_map.py',)


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def map_spec(assets,prefix_ids):
    return dict(schema='codex-qwen-point-map-v1',pinned_assets=assets,prefix_ids=list(prefix_ids),
        dimension=DIMENSION,period=PERIOD,layers=LAYERS,full_vocabulary=151936,steering_layer=12,
        alpha_ratio=list(base.ALPHA.as_integer_ratio()),log10_temperature_ratio=list(base.LOG10_TEMPERATURE.as_integer_ratio()),
        temperature='exp(log(10)*exact_dyadic_log10T)',rotary='real base^(-2j/head_dim), real sin/cos',
        weights_and_constants='exact stored binary rationals',
        state='one appended embedding after the fixed alpha-steered prefix; full-vocabulary expectation')


def backend(torch_version):
    return dict(python=sys.version,python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,
        numpy=np.__version__,torch=str(torch_version),precision=ctx.prec,threads=ctx.threads)


def admit(request,assets,prefix_ids,torch_version,source_commit):
    keys={'schema','source_commit','source_sha256','map','map_sha256','backend','backend_sha256',
          'centers','centers_sha256','prefix_attempt','prefix_receipt_sha256',
          'prefix_map_source_sha256','prefix_cache_sha256','precision','selection_provenance'}
    if type(request)is not dict or set(request)!=keys or request['schema']!=SCHEMA:
        raise ValueError('complete exact phase request required')
    if not base.valid_hash(source_commit,40) or request['source_commit']!=source_commit:
        raise ValueError('bounded source snapshot identity required')
    pins={path:base.digest(ROOT/path) for path in SOURCES}
    if request['source_sha256']!=pins:raise ValueError('complete current worker source bundle differs')
    mapping=map_spec(assets,prefix_ids);current_backend=backend(torch_version)
    if (request['map']!=mapping or request['map_sha256']!=canonical(mapping)
            or canonical(request['map'])!=request['map_sha256']):
        raise ValueError('independent exact map identity differs')
    if (request['backend']!=current_backend or request['backend_sha256']!=canonical(current_backend)
            or canonical(request['backend'])!=request['backend_sha256']):
        raise ValueError('arithmetic backend differs')
    if type(request['precision'])is not int or request['precision']!=2048 or ctx.prec!=2048 or ctx.threads!=1:
        raise ValueError('2048-bit CPU1 phase evaluation required')
    if request['centers_sha256']!=canonical(request['centers']):
        raise ValueError('complete centre tuple hash mismatch')
    centers=saved.decoded_centers(request['centers'],PERIOD,DIMENSION)
    if saved.center_records(centers)!=request['centers']:
        raise ValueError('canonical exact global centre data required')
    if type(request['selection_provenance'])is not dict or not request['selection_provenance']:
        raise ValueError('candidate-selection provenance required; it is not proof')
    for key in ('prefix_receipt_sha256','prefix_map_source_sha256','prefix_cache_sha256'):
        if not base.valid_hash(request[key]):raise ValueError('externally pinned prefix chain required')
    if type(request['prefix_attempt'])is not str:raise ValueError('owned prefix path required')
    prefix=Path(request['prefix_attempt']).resolve()
    if prefix.parent!=OWNED.resolve() or not prefix.is_dir():raise ValueError('existing owned prefix attempt required')
    return centers,prefix


def evaluate_phase(ball,centers,index,out,event):
    if type(index)is not int or not 0<=index<PERIOD:raise ValueError('one global phase index required')
    if len(centers)!=PERIOD or any(len(row)!=DIMENSION for row in centers):
        raise ValueError('complete global centre tuple required')
    center=centers[index];next_center=centers[(index+1)%PERIOD]
    if any(not v.is_finite() or not v.is_exact() for row in centers for v in row):
        raise ValueError('all nominal centres must be exact finite points')
    base.write_json(out/'input.json',dict(phase_index=index,center=[base.packed(v) for v in center],
        next_center=[base.packed(v) for v in next_center]))
    starts=[];ends=[];live_seen=False
    def progress(kind,layer,hidden):
        nonlocal live_seen
        if kind=='step_start':
            if layer!=len(starts) or len(starts)!=len(ends):raise ValueError('phase decoder start order mismatch')
            if layer==0:
                if len(hidden)!=1 or len(hidden[0])!=DIMENSION:raise ValueError('complete first live point required')
                live=[q.value for q in hidden[0]];tangents=[q.tangent for q in hidden[0]]
                if any(not a.is_exact() or a!=b for a,b in zip(live,center)):
                    raise ValueError('live receiver is not the exact requested centre')
                if any(not v.is_exact() or v!=0 for v in tangents):raise ValueError('exact zero live point tangents required')
                base.write_json(out/'live-input.json',dict(phase_index=index,
                    center=[base.packed(v) for v in live],tangent=[base.packed(v) for v in tangents]))
                live_seen=True
            starts.append(layer)
        elif kind=='step_end':
            if layer!=len(ends) or len(starts)!=len(ends)+1:raise ValueError('phase decoder end order mismatch')
            ends.append(layer)
        else:raise ValueError('unexpected uncached-prefix or decoder event')
        event(kind,layer=layer)
    ball.progress=progress;started=time.monotonic();value,tangent=ball.forward(center)
    if not live_seen or starts!=list(range(LAYERS)) or ends!=list(range(LAYERS)):
        raise ValueError('all ordered decoder layers and live inputs required')
    if (len(value)!=DIMENSION or len(tangent)!=DIMENSION
            or any(not v.is_finite() for v in value+tangent)
            or any(not v.is_exact() or v!=0 for v in tangent)):
        raise ValueError('complete finite point image and exact zero tangents required')
    base.write_json(out/'point.json',dict(value=[base.packed(v) for v in value],
        tangent=[base.packed(v) for v in tangent]))
    return dict(phase_index=index,seconds=time.monotonic()-started,
        input_sha256=base.digest(out/'input.json'),live_input_sha256=base.digest(out/'live-input.json'),
        point_sha256=base.digest(out/'point.json'),maximum_radius=base.packed(max(v.rad() for v in value)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True);parser.add_argument('--request-sha256',required=True)
    parser.add_argument('--phase',type=int,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();request_path=args.request.resolve()
    if (len(out.parents)<4 or out.parents[3]!=OWNED.resolve()
            or out.parents[1].name!='run' or out.name!=f'phase-{args.phase:02d}'
            or not out.parent.name.startswith('candidate-')
            or request_path!=out.parent/'request.json'
            or (out.parents[2]/'receipt.json').exists()):
        raise ValueError('one live owned candidate request and exclusive phase output required')
    if type(args.phase)is not int or not 0<=args.phase<PERIOD:raise ValueError('global phase index required')
    if not base.valid_hash(args.request_sha256):raise ValueError('external request SHA256 required')
    raw=request_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=args.request_sha256:
        raise ValueError('externally pinned original request bytes required')
    request=json.loads(raw)
    if type(request)is not dict or type(request.get('prefix_attempt'))is not str:
        raise ValueError('complete request with immutable prefix path required')
    if out.is_relative_to(Path(request['prefix_attempt']).resolve()):
        raise ValueError('prefix evidence is immutable')
    out.mkdir(parents=True,exist_ok=False)
    (out/'request.json').write_bytes(raw);started=time.monotonic();stage='admission'
    def event(name,**fields):
        row=dict(event=name,stage=stage,phase_index=args.phase,elapsed_seconds=time.monotonic()-started,
            utc=datetime.now(timezone.utc).isoformat(),**fields)
        with (out/'events.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
    try:
        # The outer controller owns stronger aggregate/watchdog controls. This
        # local cap prevents an accidental unconstrained direct worker launch.
        cap=18*1024**3;soft,hard=resource.getrlimit(resource.RLIMIT_AS)
        cap=min([cap]+[bound for bound in (soft,hard) if bound>=0])
        resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
        if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
        os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
        import torch
        from threadpoolctl import threadpool_limits
        sys.path.insert(0,str(ROOT/'codex/crosscheck'))
        from qwen_map import ManualQwenSoftMap,PINNED_SHA256,PREFIX_IDS
        torch.set_num_threads(1);torch.set_num_interop_threads(1);ctx.prec=2048;ctx.threads=1
        with threadpool_limits(limits=1):
            centers,prefix=admit(request,PINNED_SHA256,PREFIX_IDS,torch.__version__,os.environ.get('CERT_SOURCE_COMMIT',''))
            if out.is_relative_to(prefix):raise ValueError('prefix evidence is immutable')
            base.write_json(out/'inputs.json',dict(request_sha256=args.request_sha256,
                source_commit=request['source_commit'],phase_index=args.phase,pid=os.getpid(),
                map_sha256=request['map_sha256'],backend_sha256=request['backend_sha256'],
                source_sha256=request['source_sha256'],certificate_passed=False))
            stage='model_load';model=ManualQwenSoftMap.from_pinned_files()
            if model.d!=DIMENSION or model.layers!=LAYERS or model.steering_layer!=12 or model.config['vocab_size']!=151936:
                raise ValueError('complete assigned architecture required')
            weights={name:value.numpy() for name,value in model.weights.items()}
            for weight in weights.values():weight.flags.writeable=False
            vector=model.vector.numpy();vector.flags.writeable=False
            ball=base.BallQwen(model.config,weights,PREFIX_IDS,vector,alpha=base.ALPHA,
                log10_temperature=base.LOG10_TEMPERATURE,
                linear_options=dict(chunk_rows=64,chunk_cols=896,cache=base.DyadicWeightCache(max_elements=262144)))
            stage='prefix_restore';binding=base.load_prefix(prefix,ball,
                expected_receipt_sha256=request['prefix_receipt_sha256'],expected_map_source_sha256=request['prefix_map_source_sha256'],
                expected_assets=PINNED_SHA256,expected_config=model.config,expected_weights=weights,
                expected_vector=vector,expected_steering_layer=model.steering_layer)
            if binding['prefix_sha256']!=request['prefix_cache_sha256']:raise ValueError('pinned prefix cache bytes differ')
            base.write_json(out/'prefix-binding.json',binding)
            stage='one_point_image';result=evaluate_phase(ball,centers,args.phase,out,event)
            if base.digest(request_path)!=args.request_sha256:raise ValueError('request changed during phase evaluation')
        base.write_json(out/'status.json',dict(status='phase_point_completed',certificate_passed=False,
            request_sha256=args.request_sha256,**result,elapsed_seconds=time.monotonic()-started,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        event('completed')
    except Exception as error:
        base.write_json(out/'status.json',dict(status='phase_point_failed',certificate_passed=False,phase_index=args.phase,
            stage=stage,error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        raise


if __name__=='__main__':main()
