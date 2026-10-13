"""Bounded target setup: recover a numerical P8 proposal; measure Arb enclosures.

This pilot cannot issue a certificate. It records exact-real one-step primal
and derivative-box diagnostics, including failures, to design the later proof.
"""
import argparse
from datetime import datetime,timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback
import flint
import numpy as np
import torch
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(ROOT/'codex/crosscheck'))
from qwen_map import ManualQwenSoftMap,PINNED_SHA256,PREFIX_IDS,SNAPSHOT,sha256
from ball_qwen import BallQwen,exact
from ball_linear import DyadicWeightCache
SEED=Path('/data2/neural-shrimp-data/codex/task5-fresh-prediction/qualification-B-02/run/setup.npz')
SEED_SHA='d3bccdc48595c2d41206db8147b42c3cf668a9c977ac399063b036b1794da48e'
OWNED=Path('/data2/neural-shrimp-data/codex/llm-p8-certificate')


def packed(value):
    if not value.is_finite():
        raise ValueError('cannot serialize nonfinite proof enclosure')
    return dict(mid=[int(v) for v in value.mid().man_exp()],rad=[int(v) for v in value.rad().man_exp()])


def jsonwrite(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--precision',type=int,default=384);p.add_argument('--threads',type=int,default=1)
    p.add_argument('--box-radius-power',type=int,default=-80);a=p.parse_args()
    if a.precision<128 or a.threads!=1 or a.box_radius_power>=0:
        raise ValueError('pilot requires high precision, one CPU thread and a small positive dyadic radius')
    out=a.output.resolve()
    if not out.is_relative_to(OWNED) or out==OWNED:raise ValueError('owned exclusive output required')
    out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();phase='initialization'
    def event(name,**info):
        row=dict(event=name,phase=phase,seconds=time.monotonic()-start,utc=datetime.now(timezone.utc).isoformat(),**info)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
    def progress(kind,index,values):
        info={}
        if kind.endswith('_end'):
            balls=[q for row in values for q in row]
            info=dict(max_primal_radius=str(max(q.value.rad() for q in balls)),
                      max_primal_abs=str(max(q.value.abs_upper() for q in balls)),
                      max_tangent_abs=str(max(q.tangent.abs_upper() for q in balls)))
        event(kind,layer=index,**info)
    try:
        torch.set_num_threads(1);torch.set_num_interop_threads(1);ctx.threads=1;ctx.prec=a.precision
        if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
        source=os.environ.get('CERT_SOURCE_COMMIT','')
        if len(source)!=40 or any(v not in '0123456789abcdef' for v in source):raise ValueError('exact source commit required')
        if sha256(SEED)!=SEED_SHA:raise ValueError('seed hash mismatch')
        jsonwrite(out/'inputs.json',dict(source_commit=source,seed=str(SEED),seed_sha256=SEED_SHA,
                  versions=dict(python=sys.version,python_flint=flint.__version__,flint=getattr(flint,'__FLINT_VERSION__','unavailable'),numpy=np.__version__,torch=torch.__version__),
                  arb_context=dict(precision=ctx.prec,threads=ctx.threads),
                  pinned_assets=PINNED_SHA256,prefix_ids=PREFIX_IDS,alpha_ratio=list((.071674).as_integer_ratio()),
                  log10_temperature_ratio=list((-.115).as_integer_ratio()),precision=a.precision,
                  temperature='exp(log(10)*exact_dyadic_log10T)',rotary='real base^(-2j/head_dim), real sin/cos',
                  prefix='fully recomputed exact-real enclosure with steering',box_radius_power=a.box_radius_power,
                  statement='setup diagnostics only; no Krawczyk or attraction certificate',pid=os.getpid()))
        phase='numerical_proposal';event('load_model')
        model=ManualQwenSoftMap.from_pinned_files();x=torch.tensor(np.load(SEED,allow_pickle=False)['x_c'].copy(),dtype=torch.float64)
        cache=model.prefix_cache(.071674);history=[];passed=False
        for cycle in range(40):
            y=model.period_map(x,.071674,-.115,8,cache)
            residual=float(torch.linalg.vector_norm(y-x)/torch.linalg.vector_norm(x))
            history.append(residual);x=y
            np.savez(out/'candidate-latest.npz',state=x.numpy().copy())
            jsonwrite(out/'candidate-history.json',dict(relative_returns=history))
            event('proposal_cycle',cycle=cycle,relative_residual=residual)
            if residual<1e-12 and len(history)>=2 and history[-2]<1e-12:
                passed=True;break
        if not passed:
            jsonwrite(out/'proposal.json',dict(passed=False,reason='nonconverged',history=history,candidate_sha256=sha256(out/'candidate-latest.npz')))
            raise ValueError('assigned P8 numerical proposal did not converge')
        points=[x.numpy().copy()]
        for _ in range(8):x=model.forward(x,.071674,-.115,cache);points.append(x.numpy().copy())
        points=np.array(points);norm=np.linalg.norm(points[0]);returns={str(d):float(np.linalg.norm(points[d]-points[0])/norm) for d in [1,2,4,8]}
        np.savez(out/'proposal.npz',points=points)
        from tokenizers import Tokenizer
        tokenizer_path=SNAPSHOT/'tokenizer.json'
        tokenizer=Tokenizer.from_file(str(tokenizer_path))
        expected=[tokenizer.encode(t,add_special_tokens=False).ids for t in ['.',' This',' is',' there',' is',' a',' is',' a']]
        if any(len(v)!=1 for v in expected):raise ValueError('assigned token labels are not single vocabulary items')
        expected=[v[0] for v in expected]
        token_ids=[int(model.logits(torch.tensor(state,dtype=torch.float64),.071674,cache).argmax()) for state in points[:-1]]
        label_pass=any(token_ids==expected[j:]+expected[:j] for j in range(8))
        labels=dict(tokenizer_sha256=sha256(tokenizer_path),expected_ids=expected,actual_ids=token_ids,cyclic_label_match=label_pass)
        proposal_pass=returns['8']<1e-10 and min(returns[str(d)] for d in [1,2,4])>1e-6 and label_pass
        jsonwrite(out/'proposal.json',dict(passed=proposal_pass,history=history,returns=returns,labels=labels,arrays_sha256=sha256(out/'proposal.npz')))
        if not proposal_pass:
            raise ValueError('numerical primitive/closure or assigned-token-label gate failed')
        weights={k:t.numpy() for k,t in model.weights.items()}
        for w in weights.values():w.flags.writeable=False
        options=dict(chunk_rows=64,chunk_cols=896,cache=DyadicWeightCache(max_elements=262144))
        ball=BallQwen(model.config,weights,PREFIX_IDS,model.vector.numpy(),alpha=.071674,
                      log10_temperature=-.115,linear_options=options,progress=progress)
        phase='rigorous_prefix';event('prefix_begin');prefix=ball.prefix_cache()
        with gzip.open(out/'prefix-cache.json.gz','wt') as f:
            json.dump([[[[[packed(q.value) for q in head] for head in token] for token in field] for field in layer] for layer in prefix],f)
        event('prefix_saved',sha256=sha256(out/'prefix-cache.json.gz'))
        phase='rigorous_point';f,df=ball.forward(points[0]);
        jsonwrite(out/'point-step.json',dict(value=[packed(v) for v in f],tangent=[packed(v) for v in df],
                  defect_from_next_proposal=[packed(v-exact(y)) for v,y in zip(f,points[1])]))
        event('point_step_done',max_radius=str(max(v.rad() for v in f)),
              max_defect=str(max((v-exact(y)).abs_upper() for v,y in zip(f,points[1]))))
        phase='rigorous_box_jvp';radius=arb((1,a.box_radius_power))
        box=[arb(exact(v),(1,a.box_radius_power)) for v in points[0]]
        tangent=[arb(0,1) for _ in points[0]]
        jsonwrite(out/'box-inputs.json',dict(state=[packed(v) for v in box],tangent=[packed(v) for v in tangent]))
        f,df=ball.forward(box,tangent)
        norm=max(v.abs_upper() for v in df)
        jsonwrite(out/'box-step.json',dict(value=[packed(v) for v in f],derivative_action=[packed(v) for v in df],
                  input_radius=packed(radius),induced_infinity_upper=packed(norm)))
        event('box_step_done',norm_upper=str(norm),max_radius=str(max(v.rad() for v in f)))
        jsonwrite(out/'status.json',dict(status='setup_completed',certificate_passed=False,
                  reason='one-step diagnostics only; no root, primitive or monodromy proof',elapsed=time.monotonic()-start))
    except Exception as error:
        jsonwrite(out/'status.json',dict(status='setup_failed',certificate_passed=False,phase=phase,
                  error=repr(error),traceback=traceback.format_exc(),elapsed=time.monotonic()-start))
        raise

if __name__=='__main__':main()
