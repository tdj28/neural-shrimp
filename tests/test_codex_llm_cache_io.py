"""Synthetic provenance and exact-enclosure checks; no target artifacts used."""
import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import flint
import numpy as np
from flint import arb,ctx
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'codex/certify-llm'))
from cache_io import load_prefix,unpack,digest
from pilot import packed
from ball_qwen import Dual,exact


def bundle(tmp_path):
    source=tmp_path/'source/codex/certify-llm/ball_qwen.py';source.parent.mkdir(parents=True);source.write_text('# synthetic evaluator identity')
    run=tmp_path/'run';run.mkdir()
    inputs=dict(source_commit='a'*40,pinned_assets={'weights':'b'*64},prefix_ids=[2,3],alpha_ratio=[1,8],
                log10_temperature_ratio=[-1,4],precision=ctx.prec,versions=dict(python_flint=flint.__version__,flint=flint.__FLINT_VERSION__),
                temperature='exp(log(10)*exact_dyadic_log10T)',rotary='real base^(-2j/head_dim), real sin/cos')
    (run/'inputs.json').write_text(json.dumps(inputs))
    value=arb(1,(1,-80));data=[[[[[packed(value),packed(value)]] for _ in range(2)] for _ in range(2)]]
    with gzip.open(run/'prefix-cache.json.gz','wt') as f:json.dump(data,f)
    receipt=dict(source_commit='a'*40,artifacts=[dict(path=str(p.relative_to(tmp_path)),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(tmp_path.rglob('*')) if p.is_file()])
    (tmp_path/'receipt.json').write_text(json.dumps(receipt))
    config={'synthetic_config':1};weights={'synthetic':np.array([1.])};vector=np.array([.25,.5])
    model=SimpleNamespace(config=dict(config),weights=weights,vector=vector,steering_layer=12,prefix_ids=(2,3),alpha=exact(.125),log10_temperature=exact(-.25),precision=ctx.prec,
                          layers=1,kv_heads=1,hdim=2,scalar=Dual.make,cache=None)
    kwargs=dict(expected_receipt_sha256=digest(tmp_path/'receipt.json'),expected_map_source_sha256=digest(source),expected_assets={'weights':'b'*64},
                expected_config=config,expected_weights=weights,expected_vector=vector,expected_steering_layer=12)
    return model,kwargs,value


def test_prefix_keeps_radius_and_zero_tangents(tmp_path):
    with ctx.workprec(160):
        model,kw,v=bundle(tmp_path);load_prefix(tmp_path,model,**kw)
        q=model.cache[0][0][0][0][0]
        assert q.value.contains(v) and v.contains(q.value)
        assert q.value.rad()>0 and q.tangent==0


@pytest.mark.parametrize('failure',['receipt','source','asset','parameter','prefix','precision','bytes',
     'config','weights_equal_copy','vector_equal_copy','steering','expected_steering','steering_float'])
def test_corrupt_or_incompatible_prefix_fails(tmp_path,failure):
    with ctx.workprec(160):
        model,kw,_=bundle(tmp_path)
        if failure=='receipt':kw['expected_receipt_sha256']='0'*64
        elif failure=='source':kw['expected_map_source_sha256']='0'*64
        elif failure=='asset':kw['expected_assets']={}
        elif failure=='parameter':model.alpha=exact(.25)
        elif failure=='prefix':model.prefix_ids=(3,2)
        elif failure=='precision':model.precision+=1
        elif failure=='bytes':(tmp_path/'run/prefix-cache.json.gz').write_bytes(b'corrupt')
        elif failure=='config':model.config={'synthetic_config':2}
        elif failure=='weights_equal_copy':model.weights=dict(model.weights)
        elif failure=='vector_equal_copy':model.vector=model.vector.copy()
        elif failure=='steering':model.steering_layer=11
        elif failure=='expected_steering':kw['expected_steering_layer']=11;model.steering_layer=11
        elif failure=='steering_float':model.steering_layer=12.
        with pytest.raises(ValueError):load_prefix(tmp_path,model,**kw)
        assert model.cache is None


def test_bad_ball_encoding_fails():
    for bad in [{'mid':[0,0],'rad':[-1,-8]},{'mid':[.1,0],'rad':[0,0]},{'mid':[0,0]}]:
        with pytest.raises(ValueError):unpack(bad)


def test_explicit_receiver_inputs_are_required(tmp_path):
    with ctx.workprec(160):
        model,kw,_=bundle(tmp_path)
        for field in ('expected_config','expected_weights','expected_vector','expected_steering_layer'):
            missing=dict(kw);missing.pop(field)
            with pytest.raises(TypeError):load_prefix(tmp_path,model,**missing)
            assert model.cache is None


def test_restored_prefix_uses_all_zero_jet_directions(tmp_path):
    from ball_jet import Jet
    with ctx.workprec(160):
        model,kw,v=bundle(tmp_path)
        model.scalar=lambda value:Jet.make(value,[0]*8)
        binding=load_prefix(tmp_path,model,**kw)
        q=model.cache[0][0][0][0][0]
        assert q.value.contains(v) and v.contains(q.value) and q.value.rad()>0
        assert q.tangent==(arb(0),)*8
        assert binding['receiver_construction_checked'] is True
        assert binding['hashes_to_arrays_authenticated_by_this_loader'] is False
        assert binding['steering_layer']==12
