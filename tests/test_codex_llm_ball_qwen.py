"""Synthetic tiny-transformer cross-checks; no real checkpoint or target run."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'codex/certify-llm'))
from ball_qwen import BallQwen,Dual,softmax,rmsnorm,exact
spec=importlib.util.spec_from_file_location('synthetic_independent_qwen',ROOT/'codex/crosscheck/qwen_map.py')
manual=importlib.util.module_from_spec(spec);sys.modules[spec.name]=manual;spec.loader.exec_module(manual)

@pytest.fixture(autouse=True)
def precision():
    with ctx.workprec(160):
        yield


def fixture():
    config=dict(hidden_size=4,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,
                intermediate_size=6,vocab_size=7,rms_norm_eps=1e-6,rope_theta=10000.,
                tie_word_embeddings=True,hidden_act='silu',attention_dropout=0.)
    rng=np.random.default_rng(15001); w={}
    def add(k,shape,mean=0):w[k]=np.round(rng.normal(mean,.08,size=shape)*256)/256
    add('model.embed_tokens.weight',(7,4));add('model.norm.weight',(4,),1)
    for i in range(2):
        n=f'model.layers.{i}.'
        add(n+'input_layernorm.weight',(4,),1);add(n+'post_attention_layernorm.weight',(4,),1)
        for p,s in [('q',4),('k',2),('v',2)]:
            add(n+f'self_attn.{p}_proj.weight',(s,4));add(n+f'self_attn.{p}_proj.bias',(s,))
        add(n+'self_attn.o_proj.weight',(4,4))
        for p in ['gate','up']:add(n+f'mlp.{p}_proj.weight',(6,4))
        add(n+'mlp.down_proj.weight',(4,6))
    vector=np.array([.25,-.125,.125,.375]);prefix=[1,3,2]
    ref=manual.ManualQwenSoftMap(config,{k:torch.tensor(v,dtype=torch.float64) for k,v in w.items()},prefix,vector,steering_layer=1)
    model=BallQwen(config,w,prefix,vector,steering_layer=1,alpha=.071674,log10_temperature=-.115)
    return model,ref


def test_exact_dyadics_and_real_temperature():
    value=.1;n,d=value.as_integer_ratio()
    assert exact(value)==arb(n)/d
    model,_=fixture()
    assert model.temperature.overlaps((arb(10).log()*exact(-.115)).exp())
    assert not model.temperature.contains(10.**-.115)


def test_full_tiny_map_and_analytic_jvp_match_independent_torch():
    model,ref=fixture();x=np.array([.08,-.11,.16,.03]);v=np.array([.1,.2,-.15,.07])
    value,tangent=model.forward(x,v)
    xt=torch.tensor(x,dtype=torch.float64);vt=torch.tensor(v,dtype=torch.float64)
    expected=ref.forward(xt,.071674,-.115).numpy();expected_v=ref.jvp(xt,vt,.071674,-.115).numpy()
    np.testing.assert_allclose([float(a.mid()) for a in value],expected,atol=3e-14,rtol=3e-12)
    np.testing.assert_allclose([float(a.mid()) for a in tangent],expected_v,atol=3e-14,rtol=3e-12)
    assert max(float(a.rad()) for a in value)<1e-30
    assert max(float(a.rad()) for a in tangent)<1e-30


def test_interval_forward_and_tangent_enclose_interior_points():
    model,_=fixture();x=np.array([.08,-.11,.16,.03]);v=np.array([.1,.2,-.15,.07])
    box=[arb(exact(a),(1,-24)) for a in x];directions=[arb(exact(a),(1,-25)) for a in v]
    f,j=model.forward(box,directions)
    rng=np.random.default_rng(15002)
    for _ in range(5):
        sample=x+rng.uniform(-1,1,4)*2**-24;sv=v+rng.uniform(-1,1,4)*2**-25
        sf,sj=model.forward(sample,sv)
        assert all(a.contains(b) for a,b in zip(f,sf))
        assert all(a.contains(b) for a,b in zip(j,sj))


def test_softmax_shift_and_zero_sum_derivative():
    q=[Dual.make(10000.,2.),Dual.make(9999.,-1.)]
    p=softmax(q)
    assert sum((v.value for v in p),arb(0)).contains(1)
    assert sum((v.tangent for v in p),arb(0)).contains(0)
    shifted=softmax([v-10000 for v in q])
    assert all(a.value.overlaps(b.value) and a.tangent.overlaps(b.tangent) for a,b in zip(p,shifted))


def test_precision_change_invalidates_prefix_cache():
    model,_=fixture()
    with ctx.workprec(200):
        with pytest.raises(ValueError,match='precision'):model.forward([0.1]*4)


def test_rmsnorm_refuses_unproved_positive_denominator():
    with pytest.raises(ValueError,match='positive'):
        rmsnorm([Dual.make(0)], [1], arb(0,1))


def test_cache_rebinding_rejected_and_wide_constant_not_silently_rounded():
    model,_=fixture();model.alpha=exact(.0717)
    with pytest.raises(ValueError,match='dependencies'):model.forward([.1]*4)
    if np.dtype(np.longdouble).itemsize>8:
        with pytest.raises(TypeError,match='wider'):exact(np.longdouble('.1'))


def test_exact_ball_serialization_round_trip():
    from pilot import packed
    for value in [arb.pi(),arb(0,(1,-80)),arb(-3,(1,-110)),arb((1,-200))]:
        data=packed(value)
        recovered=arb(tuple(data['mid']),tuple(data['rad']))
        assert recovered.contains(value) and value.contains(recovered)
