"""Small analytic and independent-map controls; no target/model assets loaded."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

BASE=Path(__file__).resolve().parents[1]
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
analysis=load(BASE/'codex/loop-gain-crosscheck/analysis.py','loop_gain_analysis')
run=load(BASE/'codex/loop-gain-crosscheck/run.py','loop_gain_run')
qpath=BASE/'codex/crosscheck/qwen_map.py'
if not qpath.exists():qpath=BASE/'qwen_map_original.py'
qwen=load(qpath,'tiny_qwen_original')
torch.set_num_threads(1)


def toy():
    rng=np.random.default_rng(810)
    def t(shape):return torch.tensor(rng.normal(0,.12,shape),dtype=torch.float64)
    config=dict(hidden_size=4,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,
        intermediate_size=7,vocab_size=9,rms_norm_eps=1e-6,rope_theta=10000.,
        tie_word_embeddings=True,hidden_act='silu',attention_dropout=0.)
    w={'model.embed_tokens.weight':t((9,4)),'model.norm.weight':torch.ones(4,dtype=torch.float64)}
    for i in range(2):
        p=f'model.layers.{i}.'
        for n in ['input_layernorm.weight','post_attention_layernorm.weight']:w[p+n]=torch.ones(4,dtype=torch.float64)
        for n,k in [('q',4),('k',2),('v',2)]:
            w[p+f'self_attn.{n}_proj.weight']=t((k,4));w[p+f'self_attn.{n}_proj.bias']=t((k,))
        w[p+'self_attn.o_proj.weight']=t((4,4))
        for n in ['gate','up']:w[p+f'mlp.{n}_proj.weight']=t((7,4))
        w[p+'mlp.down_proj.weight']=t((4,7))
    return qwen.ManualQwenSoftMap(config,w,[1,2,3],t((4,)),steering_layer=1)


def test_centered_covariance_matches_exact_two_point_variance():
    e=torch.tensor([[1.,2.],[3.,-2.]],dtype=torch.float64)
    p=torch.tensor([.25,.75],dtype=torch.float64)
    cov,mean=analysis.covariance(e,p,chunk=1)
    delta=e[0]-e[1]
    torch.testing.assert_close(cov,.25*.75*torch.outer(delta,delta),rtol=0,atol=1e-14)
    torch.testing.assert_close(mean,.25*e[0]+.75*e[1])


def test_covariance_translation_invariance_and_dense_formula():
    torch.manual_seed(7);e=torch.randn(11,4,dtype=torch.float64)
    p=torch.softmax(torch.randn(11,dtype=torch.float64),0)
    cov,_=analysis.covariance(e,p,chunk=3)
    shifted,_=analysis.covariance(e+torch.tensor([2.,3.,-4.,5.]),p,chunk=5)
    exact=e.T@(torch.diag(p)-torch.outer(p,p))@e
    torch.testing.assert_close(cov,exact,rtol=1e-13,atol=1e-13)
    torch.testing.assert_close(cov,shifted,rtol=1e-13,atol=1e-13)


@pytest.mark.parametrize('p',[[.3,.4],[-.1,1.1],[float('nan'),.5]])
def test_covariance_refuses_invalid_probabilities(p):
    with pytest.raises(ValueError):analysis.covariance(torch.ones((2,3),dtype=torch.float64),torch.tensor(p,dtype=torch.float64))


def test_monodromy_order_and_signed_eigenvalues():
    a=np.array([[2.,1.],[0.,1.]])
    b=np.array([[1.,0.],[3.,-.25]])
    result=analysis.loop_spectra([a,b])
    np.testing.assert_array_equal(result['product'],b@a)
    assert not np.allclose(result['product'],a@b)
    assert any(np.real(result['monodromy_eigenvalues'])<0)


def test_cyclic_eigenvalues_same_but_singular_norm_not_same():
    a=np.array([[2.,3.],[0.,.1]])
    b=np.array([[.2,0.],[1.,.3]])
    x,y=analysis.loop_spectra([a,b]),analysis.loop_spectra([b,a])
    np.testing.assert_allclose(np.sort_complex(x['monodromy_eigenvalues']),np.sort_complex(y['monodromy_eigenvalues']))
    assert abs(x['monodromy_singular_values'][0]-y['monodromy_singular_values'][0])>.1


def test_hidden_wrapper_matches_original_map_and_prefix_is_bound():
    model=toy();x=torch.tensor([.2,-.1,.8,.5],dtype=torch.float64);cache=model.prefix_cache(.03)
    h,image=analysis.joint_map(model,x,.03,-.1,cache)
    torch.testing.assert_close(image,model.forward(x,.03,-.1,cache),rtol=1e-14,atol=1e-14)
    torch.testing.assert_close(h@model.embedding.T,model.logits(x,.03,cache),rtol=1e-14,atol=1e-14)
    with pytest.raises(ValueError):analysis.normalized_hidden(model,x,.031,cache)


def test_full_phase_factorization_matches_independent_original_ad():
    model=toy();x=torch.tensor([.2,-.1,.8,.5],dtype=torch.float64);events=[]
    summary,matrix=analysis.phase(model,x,.03,-.1,cache=model.prefix_cache(.03),batch=3,seed=18001,
        emit=lambda n,v:events.append((n,v)),check=lambda:None)
    assert matrix.shape==(4,4)
    assert summary['checks']['factorization_pass'] and summary['checks']['dense_direction_pass']
    assert len(summary['directions'])==8
    assert [x[1]['first_column'] for x in events if x[0]=='basis']==[0,3]


def test_orbit_gate_refuses_fixed_point_claimed_as_period_eight():
    class Fixed:
        prefix_ids=torch.tensor([0])
        def initial_state(self):return torch.tensor([1.,2.],dtype=torch.float64)
        def forward(self,x,*a,**kw):return x
    with pytest.raises(ValueError,match='closure/primitive'):
        analysis.orbit(Fixed(),0.,0.,transient=1,period=8,cache=None,emit=lambda *a:None,check=lambda:None)


def valid_plan():
    return dict(schema='codex-loop-gain-v1',admission='reviewed',envelope=run.CAPS.copy(),
        source_sha256={x:'a'*64 for x in run.SOURCES},scope='synthetic fixture',
        parameters=dict(alpha=.071674,log10_temperature=-.115,period=8,transient=1000,dimension=896,
            vocabulary=151936,batch_directions=16,random_directions=8,first_seed=18001))


@pytest.mark.parametrize('kind',['prepared','boolean-int','changed-seed','missing-source','changed-cap'])
def test_plan_refuses_scope_drift(kind):
    p=valid_plan()
    if kind=='prepared':p['admission']='prepared'
    elif kind=='boolean-int':p['parameters']['period']=True
    elif kind=='changed-seed':p['parameters']['first_seed']=18002
    elif kind=='missing-source':p['source_sha256'].pop(run.SOURCES[0])
    else:p['envelope']['timeout_seconds']=3601
    with pytest.raises(ValueError):run.validate(p)


def test_exact_plan_accepts():
    assert run.validate(valid_plan())['parameters']['period']==8


def test_composed_controls_check_primal_clock_and_product_orientation():
    class Linear:
        d=2
        def period_map(self,x,alpha,temperature,period,cache=None):
            m=torch.tensor([[.2, .4],[0.,-.1]],dtype=torch.float64)
            for _ in range(period):x=m@x
            return x
    model=Linear();x=torch.tensor([1.,2.],dtype=torch.float64)
    m=np.array([[.2,.4],[0.,-.1]])
    image=model.period_map(x,0.,0.,8)
    assert analysis.composed_controls(model,x,0.,0.,period=8,cache=None,matrix=np.linalg.matrix_power(m,8),
        expected_image=image,seed=18009,emit=lambda *a:None,check=lambda:None)['passed']
    with pytest.raises(ValueError,match='composed original-map'):
        analysis.composed_controls(model,x,0.,0.,period=8,cache=None,matrix=np.linalg.matrix_power(m,7),
            expected_image=image,seed=18009,emit=lambda *a:None,check=lambda:None)
    with pytest.raises(ValueError,match='composed original-map'):
        analysis.composed_controls(model,x,0.,0.,period=8,cache=None,matrix=np.linalg.matrix_power(m,8),
            expected_image=image*2,seed=18009,emit=lambda *a:None,check=lambda:None)
