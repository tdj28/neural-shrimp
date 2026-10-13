"""Tiny all-token head and inherited-transformer tests; no checkpoint calls."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from flint import arb, arb_mat, ctx, fmpq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'codex/certify-llm'))
import streamed_head as stream
from ball_jet import BallQwenJet, Jet, linear_sequence, softmax
from ball_linear import DyadicWeightCache
from ball_qwen import exact

spec = importlib.util.spec_from_file_location('stream_tiny_fixture', ROOT/'tests/test_codex_llm_ball_qwen.py')
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


@pytest.fixture(autouse=True)
def arithmetic():
    threads = ctx.threads
    ctx.threads = 1
    with ctx.workprec(192): yield
    ctx.threads = threads


def old_head(embedding, hidden, temperature):
    logits = linear_sequence(embedding, [hidden])[0]
    probabilities = softmax([v/temperature for v in logits])
    output = linear_sequence(embedding.T, [probabilities])[0]
    return [v.value for v in output], arb_mat([[v for v in q.tangent] for q in output])


def overlap(left, right):
    lf, lj = left; rf, rj = right
    assert len(lf) == len(rf)
    assert (lj.nrows(), lj.ncols()) == (rj.nrows(), rj.ncols())
    assert all(a.overlaps(b) for a,b in zip(lf,rf))
    assert all(lj[i,k].overlaps(rj[i,k]) for i in range(lj.nrows()) for k in range(lj.ncols()))


@pytest.mark.parametrize('chunk',[1,2,3,16])
def test_two_token_exact_analytic_tanh_derivative(chunk):
    embedding = np.array([[-1.],[1.]])
    h, temperature = arb(3)/8, arb(5)/4
    result, jac = stream.streamed_head(embedding,[Jet(h,(arb(1),arb(-2),arb(0)))],temperature,token_chunk=chunk)
    exp2 = (2*h/temperature).exp()
    expected = (exp2-1)/(exp2+1)
    derivative = 4*exp2/((exp2+1)*(exp2+1)*temperature)
    assert result[0].overlaps(expected)
    assert jac[0,0].overlaps(derivative) and jac[0,1].overlaps(-2*derivative)
    assert jac[0,2].is_exact() and jac[0,2] == 0


@pytest.mark.parametrize('chunk',[1,2,4,32])
def test_all_token_existing_jet_comparison(chunk):
    embedding = np.array([[.25,-1.],[1.,.5],[-.5,.75],[2.,-2.],[-.75,1.5]])
    hidden = [Jet.make(.125,[1.,-.5,0.]),Jet.make(-.25,[-.25,1.,0.])]
    temp = (arb(10).log()*exact(-.115)).exp()
    overlap(stream.streamed_head(embedding,hidden,temp,token_chunk=chunk),old_head(embedding,hidden,temp))


def test_last_token_contributes_to_primal_and_jacobian():
    embedding = np.array([[0.],[0.],[2.]])
    value,jac = stream.streamed_head(embedding,[Jet.make(0.,[1.])],arb(1),token_chunk=2)
    assert value[0].contains(arb(fmpq(2,3)))
    assert jac[0,0].contains(arb(fmpq(8,9)))
    dropped,dropped_j = stream.streamed_head(embedding[:2],[Jet.make(0.,[1.])],arb(1),token_chunk=2)
    assert dropped[0] == 0 and dropped_j[0,0] == 0
    assert not value[0].contains(0) and not jac[0,0].contains(0)


def test_huge_common_shift_and_mixed_signed_cancellation():
    embedding = np.array([[10000.,1.],[10000.,-1.]])
    hidden = [Jet.make(1.,[1.,0.,0.]),Jet.make(0.,[0.,1.,-1.])]
    value,jac = stream.streamed_head(embedding,hidden,arb(1),token_chunk=1)
    assert value[0].contains(10000) and value[1].contains(0)
    assert all(jac[i,0].contains(0) for i in range(2))
    assert jac[1,1].contains(1) and jac[1,2].contains(-1)
    assert max(float(jac[i,0].abs_upper()) for i in range(2)) < 1e-45


def test_interval_hidden_and_direction_balls_enclose_sampled_derivatives():
    embedding = np.array([[.25,-1.],[1.,.5],[-.5,.75],[2.,-2.],[-.75,1.5]])
    h = np.array([.125,-.25]); u = np.array([[1.,-.5],[-.25,1.]])
    hidden = [Jet(arb(exact(h[i]),(1,-20)),tuple(arb(exact(u[i,k]),(1,-21)) for k in range(2))) for i in range(2)]
    outer_f,outer_j = stream.streamed_head(embedding,hidden,arb(1),token_chunk=2)
    rng = np.random.default_rng(1031)
    for _ in range(5):
        hs = h+rng.uniform(-1,1,2)*2**-20
        us = u+rng.uniform(-1,1,(2,2))*2**-21
        sampled = [Jet.make(hs[i],list(us[i])) for i in range(2)]
        inner_f,inner_j = old_head(embedding,sampled,arb(1))
        assert all(a.contains(b) for a,b in zip(outer_f,inner_f))
        assert all(outer_j[i,k].contains(inner_j[i,k]) for i in range(2) for k in range(2))


def test_affine_calls_and_progress_cover_all_rows_without_full_vocab_tangents(monkeypatch):
    rng = np.random.default_rng(42)
    embedding = np.round(rng.normal(size=(17,3))*16)/16
    hidden = [Jet.make(.125,[1.,-1.,.5,0.,.25]) for _ in range(3)]
    actual_affine = stream.affine; calls=[]; events=[]
    def audit_affine(weights, values, **kwargs):
        columns = values.ncols() if isinstance(values,arb_mat) else 1
        calls.append((weights.shape, columns))
        assert weights.shape not in ((17,3),(3,17))
        assert columns in (1,5,6)
        return actual_affine(weights,values,**kwargs)
    monkeypatch.setattr(stream,'affine',audit_affine)
    stream.streamed_head(embedding,hidden,arb(1),token_chunk=4,progress=lambda *args:events.append(args))
    assert len(calls) == 15
    for name in ('head_logits','head_derivatives'):
        intervals = [(a,b) for kind,a,b in events if kind == name]
        assert intervals == [(0,4),(4,8),(8,12),(12,16),(16,17)]
        assert [v for a,b in intervals for v in range(a,b)] == list(range(17))


def test_weight_cache_uses_new_exact_bytes_between_independent_calls():
    embedding = np.array([[-1.],[1.]])
    options = dict(cache=DyadicWeightCache(max_elements=8))
    hidden = [Jet.make(.5,[1.])]
    old = stream.streamed_head(embedding,hidden,arb(1),token_chunk=1,linear_options=options)
    embedding[1,0] = 2.
    new = stream.streamed_head(embedding,hidden,arb(1),token_chunk=1,linear_options=options)
    overlap(new,old_head(embedding,hidden,arb(1)))
    assert not old[0][0].overlaps(new[0][0])


@pytest.mark.parametrize('temperature',[arb(0),arb(-1),arb(0,1),arb('nan'),arb('inf')])
def test_temperature_must_be_finite_and_provably_positive(temperature):
    with pytest.raises((ValueError,TypeError)):
        stream.streamed_head(np.ones((2,1)),[Jet.make(0.,[1.])],temperature)


@pytest.mark.parametrize('embedding',[np.zeros((0,1)),np.zeros((2,0)),np.ones(2),np.ones((2,1),dtype=int),
    np.ones((2,1),dtype=np.longdouble),np.array([[1.],[np.nan]]),np.array([[1.],[np.inf]])])
def test_bad_weight_shape_dtype_or_tail_refused(embedding):
    with pytest.raises(ValueError):stream.streamed_head(embedding,[Jet.make(0.,[1.])],arb(1))


@pytest.mark.parametrize('hidden',[[],[arb(1)],[Jet(arb('nan'),(arb(1),))],
    [Jet(arb(1),(arb('nan'),))],[Jet.make(1.,[1.]),Jet.make(2.,[1.,2.])]])
def test_incomplete_nonfinite_or_mixed_hidden_refused(hidden):
    with pytest.raises((TypeError,ValueError)):
        stream.streamed_head(np.ones((2,max(1,len(hidden)))),hidden,arb(1))


@pytest.mark.parametrize('chunk',[0,-1,True,1.5])
def test_bad_chunk_refused(chunk):
    with pytest.raises(ValueError):stream.streamed_head(np.ones((2,1)),[Jet.make(0.,[1.])],arb(1),token_chunk=chunk)


def pair(count=3):
    scalar,_ = fixtures.fixture()
    kwargs = dict(steering_layer=scalar.steering_layer,alpha=.071674,log10_temperature=-.115,direction_count=count)
    old = BallQwenJet(scalar.config,scalar.weights,scalar.prefix_ids,scalar.vector,**kwargs)
    new = stream.StreamedHeadQwenJet(scalar.config,scalar.weights,scalar.prefix_ids,scalar.vector,token_chunk=2,**kwargs)
    return old,new


def test_full_inherited_transformer_matches_old_jet_and_default_zero():
    old,new = pair(); state=np.array([.08,-.11,.16,.03])
    directions=np.array([[1.,0.,-1.,0.],[.1,-.2,.3,-.4],[0.,0.,0.,0.]])
    overlap(new.forward(state,directions),old.forward(state,directions))
    _,zero = new.forward(state)
    assert all(zero[i,k].is_exact() and zero[i,k] == 0 for i in range(4) for k in range(3))


def test_full_map_box_and_direction_radii_survive_admission():
    old,new = pair(2); state=[arb(exact(v),(1,-26)) for v in [.08,-.11,.16,.03]]
    directions=arb_mat([[arb(exact((i+1)/8),(1,-28)),arb(exact(-(i+1)/8),(1,-29))] for i in range(4)])
    outer = new.forward(state,directions)
    overlap(outer,old.forward(state,directions))
    point = old.forward([v.mid() for v in state],arb_mat([[directions[i,k].mid() for k in range(2)] for i in range(4)]))
    assert all(a.contains(b) for a,b in zip(outer[0],point[0]))
    assert all(outer[1][i,k].contains(point[1][i,k]) for i in range(4) for k in range(2))


@pytest.mark.parametrize('directions',[np.ones((4,2)),np.ones((2,4),dtype=int),np.ones((2,4),dtype=np.longdouble),
    np.full((2,4),np.nan),arb_mat(2,4),[[1.]*4]*2,arb_mat([[arb('nan'),arb(1)] for _ in range(4)])])
def test_subclass_bad_direction_admission(directions):
    _,new=pair(2)
    with pytest.raises((TypeError,ValueError)):new.forward([.1]*4,directions)


@pytest.mark.parametrize('mutation',['precision','alpha','embedding','weight_entry'])
def test_subclass_retains_cache_and_embedding_rebinding_guards(mutation):
    _,new=pair(2);new.prefix_cache()
    if mutation == 'precision':
        with ctx.workprec(256):
            with pytest.raises(ValueError,match='precision'):new.forward([.1]*4)
        return
    if mutation == 'alpha':new.alpha=exact(.07)
    elif mutation == 'embedding':new.embedding=new.embedding.copy()
    else:new.weights['model.embed_tokens.weight']=new.embedding.copy()
    with pytest.raises(ValueError):new.forward([.1]*4)


def test_no_bias_override_or_wrong_state_dimension():
    with pytest.raises(ValueError,match='bias'):
        stream.streamed_head(np.ones((2,1)),[Jet.make(0.,[1.])],arb(1),linear_options={'bias_columns':()})
    _,new=pair(2)
    with pytest.raises(ValueError,match='state'):new.forward([.1]*3)


def test_strict_argmax_checks_every_competitor_and_records_exact_margin():
    values = [arb(-10),arb(2,(1,-4)),arb(0),arb(3,(1,-4))]
    record = stream.argmax_diagnostic(values)
    assert record['candidate_index'] == 3
    assert record['vocabulary_count'] == 4 and record['competitor_count'] == 3
    assert record['strict_margin_established'] is True and record['certificate_passed'] is False
    margin = arb(tuple(record['strict_margin']['mid']),tuple(record['strict_margin']['rad']))
    assert margin.contains(arb(7)/8) and margin > 0


@pytest.mark.parametrize('values',[[arb(1),arb(1)],[arb(2,1),arb(1,1)],[arb(1),arb(2,2)],
    [arb(0),arb(1),arb(1)],[arb(1)]])
def test_tie_or_overlapping_logit_intervals_do_not_establish_strict_label(values):
    assert stream.argmax_diagnostic(values)['strict_margin_established'] is False


def test_optional_head_metadata_and_subclass_metadata_are_current_only():
    record={}
    stream.streamed_head(np.array([[-1.],[1.]]),[Jet.make(.5,[1.])],arb(1),token_chunk=1,diagnostics=record)
    assert record['candidate_index'] == 1 and record['strict_margin_established'] is True
    assert record['vocabulary_count'] == 2
    with pytest.raises(ValueError,match='empty'):
        stream.streamed_head(np.ones((2,1)),[Jet.make(0.,[1.])],arb(1),diagnostics=record)
    _,new=pair(2);new.forward([.1]*4)
    assert new.last_head_diagnostics['vocabulary_count'] == 7
    with pytest.raises(ValueError):new.forward([.1]*3)
    assert new.last_head_diagnostics is None
