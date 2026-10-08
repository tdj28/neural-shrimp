"""72D analytic parity and exact-input failure controls; no research target run."""
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
from flint import ctx
from neural_shrimp.gdmaps import TwoLayerNet
from neural_shrimp.rigorous_orbits import BallTanhMap, exact
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import certify_s1r_period2 as subject
from audit_s1r_branch_outcome import loss_gradient_hessian


def case():
    rng=np.random.default_rng(127)
    model=asdict(TwoLayerNet(d=8,width=8,bias=False,in_scale=.5,out_scale=.125,ridge=.001))
    return model,rng.normal(size=(72,8)),rng.normal(size=72),rng.normal(size=72),np.array([.3,.2])


def test_exact_binary64_translation_and_invalid_inputs():
    m,x,y,t,r=case();a=subject.exact_inputs(m,x,y,t,r,{})
    assert Fraction(a['network']['ridge'])==Fraction(float(m['ridge']))
    assert Fraction(a['lr_hidden'])==Fraction(float(r[0]))
    assert all(Fraction(q)==Fraction(float(v)) for q,v in zip(a['approximation'],t))
    bad=t.copy();bad[0]=np.nan
    with pytest.raises(ValueError,match='finite'):subject.exact_inputs(m,x,y,bad,r,{})
    with pytest.raises(ValueError,match='positive'):subject.exact_inputs(m,x,y,t,-r,{})
    with pytest.raises(ValueError,match='dimensions'):subject.exact_inputs(m,x[:-1],y,t,r,{})


def test_wide_map_and_hessian_against_independent_sample_formula():
    # Three synthetic samples keep this a cheap dimensional/indexing control.
    m,x,y,t,r=case();a=subject.exact_inputs(m,x,y,t,r,{})
    a['x']=a['x'][:3];a['y']=a['y'][:3]
    _,g,h=loss_gradient_hessian(t,x[:3],y[:3],m)
    rates=np.r_[np.full(64,r[0]),np.full(8,r[1])]
    with ctx.workprec(192):state,jac=BallTanhMap(a).step([exact(q) for q in a['approximation']])
    np.testing.assert_allclose([float(v.mid()) for v in state],t-rates*g,rtol=2e-13,atol=2e-13)
    numeric=np.array([[float(jac[i,j].mid()) for j in range(72)] for i in range(72)])
    np.testing.assert_allclose(numeric,np.eye(72)-rates[:,None]*h,rtol=2e-13,atol=2e-13)


def test_profile_rejects_tampered_or_unqualified_receipt(tmp_path):
    from neural_shrimp.rigorous_orbits import canonical_bytes
    def write(name,value):(tmp_path/name).write_bytes(canonical_bytes(value))
    sources={'source':'hash'};inputs={'test':'value'}
    write('runtime-start.json',{'source_sha256':sources,'settings':subject.SETTINGS})
    write('inputs.json',inputs);write('result.json',{'passed':True,'projected_seconds':20,'max_rss_kib':100})
    subject.manifest(tmp_path,'profile_passed')
    subject.profile_gate(tmp_path,sources,inputs)
    write('result.json',{'passed':True,'projected_seconds':200,'max_rss_kib':100})
    with pytest.raises(ValueError,match='hash'):subject.profile_gate(tmp_path,sources,inputs)
    subject.manifest(tmp_path,'profile_passed')
    with pytest.raises(ValueError,match='gate'):subject.profile_gate(tmp_path,sources,inputs)
    write('result.json',{'passed':True,'projected_seconds':20,'max_rss_kib':100});subject.manifest(tmp_path,'profile_passed')
    with pytest.raises(ValueError,match='source'):subject.profile_gate(tmp_path,{'changed':'hash'},inputs)
