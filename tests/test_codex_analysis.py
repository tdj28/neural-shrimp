"""Synthetic differentiable integration tests; no model weights are loaded."""
import numpy as np
import pytest
import torch

from codex.crosscheck.analysis import orbit_crosscheck, fold_crosscheck


class Toy:
    def __init__(self, kind):
        self.kind = kind
    def prefix_cache(self, alpha):
        return float(alpha)
    def forward(self, x, alpha, temperature, cache=None):
        if self.kind == "logistic":
            return 3.2*x*(1-x)
        # Local p2 fold from F(x)=-x+a*x+x^3 at x=0 would be period1;
        # use a flip between two disjoint affine-coordinate charts instead.
        # Chart centers -1,+1; local q maps to q+a+q² on the first leg only.
        if bool(x[0].detach() < 0):
            q = x[0]+1
            return torch.stack([1+q+2*alpha+3*q*q, .5*x[1]])
        return torch.stack([x[0]-2, .5*x[1]])
    def period_map(self, x, alpha, temperature, period, cache=None):
        for _ in range(period):
            x = self.forward(x, alpha, temperature, cache)
        return x
    def jvp(self,x,v,alpha,temperature,period=1,cache=None):
        return torch.func.jvp(lambda y:self.period_map(y,alpha,temperature,period,cache),(x,),(v,))[1]
    def jacobian(self,x,alpha,temperature,period=1,chunk_size=16,cache=None):
        return torch.func.jacfwd(lambda y:self.period_map(y,alpha,temperature,period,cache))(x)


def test_orbit_analysis_records_perturbation_and_stages():
    saved=[]
    r=3.2
    state=np.array([(r+1-np.sqrt((r-3)*(r+1)))/(2*r)])
    out=orbit_crosscheck(Toy("logistic"),state,alpha=0,log_temperature=0,period=2,
                         save=lambda stage,payload:saved.append((stage,payload)))
    assert out["passed"],out
    assert out["solver"]["history"][0]["relative_residual"]>1e-11
    assert abs(np.linalg.norm(out["initial_state"]-state)-1e-8)<1e-16
    assert out["diagnostics"]["numerically_attracting"]
    assert any(stage.startswith("monodromy-")for stage,_ in saved)
    assert saved[-1][0].startswith("orbit-result-")


def test_fold_analysis_primitive_two_nested_derivatives_and_prefix_callback():
    saved=[]
    out=fold_crosscheck(Toy("fold"),np.array([-1.,0.]),np.array([1.,0.]),
                        alpha=0.,period=2,save=lambda stage,payload:saved.append((stage,payload)),
                        reference={"a":2.,"b":3.,"ab":6.})
    assert out["passed"],out
    assert abs(out["solver"]["alpha"])<1e-12
    assert abs(out["ad_coefficients"]["a"]-2)<1e-12
    assert abs(out["ad_coefficients"]["b"]-3)<1e-12
    assert all(out["prerequisite_gates"].values())
    assert len(out["phase_closures"])==2
    assert saved[-1][0].startswith("fold-result-")
    assert out["operation_counts"]["critical_gradient"]>0


def test_repelling_primitive_cycle_does_not_pass_attracting_target():
    class Repelling(Toy):
        def forward(self, x, alpha, temperature, cache=None):
            return 3.5*x*(1-x)
    r=3.5
    state=np.array([(r+1-np.sqrt((r-3)*(r+1)))/(2*r)])
    out=orbit_crosscheck(Repelling("logistic"),state,alpha=0,log_temperature=0,period=2)
    assert out["diagnostics"]["closure_passed"]
    assert out["diagnostics"]["primitive_separation_passed"]
    assert not out["diagnostics"]["spectral_contraction_passed"]
    assert not out["passed"]


def test_wrong_full_monodromy_dimension_is_rejected():
    class WrongMatrix(Toy):
        def jacobian(self, *args, **kwargs):
            return torch.eye(2,dtype=torch.float64)
    r=3.2
    state=np.array([(r+1-np.sqrt((r-3)*(r+1)))/(2*r)])
    with pytest.raises(ValueError, match="full state dimension"):
        orbit_crosscheck(WrongMatrix("logistic"),state,alpha=0,log_temperature=0,period=2)
