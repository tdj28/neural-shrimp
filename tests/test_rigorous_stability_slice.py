import sys
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_stability_slice import stability_slice,uniform_tangent
from neural_shrimp.rigorous_orbits import exact
from neural_shrimp.rigorous_connection_frames import restore_matrix
from test_rigorous_arclength_slices import StableFold,parent


def chart():
    out=parent();out['exact_weights']=['1','1'];return out


def test_complete_synthetic_fold_branch():
    with ctx.workprec(256):
        events=[]
        result=stability_slice(StableFold(),StableFold(),chart(),'1/32','1/16',observe=lambda k,r:events.append((k,r)))
        assert result['passed'] and result['status']=='complete_slice_bounds'
        assert sum(k.endswith('slice_intersection') for k,r in events)==6
        assert [r['phase'] for k,r in events if k=='phase_derivatives']==[0]
        inp=next(r for k,r in events if k=='physical_inputs')
        assert inp['center_parameter']=='3/64' and inp['exact_endpoints']==['1/32','1/16']
        assert result['norm_cocycle']['phases']==1


def test_tangent_matches_prior_arithmetic_except_scope():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
    from analyze_s2_period18_slice_budget import tangent_bound
    with ctx.workprec(256):
        pre=arb_mat([[1,0],[0,1]]);jac=arb_mat([[arb(1,'1/100'),arb(0,'1/200')],[0,1]])
        a=uniform_tangent(pre,jac,[arb(1)]*2);b=tangent_bound(pre,jac,[arb(1)]*2)
        a.pop('scope');b.pop('scope');assert a==b
        assert restore_matrix(a['derivative_box'])[1,0].contains(1)


def test_failed_tangent_stops_before_center_and_mixed_acquisition(monkeypatch):
    import neural_shrimp.rigorous_stability_slice as m
    monkeypatch.setattr(m,'uniform_tangent',lambda *a:{'passed':False})
    class NoJet(StableFold):
        def jet_step(self,*a):raise AssertionError('no derivative after failed uniform gate')
    events=[]
    result=stability_slice(StableFold(),NoJet(),chart(),'1/32','1/16',observe=lambda k,r:events.append(k))
    assert not result['passed'] and result['status']=='inconclusive_uniform_tangent'
    assert not any(k.startswith('center_') for k in events)


def test_interruption_and_storage_errors_are_not_numerical_failures():
    def fail(k,r):raise OSError('synthetic storage cap')
    with pytest.raises(OSError,match='storage cap'):
        stability_slice(StableFold(),StableFold(),chart(),'1/32','1/16',observe=fail)


@pytest.mark.parametrize('left,right',[('0','0'),('1','0'),('-1','1')])
def test_invalid_or_out_of_parent_domains_rejected(left,right):
    with pytest.raises(ValueError):stability_slice(StableFold(),StableFold(),chart(),left,right)


def test_exact_positive_weights_required():
    for weight in (arb(0),arb(-1),arb(1,'1/10')):
        with pytest.raises(ValueError):uniform_tangent(arb_mat([[1]]),arb_mat([[1]]),[weight])
