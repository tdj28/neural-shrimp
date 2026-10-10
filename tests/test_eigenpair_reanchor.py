from fractions import Fraction
import copy
import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import exact,eye
from neural_shrimp.rigorous_eigenpair_path import quadratic_tube
from neural_shrimp.rigorous_eigenpair_reanchor import (localization_identity,localize_endpoint,tangent_proposal,reanchor)
from neural_shrimp.rigorous_eigenpair_march import decode
from test_rigorous_eigenpair_locus import CoupledCycle,settings,Cusp


def parent(sigma=1):
    model=CoupledCycle(sigma);z,kw=settings(model,sigma=sigma)
    r,_=quadratic_tube(model,z,**kw);assert r['passed'];return model,r


@pytest.mark.parametrize('sigma',[-1,1])
def test_native_sign_phase_and_full_anchor_survive_reanchor(sigma):
    with ctx.workprec(256):
        model,r=parent(sigma);events=[];ticks=[]
        out=reanchor(model,r,tick=lambda:ticks.append(1),observe=lambda k,v:events.append((k,v)))
        assert out['passed'] and len(out['charts'])==2 and len(ticks)==8+2*15
        assert out['charts'][0]['sigma']==sigma and out['charts'][0]['normal']!=r['normal']
        raw=[v for k,v in events if k=='chart_return'];anchor=raw[0]['anchor_check']
        assert anchor['passed'] and anchor['parameter']==anchor['projection']
        assert len([v for k,v in events if k=='localization_intersection'])==3
        assert raw[0]['parameter_endpoints'][1]==raw[1]['parameter_endpoints'][0]


def test_known_root_identity_uses_hull_when_center_excluded_and_singular_C_allowed():
    with ctx.workprec(256):
        center=[arb(0),arb(0)];box=[arb('1 +/- .125'),arb('2 +/- .125')]
        # Coupled affine H(x)=J(x-[1,2]); exact known root not equal center.
        J=arb_mat([[2,1],[-1,3]]);g=arb_mat([[-4],[-5]]);C=J.inv()
        # This inverse is rational with denominator7; choose any exact C.
        C=arb_mat([[arb('.5'),0],[0,arb('.25')]])
        image,D=localization_identity(center,g,J,C,box)
        assert image[0,0].contains(1) and image[1,0].contains(2)
        nochange,_=localization_identity(center,g,J,arb_mat(2,2),box)
        assert nochange[0,0].contains(box[0]) and nochange[1,0].contains(box[1])


def test_radius_transport_exact_positive_orientation_and_width():
    with ctx.workprec(256):
        m,r=parent();_,box=localize_endpoint(m,r);p=tangent_proposal(r,box)
        assert all(Fraction(v)>0 for v in p['exact_new_radii'])
        assert all(Fraction(w)*Fraction(2)**p['radius_exponent']==Fraction(v) for w,v in zip(p['weights'],p['exact_new_radii']))
        assert decode(p['normal_dot_saved_tangent'])>0 and decode(p['normalization_residual']).contains(0)
        assert all(exact(v).is_exact() for v in p['normal']+p['origin'])
        assert arb((1,p['halfwidth_exponent']))<=decode(p['tangent_norm'])*arb((1,r['settings']['halfwidth_exponent']))


def test_failed_or_unproven_parent_never_localizes():
    with ctx.workprec(256):
        m,r=parent();r['passed']=False
        with pytest.raises(ValueError):localize_endpoint(m,r)


def test_intersection_failure_retains_image(monkeypatch):
    import neural_shrimp.rigorous_eigenpair_reanchor as R
    with ctx.workprec(256):
        m,r=parent();events=[]
        monkeypatch.setattr(R,'localization_identity',lambda center,g,j,c,b:(arb_mat([[arb(1000)] for v in b]),eye(len(b))))
        with pytest.raises(ValueError):localize_endpoint(m,r,observe=lambda k,v:events.append(k))
        assert events[-1]=='localization_image_return' and 'localization_return' not in events


def test_durable_intersection_failure_stops_next_assembly():
    with ctx.workprec(256):
        m,r=parent();ticks=[]
        def obs(k,v):
            if k=='localization_intersection':raise OSError('durability injection')
        with pytest.raises(OSError):localize_endpoint(m,r,tick=lambda:ticks.append(1),observe=obs)
        assert len(ticks)==4


def test_cusp_rate_flattening_is_separate_from_native_locus_rank():
    with ctx.workprec(256):
        z=['0','1','1','1'];normal=['1','0','0','0']
        r,_=quadratic_tube(Cusp(),z,period=1,sigma=1,normal=normal,origin=z,parameter='0',halfwidth_exponent=-12,radius_exponent=-16,anchor={'box':[exact(v) for v in z],'period':1,'sigma':1,'dimension':1})
        assert r['passed']
        # Center rate tangent exactly zero at cusp, but raw state tangent is one.
        assert decode(r['slope'][0])==1 and decode(r['slope'][-1]).is_zero() and decode(r['slope'][-2]).is_zero()
        assert all(r['gates'].values())


def test_zero_tangent_and_nonexact_preconditioner_rejected():
    with ctx.workprec(256):
        m,r=parent();z={'mid':[0,0],'rad':[0,0]};bad=copy.deepcopy(r);bad['slope']=[z]*10;bad['curvature']=[z]*10
        with pytest.raises(ValueError,match='nonzero'):tangent_proposal(bad,[decode(v) for v in r['endpoint_root_boxes'][1]])
        with pytest.raises(ValueError,match='exact fixed'):localization_identity([arb(0)],arb_mat([[0]]),arb_mat([[1]]),arb_mat([[arb('1 +/- .01')]]),[arb('0 +/- .1')])
