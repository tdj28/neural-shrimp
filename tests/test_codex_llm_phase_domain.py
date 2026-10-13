"""Exact mixed-precision domain controls; no checkpoint or model calls."""
from pathlib import Path
from fractions import Fraction
import copy
import json
import sys

import pytest
from flint import arb,ctx

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'codex/certify-llm'))
import phase_domain as domain


def point(mantissa,exponent=0):
    return {'mid':[mantissa,exponent],'rad':[0,0]}


def make(centers=None,radii=None,precision=1024):
    centers=centers if centers is not None else [[point((1<<2047)+1,-2047),point(-3,-1)]]
    radii=radii if radii is not None else [point(1,-2400)]
    artifact=domain.build_phase_domains(centers,radii,period=len(centers),dimension=len(centers[0]),target_precision=precision)
    return centers,radii,artifact


def verify(centers,radii,artifact,expected=None):
    return domain.decode_verified_domains(artifact,centers,radii,
        period=len(centers),dimension=len(centers[0]),target_precision=artifact['target_precision'],
        expected_artifact_sha256=expected or artifact['artifact_sha256'])


def exact(pair):
    return Fraction(pair[0])*(Fraction(2)**pair[1])


def test_2048bit_center_and_radius_below_rounding_ulp_are_retained():
    centers,radii,artifact=make()
    actual,checked=verify(centers,radii,artifact)
    witness=checked['witnesses'][0][0]
    assert exact(witness['center_rounding_error'])==Fraction(1,1<<2047)
    assert exact(witness['nominal_radius'])==Fraction(1,1<<2400)
    assert exact(witness['actual_radius'])>=Fraction(1,1<<2047)+Fraction(1,1<<2400)
    assert exact(witness['lower_margin'])>=0 and exact(witness['upper_margin'])>=0
    assert actual[0][0].is_finite() and checked['full_nominal_domain_contained'] is True
    assert artifact['actual_balls_sha256']==domain.canonical_hash(artifact['actual_decoded_balls'])


def test_rounding_both_signs_and_directions_at_low_precision():
    centers=[[point((1<<100)+1,-100),point((1<<100)-1,-100),
              point(-((1<<100)+1),-100),point(-((1<<100)-1),-100)]]
    radii=[point(1,-200)]
    c,r,a=make(centers,radii,precision=32);_,checked=verify(c,r,a)
    for witness in checked['witnesses'][0]:
        assert exact(witness['center_rounding_error'])>0
        assert exact(witness['radius_margin'])>=0


def test_noncanonical_exact_encodings_and_extreme_exponents():
    centers=[[point(1,1200),point(-8,-1403),point(0,-9000)]]
    radii=[point(1,-2500)]
    c,r,a=make(centers,radii,precision=64);actual,checked=verify(c,r,a)
    assert len(actual[0])==3
    assert all(exact(w['actual_radius'])>=exact(w['nominal_radius']) for w in checked['witnesses'][0])


def test_pack_unpack_may_expand_and_is_checked_on_actual_receiver_balls():
    c,r,a=make([[point(3,-1)]],[point((1<<40)+1,-120)],precision=64)
    actual,_=verify(c,r,a)
    with ctx.workprec(64):
        again=[[arb(tuple(domain.pack_ball(v)['mid']),tuple(domain.pack_ball(v)['rad'])) for v in row] for row in actual]
    widened=domain.containment_witnesses(c,r,again,period=1,dimension=1)
    assert exact(widened['witnesses'][0][0]['actual_radius'])>=exact(a['containment_witnesses'][0][0]['actual_radius'])
    assert widened['actual_balls_sha256']==domain.canonical_hash(widened['actual_balls'])


def test_nominal_radius_never_shrunk_and_exact_endpoint_arithmetic():
    c,r,a=make([[point(0),point(5,-2)]],[point((1<<150)+3,-200)],precision=32)
    for witness in a['containment_witnesses'][0]:
        assert exact(witness['actual_lower'])<=exact(witness['nominal_lower'])
        assert exact(witness['actual_upper'])>=exact(witness['nominal_upper'])
        assert exact(witness['actual_radius'])>=exact(witness['nominal_radius'])


@pytest.mark.parametrize('bad',[1.0,arb(1),{'mid':[1.,0],'rad':[0,0]},
    {'mid':[True,0],'rad':[0,0]},{'mid':[1,0],'rad':[1,-100]},
    {'mid':[1,0],'rad':[-1,-100]},{'mid':[1,0]},float('nan'),float('inf')])
def test_float_nonfinite_and_nonexact_center_records_refused(bad):
    with pytest.raises(ValueError):domain.build_phase_domains([[bad]],[point(1,-20)],period=1,dimension=1,target_precision=64)


@pytest.mark.parametrize('radius',[point(0),point(-1,-20),1e-30,
    {'mid':[1,-20],'rad':[1,-100]}])
def test_radius_must_be_exact_positive(radius):
    with pytest.raises(ValueError):domain.build_phase_domains([[point(1)]],[radius],period=1,dimension=1,target_precision=64)


def test_missing_phases_coordinates_radii_and_wrong_dimensions_refused():
    for c,r,p,n in [([[point(1)]],[point(1,-20)],2,1),
                    ([[point(1)]],[point(1,-20)],1,2),
                    ([[point(1)]],[],1,1),([],[],0,1),
                    ([[point(1)]],[point(1,-20)],True,1)]:
        with pytest.raises(ValueError):domain.build_phase_domains(c,r,period=p,dimension=n,target_precision=64)


@pytest.mark.parametrize('precision',[0,1,15,True,64.0])
def test_bad_precision_refused(precision):
    with pytest.raises(ValueError):domain.build_phase_domains([[point(1)]],[point(1,-20)],period=1,dimension=1,target_precision=precision)


@pytest.mark.parametrize('field',['encoded_balls','actual_decoded_balls','containment_witnesses',
    'nominal_centers','nominal_radii','actual_balls_sha256','target_precision'])
def test_any_bound_artifact_tamper_rejected(field):
    c,r,a=make();bad=copy.deepcopy(a);bad[field]=None
    with pytest.raises(ValueError):verify(c,r,bad,expected=a['artifact_sha256'])
    body={k:v for k,v in bad.items() if k!='artifact_sha256'}
    bad['artifact_sha256']=domain.canonical_hash(body)
    with pytest.raises(ValueError):verify(c,r,bad,expected=a['artifact_sha256'])


def test_nominal_input_tamper_and_missing_decoded_coordinate_refused():
    c,r,a=make();changed=copy.deepcopy(c);changed[0][0]=point(1)
    with pytest.raises(ValueError):verify(changed,r,a)
    actual,_=verify(c,r,a);actual[0].pop()
    with pytest.raises(ValueError):domain.containment_witnesses(c,r,actual,period=1,dimension=2)


def test_rehashed_narrow_ball_still_fails_independent_endpoint_check():
    c,r,a=make([[point((1<<150)+1,-150)]],[point(1,-220)],precision=64)
    bad=copy.deepcopy(a);bad['encoded_balls']=[[point(1)]]
    bad['encoded_balls_sha256']=domain.canonical_hash(bad['encoded_balls'])
    bad['artifact_sha256']=domain.canonical_hash({k:v for k,v in bad.items() if k!='artifact_sha256'})
    with pytest.raises(ValueError):verify(c,r,bad)


def test_json_roundtrip_and_context_preservation():
    with ctx.workprec(2048):
        c,r,a=make(precision=1024)
        assert ctx.prec==2048
        recovered=json.loads(json.dumps(a))
        actual,checked=verify(c,r,recovered)
        assert ctx.prec==2048 and checked['full_nominal_domain_contained']
        assert domain.canonical_hash(recovered)==domain.canonical_hash(a)


def test_nonfinite_receiver_and_midpoint_only_receiver_refused():
    c,r,a=make([[point(1)]],[point(1,-100)],precision=64)
    for balls in [[[arb('nan')]],[[arb('inf')]],[[arb(1)]]]:
        with pytest.raises(ValueError):domain.containment_witnesses(c,r,balls,period=1,dimension=1)
