from copy import deepcopy
from fractions import Fraction
import pytest
from flint import arb, arb_mat, ctx
from neural_shrimp.rigorous_arclength_connection import arclength_tube
from neural_shrimp.rigorous_arclength_slices import localize_slice, slice_stability
from neural_shrimp.rigorous_connection import restore_ball
from neural_shrimp.rigorous_connection_frames import restore_matrix
from neural_shrimp.rigorous_orbits import exact, ball_json, matrix_json


class StableFold:
    n = 1
    def step(self, x, h):
        return [x[0] - x[0]*x[0] + h - 1], arb_mat([[1-2*x[0]]]), [arb(1)]
    def jet_step(self, x, h):
        return [x[0] - x[0]*x[0] + h - 1]


def parent():
    with ctx.workprec(256):
        m = StableFold()
        return arclength_tube(m, m, ['1/2','5/4'], normal=['1','0'], origin=['1/2','5/4'],
            parameter='0', weights=['1','1'], period=1, halfwidth_exponent=-4,
            radius_exponent=-12, newton_steps=0,
            anchor={'state_box':[ball_json(arb('1/2')),ball_json(arb('5/4'))]})


def test_nonpoint_quadratic_shift_retains_actual_roots_and_gauge():
    with ctx.workprec(256):
        p = parent(); old = deepcopy(p)
        r = localize_slice(StableFold(), p, '1/32', '1/16')
        c = [restore_ball(v) for v in r['start']['center']]
        s, t, e = [restore_matrix(r['start'][k]) for k in ('slope','curvature','initial_error')]
        e = restore_matrix(r['final_error'])
        for value in ('1/32','3/64','1/16'):
            d = exact(value)-exact(r['start']['center_parameter'])
            x = arb('1/2')+exact(value)
            assert (c[0]+s[0,0]*d+t[0,0]*d*d/2+e[0,0]).contains(x)
            assert (c[1]+s[1,0]*d+t[1,0]*d*d/2+e[1,0]).contains(1+x*x)
        assert p == old and len(r['intersections']) == 3
        for row in r['intersections']:
            assert restore_matrix(row['path_derivative'])[1,0].contains(0)
        assert r['start']['slope'] != p['predictor_slope']


def test_floor_and_nonpoint_metric_are_rigorous_on_supplied_boxes():
    for left,right in [('1/16','1/16'),('1/32','1/16')]:
        r = slice_stability(StableFold(), parent(), left, right)
        assert r['passed'] and r['localization']['floor_only'] == (left == right)
        assert r['stability']['phases'][0]['interval_D_positive']['passed']


def test_shifted_image_midpoint_displacement_is_retained():
    p = parent()
    p['image_correction'] = matrix_json(arb_mat([[arb('1/1000000','1/100000')],[arb('-1/1000000','1/100000')]]))
    r = localize_slice(StableFold(), p, '1/32','1/32')
    box = [restore_ball(v) for v in r['final_state_box']]
    assert box[0].contains(arb('17/32')) and box[1].contains(arb('1313/1024'))
    assert restore_matrix(r['start']['center_residual'])[1,0] != 0


@pytest.mark.parametrize('left,right',[('-1/8','0'),('0','1/8'),('1/16','0')])
def test_outside_slice_rejected(left,right):
    with pytest.raises(ValueError,match='subinterval'):localize_slice(StableFold(),parent(),left,right)


@pytest.mark.parametrize('iterations',[0,4,True])
def test_iteration_cap(iterations):
    with pytest.raises(ValueError,match='cap'):localize_slice(StableFold(),parent(),'0','0',iterations=iterations)


@pytest.mark.parametrize('mutation',[
    lambda p:p.update(passed=False),
    lambda p:p['gates'].update(primitive_period=False),
    lambda p:p.update(predictor_curvature=matrix_json(arb_mat([[arb(0,1)],[2]])))])
def test_unqualified_or_nonexact_input_rejected(mutation):
    p=parent();mutation(p)
    with pytest.raises(ValueError):localize_slice(StableFold(),p,'0','0')


def test_false_parent_image_rejected_after_durable_start():
    p=parent();p['image_correction']=matrix_json(arb_mat([[arb(1)],[arb(0)]]));events=[]
    with pytest.raises(ValueError):localize_slice(StableFold(),p,'0','0',observe=lambda k,r:events.append(k))
    assert events==['slice_start']


def test_interruption_preserves_completed_intersection():
    events=[]
    def observe(k,r):
        events.append(k)
        if k=='slice_intersection':raise TimeoutError('synthetic interruption')
    with pytest.raises(TimeoutError):localize_slice(StableFold(),parent(),'0','0',observe=observe)
    assert events==['slice_start','slice_intersection']


def test_native_call_count_no_hidden_model_acquisitions():
    class Count(StableFold):
        calls=0
        def step(self,x,h):self.calls+=1;return super().step(x,h)
    m=Count();slice_stability(m,parent(),'0','0')
    assert m.calls == 5
