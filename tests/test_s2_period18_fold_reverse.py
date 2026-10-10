from pathlib import Path
import importlib.util
import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import ball_json,vector_json,exact
from neural_shrimp.rigorous_connection import restore_ball
from neural_shrimp.rigorous_arclength_connection import two_arclength_charts
from neural_shrimp.rigorous_fold_reverse import fold_native_proposal
from test_rigorous_arclength_connection import Fold


def module():
    p=Path(__file__).parents[1]/'tools/certify_s2_period18_fold_reverse.py'
    s=importlib.util.spec_from_file_location('fold_reverse',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def proposal(target=arb('1/128')):
    return fold_native_proposal(Fold(),[[ball_json(arb(0))]],ball_json(arb(1)),[ball_json(arb(1))],vector_json([target,1+target*target]))


def test_native_fold_proposal_and_fresh_common_unique_chart():
    with ctx.workprec(256):
        p=proposal();assert p['normal']==['1','0'] and 'exact_parameter' not in p['anchor']
        assert all(0<exact(v)<=1 for v in p['weights'])
        r=two_arclength_charts(Fold(),Fold(),p,halfwidth_exponent=-6,radius_exponent=-16)
        assert r['passed']
        rows=module().io.fold_alignments(r['charts'][0],[[ball_json(arb('1/128'))]],ball_json(1+arb('1/128')**2))
        assert rows[0]['contained_in_qualified_chart']
        wrong=module().io.fold_alignments(r['charts'][0],[[ball_json(arb('1/128',arb('1/32')))]],ball_json(1+arb('1/128')**2))
        assert not wrong[0]['contained_in_qualified_chart']


def test_orientation_toward_target_and_zero_rate_normal():
    with ctx.workprec(256):
        p=proposal(arb('-1/128'));assert p['normal']==['-1','0'] and p['orientation_sign']==-1
        assert restore_ball(p['anchor']['state_box'][-1])==1


def test_unresolved_orientation_fails():
    with pytest.raises(ValueError,match='orientation'):proposal(arb(0))


def test_native_order_uses_each_phase_once_before_border():
    class Phases:
        n=1
        def __init__(self):self.calls=[]
        def step(self,x,h):
            self.calls.append(x[0]);j=arb(2) if x[0]==0 else arb('1/2')
            return [x[0]],arb_mat([[j]]),[arb(-1)]
    with ctx.workprec(256):
        m=Phases();r=fold_native_proposal(m,[[ball_json(arb(0))],[ball_json(arb(1))]],ball_json(arb(1)),[ball_json(arb(1))],vector_json([arb('1/8'),arb('5/4'),arb(1)]))
        assert m.calls==[0,1,0,1]
        assert restore_ball(r['derivative_proposals'][0]['direction_before'][0])==1
        assert restore_ball(r['derivative_proposals'][1]['direction_before'][0])==2
        assert restore_ball(r['propagated_closure_defect'][0])==0
        assert exact(r['normal'][1])==2*exact(r['normal'][0])


def test_proposal_failure_never_creates_chart():
    def stop():raise TimeoutError('fixed synthetic stop')
    with pytest.raises(TimeoutError):fold_native_proposal(Fold(),[[ball_json(arb(0))]],ball_json(arb(1)),[ball_json(arb(1))],vector_json([arb(1),arb(2)]),tick=stop)


def test_source_and_native_parent_pin():
    m=module();assert m.PIN=='c2361075dd34763265f8f4756dcf979361795e9ee7cf63700dcd470dd3f86150'
    assert m.SETTINGS==dict(precision=256,halfwidth_exponent=-7,radius_exponent=-16,newton_steps=2)
    assert 'src/neural_shrimp/rigorous_fold_reverse.py' in m.SOURCES
    assert len(m.SOURCES)==len(set(m.SOURCES))


def test_manifest_before_science(tmp_path,monkeypatch):
    m=module();monkeypatch.setattr(m,'PARENT',tmp_path);(tmp_path/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='manifest'):m.prepare()
