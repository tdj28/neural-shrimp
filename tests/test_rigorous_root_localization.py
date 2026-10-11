from copy import deepcopy
from fractions import Fraction
import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import exact,ball_json,matrix_json
from neural_shrimp.rigorous_connection import tube,restore_ball
from neural_shrimp.rigorous_connection_frames import restore_matrix
from neural_shrimp.rigorous_root_localization import localize_root,localized_stability


class SqrtRoot:
    n=1
    def step(self,x,h):
        return [x[0]+(h-x[0]*x[0])/4],arb_mat([[1-x[0]/2]]),[arb('1/4')]


def parent():
    with ctx.workprec(256):
        return tube(SqrtRoot(),[['1']],'1',halfwidth_exponent=-6,radius_exponent=-8,newton_steps=0,
                    anchor={'rate':'1','phase_boxes':[[ball_json(arb(1))]]})


def test_floor_refines_known_root_and_stability():
    r=localized_stability(SqrtRoot(),parent(),'1','1')
    assert r['passed'] and r['localization']['floor_only']
    q=r['localization'];e=restore_matrix(q['final_correction'])
    assert e[0,0].contains(0) and e[0,0].rad()<exact('1/1000000000000000')
    assert len(q['refinements'])==3


def test_nonpoint_center_and_rate_variation_enclose_exact_sqrt_roots():
    with ctx.workprec(256):
        r=localize_root(SqrtRoot(),parent(),'129/128','65/64')
        center=restore_matrix(r['center']);s=restore_matrix(r['predictor_slope']);e=restore_matrix(r['final_correction'])
        for h in ('129/128','259/256','65/64'):
            enclosure=center[0,0]+s[0,0]*(exact(h)-exact(r['center_parameter']))+e[0,0]
            assert enclosure.contains(exact(h).sqrt())
        assert e[0,0].rad()<restore_matrix(r['initial_correction'])[0,0].rad()


def test_derivative_domain_contains_predictor_when_error_excludes_zero():
    r=parent();r['image_correction']=matrix_json(arb_mat([[arb('1/10000','1/100000')]]))
    # A deliberately shifted candidate correction tests convex-segment construction;
    # its false parent premise eventually causes empty intersection, as required.
    seen=[]
    class Recorder(SqrtRoot):
        def step(self,x,h):seen.append(x[0]);return super().step(x,h)
    with pytest.raises(ValueError):localize_root(Recorder(),r,'1','1')
    assert seen[1].contains(seen[0])


def test_correction_intersection_failure_preserves_completed_start():
    r=parent();r['image_correction']=matrix_json(arb_mat([[arb(2)]]));events=[]
    with pytest.raises(ValueError):localize_root(SqrtRoot(),r,'1','1',observe=lambda k,v:events.append(k))
    assert events==['localization_start']


@pytest.mark.parametrize('change',[lambda r:r['gates'].update(uniform_root=False),
    lambda r:r['statement'].update(phase_dimension=2),lambda r:r.update(predictor_slope=matrix_json(arb_mat([[arb(1,1)]])))])
def test_invalid_parent_fails_closed(change):
    r=parent();change(r)
    with pytest.raises(ValueError):localize_root(SqrtRoot(),r,'1','1')


@pytest.mark.parametrize('left,right,it',[('0','1',3),('1','2',3),('2','1',3),('1','1',0),('1','1',4)])
def test_invalid_slice_or_budget(left,right,it):
    with pytest.raises(ValueError):localize_root(SqrtRoot(),parent(),left,right,iterations=it)


def test_no_extra_calls_after_interruption():
    count=0
    def tick():
        nonlocal count
        count+=1
        if count==3:raise TimeoutError('synthetic')
    with pytest.raises(TimeoutError):localize_root(SqrtRoot(),parent(),'1','1',tick=tick)
    assert count==3


def test_parent_data_unmodified():
    r=parent();old=deepcopy(r);localize_root(SqrtRoot(),r,'1','1');assert r==old


def test_does_not_invent_primitive_period():
    r=parent();r['gates']['primitive_period']=False
    assert not localize_root(SqrtRoot(),r,'1','1')['inherited_primitive_period']


def driver():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'tools/certify_s2_period18_localized_roots.py'
    spec=importlib.util.spec_from_file_location('localization_driver_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_exact_outer_probe_domains_and_shared_floor():
    m=driver();r=parent()
    a,b=map(Fraction,r['exact_parameter_endpoints'])
    assert m.domain(r,0,'probe')==(str(a),str(a+Fraction(1,32768)))
    assert m.domain(r,1,'probe')==(str(b-Fraction(1,32768)),str(b))
    assert m.domain(r,1,'floor')==('1','1')


def test_profile_manifest_rejected_before_any_inputs(tmp_path,monkeypatch):
    m=driver();(tmp_path/'manifest.json').write_text('{}')
    monkeypatch.setattr(m,'prepare_parent',lambda:pytest.fail('target accessed'))
    with pytest.raises(ValueError,match='manifest'):m.profile_gate(tmp_path,'0'*64)


@pytest.mark.parametrize('elapsed,size',[(-1,100),(float('nan'),100),(float('inf'),100),(20,100),(1,-1),(1,float('inf')),(1,2*1024**2)])
def test_completed_resource_contradictions_rejected(tmp_path,monkeypatch,elapsed,size):
    m=driver();runtime={**m.environment(),'mode':'profile','nice':10,'source_sha256':{}}
    outcome={'mode':'profile','exit_code':0,'elapsed_seconds':elapsed,'max_rss_kib':size}
    launch={'nice':10,'exit_code':0,'owned_live_pids':[],'elapsed_seconds':1}
    monkeypatch.setattr(m,'entry',lambda p:{'sha256':'pin'})
    monkeypatch.setattr(m,'verify_inventory',lambda *args:None)
    monkeypatch.setattr(m,'sources',lambda:{})
    monkeypatch.setattr(m,'read',lambda p:{'files':{}} if p.name=='manifest.json' else {'runtime-start.json':runtime,'outcome.json':outcome,'launcher-result.json':launch}[p.name])
    with pytest.raises(ValueError,match='resource'):m.completed_bundle(tmp_path,'pin','profile')


def test_selected_full_probe_not_tiny_partial_profile(monkeypatch):
    m=driver()
    monkeypatch.setattr(m,'completed_bundle',lambda *args:({'status':'synthetic_profile_completed','counts':{'map_returned':125,'unit_returned':250}},None,None))
    with pytest.raises(ValueError,match='coverage'):m.profile_gate(None,'pin')


def test_parent_source_inventory_has_implicit_imports_and_unchanged_checker():
    m=driver()
    for path in ('src/neural_shrimp/models.py','src/neural_shrimp/controls.py','src/neural_shrimp/rigorous_orbits.py',m.CARD):
        assert path in m.SOURCES
