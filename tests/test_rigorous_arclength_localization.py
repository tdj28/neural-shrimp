import pytest
from flint import arb,ctx
from neural_shrimp.rigorous_orbits import vector_json
from neural_shrimp.rigorous_connection import restore_ball
from neural_shrimp.rigorous_connection_frames import restore_matrix
from neural_shrimp.rigorous_arclength_localization import localize_endpoint
from test_rigorous_arclength_connection import Fold


def parent():
    return {'passed':True,'gates':{'uniform_root':True,'primitive_period':True,'anchor_identified':True},
      'statement':{'period':1,'phase_dimension':1},'normal':['1','0'],'origin':['1/2','1'],
      'exact_parameter_endpoints':['-1/64','0'],
      'endpoint_root_enclosures':[[],vector_json([arb('1/2',arb('1/256')),arb('5/4',arb('1/256'))])]}


def test_full_known_root_survives_three_intersections():
    with ctx.workprec(256):
        r=localize_endpoint(Fold(),parent());box=[restore_ball(v) for v in r['final_state_box']]
        assert len(r['intersections'])==3 and r['passed']
        assert box[0].contains(arb('1/2')) and box[1].contains(arb('5/4'))
        assert max(v.rad() for v in box)<arb('1e-50')
        assert r['start']['old_parameter']=='0'


def test_center_displacement_and_nonzero_residual_retained():
    c=parent();c['endpoint_root_enclosures'][1]=vector_json([arb('129/256',arb('1/128')),arb('321/256',arb('1/128'))])
    with ctx.workprec(256):
        r=localize_endpoint(Fold(),c)
        assert restore_ball(r['final_state_box'][0]).contains(arb('1/2'))
        assert restore_ball(r['final_state_box'][1]).contains(arb('5/4'))
        assert restore_matrix(r['start']['center_residual'])[1,0] != 0

@pytest.mark.parametrize('change',[{'passed':False},{'gates':{'uniform_root':False}}])
def test_unqualified_parent_rejected(change):
    c=parent();c.update(change)
    with pytest.raises(ValueError,match='qualified'):localize_endpoint(Fold(),c)

@pytest.mark.parametrize('iterations',[0,4,True])
def test_fixed_intersection_count(iterations):
    with pytest.raises(ValueError,match='cap'):localize_endpoint(Fold(),parent(),iterations=iterations)


def test_interruption_retains_completed_stages():
    rows=[]
    def observe(k,r):
        rows.append((k,r))
        if k=='intersection':raise TimeoutError('injected after durable intersection')
    with pytest.raises(TimeoutError):localize_endpoint(Fold(),parent(),observe=observe)
    assert [k for k,r in rows]==['localization_start','intersection']


def test_wrong_endpoint_gauge_causes_empty_intersection():
    c=parent();c['exact_parameter_endpoints'][1]='1'
    with pytest.raises(ValueError):localize_endpoint(Fold(),c)


def test_synthetic_scope_explicit():
    c=parent();c['passed']=False
    r=localize_endpoint(Fold(),c,synthetic_profile=True)
    assert not r['scientific_claim']


def driver():
    import importlib.util
    from pathlib import Path
    p=Path(__file__).parents[1]/'tools/certify_s2_period18_endpoint.py'
    s=importlib.util.spec_from_file_location('endpoint_driver',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

@pytest.mark.parametrize('change',[('elapsed_seconds',-1),('elapsed_seconds',float('nan')),('max_rss_kib',-1),('completed_intersections',2)])
def test_profile_gate_rejects_incomplete_or_bad_resources(tmp_path,monkeypatch,change):
    m=driver();rt={'mode':'profile','source_sha256':{},'python':m.platform.python_version(),'python_flint':m.flint.__version__,
      'flint':m.flint.__FLINT_VERSION__,'numpy':m.numpy.__version__,'scipy':m.scipy.__version__,'precision':256,'map_cap':72,
      'unit_derivative_cap':144,'internal_seconds':20,'tick_cap':400,'threads':dict.fromkeys(m.THREADS,'1'),'nice':10}
    result={'exit_code':0,'status':'synthetic_profile_completed','adapter_counts':{'map_attempted':72,'map_returned':72,'unit_attempted':144,'unit_returned':144,'map_cap':72,'unit_cap':144},'elapsed_seconds':1,'max_rss_kib':100,'completed_intersections':3}
    monkeypatch.setattr(m,'entry',lambda p:{'sha256':'pin'});monkeypatch.setattr(m,'verify_inventory',lambda *x:None)
    monkeypatch.setattr(m,'sources',lambda:{});monkeypatch.setattr(m,'total',lambda p:100)
    monkeypatch.setattr(m,'read',lambda p:{'manifest.json':{'files':{}},'runtime-start.json':rt,'outcome.json':result}[p.name])
    assert m.profile_gate(tmp_path,'pin')['projected_seconds']==7
    result[change[0]]=change[1]
    with pytest.raises(ValueError):m.profile_gate(tmp_path,'pin')
