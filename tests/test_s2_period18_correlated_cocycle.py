import importlib.util
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import matrix_json,vector_json


def driver():
    p=Path(__file__).parents[1]/'tools/certify_s2_period18_correlated_cocycle.py'
    s=importlib.util.spec_from_file_location('correlated_driver',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def profile_fixture(monkeypatch,m):
    rt={'mode':'profile','source_sha256':{},'python':m.platform.python_version(),'python_flint':m.flint.__version__,
      'flint':m.flint.__FLINT_VERSION__,'numpy':m.numpy.__version__,'scipy':m.scipy.__version__,'precision':256,'map_cap':90,
      'unit_derivative_cap':180,'mixed_call_cap':18,'internal_seconds':30,'tick_cap':500,'threads':dict.fromkeys(m.THREADS,'1'),'nice':10}
    r={'exit_code':0,'status':'synthetic_profile_completed','adapter_counts':{'map_attempted':90,'map_returned':90,'unit_attempted':180,'unit_returned':180,'map_cap':90,'unit_cap':180},
      'elapsed_seconds':1,'max_rss_kib':100,'completed_intersections':3,'completed_cases':1,'completed_physical_phases':18,'completed_products':18,
      'mixed_counts':{'attempted':18,'returned':18,'cap':18},'cases':[{'role':'probe','passed':True,'point_passed':True,'interval_passed':True}]}
    monkeypatch.setattr(m,'entry',lambda p:{'sha256':'pin'});monkeypatch.setattr(m,'verify_inventory',lambda *x:None)
    monkeypatch.setattr(m,'sources',lambda:{});monkeypatch.setattr(m,'total',lambda p:100)
    monkeypatch.setattr(m,'read',lambda p:{'manifest.json':{'files':{}},'runtime-start.json':rt,'outcome.json':r,'launcher-result.json':{'mode':'profile','source_sha256':{},'exit_code':0,'owned_live_pids':[],'elapsed_seconds':2}}[p.name])
    return rt,r


@pytest.mark.parametrize('key,value',[('elapsed_seconds',0),('elapsed_seconds',-1),('elapsed_seconds',float('nan')),('elapsed_seconds',20),
 ('max_rss_kib',0),('max_rss_kib',-1),('max_rss_kib',float('inf')),('max_rss_kib',1024**2),('completed_intersections',2),
 ('completed_physical_phases',17),('completed_products',17),('completed_cases',0),('cases',[]),('mixed_counts',{'attempted':18,'returned':17,'cap':18})])
def test_profile_resource_and_complete_path_controls(tmp_path,monkeypatch,key,value):
    m=driver();rt,r=profile_fixture(monkeypatch,m);assert m.profile_gate(tmp_path,'pin','pin')['projected_seconds']==14
    r[key]=value
    with pytest.raises(ValueError):m.profile_gate(tmp_path,'pin','pin')


def test_library_manifest_and_source_inventory(tmp_path,monkeypatch):
    m=driver();rt,r=profile_fixture(monkeypatch,m)
    with pytest.raises(ValueError,match='manifest'):m.profile_gate(tmp_path,'changed','pin')
    rt['python_flint']='changed'
    with pytest.raises(ValueError,match='library'):m.profile_gate(tmp_path,'pin','pin')
    for name in (m.CARD,'src/neural_shrimp/rigorous_folds.py','src/neural_shrimp/models.py','src/neural_shrimp/controls.py','tools/analyze_s2_period18_slice_budget.py'):
        assert name in m.SOURCES


def synthetic_join(m,monkeypatch):
    chart={'normal':['normal'],'origin':['origin'],'preconditioner':matrix_json(arb_mat([[1]])),
      'image_correction':['image'],'exact_weights':['1']*235}
    inp={'chart':chart,'model_inputs':{'same':True}};saved={'chart':chart.copy(),'model_inputs':{'same':True}}
    error=matrix_json(arb_mat([[arb(0)]]));box=vector_json([arb(0)]*234+[arb(1)])
    probe={'role':'probe','stage_indices':list(range(7,14)),'localization':{'exact_endpoints':m.ENDPOINTS,'final_error':error,'final_state_box':box}}
    start={'normal':chart['normal'],'origin':chart['origin'],'preconditioner':chart['preconditioner'],'inherited_image':chart['image_correction'],'exact_endpoints':m.ENDPOINTS}
    final={'iteration':2,'retained_error':error,'jacobian':matrix_json(arb_mat([[1]])),'derivative_domain':box}
    tangent={'passed':True,'derivative_box':matrix_json(arb_mat([[int(i==234)] for i in range(235)]))}
    report={'exact_endpoints':m.ENDPOINTS,'implicit_tangent':tangent.copy()}
    monkeypatch.setattr(m.budget,'tangent_bound',lambda *a:tangent)
    return inp,saved,probe,start,final,report


def test_same_gauge_uniform_tangent_join_and_mutations(monkeypatch):
    m=driver();args=synthetic_join(m,monkeypatch)
    assert m.joined_inputs(*args)['whole_tangent'][-1]['mid']==[1,0]
    for pos,key,value in [(1,'model_inputs',{'changed':True}),(2,'stage_indices',[7]),(3,'normal',['changed']),
                         (4,'iteration',1),(5,'exact_endpoints',['0','1']), (5,'implicit_tangent',{'passed':True})]:
        args=list(synthetic_join(m,monkeypatch));args[pos]=dict(args[pos]);args[pos][key]=value
        with pytest.raises(ValueError):m.joined_inputs(*args)


def test_mixed_counter_interruption_and_precision(monkeypatch):
    m=driver();monkeypatch.setattr(m.FoldTanhMap,'__init__',lambda self,inp:None)
    monkeypatch.setattr(m.FoldTanhMap,'jet_step',lambda *a:[])
    with ctx.workprec(256):
        model=m.CountedMixedMap({})
        for _ in range(18):model.jet_step([],arb(1))
        assert model.counts()=={'attempted':18,'returned':18,'cap':18}
        with pytest.raises(ValueError):model.jet_step([],arb(1))
        failed=m.CountedMixedMap({})
        def stop(*a):raise TimeoutError('injected')
        monkeypatch.setattr(m.FoldTanhMap,'jet_step',stop)
        with pytest.raises(TimeoutError):failed.jet_step([],arb(1))
        assert failed.counts()=={'attempted':1,'returned':0,'cap':18}
    with ctx.workprec(128):
        with pytest.raises(ValueError):failed.jet_step([],arb(1))


def test_input_manifest_failure_precedes_any_read(tmp_path,monkeypatch):
    m=driver();monkeypatch.setattr(m,'entry',lambda p:{'sha256':'changed'})
    monkeypatch.setattr(m,'read',lambda *a:pytest.fail('must reject first'))
    with pytest.raises(ValueError,match='manifest'):m.authenticate(tmp_path,'approved','commit')


@pytest.mark.parametrize('change',[{'exit_code':2},{'owned_live_pids':[123]},{'elapsed_seconds':0},{'source_sha256':{'changed':'hash'}}])
def test_launcher_failure_or_identity_never_admits(tmp_path,monkeypatch,change):
    m=driver();rt,r=profile_fixture(monkeypatch,m);prior=m.read
    launcher={'mode':'profile','source_sha256':{},'exit_code':0,'owned_live_pids':[],'elapsed_seconds':2};launcher.update(change)
    monkeypatch.setattr(m,'read',lambda p:launcher if p.name=='launcher-result.json' else prior(p))
    with pytest.raises(ValueError):m.profile_gate(tmp_path,'pin','pin')


def test_outer_manifest_pin_is_required(tmp_path,monkeypatch):
    m=driver();profile_fixture(monkeypatch,m)
    with pytest.raises(ValueError,match='launcher manifest'):m.profile_gate(tmp_path,'pin','changed')


def test_real_serialization_plain_and_gzip(tmp_path):
    m=driver();value={'arithmetic':'no target','integer':1}
    plain=m.save(tmp_path/'start.json',value);packed=m.save(tmp_path/'stage.json.gz',value)
    m.save(tmp_path/'stage-payload.json',packed)
    assert m.read(tmp_path/'start.json')==m.read(tmp_path/'stage.json.gz')==value
    assert plain==packed and plain['uncompressed_bytes']>0
    assert m.entry(tmp_path/'start.json')['sha256']==plain['uncompressed_sha256']
