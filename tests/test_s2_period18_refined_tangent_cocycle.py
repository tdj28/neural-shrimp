import importlib.util
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import matrix_json,vector_json,eye
from test_rigorous_arclength_slices import StableFold


def driver():
    p=Path(__file__).parents[1]/'tools/certify_s2_period18_refined_tangent_cocycle.py'
    spec=importlib.util.spec_from_file_location('refined_mixed_driver',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_acquisition_uses_refined_total_tangent_and_native_center():
    m=driver()
    with ctx.workprec(256):
        inp={'whole_box':vector_json([arb('1/2','1/64'),arb('5/4','1/64')]),'whole_tangent':vector_json([arb(1),arb(1)]),
          'center_J':[matrix_json(arb_mat([[0]]))],'bases':[matrix_json(eye(1))],'halfwidth':'1/64','exact_endpoints':['-1/64','1/64']}
        events=[];result=m.acquire(StableFold(),inp,observe=lambda k,r:events.append((k,r)))
        assert result['passed'] and len(events)==7
        phase=next(r for k,r in events if k=='refined_phase')
        assert phase['whole_total_Jprime'][0][0]['mid']==[-1,1]
        assert phase['center_J']==inp['center_J'][0]


def test_invalid_refined_tangent_rejected_before_jet():
    m=driver();inp={'whole_box':vector_json([arb(0),arb(1)]),'whole_tangent':vector_json([arb('nan'),arb(1)]),'center_J':[matrix_json(eye(1))],'bases':[matrix_json(eye(1))]}
    with pytest.raises(ValueError,match='finite'):m.acquire(StableFold(),inp)


def fixture(monkeypatch,m):
    rt={'mode':'profile','source_sha256':{},'python':m.platform.python_version(),'python_flint':m.io.flint.__version__,'flint':m.io.flint.__FLINT_VERSION__,'numpy':m.io.numpy.__version__,'scipy':m.io.scipy.__version__,
      'precision':256,'internal_seconds':10,'mixed_call_cap':18,'model_C1_calls':0,'threads':dict.fromkeys(m.THREADS,'1'),'nice':10}
    out={'exit_code':0,'passed':True,'stages':75,'mixed_counts':{'attempted':18,'returned':18,'cap':18},'elapsed_seconds':1,'max_rss_kib':100}
    launcher={'mode':'profile','source_sha256':{},'exit_code':0,'owned_live_pids':[],'elapsed_seconds':2,'max_group_rss_kib':200}
    monkeypatch.setattr(m.io,'entry',lambda p:{'sha256':'pin'});monkeypatch.setattr(m.io,'verify_inventory',lambda *a:None);monkeypatch.setattr(m.io,'total',lambda p:100);monkeypatch.setattr(m,'sources',lambda:{})
    monkeypatch.setattr(m.io,'read',lambda p:{'manifest.json':{'files':{}},'runtime-start.json':rt,'outcome.json':out,'launcher-result.json':launcher}[p.name])
    return rt,out,launcher


@pytest.mark.parametrize('key,value',[('passed',False),('stages',74),('mixed_counts',{'attempted':18,'returned':17,'cap':18}),('elapsed_seconds',0),('elapsed_seconds',float('nan')),('elapsed_seconds',7),('max_rss_kib',512*1024)])
def test_profile_rejects_incomplete_or_bad_resources(tmp_path,monkeypatch,key,value):
    m=driver();rt,out,launch=fixture(monkeypatch,m);assert m.profile_gate(tmp_path,'pin','pin')['projected_seconds']==9;out[key]=value
    with pytest.raises(ValueError):m.profile_gate(tmp_path,'pin','pin')


def test_profile_all_source_library_and_ownership_pins(tmp_path,monkeypatch):
    m=driver();rt,out,launch=fixture(monkeypatch,m)
    for w,o in [('changed','pin'),('pin','changed')]:
        with pytest.raises(ValueError):m.profile_gate(tmp_path,w,o)
    launch['owned_live_pids']=[1]
    with pytest.raises(ValueError):m.profile_gate(tmp_path,'pin','pin')
    launch['owned_live_pids']=[];rt['flint']='changed'
    with pytest.raises(ValueError):m.profile_gate(tmp_path,'pin','pin')


def test_same_actual_refinement_proof_and_native_sources():
    m=driver()
    for name in ('src/neural_shrimp/rigorous_tangent_variation.py','src/neural_shrimp/rigorous_norm_cocycle.py','src/neural_shrimp/rigorous_folds.py',m.CARD):assert name in m.SOURCES
    with ctx.workprec(256):
        inp=m.profile_input();assert len(inp['whole_box'])==235 and len(inp['center_J'])==18
        assert inp['halfwidth']=='1/2048' and inp['scientific_claim'] is False
