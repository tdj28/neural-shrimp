import importlib.util
from pathlib import Path
import pytest
from flint import arb, arb_mat, ctx
from neural_shrimp.rigorous_connection import restore_ball
from neural_shrimp.rigorous_connection_frames import restore_matrix


def module():
    p=Path(__file__).parents[1]/'tools/analyze_s2_period18_slice_budget.py'
    s=importlib.util.spec_from_file_location('slice_budget',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_implicit_derivative_orientation_and_interval_retention():
    with ctx.workprec(256):
        m=module();a=arb_mat([[arb(1,'1/100'),-1],[1,0]])
        pre=arb_mat([[0,1],[-1,1]])
        r=m.tangent_bound(pre,a,[arb(1),arb(1)])
        assert r['passed'] and len(r['refinements'])==3
        y=restore_matrix(r['derivative_box'])
        assert y[0,0].contains(1)
        assert y[1,0].contains(arb('99/100')) and y[1,0].contains(arb('101/100'))


def test_no_inverse_claim_when_neumann_fails():
    r=module().tangent_bound(arb_mat([[1]]),arb_mat([[-1]]),[arb(1)])
    assert not r['passed'] and 'derivative_box' not in r


@pytest.mark.parametrize('w',[arb(0),arb(-1),arb(1,1)])
def test_invalid_weights_rejected(w):
    with pytest.raises(ValueError,match='weights'):module().tangent_bound(arb_mat([[1]]),arb_mat([[1]]),[w])


def test_zero_j_width_has_zero_error_pieces():
    r=module().phase_budget(arb_mat([['1/2']]),arb_mat([[2]]),arb_mat([[2]]),arb_mat([[1]]))
    for key in ('left_linear','right_linear','quadratic'):
        assert restore_ball(r['unscaled_piece_inf'][key])==0
    assert restore_ball(r['unscaled_piece_inf']['point_D']).contains(arb('3/2'))


def test_nonzero_j_width_retains_linear_and_quadratic_terms():
    r=module().phase_budget(arb_mat([[arb('1/2','1/100')]]),arb_mat([[2]]),arb_mat([[2]]),arb_mat([[1]]))
    assert restore_ball(r['unscaled_piece_inf']['left_linear'])>0
    assert restore_ball(r['unscaled_piece_inf']['right_linear'])>0
    assert restore_ball(r['unscaled_piece_inf']['quadratic'])>0


def test_two_saved_cases_reach_finalization_with_dictionary_stage(tmp_path,monkeypatch):
    m=module();monkeypatch.setattr(m,'SOURCES',[])
    monkeypatch.setattr(m.os,'getpriority',lambda *args:10)
    monkeypatch.setattr(m.subprocess,'check_output',lambda args,**kw:'' if args[1]=='status' else '0'*40)
    original_entry=m.io.entry
    monkeypatch.setattr(m.io,'entry',lambda p:{'sha256':m.PIN} if p==m.PARENT/'manifest.json' else original_entry(p))
    monkeypatch.setattr(m.io,'verify_inventory',lambda *args:None)
    def read(p):
        if p.name=='manifest.json':return {'files':{}}
        if p.name=='runtime-start.json':return {'source_commit':m.EXECUTION,'mode':'target','source_sha256':{}}
        if p.name=='outcome.json':return {'exit_code':0,'completed_cases':2}
        if p.name=='inputs.json.gz':return {'chart':{}}
        if p.name.endswith('-result.json.gz'):
            role=p.name.split('-')[0];offset=0 if role=='floor' else 7
            return {'role':role,'stage_indices':list(range(offset,offset+7))}
        return {'synthetic_stage':True}
    monkeypatch.setattr(m.io,'read',read)
    monkeypatch.setattr(m,'diagnose',lambda result,start,final,chart,tick:{'implicit_tangent':{'passed':True}})
    out=tmp_path/'run'
    assert m.execute(out)==0
    import json
    result=json.loads((out/'outcome.json').read_text())
    assert result['elapsed_seconds']>=0 and len(result['cases'])==2
    assert (out/'manifest.json').is_file()
