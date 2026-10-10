import copy,importlib.util
from pathlib import Path
import pytest
from flint import ctx
from test_eigenpair_reanchor import parent
from neural_shrimp.eigenpair_gauge_diagnostics import saved_predictor_diagnostics
from neural_shrimp.rigorous_eigenpair_march import decode


def driver():
    p=Path(__file__).parents[1]/'tools/continue_eigenpair_tangent_chain.py';s=importlib.util.spec_from_file_location('tangent_driver',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_saved_rate_projection_separate_from_critical_spectrum():
    with ctx.workprec(256):
        _,r=parent();d=saved_predictor_diagnostics(r);assert len(d['rows'])==2
        assert all(decode(v['full_norm'])>0 for v in d['rows'])
        z={'mid':[0,0],'rad':[0,0]};rr=copy.deepcopy(r);rr['slope'][-2:]=[z,z];rr['curvature'][-2:]=[z,z]
        out=saved_predictor_diagnostics(rr);assert all(v['small_rate_projection_flag'] for v in out['rows'])
        assert 'not a validated tangent' in out['scope'] and 'eigenvalues' not in out


def test_cost_uses_observed_larger_memory_and_rejects_bad_controls():
    m=driver();a={'passed':True,'status':'two_reanchored_charts','ticks':326,'elapsed_seconds':65.,'peak_rss_kib':1570596,'stored_bytes_before_final':140087544};p={'passed':True,'native_calls':0,'elapsed_seconds':1.,'peak_rss_kib':100000}
    r=m.cost_gate(a,p);assert r['seconds']<300 and r['rss_kib']==1570596+128*1024
    for bad in ({'ticks':325},{'elapsed_seconds':-1},{'peak_rss_kib':float('nan')},{'stored_bytes_before_final':float('inf')},{'passed':False}):
        with pytest.raises(ValueError):m.cost_gate(dict(a,**bad),p)
    with pytest.raises(ValueError):m.cost_gate(a,dict(p,native_calls=1))


def test_complete_endpoint_and_gauge_cannot_change():
    m=driver()
    with ctx.workprec(256):
        _,r=parent();r['parameter_endpoints']=['9/128','11/128'];r['settings']['halfwidth_exponent']=-7
        c=m.compact(r);last={'call':3,'index':3,'right':'11/128'};out={'passed':True,'status':'four_joined_charts','calls':4,'progress':{'accepted':[last]*4,'last_accepted':last,'active_call':None}}
        result={'completed':True,'accepted':[{}]*4,'calls':4,'last_chart':c};accepted={'call':3,'index':3,'chart':c}
        assert m.validate_parent(out,result,accepted,r,'11/128',c)==c
        for bad in ({'completed':False},{'calls':6}):
            with pytest.raises(ValueError):m.validate_parent(out,dict(result,**bad),accepted,r,'11/128',c)
        bad=copy.deepcopy(c);bad['normal'][0]='1'
        with pytest.raises(ValueError,match='gauge'):m.validate_parent(out,result,accepted,r,'11/128',bad)
        with pytest.raises(ValueError,match='endpoint'):m.validate_parent(out,result,accepted,r,'13/128',c)


def test_full_provenance_and_old_core_unchanged():
    m=driver();root=Path(__file__).parents[1]
    assert set(m.R.SOURCES)<=set(m.SOURCES) and len(m.SOURCES)==len(set(m.SOURCES))
    assert all((root/v).is_file() for v in m.SOURCES)
    assert 'src/neural_shrimp/controls.py' in m.SOURCES
    s=(root/'tools/continue_eigenpair_tangent_chain.py').read_text();assert "observer(kind,value)" in s and 'tangent_projection_return' in s
    assert 'return M.march(model,parent' in s and "args.expected_right==str(Fraction(8*(args.unit-1)+3,128))" in s
