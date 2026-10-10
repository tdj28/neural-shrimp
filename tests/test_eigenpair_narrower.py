from fractions import Fraction
import pytest
from flint import ctx
from test_eigenpair_reanchor import parent
from neural_shrimp.eigenpair_narrower import narrower_inputs,narrow
from neural_shrimp.rigorous_eigenpair_march import decode


@pytest.mark.parametrize('sigma',[-1,1])
def test_halved_width_starts_exactly_at_parent_and_reacquires_native_proof(sigma):
    with ctx.workprec(256):
        m,c=parent(sigma);initial,kw=narrower_inputs(c);old=Fraction(2)**c['settings']['halfwidth_exponent']
        assert Fraction(kw['parameter'])-old/2==Fraction(c['parameter_endpoints'][1])
        assert kw['weights']==c['weights'] and kw['normal']==c['normal'] and kw['origin']==c['origin']
        assert all(a.contains(decode(b)) for a,b in zip(kw['anchor']['box'],c['endpoint_root_boxes'][1]))
        ticks=[];events=[];r=narrow(m,c,tick=lambda:ticks.append(1),observe=lambda k,v:events.append(k))
        assert r['passed'] and len(ticks)==30 and events.count('path_phase_return')==12 and events.count('phase_return')==16
        assert r['charts'][0]['parameter_endpoints'][0]==c['parameter_endpoints'][1]
        assert r['charts'][0]['parameter_endpoints'][1]==r['charts'][1]['parameter_endpoints'][0]
        assert all(v['sigma']==sigma and all(v['gates'].values()) for v in r['charts'])


def test_unqualified_or_inconsistent_parent_cannot_launch():
    with ctx.workprec(256):
        _,c=parent();c['parameter_endpoints'][1]='123'
        with pytest.raises(ValueError):narrower_inputs(c)
        _,c=parent();c['passed']=False
        with pytest.raises(ValueError):narrower_inputs(c)


def driver():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).parents[1]/'tools/narrow_eigenpair_locus.py'
    spec=importlib.util.spec_from_file_location('narrow_driver',path);d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d);return d


def test_complete_source_inventory_and_measured_cost():
    from pathlib import Path
    d=driver();assert set(d.R.SOURCES)<=set(d.SOURCES)
    assert all((Path(__file__).parents[1]/n).is_file() for n in d.SOURCES)
    old=dict(passed=True,ticks=127,status='one_repaired_partial_chart',elapsed_seconds=21.16,peak_rss_kib=656452,stored_bytes_before_final=44468293)
    assert d.cost_gate(old)['seconds']<150
    old['elapsed_seconds']=100
    with pytest.raises(ValueError):d.cost_gate(old)


@pytest.mark.parametrize('defect',[None,'source_gauge','failed','inclusion','anchor','divisor','full_identity','domain'])
def test_complete_parent_binding(defect):
    import copy
    from test_eigenpair_partial import fixture
    from neural_shrimp.rigorous_eigenpair_march import compact
    d=driver();_,_,_,old,_,_=fixture();full=copy.deepcopy(old)
    full.update(schema='neural-shrimp-quadratic-eigenpair-locus-v1',parameter_center='221/128',parameter_endpoints=['219/128','223/128'],center=[],slope=[],curvature=[],weights=[],endpoint_root_boxes=[],coordinate_inclusions=[True]*470,anchor_check={'passed':True,'coordinate_inclusions':[True]*470},divisors=[{'divisor':q,'excluded':True}for q in [1,2,3,6,9]])
    result={'passed':True,'charts':[compact(full)]}
    if defect=='source_gauge':old['normal']=['other']
    if defect=='failed':full['passed']=False
    if defect=='inclusion':full['coordinate_inclusions'][364]=False
    if defect=='anchor':full['anchor_check']['coordinate_inclusions'][3]=False
    if defect=='divisor':full['divisors'][3]['excluded']=False
    if defect=='full_identity':result['charts']=[]
    if defect=='domain':full['parameter_endpoints']=['223/128','227/128']
    if defect is None:assert d.validate_parent(full,result,old)==compact(full)
    else:
        with pytest.raises(ValueError):d.validate_parent(full,result,old)
