from copy import deepcopy
import importlib.util
from pathlib import Path
import pytest
from flint import arb,ctx
from test_arclength_gauge_diagnostic import chart
from test_s2_period18_fold_endpoint import fixture as old_fixture


def module():
    p=Path(__file__).parents[1]/'tools/analyze_s2_period18_near_fold_gauge.py'
    s=importlib.util.spec_from_file_location('near_fold_gauge',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def fixture():
    inp,rows,files=old_fixture()
    for c in [inp['previous'],*files.values()]:c['settings']['halfwidth_exponent']=-8
    return inp,rows,files


def test_prefix_complete_native_join():
    inp,rows,files=fixture();last,finals,initials=module().validate_prefix(inp,rows,lambda n,i:files[n],expected=2)
    assert len(finals)==2 and not initials and last['passed']

@pytest.mark.parametrize('kind',['count','index','tail','phase','gauge','rebox','fold'])
def test_prefix_rejects_corruption(kind):
    inp,rows,files=fixture();c=files['chart-001.json.gz']
    if kind=='count':rows.pop()
    if kind=='index':rows[-1]['index']=2
    if kind=='tail':c['passed']=False
    if kind=='phase':c['anchor']['state_box'][0]['mid']=[1,0]
    if kind=='gauge':c['origin'][0]='1'
    if kind=='rebox':rows[-1]['reboxed']=True
    if kind=='fold':rows[-1]['known_fold_identified']=True
    with pytest.raises((ValueError,KeyError)):module().validate_prefix(inp,rows,lambda n,i:files[n],expected=2)


def test_proposal_uses_whole_boxes_and_exact_dyadics():
    with ctx.workprec(256):
        m=module();rows=m.proposals(chart(),[arb(4,arb('1/128')),arb(5,arb('1/128'))])
        assert [r['label'] for r in rows]==['inherited','endpoint_taylor','target_informed_secant']
        for row in rows:
            assert all(m.exact(v).is_exact() for v in row['normal']+row['origin'])
            assert m.restore_ball(row['fold_gauge_box']).rad()>0
        assert m.restore_ball(rows[-1]['endpoint_gauge_box']).rad()>0
        assert m.restore_ball(rows[-1]['fold_gauge_box'])>0


def test_zero_secant_fails_without_new_claim():
    with pytest.raises(ValueError,match='nonzero'):module().proposals(chart(),[arb(2),arb(3)])


def test_dimension_rejected():
    with pytest.raises(ValueError,match='dimensions'):module().proposals(chart(),[arb(2)])


def test_singular_border_retained():
    assert not module().condition(chart(),[arb(1),arb(0)])['invertible_saved_midpoint']
