import importlib.util
from pathlib import Path
from copy import deepcopy
from fractions import Fraction
import pytest
from test_s2_period18_fold_march import chart


def module():
    p=Path(__file__).parents[1]/'tools/certify_s2_period18_fold_endpoint.py'
    s=importlib.util.spec_from_file_location('fold_endpoint',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def fixture():
    before=chart();before.update(center=[],parameter_center='1/64',predictor_slope=[],predictor_curvature=[])
    previous=before;rows=[];files={}
    for index in range(2):
        c=deepcopy(before);shared=previous['exact_parameter_endpoints'][1];end=str(Fraction(shared)+Fraction(1,64))
        c['exact_parameter_endpoints']=[shared,end];c['parameter_center']=str(Fraction(shared)+Fraction(1,128))
        c['anchor']={'state_box':deepcopy(previous['endpoint_root_enclosures'][1]),'exact_parameter':shared}
        c['known_fold_membership']={'all_cyclic_alignments':[{'contained_in_qualified_chart':False} for _ in range(18)]}
        name=f'chart-{index:03d}.json.gz';files[name]=c
        rows.append({'index':index,'artifact':{'path':name},'passed':True,'gates':c['gates'].copy(),'known_fold_identified':False,'reboxed':False})
        previous=c
    return {'previous':before},rows,files


def test_only_full_durable_prefix_is_returned():
    inp,rows,files=fixture();last,finals,initials=module().validate_prefix(inp,rows,lambda n,i:files[n],expected=2)
    assert last['exact_parameter_endpoints'][1]=='7/128'
    assert finals==['chart-000.json.gz','chart-001.json.gz'] and not initials


@pytest.mark.parametrize('kind',['count','index','gates','tail','phase','gauge','rebox','fold'])
def test_durable_prefix_corruption_rejected(kind):
    inp,rows,files=fixture();c=files['chart-001.json.gz']
    if kind=='count':rows.pop()
    if kind=='index':rows[-1]['index']=2
    if kind=='gates':rows[-1]['gates']['uniform_root']=False
    if kind=='tail':c['passed']=False
    if kind=='phase':c['anchor']['state_box'][0]['mid']=[1,0]
    if kind=='gauge':c['origin'][0]='1'
    if kind=='rebox':rows[-1]['reboxed']=True
    if kind=='fold':rows[-1]['known_fold_identified']=True
    with pytest.raises((ValueError,KeyError)):
        module().validate_prefix(inp,rows,lambda n,i:files[n],expected=2)


def test_rebox_requires_retained_unchanged_predictor():
    inp,rows,files=fixture();c=files['chart-001.json.gz']
    for key in ('preconditioner','predictor_residual','center_path_derivatives','whole_path_third_derivative'):c[key]=[]
    old=deepcopy(c);old['gates']['uniform_root']=False
    files['initial-001.json.gz']=old;c['reboxing']={'initial_artifact':{'path':'initial-001.json.gz'}};rows[-1]['reboxed']=True
    module().validate_prefix(inp,rows,lambda n,i:files[n],expected=2)
    old['center']=['tampered']
    with pytest.raises(ValueError,match='predictor'):module().validate_prefix(inp,rows,lambda n,i:files[n],expected=2)
