"""Tiny source/metadata controls only; no target files, model or Git."""
import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

HERE=Path(__file__).resolve().parent
ADAPTER=HERE/'stress_runtime.py' if (HERE/'stress_runtime.py').exists() else HERE.parent/'codex/return-map-crosscheck/stress_runtime.py'
spec=importlib.util.spec_from_file_location('task19_stress_test',ADAPTER)
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
# Literal reviewed source supplies only the AST equality control, never execution.
REPO=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp')
BASE=ADAPTER.parent/'runtime.py' if (ADAPTER.parent/'runtime.py').exists() else REPO/'codex/return-map-crosscheck/runtime.py'


def plan(index=4):
    log_t,q,_=S.cell(index)
    return dict(schema='codex-return-map-stress-v1',admission='reviewed',mode='stress',cell_index=index,
        envelope=dict(S.CAPS),source_sha256={**S.BASE_PINS,S.SELF:'a'*64},input_manifest_sha256=S.AUTHORITY_SHA,
        experiment=dict(alpha=S.ALPHA,log10_temperature=log_t,transient=1000,record=2000,
            offsets=[1e-6,1e-9],seed=0,expected_producer_lead_period=q,maximum_period=64,threshold=1e-8),scope='synthetic')


def metadata(index=4):
    log_t,q,_=S.cell(index)
    doc={'cells':[[0.,0.]]*65};info={'cells':[{} for _ in range(65)]}
    doc['cells'][index]=[log_t,S.ALPHA]
    info['args']=dict(transient=1000,record=2000,eps=[1e-6,1e-9],model='qwen0.5b',concept='hostility-vs-kindness',layer=12)
    info['cells'][index]=dict(alpha=S.ALPHA,log10_temperature=log_t,lead_period=q,period=0,eps=[1e-6,1e-9],lyapunov=[.01,.02])
    return doc,info


@pytest.mark.parametrize('index',[4,13])
def test_exact_two_cell_plan_and_separate_output(index):
    p=plan(index);assert S.validate(p)==p
    path=S.output_path(index)
    assert str(path).endswith(f'/stress-cell{index:02d}-001/run')
    assert S.output_path(4)!=S.output_path(13)
    assert path!=S.OWNED/'trajectory-001/run' and path!=S.OWNED/'saved-001/run'


@pytest.mark.parametrize('index',[0,5,True,4.,'4',None])
def test_nonallowlisted_or_inexact_index_refuses(index):
    with pytest.raises(ValueError):S.cell(index)


def test_plan_refuses_prepared_and_any_changed_science_or_source():
    mutations=[('admission','prepared'),('mode','trajectory'),('schema','codex-return-map-v1')]
    for key,value in mutations:
        p=plan();p[key]=value
        with pytest.raises(ValueError):S.validate(p)
    for key,value in [('log10_temperature',.18),('expected_producer_lead_period',7),('seed',1),('record',1000),('primary_phase',6)]:
        p=plan();p['experiment'][key]=value
        with pytest.raises(ValueError):S.validate(p)
    p=plan();p['envelope']['timeout_seconds']=4201
    with pytest.raises(ValueError):S.validate(p)
    p=plan();p['source_sha256'][next(iter(S.BASE_PINS))]='f'*64
    with pytest.raises(ValueError):S.validate(p)
    p=plan();p['source_sha256'].pop(S.SELF)
    with pytest.raises(ValueError):S.validate(p)


@pytest.mark.parametrize('index',[4,13])
def test_identity_binds_exact_original_row_before_native(index):
    doc,info=metadata(index);r=S.bind_cell(doc,info,index)
    assert r['index']==index and r['expected_period_is_comparison_only']
    bad=copy.deepcopy(info);bad['cells'].pop()
    with pytest.raises(ValueError,match='missing_or_incomplete'):S.bind_cell(doc,bad,index)
    bad=copy.deepcopy(info);bad['cells'][index]['lead_period']=7
    with pytest.raises(ValueError,match='invalid_frozen'):S.bind_cell(doc,bad,index)
    bad=copy.deepcopy(doc);bad['cells'][index]=[S.cell(13 if index==4 else 4)[0],S.ALPHA]
    with pytest.raises(ValueError,match='invalid_frozen'):S.bind_cell(bad,info,index)
    bad=copy.deepcopy(info);bad['cells'][index]={}
    with pytest.raises(ValueError,match='invalid_frozen'):S.bind_cell(doc,bad,index)
    bad=copy.deepcopy(info);bad['args']['record']=1000
    with pytest.raises(ValueError,match='invalid_original_trajectory_recipe'):S.bind_cell(doc,bad,index)
    bad=copy.deepcopy(info);bad['cells'][index]['lyapunov'][0]=float('nan')
    with pytest.raises(ValueError,match='invalid_original_optional_twin_comparison'):S.bind_cell(doc,bad,index)


def test_fits_use_only_observed_period_and_all_dispositions(tmp_path):
    calls=[];saved=[]
    rows=[dict(phase=i,disposition='constant_coordinate') for i in range(3)]
    def fits(p,q):calls.append(('fits',q,p));return rows
    def aggregate(r,q,full):calls.append(('aggregate',q,full));assert r is rows;return {'actual':q}
    base=SimpleNamespace(write=lambda path,value:saved.append((path,value)))
    reducer=SimpleNamespace(phase_fits=fits,aggregate=aggregate)
    # Observed3 is deliberately neither frozen producer10 nor28: no forcing.
    got,summary=S.fit_rows(base,reducer,[.5]*100,3,-.1,tmp_path)
    assert got is rows and summary=={'actual':3}
    assert [c[1] for c in calls]==[3,3] and saved[0][1] is rows
    calls.clear();saved.clear()
    assert S.fit_rows(base,reducer,[.5]*100,0,.2,tmp_path)==([],None)
    assert calls==[] and saved==[]


def test_comparison_projection_retains_historical_rank_rule():
    rows=[]
    for phase,residual in [(0,.02),(1,.001)]:
        fit=dict(degree=8,points=199,residual_relative=residual,producer_floor_exponent=.4,
            stationary_count=1,producer_stationary_count=2,rank_full=False)
        rows.append(dict(phase=phase,disposition='fitted',original=fit,
            amended_random=dict(producer_value_fit_qualified=(phase==0),qualified_value_fit=False,
                selected_degree=4,cv_residual_relative=.005,fit=fit)))
    summary={'original':dict(estimate=.04,full_twin_mean=.03,relative_discrepancy=1/3),
        'amended_random':dict(exponents_per_step=[.04],estimate=.04,spread=0.,full_twin_mean=.03,relative_discrepancy=1/3)}
    forms=S.producer_forms(rows,summary)
    assert forms['original']['best_phase']==1
    assert forms['amended']['single_valued_phases']==[0] # no new rank exclusion
    assert forms['amended']['unimodal_phases']==0 # physical producer stationary2
    assert S.producer_forms([],None)=={}


def test_comparison_binds_index_and_preserves_missing_and_unqualified():
    compare=lambda own,old:[dict(path='/x',own=own['x'],producer=old['x'],difference=own['x']-old['x'])]
    base=SimpleNamespace(compare_values=compare)
    log_t,_,_=S.cell(4)
    data={'rows':[{} for _ in range(65)]};data['rows'][4]=dict(log10_temperature=log_t,alpha=S.ALPHA,x=2.,amended=None)
    r=S.compare_row(base,data,4,{'original':{'x':1.}})
    assert r['rows'][0]['difference']==-1. and r['rows'][1]['disposition']=='producer_missing'
    r=S.compare_row(base,data,4,{})
    assert r['rows'][0]['disposition']=='own_unqualified'
    with pytest.raises(ValueError,match='missing_or_incomplete'):S.compare_row(base,{'rows':[]},4,{})
    with pytest.raises(ValueError,match='invalid_original'):S.compare_row(base,data,13,{})


def test_native_map_and_retention_block_is_identical_to_mandatory_source():
    old=ast.parse(BASE.read_text());new=ast.parse(ADAPTER.read_text())
    a=next(x for x in old.body if isinstance(x,ast.FunctionDef) and x.name=='run_trajectory')
    b=next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name=='run_trajectory')
    def native(fn):
        start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Import) and n.names[0].name=='numpy')
        stop=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='T.advance')
        return [ast.dump(n,include_attributes=False) for n in fn.body[start:stop+1]]
    assert native(a)==native(b)
    # The new adapter calls unchanged twin/reducer helpers, and cannot accidentally
    # compare the original trace-best cell or introduce mandatory phase6/seven fits.
    text=ADAPTER.read_text()
    for forbidden in ('trace-best','primary_phase','fixed_phase6','primary_seven','phase_fits(pmax,7)'):
        assert forbidden not in text
