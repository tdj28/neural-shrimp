import importlib.util
from pathlib import Path
import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]/'codex/return-map-crosscheck'
def load(name):
    s=importlib.util.spec_from_file_location('task19_'+name,HERE/(name+'.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
R=load('reduce');T=load('trajectory');W=load('runtime')


def test_svd_quadratic_and_maximum():
    x=np.linspace(.03,.97,201);y=4*x*(1-x)
    for basis in ('power','chebyshev'):
        f=R.details(R.polynomial(x,y,2,basis),x,y)
        np.testing.assert_allclose(R.evaluate(f,x),y,atol=2e-15)
        np.testing.assert_allclose(R.evaluate(f,x,1),4-8*x,atol=1e-13)
        assert f['rank_full'] and f['stationary_count']==1
        assert f['critical_points'][0]['kind']=='quadratic_maximum'
        assert abs(f['critical_points'][0]['x']-.5)<1e-14
        assert abs(f['critical_points'][0]['curvature']+8)<1e-12


def test_minimum_not_counted_as_maximum():
    x=np.linspace(-1,1,101);f=R.details(R.polynomial(x,x*x,2),x,x*x)
    assert f['stationary_count']==1 and f['fitted_quadratic_maximum_count']==0
    assert f['critical_points'][0]['kind']=='quadratic_minimum'


def test_purged_pairs_never_share_scalar_endpoints():
    for train,test in R.folds(71,'purged_contiguous'):
        tr=set(train)|set(train+1);te=set(test)|set(test+1)
        assert tr.isdisjoint(te)
    assert np.array_equal(np.sort(np.concatenate([t for _,t in R.folds(71,'random')])),np.arange(71))


def test_every_cv_candidate_and_fold_retained():
    x=np.linspace(.05,.95,120);y=3.8*x*(1-x)
    for kind in ('random','purged_contiguous'):
        c=R.crossvalidate(x,y,kind)
        assert [r['degree'] for r in c['candidates']]==list(range(4,20,2))
        assert all(len(r['folds'])==5 for r in c['candidates'])
        assert c['qualified_value_fit']


def test_short_constant_invalid_and_nonfinite_retained_or_refused():
    assert R.phase_fits(np.full(100,.5),4)[0]['disposition']=='insufficient_repetitions'
    assert R.phase_fits(np.full(200,.5),4)[0]['disposition']=='constant_coordinate'
    with pytest.raises(ValueError):R.phase_fits(np.array([np.nan]*200),4)
    with pytest.raises(ValueError):R.phase_fits(np.full(200,1.1),4)


def test_sample_distance_is_not_centroid_distance():
    states=np.array([[0.,0.],[4.,0.],[2.,0.],[6.,0.]])
    b=R.sample_bands(states,2,block=1)
    assert b['pairs'][0]['sample_distance']==2
    assert b['pairs'][0]['centroid_distance']==4
    assert b['pairs'][0]['bounding_ball_gap']==2
    assert b['pairs'][0]['sample_indices']==[2,1]


def test_twin_affine_growth_clock_and_retention():
    events=[]
    def step(x):
        return dict(state=2*x,lead=np.zeros(len(x),dtype=np.int64),
                    pmax=np.full(len(x),.75),entropy=x[:,0].copy())
    def sink(*args):events.append(args)
    result=T.advance(step,np.array([.01,.02]),sink,transient=1,record=8)
    np.testing.assert_allclose(result['lyapunov'],np.log(2),atol=2e-6)
    original=next(e[3]['state'] for e in events if e[:3]==('input','record',0))
    returned=next(e[3] for e in events if e[:3]==('return','record',0))
    np.testing.assert_array_equal(returned['entropy'],original[:,0])
    np.testing.assert_array_equal(returned['state'],2*original)
    assert result['row_forwards']==25 and len(result['disjoint_windows'])==4


def test_completed_nonfinite_and_thrown_native_keep_input():
    events=[]
    def bad(x):return dict(state=np.full_like(x,np.nan),lead=np.zeros(len(x),dtype=np.int64),pmax=np.ones(len(x)),entropy=np.ones(len(x)))
    with pytest.raises(ValueError):T.advance(bad,np.array([1.,2.]),lambda *e:events.append(e),1,4)
    assert [e[0] for e in events]==['input','return'] and np.isnan(events[-1][3]['state']).all()
    events=[]
    def thrown(x):raise RuntimeError('native error')
    with pytest.raises(RuntimeError):T.advance(thrown,np.array([1.,2.]),lambda *e:events.append(e),1,4)
    assert len(events)==1 and events[0][0]=='input'


def test_period_and_finite_lag_only():
    assert R.symbolic_period(np.tile(np.arange(7),30))==7
    a=np.tile(np.arange(7),30);a[-1]=100;assert R.symbolic_period(a)==0
    states=np.column_stack([np.tile(np.arange(7),30),np.ones(210)])
    assert T.state_diagnostics(states)['period']==7


def test_aggregate_full_twin_values_do_not_change_phase_selection():
    seq=[.217]
    for _ in range(299):seq.append(3.9*seq[-1]*(1-seq[-1]))
    rows=R.phase_fits(np.asarray(seq),1)
    a=R.aggregate(rows,1,.5);b=R.aggregate(rows,1,17.)
    for method in a:
        assert a[method]['phases']==b[method]['phases']
        assert a[method]['estimate']==b[method]['estimate']


def test_complete65_gate_rejects_order_missing_nonfinite():
    cells=[[i/1000,.0655] for i in range(65)]
    info=[dict(log10_temperature=c[0],alpha=c[1],lyapunov=[.1,.1],eps=[1e-6,1e-9],period=0,lead_period=1) for c in cells]
    lead=np.zeros((2000,65),dtype=np.int64);p=np.full((2000,65),.5)
    W.admit_sequences(cells,info,lead,p)
    with pytest.raises(ValueError):W.admit_sequences(cells[::-1],info,lead,p)
    with pytest.raises(ValueError):W.admit_sequences(cells[:-1],info,lead,p)
    lead[0,0]=151936
    with pytest.raises(ValueError):W.admit_sequences(cells,info,lead,p)
    lead[0,0]=0;info[0]['period']=1.5
    with pytest.raises(ValueError):W.admit_sequences(cells,info,lead,p)
    info[0]['period']=0
    p[0,0]=np.nan
    with pytest.raises(ValueError):W.admit_sequences(cells,info,lead,p)


def test_recording_retains_nonfinite_and_inputs(tmp_path):
    rec=W.Recording(tmp_path,lambda:None,transient=1,record=4,dimension=2)
    rec('input','transient',0,{'state':np.ones((1,2))})
    rec('return','transient',0,dict(state=np.full((1,2),np.nan),lead=np.array([1]),pmax=np.array([.5]),entropy=np.array([1.])))
    rec.flush();rec.journal.close()
    np.testing.assert_array_equal(np.load(tmp_path/'transient-input.npy'),np.ones((1,1,2)))
    assert np.isnan(np.load(tmp_path/'transient-state.npy')).all()
    assert len((tmp_path/'events.jsonl').read_text().splitlines())==2


def test_plan_refuses_unreviewed_changed_science_and_caps():
    plan=dict(schema='codex-return-map-v1',admission='reviewed',mode='trajectory',envelope=dict(W.CAPS['trajectory']),
        source_sha256={s:'a'*64 for s in W.SOURCES},input_manifest_sha256=W.AUTHORITY_SHA,scope='test',
        experiment=dict(alpha=W.ALPHA,log10_temperature=W.LOG_T,transient=1000,record=2000,offsets=[1e-6,1e-9],seed=0,primary_phase=6,maximum_period=64,threshold=1e-8))
    assert W.validate(plan)==plan
    import copy
    for field,value in [('admission','prepared'),('envelope',{}),('source_sha256',{})]:
        bad=copy.deepcopy(plan);bad[field]=value
        with pytest.raises(ValueError):W.validate(bad)
    bad=copy.deepcopy(plan);bad['experiment']['primary_phase']=5
    with pytest.raises(ValueError):W.validate(bad)


def test_all_leaf_differences_retained():
    rows=W.compare_values({'x':[1.,2.],'y':None},{'x':[1.,2.00001],'y':3.})
    assert len(rows)==3 and rows[1]['difference']!=0 and rows[2]['own'] is None


def test_rank_warning_does_not_silently_change_historical_selection():
    x=np.linspace(.03,.97,101);f=R.details(R.polynomial(x,4*x*(1-x),2),x,4*x*(1-x))
    f['rank_full']=False
    row=dict(phase=0,disposition='fitted',original=f)
    for method in ('amended_random','purged_contiguous'):
        row[method]=dict(fit=f,producer_value_fit_qualified=True,qualified_value_fit=False)
    summary=R.aggregate([row],1,.5)
    assert summary['amended_random']['phases']==[0]
    assert summary['amended_random']['rank_warning_phases']==[0]
    assert summary['purged_contiguous']['phases']==[]


def test_physical_imaginary_tolerance_is_only_historical_chebyshev_count():
    f=dict(basis='chebyshev',degree=3,shift=.25,scale=1e-5,
        coefficients=[0.,.25+1e-14,0.,1/12],rank=4,rank_full=True,singular_values=[1.]*4,condition=1.,points=101)
    x=np.linspace(.25-1e-5,.25+1e-5,101);y=R.evaluate(f,x)
    d=R.details(f,x,y)
    assert d['stationary_count']==0 and d['producer_stationary_count']==2
    assert d['fitted_quadratic_maximum_count']==0
