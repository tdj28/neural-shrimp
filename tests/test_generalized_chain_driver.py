import importlib.util,copy
from pathlib import Path
import pytest
from flint import arb
p=Path(__file__).parents[1]/'tools/probe_eigenpair_generalized_chain.py';s=importlib.util.spec_from_file_location('chain_probe',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_cost_is_measured_finite_complete_and_bounded():
    good={'ticks':72,'status':'synthetic_profile_completed','evaluation_seconds':[.1,.2]}
    assert m.cost(good)==24
    for bad in [dict(good,ticks=36),dict(good,status='failed'),dict(good,evaluation_seconds=[.1,float('nan')]),dict(good,evaluation_seconds=[.1,4.1])]:
        with pytest.raises(ValueError):m.cost(bad)

def test_neighborhood_uses_actual_rates_native_order_and_bounded_r():
    z=[arb(0)]*41
    assert m.near(z,z)
    for index,change in ((0,1),(13,2),(14,arb(1)/50),(15,1),(28,1024)):
        b=list(z);b[index]=change
        assert not m.near(z,b)

def test_synthetic_profile_uses_full41d_13_native_model():
    inp,z=m.synthetic();assert m.TwoRateCuspMap(inp).n==13 and len(z)==41 and len(inp['x'])==24

def event_fixture(monkeypatch):
    c={'parameter_endpoints':['273/128','275/128'],'parameter_center':'137/64'}
    rec={'index':3,'call':3,'right':'275/128'}
    flags=['nontrivial_plus_one_gap_at_most_0.03','nontrivial_unit_circle_margin_at_most_0.03']
    out={'passed':True,'status':'qualified_event_stop','ticks':508,'progress':{'accepted':[{}, {}, {},rec],'active_call':None,'last_accepted':rec},'event_flags':flags}
    a={'chart':c,'index':3,'call':3,'diagnostics':{}}
    monkeypatch.setattr(m.B,'accepted_check',lambda full,end:c)
    monkeypatch.setattr(m.B,'event_flags',lambda diag:flags)
    return out,{},a

def test_only_exact_retained_event_is_admitted(monkeypatch):
    args=event_fixture(monkeypatch);assert m.qualified_event(*args)['parameter_center']=='137/64'

@pytest.mark.parametrize('kind',['status','count','call','right','center','compact','flags'])
def test_event_corruptions_fail_closed(monkeypatch,kind):
    args=event_fixture(monkeypatch);out,full,a=args
    if kind=='status':out['status']='four_joined_charts'
    elif kind=='count':out['progress']['accepted'].pop(0)
    elif kind=='call':a['call']=2
    elif kind=='right':out['progress']['last_accepted']['right']='273/128'
    elif kind=='center':a['chart']['parameter_center']='139/64'
    elif kind=='compact':a['chart']=dict(a['chart'],different=True)
    else:out['event_flags']=[]
    with pytest.raises(ValueError):m.qualified_event(*args)


def test_actual_launcher_selects_new_worker_and_precursor_arguments():
    p=Path(__file__).parents[1]/'tools/launch_eigenpair_generalized_chain.py';s=importlib.util.spec_from_file_location('chain_launcher',p);l=importlib.util.module_from_spec(s);s.loader.exec_module(l)
    args=m.parser().parse_args(['--mode','target','--out',str(m.BASE/'attempt001-target/run'),'--profile','/unused/profile','--profile-manifest','a'*64,'--profile-outer-manifest','b'*64,'--proposal','/unused/proposal','--proposal-manifest','c'*64,'--proposal-outer-manifest','d'*64])
    l.ARGS=args;l.OUT=m.BASE/'attempt001-target';l.LIMIT=140
    cmd=l.command();assert Path(l.C.__file__).name=='probe_eigenpair_generalized_chain.py'
    assert str(m.ROOT/'tools/probe_eigenpair_generalized_chain.py')in cmd
    assert '--unit'not in cmd and cmd[cmd.index('--proposal-manifest')+1]=='c'*64
