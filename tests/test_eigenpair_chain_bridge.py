import importlib.util,copy
from pathlib import Path
import pytest
p=Path(__file__).parents[1]/'tools/continue_eigenpair_chain_bridge.py';s=importlib.util.spec_from_file_location('chain_bridge',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def args(unit=1):
    parent=m.L.C.PARENT if unit==1 else m.BASE/f'unit{unit-1:03d}'
    source,outer,worker=[m.L.C.PINS[k]for k in ('source','outer','worker')]
    return m.parser().parse_args(['--out',str(m.BASE/f'unit{unit:03d}/run'),'--unit',str(unit),'--parent',str(parent),'--parent-source',source,'--parent-manifest',outer,'--parent-run-manifest',worker,'--expected-right',str(m.Fraction(275+8*(unit-1),128))])

def test_only_fixed_units_parent_and_native_extent():
    for n in [1,2,3]:m.install(args(n))
    for field,value in [('unit',4),('expected_right','273/128'),('parent',m.BASE/'unqualified'),('out',m.BASE/'other/run'),('parent_source','0'*40)]:
        a=args();setattr(a,field,value)
        with pytest.raises(ValueError):m.install(a)

def fixture(monkeypatch):
    c={'parameter_endpoints':['281/128','283/128']};rec={'call':3,'index':3}
    out={'passed':True,'status':'four_joined_charts','calls':4,'repairs':0,'progress':{'accepted':[{}, {}, {}, rec],'last_accepted':rec,'active_call':None}}
    result={'completed':True,'calls':4,'repairs':0,'attempts':[{'call':i,'passed':True}for i in range(4)],'accepted':[{}]*4,'last_chart':c}
    accepted={'call':3,'index':3,'chart':c};monkeypatch.setattr(m.B,'accepted_check',lambda full,expected:c)
    return out,result,accepted,{},'283/128'

def test_completed_prefix_is_admitted_but_membership_stop_is_terminal(monkeypatch):
    a=fixture(monkeypatch);assert m.complete_parent(*a)==a[2]['chart']
    a[0]['status']='rigorous_chain_membership'
    with pytest.raises(ValueError):m.complete_parent(*a)

@pytest.mark.parametrize('kind',['partial','active','calls','attempts','compact','index'])
def test_partial_or_mismatched_native_prefix_fails_closed(monkeypatch,kind):
    a=fixture(monkeypatch);out,result,accepted,_,_=a
    if kind=='partial':result['accepted'].pop()
    elif kind=='active':out['progress']['active_call']=4
    elif kind=='calls':result['calls']=5
    elif kind=='attempts':result['attempts'][1]['passed']=False
    elif kind=='compact':accepted['chart']={'different':True}
    else:accepted['index']=2
    with pytest.raises(ValueError):m.complete_parent(*a)

def test_launcher_targets_actual_new_worker_and_exclusive_path():
    p=Path(__file__).parents[1]/'tools/launch_eigenpair_chain_bridge.py';s=importlib.util.spec_from_file_location('chain_bridge_launcher',p);l=importlib.util.module_from_spec(s);s.loader.exec_module(l)
    assert l.BASE==m.BASE;l.ARGS=args();l.OUT=l.BASE/'unit001';l.LIMIT=420
    command=l.command();assert str(m.ROOT/'tools/continue_eigenpair_chain_bridge.py')in command
    assert command[command.index('--out')+1]==str(m.BASE/'unit001/run')
