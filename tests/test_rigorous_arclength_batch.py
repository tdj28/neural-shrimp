import copy,importlib.util
from pathlib import Path
import pytest
from neural_shrimp.rigorous_arclength_connection import two_arclength_charts
from neural_shrimp.rigorous_arclength_batch import compact_chart,next_arguments,stream_charts
from test_rigorous_arclength_connection import Fold,inputs


def parent():return compact_chart(two_arclength_charts(Fold(),Fold(),inputs())['charts'][-1])


def test_streamed_fold_continuation_exact_gauge_and_every_join():
    saved=[]
    def emit(i,row):saved.append(copy.deepcopy(row));return {'name':str(i),'durable':True}
    old=parent();r=stream_charts(Fold(),Fold(),old,count=3,emit_chart=emit)
    assert r['passed'] and r['qualified_charts']==3 and len(saved)==3
    for row in saved:
        assert row['normal']==old['normal'] and row['origin']==old['origin']
        assert row['anchor']['exact_parameter']==old['exact_parameter_endpoints'][1]
        assert row['anchor']['state_box']==old['endpoint_root_enclosures'][1]
        assert row['exact_parameter_endpoints'][0]==old['exact_parameter_endpoints'][1]
        old=row


def test_failed_gate_retained_and_stops_without_next_chart():
    old=parent();old['endpoint_root_enclosures'][1][1]['mid']=[99,0]
    rows=[];r=stream_charts(Fold(),Fold(),old,count=3,emit_chart=lambda i,row:rows.append(row))
    assert not r['passed'] and r['completed_charts']==1 and r['qualified_charts']==0
    assert len(rows)==1 and not rows[0]['gates']['anchor_identified']


def test_storage_failure_never_advances_parent():
    old=parent();before=copy.deepcopy(old);events=[]
    def emit(i,row):events.append(i);raise OSError('synthetic full disk')
    with pytest.raises(OSError):stream_charts(Fold(),Fold(),old,count=3,emit_chart=emit)
    assert events==[0] and old==before


def test_pending_stage_failure_is_retained_without_new_chart():
    calls=[]
    def observe(kind,row):
        calls.append(kind)
        if kind=='chart_committed':raise TimeoutError('synthetic deadline')
    with pytest.raises(TimeoutError):stream_charts(Fold(),Fold(),parent(),count=3,observe=observe)
    assert calls.count('chart_committed')==1


@pytest.mark.parametrize('count',[0,17,True])
def test_count_cap(count):
    with pytest.raises(ValueError):stream_charts(Fold(),Fold(),parent(),count=count)


def test_unproved_predecessor_and_wrong_exact_endpoint_rejected():
    old=parent();old['passed']=False
    with pytest.raises(ValueError):stream_charts(Fold(),Fold(),old)
    old=parent();old['exact_parameter_endpoints'][1]='3'
    with pytest.raises(ValueError):next_arguments(old)


def driver():
    spec=importlib.util.spec_from_file_location('batchdriver',Path(__file__).parents[1]/'tools/certify_s2_period18_arclength_batch.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_profile_resource_and_library_gate():
    m=driver();rt={'python':m.platform.python_version(),'python_flint':m.flint.__version__,'flint':m.flint.__FLINT_VERSION__,
        'numpy':m.numpy.__version__,'scipy':m.scipy.__version__,'internal_seconds':20,'tick_cap':300,'map_cap':144,'unit_derivative_cap':288,'jet_cap':108,
        'mode':'profile','threads':dict.fromkeys(m.THREADS,'1'),'nice':10}
    row={'elapsed_seconds':1,'max_rss_kib':100,'exit_code':0};m.profile_environment(rt,row)
    rt['numpy']='wrong'
    with pytest.raises(ValueError,match='library'):m.profile_environment(rt,row)


def test_profile_gate_full_counts_and_projected16_cost(tmp_path,monkeypatch):
    m=driver();(tmp_path/'manifest.json').write_text('{}');pin=m.entry(tmp_path/'manifest.json')['sha256']
    runtime={'source_sha256':{'x':'y'}}
    outcome={'elapsed_seconds':6,'max_rss_kib':100,'exit_code':0,'status':'synthetic_profile_completed',
        'adapter_counts':{'map_attempted':144,'map_returned':144,'unit_attempted':288,'unit_returned':288,'map_cap':144,'unit_cap':288},
        'jet_counts':{'attempted':108,'returned':108,'cap':108}}
    objects={'manifest.json':{'files':{}},'runtime-start.json':runtime,'outcome.json':outcome,
             'batch.json':{'charts':[{'gates':{'uniform_root':True}}]*2}}
    monkeypatch.setattr(m,'read',lambda p:objects[p.name]);monkeypatch.setattr(m,'verify_inventory',lambda *args:None)
    monkeypatch.setattr(m,'profile_environment',lambda *args:None);monkeypatch.setattr(m,'sources',lambda:runtime['source_sha256'])
    r=m.profile_gate(tmp_path,pin);assert r['projected_seconds']==101 and r['projected_bytes']==16*18*1024**2
    outcome['jet_counts']['returned']=107
    with pytest.raises(ValueError,match='path-jet'):m.profile_gate(tmp_path,pin)


def test_fixed_predecessor_pin_before_read(tmp_path,monkeypatch):
    m=driver();monkeypatch.setattr(m,'PARENT',tmp_path);monkeypatch.setattr(m,'original_anchor',lambda:{})
    (tmp_path/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='predecessor manifest'):m.prepare()
