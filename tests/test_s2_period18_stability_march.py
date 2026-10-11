import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

def driver():
    p=Path(__file__).parents[1]/'tools/certify_s2_period18_stability_march.py';s=importlib.util.spec_from_file_location('stability_march_driver',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_actual_runner_cannot_hide_modified_new_source(monkeypatch):
    m=driver();monkeypatch.setattr(m.io,'entry',lambda p:{'sha256':'current'})
    m.actual_runner({'source_sha256':{m.FIRST:'current'}},7)
    m.actual_runner({'source_sha256':{m.CURRENT:'current',m.FIRST:'current'}},6)
    for inv,index in [({m.CURRENT:'tampered',m.FIRST:'current'},6),({m.FIRST:'current'},6),({m.CURRENT:'current'},7)]:
        with pytest.raises(ValueError):m.actual_runner({'source_sha256':inv},index)

def test_adapter_restores_base_on_success_and_failure(monkeypatch,tmp_path):
    m=driver();old=(m.base.sources,m.base.input_for)
    def run(*args):
        assert m.base.sources is m.sources
        return 17
    monkeypatch.setattr(m.base,'execute',run);assert m.execute(tmp_path,'profile')==17;assert (m.base.sources,m.base.input_for)==old
    monkeypatch.setattr(m.base,'execute',lambda *a:(_ for _ in ()).throw(OSError('save')))
    with pytest.raises(OSError):m.execute(tmp_path,'profile')
    assert (m.base.sources,m.base.input_for)==old

@pytest.mark.parametrize('index',[-1,8,True])
def test_invalid_index_rejected_before_base(monkeypatch,tmp_path,index):
    m=driver();monkeypatch.setattr(m.base,'execute',lambda *a:pytest.fail('must not start'))
    with pytest.raises(ValueError):m.execute(tmp_path,'target',index=index,previous=tmp_path,previous_pin='p',previous_outer_pin='q')

def test_no_target_read_for_profile(monkeypatch):
    m=driver();monkeypatch.setattr(m,'ORIGINAL_INPUT',lambda mode:{'synthetic':mode});monkeypatch.setattr(m,'previous_coverage',lambda *a:pytest.fail('target input'))
    assert m.input_for('profile')=={'synthetic':'profile'}

def test_exact_selected_chart_and_native_join(monkeypatch,tmp_path):
    m=driver();chart={'passed':True,'gates':{'uniform_root':True,'primitive_period':True,'anchor_identified':True},'statement':{'period':18,'phase_dimension':13},'normal':['n'],'origin':['o'],'exact_parameter_endpoints':['21/64','43/128']}
    following=dict(chart,exact_parameter_endpoints=['43/128','11/32']);calls=[]
    predecessor={'chart':following,'model_inputs':{'exact':'model'}}
    monkeypatch.setattr(m,'previous_coverage',lambda *a:(predecessor,{'index':8}))
    parent=SimpleNamespace(PARENT=tmp_path,PIN='pin',EXECUTION='source',partial_input=lambda:(calls.append('fullprefix') or {'model_inputs':{'exact':'model'}}),read=lambda p:chart if '007' in p.name else following,io=SimpleNamespace(compact_chart=lambda c:c,joined=lambda a,b:calls.append('joined')))
    monkeypatch.setattr(m.io.io,'parent',parent);monkeypatch.setattr(m.io,'entry',lambda p:{'sha256':'chart','bytes':10})
    value=m.input_for('target',7,tmp_path,'p','q');assert calls==['fullprefix','joined'] and value['selected_artifact']['index']==7 and len(value['slices'])==8
    chart['gates']['primitive_period']=False
    with pytest.raises(ValueError):m.input_for('target',7,tmp_path,'p','q')


def test_predecessor_requires_complete_exact_coverage(monkeypatch,tmp_path):
    m=driver();path=tmp_path/'run';ends=['43/128','11/32']
    rows=[{'passed':True,'exact_endpoints':x} for x in m.base.partition(ends)]
    rt={'source_sha256':{m.FIRST:'current'},'mode':'target','precision':256,'nice':10,'threads':dict.fromkeys(m.THREADS,'1'),'source_commit':'commit'}
    out={'passed':True,'exit_code':0,'completed_slices':8,'stages':1288,'rows':rows,'counts':{'map_attempted':1296,'map_returned':1296,'unit_attempted':2592,'unit_returned':2592,'mixed_attempted':288,'mixed_returned':288},'complete_coverage':ends}
    inp={'selected_artifact':{'index':8},'coverage_endpoints':ends,'chart':{'exact_parameter_endpoints':ends}}
    launch={'exit_code':0,'source_commit':'commit','owned_live_pids':[]}
    monkeypatch.setattr(m.io,'entry',lambda p:{'sha256':'pin' if p.name=='manifest.json' else 'current'})
    monkeypatch.setattr(m.io,'verify_inventory',lambda *a:None);monkeypatch.setattr(m.io,'sha256',lambda *a:'current');monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:b'source')
    monkeypatch.setattr(m.io,'read',lambda p:{'manifest.json':{'files':{}},'runtime-start.json':rt,'outcome.json':out,'launcher-result.json':launch,'inputs.json.gz':inp}[p.name])
    assert m.previous_coverage(path,'pin','pin',7)[0] is inp
    out['passed']=False
    with pytest.raises(ValueError):m.previous_coverage(path,'pin','pin',7)
    out['passed']=True;inp['selected_artifact']['index']=7
    with pytest.raises(ValueError):m.previous_coverage(path,'pin','pin',7)
    inp['selected_artifact']['index']=8;launch['owned_live_pids']=[123]
    with pytest.raises(ValueError):m.previous_coverage(path,'pin','pin',7)
