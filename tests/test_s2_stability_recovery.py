import copy, importlib.util
from pathlib import Path
import pytest

def modules():
    path=Path(__file__).parents[1]/'tools/resume_s2_period18_stability_march.py'
    spec=importlib.util.spec_from_file_location('recovery_resume',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m,m.recovery

def fixture(r):
    rt={'source_commit':r.EXECUTION,'mode':'target','precision':256,'nice':10,'threads':dict.fromkeys(r.old.THREADS,'1'),'source_sha256':{'x':'a'}}
    ends=['39/128','5/16'];rows=[{'passed':True,'exact_endpoints':e} for e in r.old.base.partition(ends)]
    out={'status':'fixed_chart_slices_completed','passed':True,'exit_code':0,'completed_slices':8,'stages':1288,'rows':rows,'counts':{'map_attempted':1296,'map_returned':1296,'unit_attempted':2592,'unit_returned':2592,'mixed_attempted':288,'mixed_returned':288},'elapsed_seconds':230.,'max_rss_kib':694052,'complete_coverage':ends}
    inp={'selected_artifact':{'index':4},'coverage_endpoints':ends,'chart':{'exact_parameter_endpoints':ends}}
    launch={'source_commit':r.EXECUTION,'mode':'target','source_sha256':rt['source_sha256']}
    return rt,out,inp,launch

def test_recovery_complete_worker():
    _,r=modules();r.worker_checks(*fixture(r))

@pytest.mark.parametrize('defect',['incomplete','failed','counts','index','gap','source','time','rss'])
def test_recovery_fails_closed(defect):
    _,r=modules();rt,out,inp,launch=fixture(r)
    if defect=='incomplete':out['completed_slices']=7
    if defect=='failed':out['rows'][0]['passed']=False
    if defect=='counts':out['counts']['mixed_returned']=287
    if defect=='index':inp['selected_artifact']['index']=3
    if defect=='gap':out['rows'][0]['exact_endpoints'][0]='1/2'
    if defect=='source':launch['source_commit']='other'
    if defect=='time':out['elapsed_seconds']=float('nan')
    if defect=='rss':out['max_rss_kib']=2*1024**2
    with pytest.raises(ValueError):r.worker_checks(rt,out,inp,launch)

def test_resume_rejects_old_runner(monkeypatch,tmp_path):
    m,_=modules();monkeypatch.setattr(m.io,'read',lambda p:{'source_sha256':{}});monkeypatch.setattr(m.io,'entry',lambda p:{'sha256':'current'})
    monkeypatch.setattr(m,'ORIGINAL_PREVIOUS',lambda *a:pytest.fail('must not fall through'))
    with pytest.raises(ValueError):m.previous_coverage(tmp_path,'x','y',2)

def test_wrapper_restores_on_error(monkeypatch,tmp_path):
    m,_=modules();saved=m.old.sources,m.old.previous_coverage
    monkeypatch.setattr(m.old,'execute',lambda *a:(_ for _ in ()).throw(OSError('save')))
    with pytest.raises(OSError):m.execute(tmp_path,'profile')
    assert (m.old.sources,m.old.previous_coverage)==saved

@pytest.mark.parametrize('index',[True,-1,4])
def test_invalid_index_before_execute(monkeypatch,tmp_path,index):
    m,_=modules();monkeypatch.setattr(m.old,'execute',lambda *a:pytest.fail('must not execute'))
    with pytest.raises(ValueError):m.execute(tmp_path,'target',index=index)
