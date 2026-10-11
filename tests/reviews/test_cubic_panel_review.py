import copy,json,os,subprocess,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/reviews'),str(ROOT/'tools'),str(ROOT/'src')]
import review_cubic_panel as D
import launch_cubic_panel_review as L


def card():
    c=D.read(ROOT/D.CARD);c['status']='reviewed'
    for k in ('producer','initial_independent'):c[k]={v:'nonadmission-test'for v in ('root','source','outer','worker')}
    return c


def test_prepared_and_missing_independent_or_producer_authority_block():
    c=card();D.card_check(c);L.card_check(c);assert D.OUT.parent==L.OUT
    for key in ('producer','initial_independent'):
        for field in ('root','source','outer','worker'):
            bad=copy.deepcopy(c);bad[key][field]=None
            with pytest.raises(ValueError):D.card_check(bad)
            with pytest.raises(AssertionError):L.card_check(bad)
    with pytest.raises(ValueError):D.card_check(dict(c,status='prepared'))
    with pytest.raises(ValueError):D.card_check(dict(c,ordinals=[32,33]))
    with pytest.raises(ValueError):D.card_check(dict(c,maximum_centers=97))


def fixture(tmp_path,monkeypatch):
    pc=D.P.read(ROOT/D.P.CARD);pc['status']='reviewed';pc['initial_independent']={k:'fixture-only'for k in pc['initial_independent']}
    monkeypatch.setattr(D.P,'initial',lambda *a:{'fixture':True})
    monkeypatch.setattr(D.P.G,'bind',lambda w,s,prior:s['ordinal'])
    def domain(w,row,context,domains):
        first=w.calls+1;w.calls+=90;w.save('domain-outcome.json',dict(passed=True,domain=row,dimension=705,native_calls=90,first_call=first,last_call=w.calls));return {},{},[]
    def cell(w,row,context,*rest):
        first=w.calls+1;w.calls+=126;passed=not(context==33 and row['domain']==1 and row['depth']==0)
        r=dict(cell=row,passed=passed,completed=True,native_calls=126,dimension=705,complement_dimension=11,first_call=first,last_call=w.calls)
        w.save('complement.json',dict(passed=passed));return r
    monkeypatch.setattr(D.P.C,'derivative_domain',domain);monkeypatch.setattr(D.P.C,'fresh_cell',cell)
    w=D.P.Work();w.output_dir=tmp_path;w.calls=144;out=D.P.target(w,pc)
    out.update(domains_done=w.domains_done,pending_cells=w.pending,completed_attempts=w.attempts,completed_charts=w.completed_charts,accepted_cells=w.accepted,native_calls=w.calls,profile_native_calls=144,target_native_calls=w.calls-144)
    monkeypatch.setattr(D.B,'pairs',lambda *a:None)
    return pc,out


def test_actual_producer_record_schema_and_global_history(tmp_path,monkeypatch):
    pc,out=fixture(tmp_path,monkeypatch);domains,cells=D.history(out,tmp_path,pc)
    assert len(domains)==16 and len(cells)==17 and len(out['completed_attempts'])==18
    assert domains[8]['chart']==34 and domains[8]['domain']['index']==1
    assert cells[0]['ordinal']==2 and cells[0]['cell']['depth']==1
    bad=copy.deepcopy(out);bad['completed_attempts'][2]['first_call']-=1
    with pytest.raises(ValueError,match='global ordered'):D.history(bad,tmp_path,pc)
    bad=copy.deepcopy(out);bad['accepted_cells']=bad['accepted_cells'][:-1]
    with pytest.raises(ValueError,match='all accepted'):D.history(bad,tmp_path,pc)
    bad=copy.deepcopy(out);bad['native_calls']-=126
    with pytest.raises(ValueError,match='complete global'):D.history(bad,tmp_path,pc)


def test_exact_union_requires_entire_closed_chart(tmp_path,monkeypatch):
    pc,out=fixture(tmp_path,monkeypatch);domains,cells=D.history(out,tmp_path,pc)
    for chart in pc['charts']:
        rows=[dict(r,passed=True,complement_dimension=11)for r in cells if r['chart']==chart['ordinal']]
        assert D.exact_union(rows,chart['parameter_endpoints'])['passed']
        with pytest.raises(ValueError):D.exact_union(rows[:-1],chart['parameter_endpoints'])
        with pytest.raises(ValueError):D.exact_union(rows+rows[:1],chart['parameter_endpoints'])


def test_finalization_reserve_and_archive_closure(tmp_path,monkeypatch):
    monkeypatch.setattr(D,'OUT',tmp_path/'result');w=D.Work();w.used=508*1024**2-1
    with pytest.raises(ValueError):w.save('blocked.json',{'x':1})
    w.prefix='';w.save('outcome.json',{'passed':False},True)
    archive=tmp_path/'source'
    for name in D.FILES:
        source=ROOT/name;assert source.is_file(),name
        p=archive/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(source.read_bytes())
    script="""import sys
from pathlib import Path
root=Path.cwd();sys.path[:0]=['tools/reviews','tools','src']
import review_cubic_panel as D
import launch_cubic_panel_review as L
loaded={str(Path(m.__file__).resolve().relative_to(root))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(root)}
assert loaded<=set(D.FILES),loaded-set(D.FILES)
assert D.OUT.parent==L.OUT
"""
    r=subprocess.run([sys.executable,'-I','-B','-c',script],cwd=archive,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
    assert r.returncode==0,r.stderr
