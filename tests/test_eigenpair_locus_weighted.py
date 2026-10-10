"""Synthetic metadata/radius controls only; no target/model calls."""
from fractions import Fraction
import importlib.util,sys
from pathlib import Path
from types import SimpleNamespace
import pytest

def load():
    path=Path(__file__).parents[1]/'tools/certify_eigenpair_locus_weighted.py'
    spec=importlib.util.spec_from_file_location('weighted_adapter',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_exact_radii_and_no_normalization_or_clipping():
    m=load();r=[2.**-24]*468+[2.5e-6,2.2e-18]
    w,rr=m.exact_weights({'linear_radii_proposal':{'radii':r}})
    assert len(w)==470 and all(Fraction(a)*Fraction(2)**-18==Fraction.from_float(b) for a,b in zip(w,r))
    assert [Fraction(v) for v in rr]==[Fraction.from_float(v) for v in r]
    assert max(Fraction(v) for v in w)<1
    for bad in (0.,-1.,True,float('nan'),float('inf'),2.**-17):
        q=r.copy();q[0]=bad
        with pytest.raises(ValueError):m.exact_weights({'linear_radii_proposal':{'radii':q}})
    with pytest.raises(ValueError):m.exact_weights({'linear_radii_proposal':{'radii':r[:-1]}})

def fixture(m):
    o={'status':'proof_gate_inconclusive','passed':False,'ticks':109,'event_counts':{'anchor_phase_return':18,'chart_return':1,'phase_return':90,'preconditioner_return':1,'proposal':2},'elapsed_seconds':24.,'peak_rss_kib':604744,'stored_bytes_before_final':66745722}
    r={'commit':m.BASE_EXEC,'settings':{'period':18,'sigma':1,'halfwidth_exponent':-20,'radius_exponent':-24,'newton_steps':2,'precision':256},'python':sys.version,'numpy':m.B.np.__version__,'flint':m.B.flint.__version__,'flint_library':m.B.flint.__FLINT_VERSION__,'threads':dict.fromkeys(m.B.THREADS,'1'),'nice':10}
    l={'commit':m.BASE_EXEC,'exit_code':2,'stop_reason':None,'owned_live_pids':[],'elapsed_seconds':25.}
    return o,r,l

def test_measured_admission_is_failed_science_not_resource_failure():
    m=load();o,r,l=fixture(m);g=m.measured_gate(o,r,l);assert g['seconds']==106 and g['bytes']==3*66745722+8*1024**2
    for upd in ({'passed':True},{'ticks':108},{'elapsed_seconds':28.},{'stored_bytes_before_final':100*1024**2},{'peak_rss_kib':float('nan')}):
        with pytest.raises(ValueError):m.measured_gate(dict(o,**upd),r,l)
    for upd in ({'stop_reason':'RSS'},{'owned_live_pids':[1]},{'exit_code':0},{'elapsed_seconds':-1.}):
        with pytest.raises(ValueError):m.measured_gate(o,r,dict(l,**upd))
    with pytest.raises(ValueError):m.measured_gate(o,dict(r,numpy='wrong'),l)

def test_install_changes_only_radius_and_weights_not_kernel(monkeypatch):
    m=load();calls=[];base=SimpleNamespace(SOURCES=['old.py'],SETTINGS={'radius_exponent':-24,'halfwidth_exponent':-20,'precision':256},bind_inputs=lambda:('data','proof',{'original':True}))
    monkeypatch.setattr(m,'B',base)
    def bind(sources):calls.append(sources);return ['1/2']*470,{'fixed':'proposal'}
    monkeypatch.setattr(m,'bind_proposal',bind);m.install()
    data,proof,b=base.bind_inputs()
    assert data=='data' and proof=='proof' and b['original'] and calls==[['old.py']]
    assert base.SETTINGS=={'radius_exponent':-18,'halfwidth_exponent':-20,'precision':256,'weights':['1/2']*470}
    assert base.SOURCES==['old.py']+m.EXTRA

def test_exclusive_new_launcher_and_old_bytes_remain_bound():
    root=Path(__file__).parents[1];s=(root/'tools/launch_eigenpair_locus_weighted.py').read_text()
    assert 'attempt-002-weighted-target' in s and 'certify_eigenpair_locus_weighted.py' in s
    assert 'RSS_CAP_KIB=2*1024**2' in s and "'135s'" in s
    assert 'unchanged target kernel/adapter' in (root/'tools/certify_eigenpair_locus_weighted.py').read_text()
