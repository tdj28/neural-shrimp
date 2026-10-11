"""Synthetic admission and immutable source-inventory controls; no target reads."""
import copy,importlib.util,sys
from pathlib import Path
import pytest


def load():
    p=Path(__file__).parents[1]/'tools/certify_eigenpair_locus_quadratic.py'
    s=importlib.util.spec_from_file_location('quadratic_target_adapter',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def fixture(m):
    o={'passed':True,'target_acquired':False,'elapsed_seconds':24.,'peak_rss_kib':484776,
       'ticks':278,'event_counts':{'chart_return':2,'curvature_proposal':2,'path_assembly_return':18,'path_phase_return':120,'phase_return':144,'preconditioner_return':2,'proposal':4},
       'chart_times':[8.,8.],'mixed_times':[.011]*12,'path_times':[.021]*12,'stored_bytes_before_final':3700000}
    o['projection']=m.QP.projection(o['chart_times'],o['mixed_times'],o['path_times'],o['peak_rss_kib'],o['stored_bytes_before_final'])
    r={'mode':'synthetic_only','commit':m.PROFILE_COMMIT,'precision':256,'nice':10,'threads':dict.fromkeys(m.THREADS,'1'),'tick_cap':300,
       'python':sys.version,'numpy':m.np.__version__,'flint':m.flint.__version__,'flint_library':m.flint.__FLINT_VERSION__}
    l={'commit':m.PROFILE_COMMIT,'exit_code':0,'owned_live_pids':[],'stop_reason':None,'elapsed_seconds':25.}
    return o,r,l


def test_profile_admits_fixed_cost_with_separate_storage_reserve():
    m=load();o,r,l=fixture(m);p=m.profile_gate(o,r,l)
    assert p['raw_bytes']==265402554 and 248*1024**2<p['raw_bytes']<376*1024**2
    assert 100<p['two_chart_seconds']<120


@pytest.mark.parametrize('key,value',[
 ('passed',False),('target_acquired',True),('ticks',277),('elapsed_seconds',-1.),
 ('elapsed_seconds',float('nan')),('elapsed_seconds',120.),('peak_rss_kib',float('inf')),
 ('peak_rss_kib',-1.),('event_counts',{}),('projection',{})])
def test_bad_profile_is_rejected(key,value):
    m=load();o,r,l=fixture(m);o[key]=value
    with pytest.raises(ValueError):m.profile_gate(o,r,l)


@pytest.mark.parametrize('key,value',[
 ('mode','target'),('commit','other'),('precision',128),('nice',0),('threads',{}),
 ('tick_cap',299),('python','other'),('numpy','other'),('flint','other'),('flint_library','other')])
def test_runtime_identity_is_exact(key,value):
    m=load();o,r,l=fixture(m);r[key]=value
    with pytest.raises(ValueError):m.profile_gate(o,r,l)


@pytest.mark.parametrize('key,value',[
 ('commit','other'),('exit_code',2),('owned_live_pids',[12]),('stop_reason','RSS'),
 ('elapsed_seconds',-1.),('elapsed_seconds',float('nan')),('elapsed_seconds',142.)])
def test_launcher_success_and_resources_required(key,value):
    m=load();o,r,l=fixture(m);l[key]=value
    with pytest.raises(ValueError):m.profile_gate(o,r,l)


def test_false_projection_and_excess_cost_are_rejected():
    m=load();o,r,l=fixture(m);o['projection']=dict(o['projection'],two_chart_seconds=1.)
    with pytest.raises(ValueError):m.profile_gate(o,r,l)
    o,r,l=fixture(m);o['chart_times']=[80.,80.]
    o['projection']=m.QP.projection(o['chart_times'],o['mixed_times'],o['path_times'],o['peak_rss_kib'],o['stored_bytes_before_final'])
    with pytest.raises(ValueError):m.profile_gate(o,r,l)


def test_complete_prior_inventory_fixed_controls_and_exclusive_launcher():
    m=load();root=Path(__file__).parents[1]
    assert set(m.B.SOURCES+m.W.EXTRA+m.QP.SOURCES)<=set(m.SOURCES)
    assert len(m.SOURCES)==len(set(m.SOURCES))
    assert all((root/p).is_file() for p in m.SOURCES)
    assert {'src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py'}<=set(m.SOURCES)
    assert m.SETTINGS=={'period':18,'sigma':1,'halfwidth_exponent':-15,'radius_exponent':-18,'newton_steps':2,'precision':256}
    s=(root/'tools/launch_eigenpair_locus_quadratic.py').read_text()
    assert 'attempt-003-quadratic-target' in s and 'certify_eigenpair_locus_quadratic.py' in s
    assert '385*1024**2' in s and "'135s'" in s and 'RSS_CAP_KIB=2*1024**2' in s
    worker=(root/'tools/certify_eigenpair_locus_quadratic.py').read_text()
    assert 'ticks[0]>288' in worker and '(384 if final else 376)*1024**2' in worker
    assert 'with ctx.workprec(256)' in worker and 'B.propagate_anchor' in worker
