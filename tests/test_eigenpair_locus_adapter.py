"""Source-only adapter controls; no saved target inputs are loaded."""
import importlib.util,sys
from copy import deepcopy
from pathlib import Path
import pytest
from flint import arb,ctx

def load(name):
    path=Path(__file__).parents[1]/'tools'/name
    spec=importlib.util.spec_from_file_location(name.replace('.','_'),path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_profile_rejects_bad_resource_library_and_source_metadata():
    m=load('certify_eigenpair_locus.py');p=m.PROFILE.projection([1.,1.],[.001]*12,100000,1000)
    o={'passed':True,'target_acquired':False,'ticks':194,'event_counts':{'phase_return':180,'proposal':4,'preconditioner_return':2,'chart_return':2},'chart_times':[1.,1.],'phase_times':[.001]*12,'elapsed_seconds':3.,'peak_rss_kib':100000,'stored_bytes_before_final':1000,'projection':p}
    r={'mode':'synthetic_only','commit':m.PROFILE_COMMIT,'precision':256,'nice':10,'threads':dict.fromkeys(m.THREADS,'1'),'python':sys.version,'numpy':m.np.__version__,'flint':m.flint.__version__,'flint_library':m.flint.__FLINT_VERSION__}
    l={'commit':m.PROFILE_COMMIT,'exit_code':0,'owned_live_pids':[],'elapsed_seconds':4.}
    assert m.profile_gate(o,r,l)==p
    for key in ('elapsed_seconds','peak_rss_kib'):
        for v in (-1.,float('nan'),float('inf'),True):
            with pytest.raises(ValueError):m.profile_gate(dict(o,**{key:v}),r,l)
    for override in ({'ticks':193},{'passed':False},{'target_acquired':True},{'projection':dict(p,two_chart_seconds=1.)}):
        with pytest.raises(ValueError):m.profile_gate(dict(o,**override),r,l)
    for override in ({'precision':53},{'nice':9},{'numpy':'wrong'},{'commit':'wrong'}):
        with pytest.raises(ValueError):m.profile_gate(o,dict(r,**override),l)
    for override in ({'exit_code':2},{'owned_live_pids':[999]},{'elapsed_seconds':-1.}):
        with pytest.raises(ValueError):m.profile_gate(o,r,dict(l,**override))


def test_propagated_anchor_uses_full_boxes_and_native_mode():
    m=load('certify_eigenpair_locus.py')
    with ctx.workprec(256):
        profile=m.PROFILE;model=profile.Synthetic13();initial,_=profile.fixture()
        # A known primitive2 supplies a valid F^18 fixed point with +1 mode.
        # The propagation helper deliberately does not authenticate period.
        box=[arb(m.exact(v),arb((1,-100))) for v in initial[:13]]+[arb(1,arb((1,-100)))]+[arb(m.exact(v),arb((1,-100))) for v in initial[13:26]]
        proof={'box':m.vector_json(box)};count=[0];events=[]
        def tick():count[0]+=1
        ini,anchor,w=m.propagate_anchor(model,proof,'1',tick,lambda k,r:events.append((k,r)))
        assert count==[18] and len(events)==18 and len(ini)==len(anchor['box'])==470
        assert anchor['exact_parameter']=='0' and anchor['sigma']==1
        assert any(v['rad'][0]>0 for v in events[0][1]['input'])
        for phase,expected in ((0,1),(1,2),(2,1)):
            assert anchor['box'][phase*26+13].contains(expected)
        assert any(v['rad'][0]>0 for v in w['full_root_box'])
        assert m.exact(ini[-1])==1


def test_anchor_interruption_retains_return_and_model_construction_precision():
    m=load('certify_eigenpair_locus.py')
    with ctx.workprec(256):
        model=m.PROFILE.Synthetic13();initial,_=m.PROFILE.fixture()
        proof={'box':m.vector_json([m.exact(v) for v in initial[:13]]+[arb(1)]+[m.exact(v) for v in initial[13:26]])}
        events=[]
        def observe(k,r):events.append((k,r));raise TimeoutError('after saved return')
        with pytest.raises(TimeoutError):m.propagate_anchor(model,proof,'1',lambda:None,observe)
        assert len(events)==1 and events[0][0]=='anchor_phase_return'
    text=(Path(__file__).parents[1]/'tools/certify_eigenpair_locus.py').read_text()
    assert 'with ctx.workprec(256):\n            model=TwoRateCuspMap' in text


def test_fixed_launch_path_caps_and_transitive_import_inventory():
    m=load('certify_eigenpair_locus.py');launch=load('launch_eigenpair_locus.py')
    assert m.SETTINGS=={'period':18,'sigma':1,'halfwidth_exponent':-20,'radius_exponent':-24,'newton_steps':2,'precision':256}
    assert len(m.SOURCES)==len(set(m.SOURCES))
    for path in ('src/neural_shrimp/controls.py','src/neural_shrimp/models.py','tools/reviews/review_clark_rnn_c1_math.py','tools/profile_eigenpair_locus.py'):
        assert path in m.SOURCES
    assert launch.RSS_CAP_KIB==2*1024**2
    assert '135s' in launch.command() and '--kill-after=5s' in launch.command()
    assert str(launch.OUT).endswith('/attempt-001-target')
