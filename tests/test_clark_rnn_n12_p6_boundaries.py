import importlib.util
from pathlib import Path
import numpy as np
import pytest
from neural_shrimp.clark_rnn import iterate

def module():
    p=Path(__file__).resolve().parents[1]/'tools/explore_clark_rnn_n12_p6_boundaries.py'
    s=importlib.util.spec_from_file_location('n12p6_control',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_native_worker_uses_six_in_twelve_dimensions_and_retains_return(tmp_path):
    m=module();w=m.Worker(tmp_path);events=[];w.emit=lambda k,v:events.append((k,v))
    x=np.linspace(.1,.8,12);j=np.eye(12)*.2
    a=w.it(x,j,2.,.7,'synthetic');b=iterate(x,j,2.,.7,6)
    np.testing.assert_array_equal(a[0],b[0]);assert a[2].shape==(6,12)
    assert events[0][0]=='native' and events[0][1]['cycle'].shape==(6,12)

def test_native_phase_and_odd_identity_rules():
    m=module();a=np.arange(72.).reshape(6,12)
    assert m.phase_match(a,a+1e-5)['passed']
    assert not m.phase_match(a,-a)['passed']
    assert not m.phase_match(a,np.roll(a,1,axis=0))['passed']

def test_interrupt_before_any_native_call(tmp_path):
    m=module();w=m.Worker(tmp_path);w.interrupted=True
    with pytest.raises(InterruptedError):w.it(np.zeros(12),np.eye(12),1.,.7,'synthetic')
    assert w.calls==0


def test_leading_complex_is_not_replaced_by_nearby_real_mode():
    m=module()
    assert m.leading_boundary_sign([1.01+.1j,1.01-.1j,.999,0.]) is None
    assert m.leading_boundary_sign([-1.02,.999,.2,.1])==-1
    assert m.leading_boundary_sign([1.02,-.999,.2,.1])==1
    assert m.leading_boundary_sign([np.nan,1.]) is None

def test_local_root_does_not_override_failed_reverse_association(tmp_path,monkeypatch):
    m=module();w=m.Worker(tmp_path);events=[];w.emit=lambda k,v:events.append((k,v))
    cycle=np.arange(72.).reshape(6,12);point={'theta':cycle[0].copy(),'cycle':cycle,'delta':.6,'eigenvalues':np.r_[1.,np.arange(11)/20+.1],'eigenvectors':np.eye(12)}
    w.aug=lambda *a:(np.zeros(25),np.eye(25),{},0.)
    monkeypatch.setattr(m,'qualify_boundary',lambda *a:({'theta':point['theta'],'cycle':cycle},{'status':'numerical_boundary_candidate'}))
    w.correct=lambda *a:({'theta':point['theta']+.01},'closed')
    result=w.localize(point,np.eye(12),5.,1,'toy')
    assert result['status']=='boundary_identity_unresolved' and result['reverse_error']>1e-7
    assert events[-1][0]=='boundary_result' and any(k=='boundary_reverse' for k,v in events)


def test_start_must_be_attracting_before_first_loss_claim():
    m=module();assert m.initial_attracting({'radius':.9})
    for rho in [1.,1.1,np.nan,np.inf,1-1e-7]:assert not m.initial_attracting({'radius':rho})


def test_proper_divisor_one_collapse_is_rejected(tmp_path):
    m=module();w=m.Worker(tmp_path);w.emit=lambda *a:None
    c=np.arange(72.).reshape(6,12);c[1]=c[0]
    w.it=lambda *a:(c[0],np.eye(12)*.2,c,np.tile(np.eye(12),(6,1,1)),0.)
    p,status=w.correct(c[0],np.eye(12),3.,.8,'synthetic');assert p is None and status=='collapsed'

def test_nonfinite_return_fails_closed(tmp_path):
    m=module();w=m.Worker(tmp_path);w.emit=lambda *a:None
    w.it=lambda *a:(np.full(12,np.nan),np.eye(12),np.ones((6,12)),np.ones((6,12,12)),0.)
    p,status=w.correct(np.ones(12),np.eye(12),3.,.8,'synthetic');assert p is None and status=='nonfinite_native'


def test_localization_uses_twelve_state_coordinates_and_delta_slot(tmp_path,monkeypatch):
    m=module();w=m.Worker(tmp_path);w.emit=lambda *a:None
    cycle=np.arange(72.).reshape(6,12);point={'theta':cycle[0].copy(),'cycle':cycle,'delta':.6,'eigenvalues':np.arange(12)/20+.2,'eigenvectors':np.eye(12)};point['eigenvalues'][0]=1.
    w.aug=lambda z,*a:(np.zeros(25),np.eye(25),{},0.)
    monkeypatch.setattr(m,'qualify_boundary',lambda z,*a:({'theta':z[:12],'cycle':cycle},{'status':'numerical_boundary_candidate'}))
    seen=[]
    def correct(x,j,g,d,tag):seen.append((x.copy(),d));return {'theta':x.copy()},'closed'
    w.correct=correct;r=w.localize(point,np.eye(12),7.,1,'synthetic')
    assert seen[0][0].shape==(12,) and seen[0][1]==.6 and r['delta']==.6 and r['status']=='numerical_boundary_candidate'
    assert r['adjacent_native_identity_error']==0

def test_nonfinite_augmented_fails_closed(tmp_path):
    m=module();w=m.Worker(tmp_path);w.emit=lambda *a:None
    p={'theta':np.ones(12),'cycle':np.ones((6,12)),'delta':.6,'eigenvalues':np.r_[1.,np.arange(11)/20+.1],'eigenvectors':np.eye(12)}
    w.aug=lambda *a:(np.full(25,np.nan),np.eye(25),{},0.)
    assert w.localize(p,np.eye(12),7.,1,'synthetic')['status']=='nonfinite_augmented'


def launcher():
    p=Path(__file__).resolve().parents[1]/'tools/launch_clark_rnn_n12_p6_boundaries.py'
    spec=importlib.util.spec_from_file_location('n12p6_launch',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_oversized_launcher_log_fails_closed(tmp_path):
    (tmp_path/'process.log').write_bytes(b'x'*(16*1024**2))
    g=launcher().final_byte_gate(tmp_path,{'exit_code':0})
    assert not g['passed'] and g['projected_launcher_bytes_upper']>16*1024**2

def test_final_metadata_allowance_is_included(tmp_path):
    (tmp_path/'run').mkdir();(tmp_path/'run/result.json').write_text('{}')
    record={'exit_code':0};m=launcher();g=m.final_byte_gate(tmp_path,record)
    assert g['passed'] and g['projected_total_bytes_upper']>4096
    record['final_byte_gate']=g;m.save(tmp_path/'launch-end.json',record)
    files={str(f.relative_to(tmp_path)):{'sha256':'0'*64,'bytes':f.stat().st_size}for f in tmp_path.rglob('*')if f.is_file()};m.save(tmp_path/'manifest.json',{'files':files})
    assert sum(f.stat().st_size for f in tmp_path.rglob('*')if f.is_file())<=g['projected_total_bytes_upper']

def test_total_cap_independently_fails(tmp_path):
    (tmp_path/'run').mkdir();(tmp_path/'run/data').write_bytes(b'x'*20000)
    g=launcher().final_byte_gate(tmp_path,{},science_cap=1000,launcher_cap=10000)
    assert not g['passed'] and g['projected_launcher_bytes_upper']<10000

@pytest.mark.parametrize('divisor',[1,2,3])
def test_all_native_p6_divisor_collapses_fail(divisor,tmp_path):
    m=module();w=m.Worker(tmp_path);w.emit=lambda *a:None
    c=np.arange(72.).reshape(6,12);c[divisor]=c[0]
    w.it=lambda *a:(c[0],np.eye(12)*.2,c,np.tile(np.eye(12),(6,1,1)),0.)
    point,status=w.correct(c[0],np.eye(12),3.,.8,'synthetic')
    assert point is None and status=='collapsed'

def test_adjacent_native_closure_failure_is_retained(tmp_path):
    m=module();w=m.Worker(tmp_path);w.emit=lambda *a:None
    c=np.arange(72.).reshape(6,12)/100;calls=[]
    def native(*args):
        calls.append(1)
        return (c[0],np.eye(12)*.2,c,np.tile(np.eye(12),(6,1,1)),0.)if len(calls)==1 else(c[1]+.01,np.eye(12)*.2,np.roll(c,-1,axis=0),np.tile(np.eye(12),(6,1,1)),0.)
    w.it=native
    point,status=w.correct(c[0],np.eye(12),3.,.8,'synthetic')
    assert point is None and status=='adjacent_closure_or_primitive'

def test_separate_native_and_augmented_caps(tmp_path):
    m=module();w=m.Worker(tmp_path);w.emit=lambda *a:0
    w.native_calls=12000
    with pytest.raises(RuntimeError,match='12000 native'):w.it(np.zeros(12),np.eye(12),1.,.7,'synthetic')
    w.native_calls=0;w.augmented_calls=96
    with pytest.raises(RuntimeError,match='96 augmented'):w.aug(np.r_[np.zeros(12),.7,np.eye(12)[0]],np.eye(12),1.,-1,'synthetic')


def test_resource_projection_bounds_every_resource():
    m=module();args=([.0001],[.001],[.0001],[.0001],dict(native=[12000],augmented=[35000],eigen=[4000],solve=[3000]),50000)
    assert m.resource_projection(*args)['gate']
    for index,bad in [(0,[.1]),(4,dict(native=[100000],augmented=[35000],eigen=[4000],solve=[3000])),(5,200000)]:
        changed=list(args);changed[index]=bad;assert not m.resource_projection(*changed)['gate']
    changed=list(args);changed[0]=[float('nan')]
    with pytest.raises(ValueError):m.resource_projection(*changed)

def test_adjacent_native_phase_is_qualified_and_retained(tmp_path):
    m=module();w=m.Worker(tmp_path);events=[];w.emit=lambda k,v:events.append((k,v))
    c=np.arange(72.).reshape(6,12)/100
    def native(x,*args):
        cc=c if np.array_equal(x,c[0])else np.roll(c,-1,axis=0)
        return x.copy(),np.eye(12)*.2,cc,np.tile(np.eye(12),(6,1,1)),0.
    w.it=native;p,status=w.correct(c[0],np.eye(12),3.,.8,'synthetic')
    assert status=='closed'and p['closure_adjacent']==0 and len(p['eigenvalues_adjacent'])==12
    assert any(k=='native_adjacent_eigen'for k,v in events)


def test_leading_mode_preserves_same_sign_competitor_and_adjacent_identity():
    m=module();e=np.r_[1.2,.999,np.arange(10)/20]
    mode=m.leading_boundary_mode({'eigenvalues':e,'eigenvalues_adjacent':e[[1,0,*range(2,12)]]})
    assert mode['phase0_index']==0 and mode['adjacent_index']==1 and mode['sign']==1
    for adjacent in [np.r_[-1.2,.999,np.arange(10)/20],np.r_[1.19,.999,np.arange(10)/20],np.r_[1.2+.1j,1.2-.1j,np.arange(10)/20]]:
        assert m.leading_boundary_mode({'eigenvalues':e,'eigenvalues_adjacent':adjacent})is None

@pytest.mark.parametrize('mode_index,expected_index',[(0,0),(None,1)])
def test_localization_keeps_leading_eigenvector_or_labels_fold_proposal(tmp_path,mode_index,expected_index):
    m=module();w=m.Worker(tmp_path);events=[];w.emit=lambda k,v:events.append((k,v))
    cycle=np.arange(72.).reshape(6,12);point={'theta':cycle[0].copy(),'cycle':cycle,'delta':.8,'eigenvalues':np.r_[1.2,.999,np.arange(10)/20],'eigenvectors':np.eye(12)}
    seen=[]
    def augmented(z,*args):seen.append(z.copy());return np.full(25,np.nan),np.eye(25),{},0.
    w.aug=augmented
    result=w.localize(point,np.eye(12),7.,1,'synthetic',mode_index=mode_index)
    assert result['status']=='nonfinite_augmented'
    np.testing.assert_array_equal(seen[0][13:],np.eye(12)[:,expected_index])
    selection=next(v for k,v in events if k=='localization_mode_selection')
    assert selection['selection']==('actual_leading_phase0_mode'if mode_index is not None else 'nearest_plus_one_correction_failure_proposal')

def test_first_unstable_cannot_supply_a_nonleading_index(tmp_path):
    m=module();w=m.Worker(tmp_path);w.emit=lambda *a:None
    point={'eigenvalues':np.r_[1.2,.999,np.arange(10)/20],'eigenvectors':np.eye(12)}
    assert w.localize(point,np.eye(12),7.,1,'synthetic',mode_index=1)['status']=='unstable_seed_not_leading'

def test_half_wave_symmetry_tie_does_not_override_native_identity():
    m=module();half=np.arange(1.,37.).reshape(3,12);cycle=np.vstack([half,-half])
    result=m.phase_match(cycle,cycle)
    assert result['best_sign']==-1 and result['best_shift']==3
    assert result['native_error']==0 and result['passed']
    assert not m.phase_match(cycle,-cycle)['passed']
    assert not m.phase_match(cycle,np.roll(cycle,1,axis=0))['passed']
def test_small_native_distance_cannot_hide_a_better_distinct_phase():
    m=module();cycle=np.arange(72.).reshape(6,12)*1e-5
    result=m.phase_match(cycle,np.roll(cycle,1,axis=0))
    assert result['native_error']<.05 and result['best_error']==0
    assert result['native_minus_best']>result['score_roundoff_tolerance']and not result['passed']
