"""Independent synthetic controls; no target data or decomposition/model calls."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('pal_audit_independent',ROOT/'tools/reviews/audit_s1r_pseudoarclength_outcome.py')
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
M=A.M


@pytest.fixture(autouse=True)
def no_new_decompositions_or_solves(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('review must reuse saved synthetic arithmetic only')
    for name in ('svd','eig','eigh','eigvals','eigvalsh','solve','lstsq','inv','pinv'):
        monkeypatch.setattr(np.linalg,name,forbidden)


def replay():
    initial=dict(pair=np.array([[.1,.2],[-.1,-.2]]),h=.99,output=.62,carried=0,secant=np.array([0.,0.,-1.]),saved_h_component=-1.)
    r=A.Replay(initial,hidden=1)
    r.witnessed='node-0000';r.next_s=.000125
    return r


def start(r,i=0,parent='node-0000',s=None):
    r.consume('edge_start',dict(id=i,parent=parent,s=r.next_s if s is None else s,nodes=[],accepted=False))


def fail(r,i=0):
    r.consume('edge_end',dict(id=i,parent='node-0000',s=r.next_s,nodes=[],accepted=False,error='synthetic failure'))


def test_reject_retry_without_required_halving():
    r=replay();start(r);fail(r)
    with pytest.raises(ValueError):start(r,1)


def test_reject_duplicate_halving():
    r=replay();start(r);fail(r)
    r.consume('halve',dict(macro=0,parent='node-0000',new_s=r.next_s/2))
    with pytest.raises(ValueError):r.consume('halve',dict(macro=0,parent='node-0000',new_s=r.next_s/2))


def test_second_failure_is_terminal():
    r=replay();start(r);fail(r)
    r.consume('halve',dict(macro=0,parent='node-0000',new_s=r.next_s/2))
    start(r,1);fail(r,1)
    with pytest.raises(ValueError):start(r,2)


def test_rollback_uses_last_durable_parent():
    r=replay()
    with pytest.raises(ValueError):start(r,parent='qualified-but-unwitnessed')


def test_pending_operation_is_preserved_without_return():
    r=replay();counts=dict(r.attempted);counts['analytic']=1
    r.consume('operation_start',dict(id=0,kind='analytic',inputs=dict(theta=np.array([1.,2.])),attempted=counts))
    s=r.summary();assert s['pending_operation']==0
    assert s['attempted']['analytic']==1 and s['returned']['analytic']==0


def test_operation_return_counter_contradiction():
    r=replay();counts=dict(r.attempted);counts['analytic']=1
    r.consume('operation_start',dict(id=0,kind='analytic',inputs=dict(theta=np.array([1.,2.])),attempted=counts))
    with pytest.raises(ValueError):
        r.consume('operation_return',dict(id=0,kind='analytic',returned=dict(r.returned),value=dict(loss=0.,gradient=np.zeros(2),hessian=np.eye(2))))


def test_operation_cap_has_no_hidden_extra_call():
    r=replay();r.attempted['solve']=64
    counts=dict(r.attempted);counts['solve']=65
    with pytest.raises(ValueError):r.consume('operation_start',dict(id=0,kind='solve',inputs={},attempted=counts))


def test_parameter_term_is_necessary_in_native_transport():
    first=np.array([0.,0.,1.]);raw=np.array([2.,0.,1.]);second=raw/np.sqrt(5.)
    B=np.array([[1.,0.,-2.],[0.,1.,0.]])
    out=M.transport(first,second,np.eye(2),np.array([2.,0.]),B)
    assert out['residual']<1e-15 and abs(out['cosine']-1)<1e-15
    bad=M.transport(first,second,np.eye(2),np.zeros(2),B)
    assert bad['residual']>1e-10 or abs(1-bad['cosine'])>1e-7


def test_svd_order_not_inferred_from_small_residuals():
    # This is an exact factorization with deliberately noncanonical ordering.
    P=np.array([[0.,1.],[1.,0.]])
    saved=dict(matrix=np.diag([2.,1.]),U=P,singular_values=np.array([1.,2.]),Vh=P)
    with pytest.raises(ValueError):M.svd_residuals(saved['matrix'],saved)


def test_true_linear_residual_distinguishes_zero_and_false_step():
    good=M.linear_check(np.array([1.,0.]),np.eye(2),np.array([-1.,0.]),np.zeros(2))
    bad=M.linear_check(np.array([1.,0.]),np.eye(2),np.zeros(2),np.zeros(2))
    assert good['passed'] and good['true_relative_forcing']==0.
    assert not bad['passed'] and bad['true_relative_forcing']==1.
    zero=M.linear_check(np.zeros(2),np.eye(2),np.zeros(2),np.zeros(2))
    assert zero['passed'] and zero['true_relative_forcing'] is None


def test_armijo_uses_actual_slope_and_fixed_factor():
    g=np.array([1.,0.]);j=np.eye(2);p=np.array([-1.,0.])
    assert M.trial_decision(g,j,p,np.array([.5,0.]),1)['accepted']
    assert not M.trial_decision(g,j,p,np.array([1.,0.]),12)['accepted']
    assert not M.trial_decision(g,j,-p,np.zeros(2),0)['accepted']
    with pytest.raises(ValueError):M.trial_decision(g,j,p,np.zeros(2),13)


def test_native_phase_swap_and_ambiguous_fixed_point():
    pair=np.array([[.1,.2],[-.1,-.2]])
    m=M.phase_match(pair,pair[::-1],True)
    assert m['passed'] and m['phase']==1
    assert not M.phase_match(np.zeros((2,2)),np.zeros((2,2)),True)['passed']


def test_transition_agreement_cannot_replace_second_phase_closure():
    pair=np.array([[.1,.2],[-.1,-.2]]);images=pair[::-1].copy();second=pair.copy();second[1,0]+=.001
    assert not M.state_checks(pair,images,second)['passed']


def test_reverse_plane_intersection_retains_forward_tangent():
    parent=np.zeros(3);t=np.array([0.,0.,1.]);child=np.array([0.,0.,.001])
    pred,meta=M.predictor(child,t,parent,t,0.)
    assert meta['length']<0 and np.array_equal(pred,parent)
    with pytest.raises(ValueError):M.predictor(child,-t,parent,t,0.)


def test_converged_record_cannot_install_unevaluated_microstep():
    r=replay();z=np.array([.1,.2,.99]);p=np.array([1e-13,0.,0.]);gv=-p;predictor=z.copy()
    r.correctors[0]=dict(current=z+p,predictor=predictor,seed=z-np.array([0.,0.,.0001]),solves=[0])
    r.linear[(0,0)]=dict(passed=True,theta_relative_correction=1e-13,h_correction=0.,op=dict(inputs=dict(z=z,G=gv)))
    with pytest.raises(ValueError):r.consume('corrector_end',dict(id=0,state=z+p,converged=True,iterations=1))


def four_leg_replay():
    r=replay();pair=r.initial['pair']
    for i in range(5):
        k=f'node-{i:04d}'
        r.nodes[k]=dict(id=k,pair=pair.copy(),carried_phase=0,h=.99,z=np.r_[pair[0],.99],tangent=np.array([0.,0.,-1.]))
    start(r);r.edges[0]['correctors']=[0,1,2,3]
    names=[f'node-{i:04d}' for i in range(1,5)]
    for i,k in enumerate(names):r.qualifiers[k]=dict(corrector_id=i,end=dict(id=k,accepted=True))
    r.matches=[dict(parent=a,child=b,replay=rp,passed=True) for a,b,rp in (
        ('node-0000','node-0001',False),('node-0000','node-0002',False),
        ('node-0002','node-0003',False),('node-0001','node-0003',True),('node-0000','node-0004',True))]
    r.common={0:dict(reference='node-0001',candidate='node-0003',end=dict(passed=True)),
              1:dict(reference='node-0000',candidate='node-0004',end=dict(passed=True))}
    end=dict(id=0,parent='node-0000',s=r.next_s,nodes=names,accepted=True,direct='node-0001')
    return r,end


def test_four_leg_acceptance_advances_only_direct_node():
    r,p=four_leg_replay();r.consume('edge_end',p)
    assert r.witnessed=='node-0001' and r.accepted==[0]
    assert set(r.summary()['qualified_unwitnessed_nodes'])=={'node-0002','node-0003','node-0004'}


def test_failed_phase_link_cannot_count_as_witness():
    r,p=four_leg_replay();r.matches[0]['passed']=False
    with pytest.raises(ValueError):r.consume('edge_end',p)


def test_missing_reverse_common_control_cannot_advance():
    r,p=four_leg_replay();del r.common[1]
    with pytest.raises(ValueError):r.consume('edge_end',p)


def test_cyclic_phase_carry_is_not_relabelled():
    r,_=four_leg_replay();r.nodes['node-0001']['carried_phase']=1
    p=dict(parent='node-0000',child='node-0001',**M.phase_match(r.initial['pair'],r.initial['pair']))
    with pytest.raises(ValueError):r.consume('pair_match',p)


def test_common_end_cannot_replace_its_declared_endpoints():
    r,_=four_leg_replay();r.common={}
    startp=dict(id=0,reference='node-0001',candidate='node-0003',canonical_h=.99,candidate_original_h=.99,h_difference=0.,passed=False)
    r.consume('common_start',startp)
    with pytest.raises(ValueError):r.consume('common_end',dict(startp,candidate='node-0004',passed=False,error='synthetic'))


def test_close_h_is_not_byte_identical_common_control():
    r,_=four_leg_replay();r.common={}
    p=dict(id=0,reference='node-0001',candidate='node-0003',canonical_h=.99,candidate_original_h=.99,h_difference=0.,passed=False)
    r.consume('common_start',p)
    r.states['common0']=dict(pair=r.initial['pair'].copy(),h=np.nextafter(.99,1.),passed=True)
    with pytest.raises(ValueError):r.consume('common_end',dict(p,passed=True))


def test_failed_svd_residual_remains_a_retained_diagnostic():
    mat=np.diag([2.,1.]);u=np.eye(2);u[0,0]=1.1
    d=M.svd_residuals(mat,dict(matrix=mat,U=u,singular_values=np.array([2.,1.]),Vh=np.eye(2)))
    assert d['reconstruction']>1e-10 and d['orthogonality']>1e-10


def test_complex_left_residual_uses_hermitian_convention():
    mat=np.array([[0.,-1.],[1.,0.]])
    vectors=np.array([[1.,1.],[-1j,1j]],dtype=np.complex128)/np.sqrt(2.)
    eig=dict(values=np.array([1j,-1j]),left=vectors.copy(),right=vectors.copy())
    nsvd=dict(matrix=mat,U=mat.copy(),singular_values=np.ones(2),Vh=np.eye(2))
    condmat=mat-np.eye(2)
    csvd=dict(matrix=condmat,U=condmat/np.sqrt(2.),singular_values=np.full(2,np.sqrt(2.)),Vh=np.eye(2))
    d=M.spectrum(mat,eig,nsvd,csvd)
    assert max(d['right_residual'])<1e-15 and max(d['left_residual'])<1e-15
    eig['left']=eig['left'][:,::-1].copy()
    failed=M.spectrum(mat,eig,nsvd,csvd)
    assert max(failed['left_residual'])>1e-10


def synthetic_trace():
    import base64,gzip,json
    p=ROOT/'tests/reviews/fixtures/pseudoarclength-synthetic-two-edges.json.gz'
    def decode(v):
        if isinstance(v,dict) and 'synthetic_array_bytes_b64' in v:
            return np.frombuffer(base64.b64decode(v['synthetic_array_bytes_b64']),dtype=v['dtype']).reshape(v['shape']).copy()
        if isinstance(v,dict):return {k:decode(x) for k,x in v.items()}
        if isinstance(v,list):return [decode(x) for x in v]
        return v
    return decode(json.loads(gzip.decompress(p.read_bytes())))


def replay_trace(d):
    r=A.Replay(d['initial'],hidden=d['hidden'])
    for kind,payload in d['events']:r.consume(kind,payload)
    return r


def test_full_saved_synthetic_trace_without_decompositions():
    d=synthetic_trace();r=replay_trace(d)
    assert r.accepted==[0,1] and r.witnessed==d['result']['last_witnessed']
    assert r.attempted==d['attempted'] and r.returned==d['returned']
    assert r.summary()['pending_operation'] is None


@pytest.mark.parametrize('mutation',('callback_order','map_image','trial_state','missing_reverse','accepted_phase_link'))
def test_saved_synthetic_graph_rejects_consistent_local_corruptions(mutation):
    d=synthetic_trace();events=d['events']
    if mutation=='callback_order':
        i=next(i for i,(kind,p) in enumerate(events) if kind=='operation_start')
        events[i],events[i+1]=events[i+1],events[i]
    elif mutation=='map_image':
        p=next(p for kind,p in events if kind=='operation_return' and p['kind']=='map')
        p['value']['image']=p['value']['image'].copy();p['value']['image'][0]+=.001
    elif mutation=='trial_state':
        p=next(p for kind,p in events if kind=='armijo_trial')
        p['state']=p['state'].copy();p['state'][0]+=.001
    elif mutation=='missing_reverse':
        i=next(i for i,(kind,p) in enumerate(events) if kind=='common_end' and p['id']==1)
        del events[i]
    else:
        p=next(p for kind,p in events if kind=='pair_match' and not p['replay'])
        p['passed']=False
    with pytest.raises((ValueError,KeyError)):
        replay_trace(d)


def test_frozen_producer_source_inventory_is_exact():
    freeze=A.G.read(ROOT/A.FROZEN)
    A.source_check(freeze['files'],freeze['commit'])


def test_frozen_producer_source_rejects_changed_identity():
    import copy
    freeze=A.G.read(ROOT/A.FROZEN);changed=copy.deepcopy(freeze['files'])
    changed[next(iter(changed))]['sha256']='0'*64
    with pytest.raises(ValueError):A.source_check(changed,freeze['commit'])


def incomplete_provenance(monkeypatch,tmp_path):
    raw=tmp_path/'absent-raw';launch=tmp_path/'launch'
    monkeypatch.setattr(A,'RAW',raw);monkeypatch.setattr(A,'LAUNCH',launch)
    pins=dict(execution_commit='a'*40,source_review_commit='9df5e138fbba1b283c18a8d47f49ebe9a69b15ae',raw_manifest_sha256=None,launcher_manifest_sha256='b'*64)
    files={k:dict(bytes=1,sha256='c'*64) for k in ('launcher-source.py','runtime-start.json','stdout.log','receipt.json','manifest.json')}
    source={'tools/launch_s1r_pseudoarclength.py':files['launcher-source.py']}
    ls=dict(source_commit=pins['execution_commit'],review_commit=pins['source_review_commit'],sources=source,
        out=str(raw),launcher=str(launch),command=['timeout','--signal=TERM','--kill-after=5s','75s','nice','-n','10','/data2/neural-shrimp/.venv/bin/python','tools/explore_s1r_pseudoarclength.py','--out',str(raw),'--review-commit',pins['source_review_commit']],
        nice=10,thread_env={k:'1' for k in A.THREADS},gpu_used=False,external_seconds=75,kill_grace_seconds=5,
        supervisor_seconds=80.,rss_cap_bytes=2*1024**3,store_bytes=127*1024**2,launcher_bytes=1024**2,
        log_bytes=128*1024,group_pid=123)
    lr=dict(source_commit=pins['execution_commit'],review_commit=pins['source_review_commit'],group_pid=123,
        owned_live_pids_remaining=[],elapsed_seconds=1.,max_group_rss_bytes=0,raw_inventory={},raw_file_count=0,
        raw_bytes=0,within_store_cap=True,raw_manifest_verified=False,successful=False)
    monkeypatch.setattr(A.G,'inventory',lambda folder,pin:files)
    monkeypatch.setattr(A.G,'read',lambda path:ls if path.name=='runtime-start.json' else lr)
    monkeypatch.setattr(A,'source_check',lambda sources,commit:None)
    return pins,ls,lr,files


def test_absent_raw_is_inventory_only_not_science(monkeypatch,tmp_path):
    pins,ls,lr,files=incomplete_provenance(monkeypatch,tmp_path)
    rows=[];rf,result,early=A.producer_provenance(pins,lambda name,payload:rows.append(name))
    assert rf=={} and result is None and not early['arithmetic_qualified']
    assert early['verdict']=='incomplete_inventory_only' and rows==['provenance.json']


@pytest.mark.parametrize('mutation',('undeclared_file','command','live_pid','inventory','elapsed','false_success'))
def test_launcher_provenance_contradictions_rejected(monkeypatch,tmp_path,mutation):
    pins,ls,lr,files=incomplete_provenance(monkeypatch,tmp_path)
    if mutation=='undeclared_file':files['extra.bin']=dict(bytes=1,sha256='d'*64)
    elif mutation=='command':ls['command'][3]='750s'
    elif mutation=='live_pid':lr['owned_live_pids_remaining']=[999]
    elif mutation=='inventory':lr['raw_inventory']={'unbound.npy':dict(bytes=1,sha256='e'*64)}
    elif mutation=='elapsed':lr['elapsed_seconds']=82.
    else:lr['successful']=True
    with pytest.raises(ValueError):A.producer_provenance(pins,lambda *args:None)


def header_events(bound):
    cost=2*(1.1480005020275712*max(256/12,102/24)+38.103352084057406*(16/473))+5
    return [['bound_inputs',bound],['cost_gate',dict(projected_seconds=cost,limit_seconds=60.,passed=True,scope='historical heuristic only')]]


def save_synthetic_records(folder,events):
    spec=importlib.util.spec_from_file_location('pal_author_test_helpers',ROOT/'tests/reviews/test_s1r_pseudoarclength_audit.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    return helper.write_records(folder,events)


def test_unjournaled_stage_start_remains_partial(tmp_path):
    d=synthetic_trace();bound={'synthetic':True};folder=tmp_path/'prefix'
    files=save_synthetic_records(folder,header_events(bound))
    result=dict(status='stopped',operation_attempted=dict.fromkeys(A.CAPS,0),operation_returned=dict.fromkeys(A.CAPS,0),
        stage_counts=dict(correctors=0,qualifiers=1,common=0,edges=0,assemblies=0),accepted_edges=[],last_witnessed=None)
    out=A.records(folder,files,result,d['initial'],bound,hidden=1)
    assert out['arithmetic_complete'] is False


@pytest.mark.parametrize('returned_ahead',(False,True))
def test_pending_analytic_return_boundary_is_partial(tmp_path,returned_ahead):
    d=synthetic_trace();bound={'synthetic':True};events=header_events(bound)
    for kind,p in d['events']:
        events.append([kind,p])
        if kind=='operation_start':break
    r=A.Replay(d['initial'],hidden=1)
    for kind,p in events:r.consume(kind,p)
    folder=tmp_path/'pending';files=save_synthetic_records(folder,events)
    returned=dict(r.returned)
    if returned_ahead:returned['analytic']+=1
    result=dict(status='stopped',operation_attempted=r.attempted,operation_returned=returned,
        stage_counts=dict(correctors=0,qualifiers=1,common=0,edges=0,assemblies=1),accepted_edges=[],last_witnessed=None)
    out=A.records(folder,files,result,d['initial'],bound,hidden=1)
    assert not out['arithmetic_complete'] and out['pending_operation']==0


def test_complete_trace_requires_bound_input_header(tmp_path):
    d=synthetic_trace();folder=tmp_path/'noheader';files=save_synthetic_records(folder,d['events'])
    result=dict(**d['result'],operation_attempted=d['attempted'],operation_returned=d['returned'],stage_counts=d['counts'])
    with pytest.raises(ValueError):A.records(folder,files,result,d['initial'],{},hidden=1)
