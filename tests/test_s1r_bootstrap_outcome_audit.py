"""Synthetic audit controls; never read S1R profile/collection outcomes."""
from copy import deepcopy
from pathlib import Path
import sys
import json
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import audit_s1r_bootstrap_outcome as a
from neural_shrimp import period2_branch as core
from neural_shrimp.s1r_bootstrap import HiddenBridge,compare_common_controls
from neural_shrimp.period2_branch_io import Store,file_entry


def fake_pair(params):
    x=np.zeros(72);x[0]=1
    return np.stack([x,-x])+np.array([.001*(params[0]-a.HIDDEN)]+[0.]*71)


def fake_node(params,reverse=False):
    pair=fake_pair(params)
    return dict(accepted=True,reason='qualified',pair=pair[::-1] if reverse else pair,
                spectra=dict(phases=[dict(stability='attracting')]*2))


def fixture(monkeypatch,reject=False):
    ev=[];calls=[0]
    def emit(k,p):ev.append((k,deepcopy(p)))
    def qualify(family,params,seeds,o,budget,emit):
        calls[0]+=1
        if reject and calls[0]>=(4 if reject is True else reject):return dict(accepted=False,reason='first_phase_corrector',corrections=[])
        return fake_node(params,calls[0]%2==0)
    monkeypatch.setattr(core,'qualify_pair',qualify)
    engine=HiddenBridge(lambda p:None,core.Options(),core.Budget(60),emit)
    inputs=dict(central_params=np.array([a.HIDDEN,a.OUTPUT]),central_pair=fake_pair([a.HIDDEN]))
    result=dict(status='running',bootstraps=[],bridge=[],association=None,link_attempts=0)
    qualified={}
    for start in a.STARTS:
        label='start_'+str(start);case=dict(label=label,params=[start,a.OUTPUT],status='running',node=None);emit('bootstrap_start',case)
        pair=fake_pair([start]);tail=np.stack([pair[1],pair[0]])
        emit('bootstrap_terminal',dict(label=label,tail=tail[None,:]))
        emit('bootstrap_screen',dict(label=label,result=dict(period=np.array([2]),status=np.array([2])),qualification_seed_indices=[40000,39999]))
        node=engine.node(np.array([start,a.OUTPUT]),pair,'bootstrap:'+label);qualified[start]=node
        case.update(period=2,screen_status=2,status='qualified_pair',node=node['id']);emit('bootstrap_result',case);result['bootstraps'].append(case)
    parent=engine.node(inputs['central_params'],inputs['central_pair'],'saved_central_requalification');result['central']=parent['id'];previous=None
    for i in range(1,13):
        target=np.array([1.1 if i==12 else a.HIDDEN+(1.1-a.HIDDEN)*i/12,a.OUTPUT])
        segment,parent,previous=engine.hidden_segment(parent,target,previous);segment['macro_index']=i;emit('bridge_segment',segment);result['bridge'].append(segment)
        if segment['status']!='connected':result['status']='bridge_gap';break
    else:
        result['association']=compare_common_controls(parent,qualified[1.1]);emit('association',result['association']);result['status']='completed_common_control_comparison'
    result['link_attempts']=len(engine.edges)
    return ev,result,inputs


def test_independent_all_twelve_links_secants_and_native_phase_propagation(monkeypatch):
    ev,r,inputs=fixture(monkeypatch);out=a.Structure(ev,r).run(inputs)
    assert out['accepted_links']==12 and out['completed_nodes']==51 and out['association']['status']=='same_pair_within_tolerance'


@pytest.mark.parametrize('mutation',['seed','phase','half','reverse','control','verdict','extra'])
def test_witness_tamper_rejected(monkeypatch,mutation):
    ev,r,inputs=fixture(monkeypatch)
    if mutation=='seed':next(p for k,p in ev if k=='node_pending' and p['purpose']=='hidden_bridge:direct')['seeds'][0][2]+=1e-6
    if mutation=='phase':next(p for k,p in ev if k=='edge_pending')['parent_phase']=1
    if mutation=='half':next(p for k,p in ev if k=='node_pending' and p['purpose']=='hidden_bridge:half')['params'][0]+=1e-6
    if mutation=='reverse':next(p for k,p in ev if k=='node_pending' and p['purpose']=='hidden_bridge:reverse')['seeds'][1][0]+=1e-6
    if mutation=='control':next(p for k,p in ev if k=='node_pending' and p['purpose']=='bootstrap:start_1.1')['rates'][0]=np.nextafter(10.**1.1,np.inf)
    if mutation=='verdict':r['association']['status']='numerically_distinct_pairs'
    if mutation=='extra':ev.append(('association',deepcopy(r['association'])))
    with pytest.raises(ValueError):a.Structure(ev,r).run(inputs)


@pytest.mark.parametrize('rejected_call',[4,5,6,7])
def test_rejected_direct_halving_and_gap_are_preserved(monkeypatch,rejected_call):
    ev,r,inputs=fixture(monkeypatch,rejected_call);out=a.Structure(ev,r).run(inputs)
    assert out['accepted_links']==0 and out['links']==4 and r['status']=='bridge_gap'
    r['status']='completed_common_control_comparison'
    with pytest.raises(ValueError):a.Structure(ev,r).run(inputs)


def test_truncated_node_keeps_only_pending_scope(monkeypatch):
    ev,r,inputs=fixture(monkeypatch);stop=next(i for i,(k,p) in enumerate(ev) if k=='node_pending' and p['purpose']=='hidden_bridge:direct')
    ev=ev[:stop+1];r.update(status='resource_stop',bridge=[],association=None,link_attempts=1)
    out=a.Structure(ev,r).run(inputs);assert out['pending_nodes']==1 and out['association'] is None
    r['status']='completed_common_control_comparison'
    with pytest.raises(ValueError):a.Structure(ev,r).run(inputs)


def test_common_controls_checked_before_matching(monkeypatch):
    n=dict(params=np.array([1.1,a.OUTPUT]),rates=np.array([10.**1.1,10.**a.OUTPUT]),**fake_node([1.1]))
    m=deepcopy(n);m['params'][0]=np.nextafter(m['params'][0],np.inf)
    monkeypatch.setattr(a,'pair_match',lambda *args:pytest.fail('matched before controls'))
    with pytest.raises(ValueError,match='identical controls'):a.common(n,m)


def snap(n=40000):
    tail=np.ones((1,max(0,n-39616),72));state=np.ones((1,72))
    return dict(completed_loops=n,planned_steps=40000,attempted_updates=np.array([n]),accepted_updates=np.array([n]),
      jvp_attempts=np.array([max(0,n-38800)]),jvp_successful_returns=np.array([max(0,n-38800)]),state=state,
      last_valid_state=state.copy(),alive=np.array([True]),diverged_at=np.array([-1]),tangent=state.copy(),
      tail=tail,tail_losses=np.ones((1,len(tail[0]))),tail_state_updates=np.arange(39617,max(39617,n+1)),
      tail_loss_updates=np.arange(39616,max(39616,n)))


def test_all_40000_counts_and_1200_jvps():
    p=snap();r=a.snapshot(p,np.ones(72));assert r['accepted']==40000 and r['jvp_returns']==1200
    p['jvp_successful_returns'][0]-=1
    with pytest.raises(ValueError):a.snapshot(p,np.ones(72))


def test_interrupted_jvp_attempt_not_successfully_returned():
    p=snap(38900);p['attempted_updates']+=1;p['jvp_attempts']+=1
    assert a.snapshot(p,np.ones(72))['jvp_returns']==100
    p['accepted_updates']+=1
    with pytest.raises(ValueError):a.snapshot(p,np.ones(72))


@pytest.mark.parametrize('key',['tail_state_updates','tail_loss_updates','state','planned_steps'])
def test_warmup_index_and_latest_state_tamper(key):
    p=snap()
    if key=='planned_steps':p[key]=39999
    else:p[key].flat[0]+=1
    with pytest.raises(ValueError):a.snapshot(p,np.ones(72))


def test_independent_recurrence_fixed_period2_and_nonfinite():
    assert a.screen_tail(np.zeros((384,72)))[0]==1
    tail=np.tile(np.array([[0.,1.],[1.,0.]]),(192,1));assert a.screen_tail(tail)[0]==2
    tail[1,0]=np.nan;assert a.screen_tail(tail)[0]==-1


def test_manifest_and_journal_tamper(tmp_path):
    root=tmp_path/'raw';store=Store(root);store.emit('bootstrap_start',dict(label='start_1.1'));store.finish(dict(status='resource_stop'))
    sha=file_entry(root/'manifest.json')['sha256'];a.inventory(root,sha);a.events(root)
    (root/'extra').write_text('foreign')
    with pytest.raises(ValueError):a.inventory(root,sha)
    (root/'extra').unlink();event=next(root.glob('*-bootstrap_start.json'));obj=json.loads(event.read_text());obj['sequence']=9;event.write_text(json.dumps(obj))
    with pytest.raises(ValueError):a.events(root)


def qualified_neutral():
    x=np.zeros(72);x[0]=1.;pair=np.stack([x,-x]);f=lambda t:-t;j=lambda t:-np.eye(72)
    corrections=[dict(seed=p.copy(),state=p.copy(),converged=True) for p in pair]
    node=dict(id='synthetic',pair=pair,params=np.array([1.,1.]),corrections=corrections,checks=core.state_checks(f,pair,core.Options()),
      independent_phase_match=core.phase_match(pair,pair,core.Options(),replay=True),spectra=core.spectral_checks(j(x),j(x),core.Options()),accepted=True,reason='qualified')
    return node,dict(seeds=list(pair)),lambda p:(f,j)


def test_complete_native_spectra_and_residuals():
    node,pending,family=qualified_neutral();r=a.qualification(node,pending,family)
    assert r['accepted'] and all(p['stability']=='neutral_or_unresolved' for p in r['phases'])
    node['spectra']['phases'][0]['values'][0]+=.01
    with pytest.raises(ValueError):a.qualification(node,pending,family)


def test_spectral_status_cannot_replace_full_pair_closure():
    node,pending,family=qualified_neutral();node['pair'][1,1]=1.
    with pytest.raises(ValueError):a.qualification(node,pending,family)


def test_independent_gradient_hessian_synthetic_finite_differences():
    rng=np.random.default_rng(7);x=rng.normal(size=(7,2));y=rng.normal(size=7);t=rng.normal(size=6)
    model=dict(d=2,width=2,activation='tanh',bias=False,in_scale=.5,out_scale=.125,loss_scale=1.,ridge=.001)
    loss,g,h=a.loss_gradient_hessian(t,x,y,model);eps=1e-5;basis=np.eye(6)
    fdg=np.array([(a.loss_gradient_hessian(t+eps*v,x,y,model)[0]-a.loss_gradient_hessian(t-eps*v,x,y,model)[0])/(2*eps) for v in basis])
    fdh=np.column_stack([(a.loss_gradient_hessian(t+eps*v,x,y,model)[1]-a.loss_gradient_hessian(t-eps*v,x,y,model)[1])/(2*eps) for v in basis])
    np.testing.assert_allclose(g,fdg,atol=2e-10,rtol=2e-8);np.testing.assert_allclose(h,fdh,atol=2e-10,rtol=2e-8)


def test_saved_corrector_step_arithmetic_without_solver(monkeypatch):
    seed=np.ones(72);step=np.zeros(72);iteration=dict(call=1,iteration=1,state=seed.copy(),step=step,residual=step,final_relative_correction=0.)
    start=dict(call=1,state=seed.copy(),seed=seed.copy());end=dict(call=1,state=seed.copy(),seed=seed.copy(),converged=True,reason='converged',iterations=1,final_relative_correction=0.,candidate_distance=0.)
    events=[('node_pending',dict(id='node-0000')),('corrector_start',start),('corrector_iteration',iteration),('corrector_end',end),('node',dict(id='node-0000',corrections=[end]))]
    monkeypatch.setattr(np.linalg,'solve',lambda *args:pytest.fail('new solve'))
    assert a.corrector_history(events,dict(status='resource_stop',phase_corrector_calls=1))['recorded_ends']==1
    iteration['step'][0]=1e-3
    with pytest.raises(ValueError):a.corrector_history(events,dict(status='resource_stop',phase_corrector_calls=1))


def test_historical_journal_uses_separate_schema(tmp_path):
    root=tmp_path/'old';store=Store(root)
    for kind in ('inputs','rate_identity','anchor_feasibility','hidden_link','segment'):store.emit(kind,dict(synthetic=True))
    store.finish(dict(status='completed_local_segment'))
    assert len(a.events(root,historical=True)[0])==5
    with pytest.raises(ValueError,match='unknown event'):a.events(root)


def launcher_fixture():
    out=Path('/synthetic/collection');profile=Path('/synthetic/profile')
    runtime=dict(mode='collect',source_commit='a'*40,pid=101)
    command=['timeout','--signal=TERM','--kill-after=5s','115s','nice','-n','10','/data2/neural-shrimp/.venv/bin/python',
      'tools/explore_s1r_bootstrap_association.py','collect','--review-commit',a.REVIEW,'--profile',str(profile),'--out',str(out)]
    start=dict(source_commit='a'*40,source_review_commit=a.REVIEW,cwd='/home/d7082791602/.codex/worktrees/tiny-lm-prediction-review/neural-shrimp',
      out=str(out),command=command,thread_env=dict.fromkeys(a.THREADS,'1'),mode='one_collection',target_collection_authorized=True,
      started_utc='2026-10-08T00:00:00+00:00')
    for key in ('uptime','memory','gpu','gpu_processes'):start[key]=dict(command=['synthetic'],exit_code=0,stdout='',stderr='')
    receipt=dict(**deepcopy(start),owned_pids_remaining=[],observed_owned_pids=[100,101],timeout_pid=100,exit_code=0,elapsed_seconds=2.,finished_utc='2026-10-08T00:00:02+00:00')
    result=dict(status='upper_bootstrap_unqualified',elapsed_seconds=1.)
    return start,receipt,result,runtime,out,profile


def test_launcher_record_binding_and_child_limit():
    args=launcher_fixture();assert a.launcher_records(*args)['exit_code']==0
    args[2]['elapsed_seconds']=121
    with pytest.raises(ValueError):a.launcher_records(*args)


@pytest.mark.parametrize('key',['command','thread_env','source_commit','exit_code','owned_pids_remaining','elapsed_seconds'])
def test_launcher_metadata_tamper(key):
    args=launcher_fixture();receipt=args[1]
    if key=='command':receipt[key][3]='999s'
    if key=='thread_env':receipt[key]['OMP_NUM_THREADS']='2'
    if key=='source_commit':receipt[key]='b'*40
    if key=='exit_code':receipt[key]=124
    if key=='owned_pids_remaining':receipt[key]=[101]
    if key=='elapsed_seconds':receipt[key]=float('nan')
    with pytest.raises(ValueError):a.launcher_records(*args)


def test_complete_warmup_counter_scope_and_saved_tail_replay(monkeypatch):
    ev=[('bootstrap_start',dict(label='start_1.1'))]
    for n in range(0,40001,1000):
        p=snap(n);p.update(label='start_1.1',last_evaluated_loss=np.array([1.]))
        ev.append(('bootstrap_checkpoint',p))
    terminal=deepcopy(ev[-1][1]);ev.append(('bootstrap_terminal',terminal))
    screen=dict(label='start_1.1',full_horizon=True,qualification_seed_indices=None,status='bootstrap_not_period_two',
      result=dict(final_state=terminal['state'],period=np.array([1]),recurrence_residual=np.array([0.]),status=np.array([1]),
        last_evaluated_loss=np.array([1.]),tail_mean_loss=np.array([1.])))
    ev.append(('bootstrap_screen',screen))
    monkeypatch.setattr(a,'loss_gradient_hessian',lambda t,x,y,m:(1.,np.zeros(72),np.eye(72)))
    result=a.warm_audit(ev,dict(theta0=np.ones(72),x=None,y=None,model=None))
    assert result[0]['accepted']==40000 and result[0]['independent_tail_transitions']==383
    assert result[1]['status']=='no_saved_snapshot'
    terminal['tail_losses'][0,12]=2.
    with pytest.raises(ValueError):a.warm_audit(ev,dict(theta0=np.ones(72),x=None,y=None,model=None))
