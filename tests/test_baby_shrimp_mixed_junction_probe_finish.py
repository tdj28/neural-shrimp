"""Independent algebra and corruption controls for BSC6 saved evidence."""
from pathlib import Path
import sys
import copy
import hashlib
import json
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
import baby_shrimp_mixed_junction_probe_finish as audit


def jet(value, gradient, hessian):
    return {"value": value, "gradient": list(gradient), "hessian": np.asarray(hessian).tolist()}


def test_two_conditional_limits_recover_distinct_signed_multiplier_jets():
    p, m = 1., -1.
    gp, gm = np.array([.3,.2]), np.array([-.1,.4])
    hp, hm = np.array([[.2,.1],[.1,-.3]]), np.array([[.4,-.2],[-.2,.1]])
    tau = jet(p+m, gp+gm, hp+hm)
    delta = jet(p*m, m*gp+p*gm, m*hp+p*hm+np.outer(gp,gm)+np.outer(gm,gp))
    result = audit.independent_limit_jets(tau,delta)
    for key,g,H in (("positive_tau_side",gp,hp),("negative_tau_side",-gm,-hm)):
        np.testing.assert_allclose(result[key]["gradient"],g,atol=1e-15)
        np.testing.assert_allclose(result[key]["hessian"],H,atol=1e-15)
    assert result["actual_tau"] == 0 and result["normal_displacement_norm"] == 0


def test_nonzero_measured_trace_is_retained_and_not_an_applied_coordinate_shift():
    tau=jet(2e-12,[1e-5,2e-5],[[.02,0],[0,-.04]])
    delta=jet(-1e-8,[1e-9,-2e-9],[[0,0],[0,0]])
    result=audit.independent_limit_jets(tau,delta)
    assert result["actual_tau"]==2e-12 and result["conditional_tau"]==0
    np.testing.assert_allclose(result["normal_displacement_estimate"],-2e-12*np.array([1e-5,2e-5])/5e-10)
    assert "unchanged coordinates" in result["scope"] and "certified" in result["scope"]


@pytest.mark.parametrize("change",["singular","delta","nan","asymmetric"])
def test_conditional_limit_refusals(change):
    tau=jet(0.,[1.,1.],[[1,0],[0,-1]])
    delta=jet(-1.,[.1,.2],[[0,0],[0,0]])
    if change=="singular":tau["gradient"]=[0,0]
    elif change=="delta":delta["value"]=0
    elif change=="nan":delta["gradient"][0]=np.nan
    else:tau["hessian"][0][1]=.1
    with pytest.raises(ValueError):audit.independent_limit_jets(tau,delta)


def test_transported_feature_reverses_only_with_reference_and_preserves_degeneracy():
    field=jet(1.,[2.,3.],[[1,0],[0,4]])
    plus=audit.independent_feature(field,[0,1]);minus=audit.independent_feature(field,[0,-1])
    assert plus["feature"]==3 and minus["feature"]==-3
    assert audit.independent_feature(field,[1,0])["status"]=="orthogonal_reference"
    assert audit.independent_feature(jet(1.,[1,1],[[1,0],[0,1]]),[1,0])["feature"] is None


def test_declared_error_balls_bound_independent_perturbed_branch_jets():
    tau=jet(1e-14,[2e-5,-1e-5],[[.02,.01],[.01,-.04]])
    delta=jet(-7e-9,[-4e-10,9e-10],[[4e-11,2e-11],[2e-11,-8e-11]])
    errors=dict(delta_value=2e-17,tau_gradient=2e-9,delta_gradient=1e-13,tau_hessian=2e-7,delta_hessian=2e-11)
    refs={"positive_tau_side":[1.,1.],"negative_tau_side":[1.,1.]}
    bounds=audit.independent_jet_error_bounds(tau,delta,errors,refs)
    original=audit.independent_limit_jets(tau,delta)
    rng=np.random.default_rng(512)
    def unit_noise(shape):
        x=rng.normal(size=shape)
        if len(shape)==2:x=(x+x.T)/2
        return x/np.linalg.norm(x,2 if len(shape)==2 else None)*rng.random()
    for _ in range(100):
        t=jet(tau['value'],np.asarray(tau['gradient'])+errors['tau_gradient']*unit_noise((2,)),
              np.asarray(tau['hessian'])+errors['tau_hessian']*unit_noise((2,2)))
        d=jet(delta['value']+errors['delta_value']*rng.uniform(-1,1),
              np.asarray(delta['gradient'])+errors['delta_gradient']*unit_noise((2,)),
              np.asarray(delta['hessian'])+errors['delta_hessian']*unit_noise((2,2)))
        changed=audit.independent_limit_jets(t,d)
        for key in refs:
            assert np.linalg.norm(np.asarray(changed[key]['gradient'])-original[key]['gradient'])<=bounds['gradient_error']
            assert np.linalg.norm(np.asarray(changed[key]['hessian'])-original[key]['hessian'],2)<=bounds['hessian_error']
            x=audit.independent_feature(original[key],refs[key]);y=audit.independent_feature(changed[key],refs[key])
            assert abs(x['feature']-y['feature'])<=bounds['branches'][key]['feature_error']


def test_error_ball_reaching_determinant_zero_is_refused():
    with pytest.raises(ValueError,match='reaches zero'):
        audit.independent_jet_error_bounds(jet(0,[1,1],[[1,0],[0,-1]]),jet(-1,[1,1],[[0,0],[0,0]]),
            dict(delta_value=1,tau_gradient=0,delta_gradient=0,tau_hessian=0,delta_hessian=0),
            {'positive_tau_side':[1,1],'negative_tau_side':[1,1]})


def test_inverse_rotation_cannot_be_hidden_by_zero_nominal_scalar_transfer():
    J=np.eye(2);dJ=np.array([[0.,0.],[.1,0.]])
    e=np.array([1.,0.]);g=np.array([0.,1.])
    result=audit.independent_location_transfer(J,dJ,e,g,0.)
    actual=np.linalg.solve(J+dJ,e)
    assert result['f_minus_nominal_location_transfer']==0
    assert abs(g@actual)==pytest.approx(.1)
    assert result['f_minus_location_transfer']>=abs(g@actual)
    assert np.linalg.norm(actual)<=result['location_error_norm']
    # The nominal component box is not claimed to contain perturbed solutions.
    assert result['nominal_location_component_box'][1]==0 and actual[1]!=0


def test_location_transfer_covers_all_box_corners_with_jacobian_and_gradient_changes():
    J=np.array([[.7,.2],[-.1,1.1]]);dJ=np.array([[.01,-.03],[.02,-.01]])
    errors=np.array([.03,.02]);g=np.array([.8,-.3]);dg=np.array([-.002,.003])
    bound=audit.independent_location_transfer(J,dJ,errors,g,np.linalg.norm(dg))
    for a in (-1,1):
        for b in (-1,1):
            actual=np.linalg.solve(J+dJ,errors*np.array([a,b]))
            assert np.linalg.norm(actual)<=bound['location_error_norm']
            assert abs((g+dg)@actual)<=bound['f_minus_location_transfer']


# The worker is used only to manufacture synthetic saved records for corruption
# tests. All scientific replay in the auditor is implemented independently.
sys.path.insert(0,str(Path(__file__).parent))
from test_baby_shrimp_mixed_junction_probe import configuration, synthetic_fields, AnalyticEvaluator
import baby_shrimp_mixed_junction_probe as worker
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    audit.os.environ[key]='1'


def test_independent_pure_replay_matches_saved_worker_geometry_correction_and_uncertainty():
    cfg=configuration();fields,_=synthetic_fields();refs=cfg['target']['references']
    geometry=worker.conditional_geometry(fields,refs,cfg)
    corrected=worker.newton_correction(fields,refs,cfg)
    replay_geom=audit.geometry_from_fields(fields,refs,cfg)
    replay_corrected=audit.correction_from_fields(fields,refs,cfg)
    audit.compare_numeric_tree(geometry,replay_geom,'geometry')
    audit.compare_numeric_tree(corrected,replay_corrected,'corrector')
    primary={'step':.0003,'fields':fields};check={'step':.0001,'fields':fields}
    spans={'tau':1e-20,'delta':1e-20}
    stored=worker.empirical_uncertainty(primary,check,geometry,geometry,corrected,corrected,spans,spans,cfg)
    rebuilt=audit.uncertainty_from_jets(primary,check,replay_geom,replay_geom,replay_corrected,replay_corrected,spans,spans,cfg)
    audit.compare_numeric_tree(stored,rebuilt,'uncertainty')


def case_fixture():
    cfg=configuration();cfg['target']['coordinate']=[2e-5,-1e-5]
    evaluate=AnalyticEvaluator();refs=cfg['target']['references'];cases=[]
    for offset in cfg['beta_offsets']:
        center,parent=worker.case_predictor(cfg['target'],offset,cases[0] if cases else None)
        case=worker.probe_case(evaluate,center,.006+offset,offset,refs,{'tau':1e-20,'delta':1e-20},cfg,lambda _:None,predictor_parent=parent)
        cases.append(case)
        if offset==0:refs=case['final_references']
    result={'complete':True,'cases':cases,'resolved_f_minus_sign_bracket':{cases[1]['f_minus']['status'],cases[2]['f_minus']['status']}=={'resolved_positive','resolved_negative'}}
    raw={p['index']:p for p in evaluate.rows}
    return result,raw,set(raw),{'tau':1e-20,'delta':1e-20},cfg


def test_complete_three_case_synthetic_measurements_and_moving_parent_replay():
    args=case_fixture()
    cases=audit.audit_cases(*args)
    assert len(cases)==3 and all(c['complete'] for c in cases)
    assert cases[1]['predictor_parent']['coordinate']==cases[0]['final_center']


@pytest.mark.parametrize('change',['actual tau','projection','common Hessian','stencil','corrector','half step','normal','location transfer','precision gate','sign label','endpoint','moving parent','dropped admission','bracket'])
def test_case_scientific_corruptions_are_rejected(change):
    result,raw,admitted,spans,cfg=case_fixture();case=result['cases'][0];row=case['iterations'][-1]
    if change=='actual tau':row['coarse_geometry']['input_tau']+=1e-7
    elif change=='projection':row['coarse_geometry']['projection']['spatial_displacement_applied']=True
    elif change=='common Hessian':row['coarse_geometry']['branches']['negative_tau_side']['jet']['hessian']=row['coarse_geometry']['branches']['positive_tau_side']['jet']['hessian']
    elif change=='stencil':row['coarse_jet']['point_indices'][0][0]=row['coarse_jet']['point_indices'][1][1]
    elif change=='corrector':row['coarse_correction']['correction'][0]+=.01
    elif change=='half step':row['coarse_correction']['jacobian_half_step'][0][0]+=.01
    elif change=='normal':row['fine_geometry']['branches']['positive_tau_side']['valley']['normal'][0]*=-1
    elif change=='location transfer':row['uncertainty']['intersection']['f_minus_location_transfer']=0.
    elif change=='precision gate':row['admission']['checks']['f_minus_total_precision']=False
    elif change=='sign label':case['f_minus']['status']='resolved_positive' if case['f_minus']['status']!='resolved_positive' else 'resolved_negative'
    elif change=='endpoint':case['final_point_index']=0
    elif change=='moving parent':result['cases'][1]['predictor_parent']['coordinate']=cfg['target']['coordinate']
    elif change=='dropped admission':admitted.remove(row['coarse_jet']['point_indices'][0][0])
    else:result['resolved_f_minus_sign_bracket']=not result['resolved_f_minus_sign_bracket']
    with pytest.raises(ValueError):audit.audit_cases(result,raw,admitted,spans,cfg)


def test_terminal_partial_case_keeps_measured_prefix_without_inventing_targets():
    result,raw,admitted,spans,cfg=case_fixture()
    case=result['cases'][0]
    case.update(complete=False,status='running')
    for name in ('final_center','final_point_index','f_minus','final_references'):case.pop(name)
    last=case['iterations'][-1]
    for name in ('fine_geometry','fine_correction','cyclic_point_index','cyclic_variations','uncertainty','admission'):last.pop(name)
    result={'complete':False,'cases':[case]}
    replay=audit.audit_cases(result,raw,admitted,spans,cfg)
    assert len(replay)==1 and not replay[0]['complete']
    assert 'fine_jet' in replay[0]['iterations'][-1] and 'admission' not in replay[0]['iterations'][-1]
    result['complete']=True
    with pytest.raises(ValueError,match='every admitted target'):audit.audit_cases(result,raw,admitted,spans,cfg)


@pytest.mark.parametrize('prefix',['coarse','fine'])
def test_derived_geometry_cannot_survive_missing_measured_stencil(prefix):
    result,raw,admitted,spans,cfg=case_fixture()
    result['cases'][0]['iterations'][-1].pop(prefix+'_jet')
    with pytest.raises(ValueError,match='measured stencil'):audit.audit_cases(result,raw,admitted,spans,cfg)


def test_sign_margin_does_not_override_absolute_precision_gate():
    result,raw,_,_,cfg=case_fixture();row=result['cases'][0]['iterations'][-1]
    cfg['maximum_f_minus_total_uncertainty']=5e-8
    row['uncertainty']['intersection']['f_minus_total_error']=5.40317e-8
    row['coarse_correction']['f_minus']=10.5*5.40317e-8
    decision=audit.admission_from_values(row['coarse_geometry'],row['coarse_correction'],row['uncertainty'],
                                        raw[row['cyclic_point_index']],cfg)
    assert not decision['passed'] and not decision['checks']['f_minus_total_precision']
    assert decision['f_minus']['status']=='not_admitted'
    assert decision['f_minus']['ungated_sign_diagnostic']=='resolved_positive'


def test_provenance_sequences_keep_status_labels_and_supporting_threads():
    provenance={'section_statuses':[{'arm_id':'ray','status':'low_signal','complete':True}],
                'retained_unresolved_stages':['remaining sections'],'supporting_audit_blas_threads':1}
    audit.compare_numeric_tree(copy.deepcopy(provenance),provenance,'prior')
    wrong=copy.deepcopy(provenance);wrong['section_statuses'][0]['status']='converged'
    with pytest.raises(ValueError,match='label'):audit.compare_numeric_tree(wrong,provenance,'prior')


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def data_root(monkeypatch,path):
    monkeypatch.setattr(audit,'DATA',path)
    monkeypatch.setattr(audit.previous,'DATA',path)


def alignment_fixture(tmp_path):
    ref={'index':0,'phases':[{'phase':p} for p in (0,6,12)]}
    raw={0:ref,1:{'index':1}}
    rng=np.random.default_rng(328)
    orbit=2+rng.normal(size=(18,4))
    for index,values in ((0,orbit),(1,orbit+1e-6)):
        directory=tmp_path/f'point-{index:04d}';directory.mkdir()
        np.save(directory/'points.npy',values)
    summary={'index':1,'qualified':True,'closure_recomputed_from_saved_endpoint':True}
    distances=[float(np.max(np.linalg.norm(np.roll(orbit+1e-6,k,axis=0)-orbit,axis=1)/np.linalg.norm(orbit,axis=1))) for k in range(18)]
    record={'point_index':1,'reference_point_index':0,'status':'passed','audit':summary,
            'all_alignment_maximum_relative_changes':distances,'best_right_roll':0,'passed':True}
    return raw,[summary],ref,record


def test_alignment_replayed_from_full_saved_orbit_and_incomplete_admissions_retained(tmp_path):
    raw,summaries,ref,row=alignment_fixture(tmp_path)
    checked,admitted=audit.audit_point_admissions(tmp_path,[row],raw,summaries,ref,configuration())
    assert admitted=={1} and checked[0]['passed']
    for status in ('started','failed'):
        row={'point_index':1,'reference_point_index':0,'status':status}
        if status=='failed':row.update(error_type='ValueError',error='saved-array check failed')
        checked,admitted=audit.audit_point_admissions(tmp_path,[row],raw,summaries,ref,configuration())
        assert admitted==set() and checked[0]['status']==status


@pytest.mark.parametrize('change',['roll','distance','summary','reference','duplicate','audit missing','contradictory failure'])
def test_state_alignment_corruption_is_rejected(tmp_path,change):
    raw,summaries,ref,row=alignment_fixture(tmp_path);records=[row]
    if change=='roll':
        path=tmp_path/'point-0001/points.npy';np.save(path,np.roll(np.load(path),1,axis=0))
    elif change=='distance':row['all_alignment_maximum_relative_changes'][0]=0.
    elif change=='summary':row['audit']=dict(row['audit'],qualified=False)
    elif change=='reference':row['reference_point_index']=1
    elif change=='duplicate':records.append(copy.deepcopy(row))
    elif change=='audit missing':row.pop('audit')
    else:row['status']='failed orbit alignment'
    with pytest.raises(ValueError):audit.audit_point_admissions(tmp_path,records,raw,summaries,ref,configuration())


def terminal_fixture(tmp_path,monkeypatch,*,complete=False,python_manifest=True):
    data_root(monkeypatch,tmp_path)
    run=tmp_path/'explore/run';run.mkdir(parents=True)
    protocol={'commit':'a'*40,'args':{'plan':'plan.json','out':str(run),'device':'cuda','seconds':5400}}
    save(run/'protocol.json',protocol)
    result={'complete':complete,'execution_finished':complete}
    save(run/('result.json' if complete else 'probe-progress.json'),result)
    if not complete:save(run/'failure.json',{'type':'TimeoutError','error':'bounded attempt exhausted'})
    if python_manifest:
        save(run/'manifest.json',{'complete':complete,'files':[audit.artifact(p,run) for p in sorted(run.rglob('*')) if p.is_file()]})
    command=['python','tools/baby_shrimp_mixed_junction_probe.py','--plan','plan.json','--out',str(run),
             '--device','cuda','--seconds','5400','--source-commit','a'*40]
    status={'worker_reaped':True,'source_commit':'a'*40,'output':str(run),'exit_code':0 if complete else 1,
            'timed_out':False,'complete':complete,'scientific_complete':complete,'ended_utc':'2026-10-22T12:00:00Z',
            'hard_seconds':5400,'termination_grace_seconds':10,'command':command,
            'worker_pid':101,'controller_pid':100,'elapsed_seconds':120.}
    directory=tmp_path/'controller';manifest=directory/'terminal-raw-manifest.json';status_path=directory/'status.json'
    save(manifest,[audit.artifact(p,run) for p in sorted(run.rglob('*')) if p.is_file()]);save(status_path,status)
    def args():return (manifest,audit.digest(manifest),status_path,audit.digest(status_path))
    return run,protocol,status,status_path,manifest,args


@pytest.mark.parametrize('complete',[False,True])
@pytest.mark.parametrize('python_manifest',[False,True])
def test_terminal_lifecycle_retains_partial_without_promoting_it(tmp_path,monkeypatch,complete,python_manifest):
    run,protocol,_,_,_,args=terminal_fixture(tmp_path,monkeypatch,complete=complete,python_manifest=python_manifest)
    checked=audit.terminal_evidence(run,protocol,*args())
    assert checked['python_finalization'] is python_manifest
    assert checked['controller']['scientific_complete'] is complete
    assert checked['controller']['worker_reaped']
    assert checked['python_complete'] is (complete and python_manifest)


@pytest.mark.parametrize('change',['source','reaped','exit','timeout','scientific','seconds','device','plan','worker','grace','raw','missing hash'])
def test_terminal_binding_and_completion_corruptions_are_rejected(tmp_path,monkeypatch,change):
    run,protocol,status,status_path,_,args=terminal_fixture(tmp_path,monkeypatch,complete=True)
    if change=='source':status['source_commit']='b'*40
    elif change=='reaped':status['worker_reaped']=False
    elif change=='exit':status['exit_code']=1
    elif change=='timeout':status['timed_out']=True
    elif change=='scientific':status['scientific_complete']=False
    elif change in ('seconds','device','plan'):
        flag='--'+change;status['command'][status['command'].index(flag)+1]='different'
    elif change=='worker':status['command'][1]='another.py'
    elif change=='grace':status['termination_grace_seconds']=11
    elif change=='raw':save(run/'unmanifested.json',{})
    save(status_path,status)
    values=list(args())
    if change=='missing hash':values[3]=None
    with pytest.raises(ValueError):audit.terminal_evidence(run,protocol,*values)


def test_python_and_controller_completion_must_agree_with_actual_result(tmp_path,monkeypatch):
    run,protocol,status,status_path,manifest,args=terminal_fixture(tmp_path,monkeypatch,complete=True)
    save(run/'result.json',{'complete':False,'execution_finished':False})
    save(manifest,[audit.artifact(p,run) for p in sorted(run.rglob('*')) if p.is_file()])
    with pytest.raises(ValueError,match='consistent reaped'):audit.terminal_evidence(run,protocol,*args())
    status.update(complete=False,scientific_complete=False);save(status_path,status)
    with pytest.raises(ValueError):audit.terminal_evidence(run,protocol,*args())


def source_fixture(tmp_path,monkeypatch):
    data_root(monkeypatch,tmp_path)
    plan={'id':'BSC6-mixed-junction-probe','period':18,'anchor_directory':'anchor','anchor_hashes':{}}
    blobs={name:('exact '+name).encode() for name in audit.EXECUTION_SOURCES}
    blobs['plan.json']=json.dumps(plan).encode()
    protocol={'commit':'a'*40,'source_sha256':{k:hashlib.sha256(v).hexdigest() for k,v in blobs.items() if k!='plan.json'},
              'args':{'plan':'plan.json'},'plan':plan,'plan_sha256':hashlib.sha256(blobs['plan.json']).hexdigest()}
    monkeypatch.setattr(audit.point_audit,'git_bytes',lambda commit,path:blobs[path])
    return protocol


def test_exact_execution_source_set_matches_frozen_worker_declaration():
    assert audit.EXECUTION_SOURCES==set(worker.SOURCES)


@pytest.mark.parametrize('missing',sorted(audit.EXECUTION_SOURCES))
def test_each_execution_dependency_is_required(tmp_path,monkeypatch,missing):
    protocol=source_fixture(tmp_path,monkeypatch)
    protocol['source_sha256'].pop(missing)
    with pytest.raises(ValueError,match='incomplete'):
        audit.previous.bind_execution(protocol,expected_id='BSC6-mixed-junction-probe',required_sources=audit.EXECUTION_SOURCES)


@pytest.mark.parametrize('change',[None,'symbolic commit','source hash','plan hash','plan contents','anchor hash'])
def test_execution_plan_and_anchor_bindings(tmp_path,monkeypatch,change):
    protocol=source_fixture(tmp_path,monkeypatch)
    if change=='symbolic commit':protocol['commit']='main'
    elif change=='source hash':protocol['source_sha256'][next(iter(audit.EXECUTION_SOURCES))]='0'*64
    elif change=='plan hash':protocol['plan_sha256']='0'*64
    elif change=='plan contents':protocol['plan']['period']=9
    elif change=='anchor hash':
        protocol['plan']['anchor_hashes']={'point.json':'0'*64}
        save(tmp_path/'anchor/point.json',{})
    call=lambda:audit.previous.bind_execution(protocol,expected_id='BSC6-mixed-junction-probe',required_sources=audit.EXECUTION_SOURCES)
    if change is None:assert call()==tmp_path/'anchor'
    else:
        with pytest.raises(ValueError):call()


def prior_fixture(tmp_path,monkeypatch,*,complete=True):
    data_root(monkeypatch,tmp_path)
    root=tmp_path/'repo';monkeypatch.setattr(audit,'ROOT',root)
    blobs={name:('prior frozen '+name).encode() for name in audit.previous.SOURCES}
    for name,data in blobs.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    monkeypatch.setattr(audit.point_audit,'git_bytes',lambda commit,path:blobs[path])
    raw=tmp_path/'explore/prior';save(raw/'point-0007/point.json',{'qualified':True,'index':7,
        'coordinate':[1.,2.],'beta':.006,'phases':[{'phase':p} for p in (0,6,12)]})
    metadata={k:k for k in audit.previous.binding.METADATA}
    receipt={'id':'BSC5-contraction-minima-offline-audit','audit_finished':True,'complete':complete,
        'center_qualified':True,'center_point_index':7,'scientific_metadata':metadata,
        'analysis_commit':'b'*40,'analysis_source_sha256':{k:hashlib.sha256(v).hexdigest() for k,v in blobs.items()},
        'lifecycle':{'controller':None,'python_finalization':True},'raw_directory':str(raw),
        'baseline':{'passed':True},'seed_replay':{'passed':True},
        'expanded_spectrum':{'complete':True,'point_index':7,'variations':{'tau':1e-13,'delta':1e-17}},
        'local_jets':[{},{}],'untested_or_incomplete_stages':[] if complete else ['remaining sections'],
        'sections':[{'arm_id':'ray','longitudinal':.002,'search':{'status':'low_signal','complete':True}}]}
    # A replay stub isolates binding/receipt controls; real point and jet replay
    # has its own saved-array tests in the predecessor suite.
    monkeypatch.setattr(audit.previous,'run_audit',lambda *args,**kwargs:copy.deepcopy(receipt))
    plan={**metadata,'id':'BSC6-mixed-junction-probe','mixed_junction':configuration()}
    plan['mixed_junction']['allow_scientifically_partial_bsc5']=not complete
    path=tmp_path/'prior-receipt.json'
    def freeze():
        save(path,receipt);plan['bsc5_audit']=audit.artifact(path,tmp_path)
        if not complete:
            plan['mixed_junction']['partial_bsc5_review']={
                'receipt_sha256':plan['bsc5_audit']['sha256'],
                'retained_unresolved_stages':copy.deepcopy(receipt['untested_or_incomplete_stages']),
                'section_statuses':[{'arm_id':'ray','longitudinal':.002,'status':'low_signal','complete':True}]}
    freeze()
    return plan,receipt,freeze,path


@pytest.mark.parametrize('complete',[False,True])
def test_predecessor_full_replay_preserves_expanded_support_and_partial_status(tmp_path,monkeypatch,complete):
    plan,receipt,_,_=prior_fixture(tmp_path,monkeypatch,complete=complete)
    result=audit.bind_bsc5(plan)
    assert result['seed']['cyclic_point_index']==7
    assert result['provenance']['scientific_complete'] is complete
    assert result['provenance']['supporting_audit_blas_threads']==1
    assert result['provenance']['section_statuses'][0]['status']=='low_signal'
    assert result['provenance']['retained_unresolved_stages']==receipt['untested_or_incomplete_stages']


@pytest.mark.parametrize('change',['receipt hash','receipt size','not audited','unqualified center','symbolic commit',
    'source set','source hash','local source','map','full replay','baseline','seed','expanded','center index','jet support','cyclic seed'])
def test_prior_bindings_and_support_are_required(tmp_path,monkeypatch,change):
    plan,receipt,freeze,path=prior_fixture(tmp_path,monkeypatch)
    if change=='not audited':receipt['audit_finished']=False
    elif change=='unqualified center':receipt['center_qualified']=False
    elif change=='symbolic commit':receipt['analysis_commit']='main'
    elif change=='source set':receipt['analysis_source_sha256'].pop(audit.previous.SOURCES[0])
    elif change=='source hash':receipt['analysis_source_sha256'][audit.previous.SOURCES[0]]='0'*64
    elif change=='local source':(audit.ROOT/audit.previous.SOURCES[0]).write_text('modified')
    elif change=='baseline':receipt['baseline']['passed']=False
    elif change=='seed':receipt['seed_replay']['passed']=False
    elif change=='expanded':receipt['expanded_spectrum']['complete']=False
    elif change=='center index':receipt['expanded_spectrum']['point_index']=8
    elif change=='jet support':receipt['local_jets'].pop()
    elif change=='cyclic seed':
        seed=Path(receipt['raw_directory'])/'point-0007/point.json';data=audit.read(seed);data['phases'].pop();save(seed,data)
    freeze()
    if change=='receipt hash':plan['bsc5_audit']['sha256']='0'*64
    elif change=='receipt size':plan['bsc5_audit']['bytes']+=1
    elif change=='map':plan['period']='another period'
    elif change=='full replay':
        bad=copy.deepcopy(receipt);bad['sections'][0]['search']['status']='converged'
        monkeypatch.setattr(audit.previous,'run_audit',lambda *a,**k:bad)
    with pytest.raises(ValueError):audit.bind_bsc5(plan)


@pytest.mark.parametrize('change',['permission','review hash','unresolved','statuses'])
def test_partial_predecessor_requires_exact_frozen_disclosure(tmp_path,monkeypatch,change):
    plan,_,_,_=prior_fixture(tmp_path,monkeypatch,complete=False)
    cfg=plan['mixed_junction']
    if change=='permission':cfg['allow_scientifically_partial_bsc5']=False
    elif change=='review hash':cfg['partial_bsc5_review']['receipt_sha256']='0'*64
    elif change=='unresolved':cfg['partial_bsc5_review']['retained_unresolved_stages']=[]
    else:cfg['partial_bsc5_review']['section_statuses'][0]['status']='converged'
    with pytest.raises(ValueError):audit.bind_bsc5(plan)


def test_real_precision_failure_is_auditable_but_cannot_continue_corrections():
    cfg=configuration();cfg['maximum_f_minus_total_uncertainty']=1e-25
    evaluate=AnalyticEvaluator();snapshots=[]
    center,parent=worker.case_predictor(cfg['target'],0.)
    with pytest.raises(ValueError):
        worker.probe_case(evaluate,center,.006,0.,cfg['target']['references'],{'tau':1e-20,'delta':1e-20},cfg,
            lambda value:snapshots.append(copy.deepcopy(value)),predictor_parent=parent)
    case=snapshots[-1];assert case['status']=='unresolved precision'
    raw={p['index']:p for p in evaluate.rows};result={'complete':False,'cases':[case]}
    replay=audit.audit_cases(result,raw,set(raw),{'tau':1e-20,'delta':1e-20},cfg)
    assert not replay[0]['complete'] and replay[0]['iterations'][-1]['admission']['f_minus']['status']=='not_admitted'
    case['iterations'][-1]['step_applied']={}
    with pytest.raises(ValueError,match='failed precision gate'):
        audit.audit_cases(result,raw,set(raw),{'tau':1e-20,'delta':1e-20},cfg)


def early_run_fixture(tmp_path,monkeypatch):
    run,protocol,status,status_path,manifest,_=terminal_fixture(tmp_path,monkeypatch,complete=False)
    plan={k:k for k in audit.previous.binding.METADATA}
    cfg=configuration();cfg.update(maximum_f_minus_total_uncertainty=5e-8,minimum_normalized_jacobian_singular_value=.2)
    plan.update(id='BSC6-mixed-junction-probe',maximum_evaluations=120,mixed_junction=cfg,bsc5_audit={'path':'bound.json'})
    protocol.update(plan=plan,plan_sha256='c'*64,source_sha256={k:'d'*64 for k in audit.EXECUTION_SOURCES})
    protocol['args']['source_commit']='a'*40
    save(run/'protocol.json',protocol)
    monkeypatch.setattr(audit.previous,'bind_execution',lambda *a,**k:tmp_path/'anchor')
    monkeypatch.setattr(audit,'bind_bsc5',lambda plan:{'provenance':{},'receipt':{'expanded_spectrum':{'variations':{'tau':1e-13,'delta':1e-17}}},'seed':{}})
    monkeypatch.setattr(audit.previous,'audit_seed',lambda *a,**k:None)
    def packet():
        save(run/'manifest.json',{'complete':False,'files':[audit.artifact(p,run) for p in sorted(run.rglob('*')) if p.is_file() and p.name!='manifest.json']})
        save(manifest,[audit.artifact(p,run) for p in sorted(run.rglob('*')) if p.is_file()])
        return dict(controller_manifest=manifest,manifest_sha=audit.digest(manifest),controller_status=status_path,status_sha=audit.digest(status_path))
    return run,protocol,packet


def test_full_partial_audit_distinguishes_finished_audit_from_unattempted_science(tmp_path,monkeypatch):
    run,_,packet=early_run_fixture(tmp_path,monkeypatch)
    args=packet()
    with pytest.raises(ValueError,match='explicit --allow-partial'):audit.run_audit(run,**args)
    result=audit.run_audit(run,allow_partial=True,**args)
    assert result['audit_finished'] and not result['complete'] and not result['execution_finished']
    assert len(result['untested_or_incomplete_stages'])==5 and result['cases']==[]
    assert result['terminal_failure']['type']=='TimeoutError'
    assert result['qualified_points']==[] and result['attempted_points']==0


def test_full_partial_audit_preserves_incomplete_model_attempt(tmp_path,monkeypatch):
    run,_,packet=early_run_fixture(tmp_path,monkeypatch)
    save(run/'runtime.json',{'norm_sensitivity':1e-15,'rotary_float64':True,'threads':8})
    (run/'point-0000').mkdir()
    save(run/'point-0001/point.json',{'index':1,'qualified':False})
    result=audit.run_audit(run,allow_partial=True,**packet())
    assert result['attempted_points']==2 and len(result['unqualified_or_incomplete_points'])==2
    assert result['qualified_points']==[] and not result['complete']


@pytest.mark.parametrize('change',['runtime','resource','source arg','primary scale','precision cap','normalized Jacobian','missing seed'])
def test_full_audit_rejects_execution_or_admission_shortcuts(tmp_path,monkeypatch,change):
    run,protocol,packet=early_run_fixture(tmp_path,monkeypatch)
    if change=='runtime':(run/'point-0000').mkdir()
    elif change=='resource':protocol['plan']['maximum_evaluations']=121
    elif change=='source arg':protocol['args']['source_commit']='b'*40
    elif change=='primary scale':protocol['plan']['mixed_junction']['primary_step']=.0001
    elif change=='precision cap':protocol['plan']['mixed_junction']['maximum_f_minus_total_uncertainty']=1e-7
    elif change=='normalized Jacobian':protocol['plan']['mixed_junction']['minimum_normalized_jacobian_singular_value']=.1
    else:save(run/'probe-progress.json',{'complete':False,'execution_finished':False,'cases':[{}]})
    save(run/'protocol.json',protocol)
    with pytest.raises(ValueError):audit.run_audit(run,allow_partial=True,**packet())
