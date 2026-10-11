"""Synthetic saved-evidence controls; no target, model, SVD or solver execution."""
import copy,hashlib,importlib.util,json,math,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('gap_saved_auditor',ROOT/'tools/reviews/audit_s1r_lower_gap_outcome.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)


def fixture_svd(matrix):
    """Closed-form decomposition for diagonal A or B=[I,(-c,0)]. No SVD call."""
    m,n=matrix.shape
    if m==n:
        U=np.eye(m);s=np.diag(matrix).copy();Vh=np.eye(m)
    else:
        assert matrix.shape==(2,3) and np.array_equal(matrix[:,:2],np.eye(2))
        c=-matrix[0,2];scale=math.sqrt(1+c*c);U=np.eye(2);s=np.array([scale,1.]);Vh=np.array([[1/scale,0.,-c/scale],[0.,1.,0.],[c/scale,0.,1/scale]])
    saved=dict(matrix=matrix,U=U,singular_values=s,Vh=Vh)
    # Independent fixture norms via explicit squares, used only for fabricated records.
    norm=lambda x:float(np.sqrt(np.sum(np.asarray(x)**2)))
    denom=max(1.,norm(matrix));V=Vh[:m].T
    right=np.array([norm((matrix@V-U*s)[:,i])/denom for i in range(m)])
    left=np.array([norm((matrix.T@U-V*s)[:,i])/denom for i in range(m)])
    low=1e-12*max(1.,float(s[0]));high=1e-10*max(1.,float(s[0]));middle=bool(np.any((s>low)&(s<=high)))
    rank=dict(status='ambiguous_rank' if middle else 'full_row_rank_resolved' if s[-1]>high else 'numerically_rank_deficient',low=low,high=high,small_count=int(np.sum(s<=low)),resolved_count=int(np.sum(s>high)),sigma_max=float(s[0]),sigma_min=float(s[-1]),condition=float(s[0]/s[-1]) if s[-1]>0 else None,isolated_soft_direction=bool(s[-2]>high and s[-1]<=.1*s[-2]))
    checks=dict(right_residual=right,left_residual=left,reconstruction=norm(matrix-(U*s)@Vh[:m])/denom,orthogonality=max(norm(U.T@U-np.eye(m)),norm(Vh@Vh.T-np.eye(n))),rank=rank)
    return saved,checks


def test_analytic_saved_returns_assemble_rate_chain_without_derivative_calls():
    theta=np.array([.3,-.5]);rates=np.array([.1,.03]);H=np.array([[2.,.2],[.2,3.]])
    def stored(t):
        g=H@t;image=t-rates*g;J=np.eye(2)-rates[:,None]*H;p=-np.array([math.log(10)*rates[0],0.])*g
        return dict(loss=float(t@H@t/2),gradient=g,hessian=H),dict(theta=t,loss=float(t@H@t/2),image=image,jacobian=J,parameter=p)
    analytic0,first=stored(theta);analytic1,second=stored(first['image']);J=first['jacobian']
    expected=dict(state=theta,map1=first['image'],map2=second['image'],partial_h_F=first['parameter'],R=second['image']-theta,M=J@J,A=J@J-np.eye(2),b=J@first['parameter']+second['parameter'])
    observed=A.assembly_check(theta,rates,analytic0,first,analytic1,second,expected,hidden=1)
    np.testing.assert_allclose(observed['b'],expected['b'],rtol=1e-14)
    corrupt=copy.deepcopy(second);corrupt['parameter'][1]=.1
    with pytest.raises(ValueError):A.assembly_check(theta,rates,analytic0,first,analytic1,corrupt,expected,hidden=1)


@pytest.mark.parametrize('mutation',['matrix','basis_shape','singular_value','right_basis','rank_label'])
def test_full_saved_svd_checks_reject_corruption(mutation):
    matrix=np.array([[1.,0.,-1.],[0.,1.,0.]]);saved,checks=fixture_svd(matrix);A.svd_check(matrix,saved,checks)
    if mutation=='matrix':saved['matrix']=matrix+.001
    elif mutation=='basis_shape':saved['Vh']=saved['Vh'][:2]
    elif mutation=='singular_value':saved['singular_values'][0]+=.01
    elif mutation=='right_basis':saved['Vh'][-1,0]+=.1
    else:checks['rank']['status']='numerically_rank_deficient'
    with pytest.raises(ValueError):A.svd_check(matrix,saved,checks)


def test_scaled_tangent_uses_column_conversion_and_inverse_scaled_secant():
    raw=np.array([[1.,0.,-1.],[0.,1.,0.]]);columns=np.array([2.,3.,.000125]);scaled=raw*columns[None,:]/np.array([2.,3.])[:,None]
    saved,checks=fixture_svd(scaled);A.svd_check(scaled,saved,checks)
    tangent=A.tangent(scaled,saved['Vh'],checks['rank'],columns,np.array([1.,0.,1.]),raw)
    np.testing.assert_allclose(tangent['tangent'],np.array([1.,0.,1.])/math.sqrt(2),rtol=1e-13)
    assert tangent['secant_cosine']>0 and tangent['raw_residual']<1e-14
    bad=A.tangent(scaled,saved['Vh'],checks['rank'],np.ones(3),np.array([1.,0.,1.]),raw)
    assert bad['status']=='abstain_null_residual'
    zero=A.tangent(scaled,saved['Vh'],checks['rank'],columns,np.zeros(3),raw)
    assert zero['status']=='abstain_zero_secant'


@pytest.mark.parametrize('s,status',[(np.array([1.,5e-11]),'ambiguous_rank'),(np.array([1.,0.]),'numerically_rank_deficient'),(np.array([1.,1e-8]),'full_row_rank_resolved')])
def test_rank_windows_and_isolation_are_numerical(s,status):
    rank=A.rank(s);assert rank['status']==status
    if status!='full_row_rank_resolved':assert A.tangent(np.zeros((2,3)),np.eye(3),rank,np.ones(3),np.ones(3),np.zeros((2,3)))['tangent'] is None
    competing=A.rank(np.array([1.,1e-11,1e-12]));assert not competing['isolated_soft_direction']


def test_phase_exchange_secants_preserve_native_and_carried_labels():
    parent=np.array([[1.,2.],[-1.,-2.]]);child=np.array([[-1.1,-2.1],[1.1,2.1]])
    sec=A.secants(parent,child,1.,.9,1)
    np.testing.assert_allclose(sec['parent',0],[.1,.1,-.1]);assert np.array_equal(sec['parent',0],sec['child',1])
    assert np.array_equal(sec['parent',1],sec['child',0])


def test_true_equation_forcing_backward_and_resolution_are_separate():
    matrix=np.eye(2);r=np.array([.01,.02]);theta=np.ones(2)
    assert A.linear_check(matrix,r,theta,-r,r)['verdict']=='linear_equation_resolved'
    assert A.linear_check(matrix,r,theta,r,r)['verdict']=='material_linear_equation_mismatch'
    assert A.linear_check(matrix,r,theta,-r,r+1e-5)['verdict']=='arithmetic_resolution_limited'
    zero=A.linear_check(matrix,np.zeros(2),theta,np.zeros(2),np.zeros(2));assert zero['true_relative_forcing'] is None and zero['verdict']=='abstain_nearzero_residual'


def trial_fixture():
    it=dict(call=473,iteration=3,state=np.array([1.,2.]),residual=np.array([.1,.2]),step=np.array([.01,-.01]))
    trials=[dict(call=473,iteration=3,halving=k,state=it['state']+2.**-k*it['step'],residual=it['residual']*2,accepted=False) for k in range(13)]
    return it,trials

@pytest.mark.parametrize('mutation',['none','state','order','acceptance','count'])
def test_thirteen_trials_use_exact_states_and_strict_infinity_merit(mutation):
    it,trials=trial_fixture()
    if mutation=='none':assert len(A.trial_check(it,trials))==13;return
    if mutation=='state':trials[4]['state'][0]+=1e-8
    elif mutation=='order':trials[4]['halving']=5
    elif mutation=='acceptance':trials[4]['accepted']=True
    else:trials.pop()
    with pytest.raises(ValueError):A.trial_check(it,trials)


def write_record(folder,index,kind,payload):
    arrays={}
    def encode(value):
        if isinstance(value,np.ndarray):
            key=f'a{len(arrays):04d}';arrays[key]=value
            return dict(array=key,dtype=value.dtype.str,shape=list(value.shape),order='C',sha256=A.digest(value.tobytes(order='C')))
        if isinstance(value,np.generic):return value.item()
        if isinstance(value,dict):return {k:encode(v) for k,v in value.items()}
        if isinstance(value,list):return [encode(v) for v in value]
        return value
    name=f'{index:05d}-{kind}.json';record=dict(kind=kind,sequence=index,payload=encode(payload))
    if arrays:
        archive=folder/(name[:-5]+'.npz');np.savez(archive,**arrays);record.update(arrays_file=archive.name,arrays_sha256=A.entry(archive)['sha256'])
    (folder/name).write_bytes(A.canonical(record));return dict(sequence=index,kind=kind,file=name)


def test_independent_record_decoder_rejects_unreferenced_array_and_signed_zero_change(tmp_path):
    event=write_record(tmp_path,0,'synthetic',dict(vector=np.array([-0.,1.])))
    record,payload=A.decode_record(tmp_path,event['file']);assert np.signbit(payload['vector'][0])
    archive=tmp_path/record['arrays_file'];np.savez(archive,a0000=np.array([-0.,1.]),extra=np.zeros(1));record['arrays_sha256']=A.entry(archive)['sha256'];(tmp_path/event['file']).write_bytes(A.canonical(record))
    with pytest.raises(ValueError,match='referenced'):A.decode_record(tmp_path,event['file'])
    np.savez(archive,a0000=np.array([0.,1.]));record['arrays_sha256']=A.entry(archive)['sha256'];(tmp_path/event['file']).write_bytes(A.canonical(record))
    with pytest.raises(ValueError,match='byte identity'):A.decode_record(tmp_path,event['file'])


def test_partial_counters_allow_return_before_failed_emit_and_no_more(tmp_path):
    journal=[]
    for i,(kind,payload) in enumerate([('bound_inputs',{}),('saved_line_search_reduction',[]),('operation_started',dict(kinds=['maps','jacobians'],attempted=dict(maps=1,jacobians=1,svds=0))),('predecessor_native_phase_0_first_phase_analytic_return',dict(gradient=np.ones(2),hessian=np.eye(2),loss=1.))]):journal.append(write_record(tmp_path,i,kind,payload))
    (tmp_path/'journal.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in journal))
    result=dict(status='failed_or_interrupted',operation_attempted=dict(maps=1,jacobians=1,svds=0),operation_returned=dict(maps=1,jacobians=1,svds=0))
    _,counts=A.records(tmp_path,A.directory(tmp_path),False,result);assert counts['pending_operation']==['maps','jacobians'] and counts['durably_returned']['maps']==0
    result['operation_returned']['maps']=2
    with pytest.raises(ValueError):A.records(tmp_path,A.directory(tmp_path),False,result)


def test_record_order_corruption_is_not_a_resource_failure(tmp_path):
    event=write_record(tmp_path,0,'saved_line_search_reduction',[]);(tmp_path/'journal.jsonl').write_text(json.dumps(event)+'\n')
    result=dict(operation_attempted=dict(maps=0,jacobians=0,svds=0),operation_returned=dict(maps=0,jacobians=0,svds=0))
    with pytest.raises(ValueError):A.records(tmp_path,A.directory(tmp_path),False,result)


def test_numerical_library_decomposition_paths_are_never_used(monkeypatch):
    def forbidden(*_,**__):pytest.fail('new decomposition/solve forbidden')
    for name in ('svd','solve','eig','eigh','inv','lstsq'):monkeypatch.setattr(np.linalg,name,forbidden)
    saved,checks=fixture_svd(np.array([[1.,0.,-1.],[0.,1.,0.]]));A.svd_check(saved['matrix'],saved,checks)
    A.linear_check(np.eye(2),np.ones(2),np.ones(2),-np.ones(2),np.ones(2))


def test_source_and_nice_guards_precede_output(tmp_path,monkeypatch):
    monkeypatch.setattr(A.os,'getpriority',lambda *_:0)
    with pytest.raises(ValueError):A.run(tmp_path/'out',tmp_path/'pins','a'*64,'b'*40)
    assert not (tmp_path/'out').exists()


def launcher():
    spec=importlib.util.spec_from_file_location('gap_audit_launcher',ROOT/'tools/reviews/launch_s1r_lower_gap_audit.py');value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def test_launcher_optimization_safe_guards(monkeypatch):
    L=launcher();monkeypatch.setattr(L.os,'getpriority',lambda *_:0)
    with pytest.raises(ValueError,match='nice10'):L.main()
    code="import importlib.util;spec=importlib.util.spec_from_file_location('l',"+repr(str(ROOT/'tools/reviews/launch_s1r_lower_gap_audit.py'))+");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.require(False,'still active')"
    completed=subprocess.run([sys.executable,'-O','-c',code],capture_output=True)
    assert completed.returncode!=0 and b'still active' in completed.stderr


def test_launcher_only_reaps_its_own_synthetic_group():
    L=launcher();code="import subprocess,sys;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);print(p.pid,flush=True)"
    process=subprocess.Popen([sys.executable,'-c',code],stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    try:
        output,_=process.communicate(timeout=3);child=int(output);assert child in [v['pid'] for v in L.group_status(process.pid) if v['state']!='Z'];assert not L.cleanup_owned(process.pid)
    finally:L.cleanup_owned(process.pid)


def six_role_fixture():
    """Hand-constructed period-two saved algebra, not a model or SVD run."""
    n=72;theta=np.zeros(n);theta[0]=.5;pair=np.stack([theta,-theta]);zero=np.zeros(n);eye=np.eye(n)
    model=dict(activation='tanh',bias=False,d=8,in_scale=.5,loss_scale=1.,out_scale=.125,ridge=.001,width=8)
    slots=[]
    for i,role in enumerate(A.ROLES):
        phase=i%2 if i<4 else None;t=pair[phase] if phase is not None else theta
        slots.append(dict(role=role,state=t.copy(),params=np.array([0. if i<2 else -.000125,0.]),rates=np.ones(2),native_phase=phase,node=dict(pair=pair,spectra=dict(phases=[dict(matrix=eye),dict(matrix=eye)])),secant=np.r_[zero,-.000125]))
    iteration=dict(call=473,iteration=3,state=theta.copy(),residual=zero.copy(),step=np.ones(n)*.01)
    trials=[dict(call=473,iteration=3,halving=k,state=theta+2.**-k*iteration['step'],residual=np.ones(n)*.1,accepted=False) for k in range(13)]
    reduction=[dict(halving=k,alpha=2.**-k,residual_inf=.1,ratio=None,accepted=False) for k in range(13)]
    original=dict(slots=slots,iteration=iteration,trials=trials,model=model,identity={},x=np.zeros((72,8)),y=np.zeros(72),scale=np.ones(72),phase_binding=dict(parent_phase=0,exchange=0,last_carried_phase=0))
    plan_identity=dict(bytes=2,sha256='a'*64)
    bound=dict(plan_file='input-plan.json',plan_identity=plan_identity,**{k:original[k] for k in ('identity','model','x','y','scale','phase_binding')},slots=[{k:s[k] for k in ('role','state','params','rates','native_phase','secant')} for s in slots],saved_iteration=iteration,saved_trials=trials)
    values=dict(bound_inputs=bound,saved_line_search_reduction=reduction)
    for slot in slots:
        role=slot['role'];t=slot['state'];first_image=t-2*t;second_image=first_image-2*first_image;c=-np.r_[np.full(64,math.log(10.)),np.zeros(8)]*(2*t);b=-2*c
        for phase,z,image,param in [('first',t,first_image,c),('second',first_image,second_image,-c)]:
            values[role+'_'+phase+'_phase_analytic_return']=dict(loss=0.,gradient=2*z,hessian=2*eye)
            values[role+'_'+phase+'_phase']=dict(theta=z,loss=0.,image=image,jacobian=-eye,parameter=param)
        assembly=dict(state=t,map1=first_image,map2=second_image,partial_h_F=c,R=zero.copy(),M=eye,A=np.zeros((n,n)),b=b)
        values[role+'_assembly']=assembly
        values[role+'_agreement']=dict(qualified_endpoint=True,closure_normalized=0.,transition_normalized=0.,primitive_separation=1.,saved_monodromy_max_absolute_error=0.) if slot['native_phase'] is not None else dict(qualified_endpoint=False,scope='nonclosed fixed-state residual geometry')
        values[role+'_scales']=dict(row_scale=np.ones(n),state_column_scale=np.ones(n),augmented_column_scale=np.r_[np.ones(n),.000125],scaled_residual=zero.copy())
        statuses={}
        for coord in A.COORDS:
            augmented=coord.endswith('_B');scale=.000125 if coord=='scaled_B' else 1.
            matrix=np.zeros((n,n+int(augmented)));U=eye.copy();s=np.zeros(n);Vh=np.eye(matrix.shape[1])
            if augmented:
                matrix[:,-1]=b*scale;s[0]=abs(float(matrix[0,-1]));Vh=np.zeros((n+1,n+1));Vh[0,-1]=np.sign(matrix[0,-1]);Vh[1:n,1:n]=np.eye(n-1);Vh[-1,0]=1.
            largest=float(s[0]);threshold=max(1.,largest)
            rank=dict(status='numerically_rank_deficient',low=1e-12*threshold,high=1e-10*threshold,small_count=n-1 if augmented else n,resolved_count=1 if augmented else 0,sigma_max=largest,sigma_min=0.,condition=None,isolated_soft_direction=False)
            name=role+'_'+coord;values[name+'_svd']=dict(matrix=matrix,U=U,singular_values=s,Vh=Vh)
            values[name+'_svd_checks']=dict(right_residual=zero.copy(),left_residual=zero.copy(),reconstruction=0.,orthogonality=0.,rank=rank)
            summary=dict(rank=rank)
            if augmented:summary['null_tangent']=dict(status='abstain_numerically_rank_deficient',tangent=None)
            else:summary.update(smallest_left_parameter_coupling=0.,normalized_left_parameter_coupling=0.,coupling_qualified=False)
            values[name+'_summary']=summary;statuses[coord]=rank['status']
        values[role+'_complete']=dict(coordinate_statuses=statuses,tangent_scope='numerical qualified-endpoint branch tangent' if slot['native_phase'] is not None else 'local residual level-set tangent at nonclosed state',accepted_orbit=False,fold_verified=False)
    linear=dict(error=zero.copy(),error_using_saved_R=zero.copy(),absolute_linear_residual=0.,true_relative_forcing=None,nearzero_threshold=1e-12,residual_norm=0.,residual_reconstruction_difference=0.,normwise_backward_error_frobenius=None,verdict='abstain_nearzero_residual')
    values['stalled_residual_agreement']=dict(max_absolute_error=0.,passed=True);values['saved_linear_equation']=linear
    values['phase_tangent_transport']=[dict(endpoint=endpoint,coordinates=coord,status='abstain_unqualified_tangent') for endpoint in ('predecessor','last') for coord in ('raw_B','scaled_B')]
    result=dict(status='diagnostic_complete',completed_roles=list(A.ROLES),saved_line_search=reduction,saved_linear_equation={k:v for k,v in linear.items() if not isinstance(v,np.ndarray)},operation_attempted=dict(maps=12,jacobians=12,svds=24),operation_returned=dict(maps=12,jacobians=12,svds=24))
    return values,original,result,plan_identity


def test_all_six_assemblies_twenty_four_saved_bases_and_phase_scopes():
    values,original,result,plan=six_role_fixture();checked=A.validate_arithmetic(values,original,True,result,plan)
    assert checked['validated_roles']==list(A.ROLES) and len(checked['phase_transport'])==4
    assert sum(len(row['checked_decompositions']) for row in checked['roles'].values())==24
    assert checked['roles'][A.ROLES[-1]]['linear_equation']['verdict']=='abstain_nearzero_residual'


@pytest.mark.parametrize('mutation',['column_scale','matrix_assembly','phase_scope','trial_aggregate'])
def test_complete_six_role_corruptions_fail(mutation):
    values,original,result,plan=six_role_fixture()
    if mutation=='column_scale':values[A.ROLES[0]+'_scales']['augmented_column_scale'][-1]=.001
    elif mutation=='matrix_assembly':values[A.ROLES[0]+'_assembly']['b'][0]+=.001
    elif mutation=='phase_scope':values[A.ROLES[-1]+'_complete']['tangent_scope']='numerical qualified-endpoint branch tangent'
    else:result['saved_line_search']=result['saved_line_search'][:-1]
    with pytest.raises(ValueError):A.validate_arithmetic(values,original,True,result,plan)


def test_complete_fabricated_record_graph_and_counter_join(tmp_path):
    values,original,result,plan=six_role_fixture();sequence=A.expected_sequence();journal=[];attempted=dict(maps=0,jacobians=0,svds=0)
    for index,kind in enumerate(sequence):
        if kind=='operation_started':
            kinds=['svds'] if sequence[index+1].endswith('_svd') else ['maps','jacobians']
            for key in kinds:attempted[key]+=1
            payload=dict(kinds=kinds,attempted=attempted.copy())
        else:payload=values[kind]
        journal.append(write_record(tmp_path,index,kind,payload))
    (tmp_path/'journal.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in journal));(tmp_path/'checkpoint.json').write_bytes(A.canonical(dict(status=result['status'],latest=journal[-1])))
    decoded,counts=A.records(tmp_path,A.directory(tmp_path),True,result);checked=A.validate_arithmetic(decoded,original,True,result,plan)
    assert counts['attempted']==counts['durably_returned']==dict(maps=12,jacobians=12,svds=24)
    assert len(checked['validated_roles'])==6


@pytest.mark.parametrize('failure',[False,True])
def test_audit_driver_durably_finishes_without_target_reads(tmp_path,monkeypatch,capsys,failure):
    root=tmp_path/'source';root.mkdir();monkeypatch.setattr(A,'ROOT',root)
    for name in A.AUDIT_FILES:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'synthetic source')
    source='a'*40;pins=dict(execution_commit='b'*40,source_review_commit='c'*40,raw_manifest_sha256='d'*64,launcher_manifest_sha256='e'*64);pins_path=tmp_path/'pins.json';pins_path.write_bytes(A.canonical(pins))
    def git(argv,**kwargs):
        if argv[1]=='rev-parse':return source+'\n'
        if argv[1]=='status':return ''
        return (root/argv[2].split(':',1)[1]).read_bytes()
    monkeypatch.setattr(A.subprocess,'check_output',git);monkeypatch.setattr(A.subprocess,'run',lambda *a,**k:None);monkeypatch.setattr(A.os,'getpriority',lambda *_:10)
    def fake(_pins,emit):
        emit('retained-stage.json',dict(saved=True))
        if failure:raise TimeoutError('synthetic interruption')
        return dict(verdict='saved_outcome_audit_pass')
    monkeypatch.setattr(A,'audit',fake);out=tmp_path/'out'
    assert A.run(out,pins_path,A.entry(pins_path)['sha256'],source)==(2 if failure else 0)
    assert A.read(out/'result.json')['status']==('failed' if failure else 'finished')
    assert (out/'retained-stage.json').exists();A.inventory(out,A.entry(out/'manifest.json')['sha256'])
    out.chmod(0o755)
    for p in out.iterdir():p.chmod(0o644)


def test_dirty_source_and_external_pin_fail_before_creation(tmp_path,monkeypatch):
    monkeypatch.setattr(A.os,'getpriority',lambda *_:10)
    monkeypatch.setattr(A.subprocess,'check_output',lambda args,**_:'a'*40 if args[1]=='rev-parse' else '?? dirty')
    with pytest.raises(ValueError,match='clean audit'):A.run(tmp_path/'out',tmp_path/'pins','0'*64,'a'*40)
    assert not (tmp_path/'out').exists()


def test_actual_complete_manifest_inventory_rejects_unlisted_file(tmp_path):
    (tmp_path/'x').write_bytes(b'x');(tmp_path/'manifest.json').write_bytes(A.canonical(dict(files={'x':A.entry(tmp_path/'x')})));pin=A.entry(tmp_path/'manifest.json')['sha256']
    A.inventory(tmp_path,pin);(tmp_path/'extra').write_bytes(b'')
    with pytest.raises(ValueError,match='complete inventory'):A.inventory(tmp_path,pin)


def test_partial_unvalidated_return_never_gets_complete_arithmetic_verdict():
    values,original,result,plan=six_role_fixture();prefix={k:values[k] for k in ('bound_inputs','saved_line_search_reduction',A.ROLES[0]+'_first_phase_analytic_return',A.ROLES[0]+'_first_phase')}
    prefix[A.ROLES[0]+'_first_phase']['jacobian'][0,0]=np.nan;result['status']='failed_or_interrupted';result['completed_roles']=[]
    checked=A.validate_arithmetic(prefix,original,False,result,plan)
    assert checked['validated_roles']==[] and checked['unvalidated_partial_roles'][0]['role']==A.ROLES[0]
    with pytest.raises(ValueError):A.validate_arithmetic(prefix,original,True,result,plan)


def synthetic_prefix(folder, stop):
    """Persist a prefix of the complete fabricated graph, including causal counters."""
    values,original,result,plan=six_role_fixture();sequence=A.expected_sequence();journal=[]
    attempted=dict(maps=0,jacobians=0,svds=0);returned=attempted.copy()
    for index,kind in enumerate(sequence[:stop]):
        if kind=='operation_started':
            kinds=['svds'] if sequence[index+1].endswith('_svd') else ['maps','jacobians']
            for key in kinds:attempted[key]+=1
            payload=dict(kinds=kinds,attempted=attempted.copy())
        else:
            payload=values[kind]
            if kind.endswith('_phase') or kind.endswith('_svd'):
                for key in (['svds'] if kind.endswith('_svd') else ['maps','jacobians']):returned[key]+=1
        journal.append(write_record(folder,index,kind,payload))
    (folder/'journal.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in journal))
    result.update(status='failed_or_interrupted',operation_attempted=attempted.copy(),operation_returned=returned.copy(),completed_roles=[r for r in A.ROLES if r+'_complete' in sequence[:stop]])
    prefix={k:values[k] for k in sequence[:stop] if k!='operation_started'}
    return prefix,original,result,plan,journal


@pytest.mark.parametrize('kind',['phase','svd'])
def test_one_unjournaled_next_attempt_is_retained_without_return(tmp_path,kind):
    sequence=A.expected_sequence()
    stop=2 if kind=='phase' else sequence.index(A.ROLES[0]+'_raw_A_svd')-1
    _,_,result,_,_=synthetic_prefix(tmp_path,stop)
    keys=['maps','jacobians'] if kind=='phase' else ['svds']
    for key in keys:result['operation_attempted'][key]+=1
    _,counts=A.records(tmp_path,A.directory(tmp_path),False,result)
    assert counts['unjournaled_start_operation']==keys
    assert counts['durably_attempted']==counts['durably_returned']
    for key in keys:result['operation_returned'][key]+=1
    with pytest.raises(ValueError,match='returned caps'):A.records(tmp_path,A.directory(tmp_path),False,result)


@pytest.mark.parametrize('mutation',['half_pair','wrong_kind','two_starts','not_next_start','pending_start','complete','boolean'])
def test_unjournaled_attempt_rejects_impossible_boundaries(tmp_path,mutation):
    stop=1 if mutation=='not_next_start' else 3 if mutation=='pending_start' else 2
    _,_,result,_,_=synthetic_prefix(tmp_path,stop)
    result['operation_attempted']['maps']+=1;result['operation_attempted']['jacobians']+=1
    if mutation=='half_pair':result['operation_attempted']['jacobians']-=1
    elif mutation=='wrong_kind':result['operation_attempted']=dict(maps=0,jacobians=0,svds=1)
    elif mutation=='two_starts':result['operation_attempted']['maps']+=1;result['operation_attempted']['jacobians']+=1
    elif mutation=='boolean':result['operation_attempted']['maps']=True
    with pytest.raises(ValueError):A.records(tmp_path,A.directory(tmp_path),mutation=='complete',result)


@pytest.mark.parametrize('checkpoint_lag',[0,1,2])
def test_checkpoint_can_lag_exactly_one_failed_emit(tmp_path,checkpoint_lag):
    _,_,result,_,journal=synthetic_prefix(tmp_path,4)
    (tmp_path/'checkpoint.json').write_bytes(A.canonical(dict(status=result['status'],latest=journal[-1-checkpoint_lag])))
    if checkpoint_lag==2:
        with pytest.raises(ValueError,match='checkpoint'):A.records(tmp_path,A.directory(tmp_path),False,result)
    else:
        _,counts=A.records(tmp_path,A.directory(tmp_path),False,result)
        assert counts['terminal_checkpoint_lag']==bool(checkpoint_lag)
        if checkpoint_lag:
            result['operation_returned']=dict(maps=1,jacobians=1,svds=0)
            with pytest.raises(ValueError,match='checkpoint'):A.records(tmp_path,A.directory(tmp_path),False,result)


def test_completed_role_corruption_cannot_hide_behind_later_failure(tmp_path):
    stop=A.expected_sequence().index(A.ROLES[1]+'_first_phase')+1
    values,original,result,plan,_=synthetic_prefix(tmp_path,stop)
    values[A.ROLES[0]+'_assembly']['b'][0]+=.001
    with pytest.raises(ValueError):A.validate_arithmetic(values,original,False,result,plan)


def test_durable_terminal_complete_remains_strict_even_before_result_append(tmp_path):
    stop=A.expected_sequence().index(A.ROLES[0]+'_complete')+1
    values,original,result,plan,_=synthetic_prefix(tmp_path,stop);result['completed_roles']=[]
    assert A.validate_arithmetic(values,original,False,result,plan)['validated_roles']==[A.ROLES[0]]
    values[A.ROLES[0]+'_assembly']['b'][0]+=.001
    with pytest.raises(ValueError):A.validate_arithmetic(values,original,False,result,plan)


def test_completed_role_result_lag_disallowed_after_later_records(tmp_path):
    stop=A.expected_sequence().index(A.ROLES[1]+'_first_phase_analytic_return')+1
    values,original,result,plan,_=synthetic_prefix(tmp_path,stop);result['completed_roles']=[]
    with pytest.raises(ValueError,match='completed-role'):A.validate_arithmetic(values,original,False,result,plan)


def test_only_unfinished_terminal_role_may_be_numeric_unvalidated(tmp_path):
    stop=A.expected_sequence().index(A.ROLES[1]+'_first_phase')+1
    values,original,result,plan,_=synthetic_prefix(tmp_path,stop)
    values[A.ROLES[1]+'_first_phase']['jacobian'][0,0]=np.nan
    checked=A.validate_arithmetic(values,original,False,result,plan)
    assert checked['validated_roles']==[A.ROLES[0]]
    assert [r['role'] for r in checked['unvalidated_partial_roles']]==[A.ROLES[1]]


@pytest.mark.parametrize('bad',[None,True,float('nan'),float('inf')])
def test_completed_phase_losses_cannot_use_abstention_or_nonfinite(bad):
    values,original,result,plan=six_role_fixture()
    for suffix in ('_first_phase_analytic_return','_first_phase'):values[A.ROLES[0]+suffix]['loss']=bad
    with pytest.raises(ValueError,match='finite numeric phase losses'):A.validate_arithmetic(values,original,True,result,plan)


def test_nonfinite_loss_retained_only_as_unvalidated_terminal_role(tmp_path):
    stop=A.expected_sequence().index(A.ROLES[0]+'_first_phase')+1
    values,original,result,plan,_=synthetic_prefix(tmp_path,stop)
    for suffix in ('_first_phase_analytic_return','_first_phase'):values[A.ROLES[0]+suffix]['loss']=None
    checked=A.validate_arithmetic(values,original,False,result,plan)
    assert checked['validated_roles']==[]
    assert checked['unvalidated_partial_roles'][0]['role']==A.ROLES[0]



def test_completed_result_lag_rejected_after_next_operation_start(tmp_path):
    stop=A.expected_sequence().index(A.ROLES[1]+'_first_phase_analytic_return')
    _,_,result,_,_=synthetic_prefix(tmp_path,stop);result['completed_roles']=[]
    with pytest.raises(ValueError,match='completed-role'):A.records(tmp_path,A.directory(tmp_path),False,result)
