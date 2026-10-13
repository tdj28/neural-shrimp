"""Synthetic retained-file audit tests, without model or producer imports."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1]/'codex/confirm-fold/audit.py'
SPEC = importlib.util.spec_from_file_location('fresh_fold_independent_audit', PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def artifact(path):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=audit.digest(path))


def fixture(tmp_path, label='A', opposite=False):
    root=tmp_path/label
    root.mkdir()
    x=np.array([1.,0.]); r=np.array([1.,0.]); l=r.copy()
    raw_a=-100. if opposite else 100.
    b=25.; U=.1/b; a=abs(raw_a)
    z=np.random.default_rng(10501 if label=='A' else 10502).normal(size=2)
    z-=r*(l@z); w=z/np.linalg.norm(z)
    starts=np.stack([x-U*r,x,x-U*r+U*w])
    M=np.diag([1.,.25])
    inputs=root/'inputs.npz'
    np.savez(inputs,x_c=x,r=r,l=l,w=w,starts=starts,matrix=M,raw_r=r,raw_l=l,
             parameter_derivative=np.array([raw_a,0.]),directional_second=np.array([2*b,0.]))
    setup=root/'setup.json'
    pc=.067 if label=='A' else -.1033
    side=-1 if opposite else 1
    write_json(setup,dict(passed=True,gates={'closed':True},raw_a=raw_a,raw_b=b,parameter_c=pc,ghost_parameter_direction=side))
    rows=[]
    for i,epsilon in enumerate(audit.EPS):
        for j,name in enumerate(audit.ENTRANCES):
            u0=0. if j==1 else -U
            n,arc=audit.scalar_prediction(u0,a,b,epsilon,U)
            rows.append(dict(id=f'{label}-{i}-{name}',target=label,group=i,entrance_index=j,eps=epsilon,
                             entrance=name,u_in=u0,u_out=U,parameter=pc+side*epsilon,quadratic_G=n,arctan_G=arc))
    target=dict(label=label,status='ready',setup=artifact(setup),setup_arrays=artifact(inputs),inputs=artifact(inputs),
                period=2 if label=='A' else 8,coordinate='alpha' if label=='A' else 'log10T',
                seed=10501 if label=='A' else 10502,parameter_c=pc,alpha=.067 if label=='A' else .0717,
                log10T=-.140 if label=='A' else pc,raw_a=raw_a,raw_b=b,a_positive=a,b_positive=b,
                U=U,ghost_direction=side,max_g=max(512,4*max(row['quadratic_G'] for row in rows)),rows=rows)
    result=dict(target=label,status='completed',rows=[],groups=[],freeze_commit='a'*40,plan_sha256='b'*64,
                freeze_pushed_at_utc='2026-10-13T01:00:00+00:00',started_utc='2026-10-13T01:00:01+00:00',
                first_target_outcome_at_utc='2026-10-13T01:00:02+00:00')
    for i,epsilon in enumerate(audit.EPS):
        states=starts.copy();counts=[None]*3
        previous=np.full_like(starts,np.nan); crossed=previous.copy()
        us=[];transverse=[];full_states=[]
        for n in range(1,target['max_g']+1):
            ur=np.full(3,np.nan);tr=np.full(3,np.nan);sr=np.full_like(starts,np.nan)
            for j in range(3):
                if counts[j] is not None:
                    continue
                old=states[j].copy()
                u=(old-x)@l
                u+=a*epsilon+b*u*u
                states[j]=x+u*r+.25*(old-x-((old-x)@l)*r)
                sr[j]=states[j]
                ur[j]=(states[j]-x)@l
                tr[j]=np.linalg.norm(states[j]-x-ur[j]*r)
                if ur[j]>U:
                    counts[j]=n;previous[j]=old;crossed[j]=states[j]
            us.append(ur);transverse.append(tr);full_states.append(sr)
            if all(k is not None for k in counts):
                break
        path=root/f'group-{i}.npz'
        np.savez(path,u=np.array(us),transverse=np.array(transverse),state_trace=np.array(full_states),initial_states=starts,
                 previous_states=previous,crossing_states=crossed,states_at_stop=states)
        result['groups'].append(dict(index=i,trace_sha256=audit.digest(path),G_evaluations=len(us),
                                      batch_model_steps=len(us)*target['period'],
                                      alpha=rows[3*i]['parameter'] if label=='A' else target['alpha'],
                                      log10T=rows[3*i]['parameter'] if label=='B' else target['log10T']))
        for j in range(3):
            row=rows[3*i+j];n=counts[j]
            de=n-row['quadratic_G'];ae=n-row['arctan_G']
            result['rows'].append(dict(**row,status='observed',measured_G=n,measured_F=target['period']*n,
                                      recorded_G=n,quadratic_error=de,arctan_error=ae,primary_pass=abs(de)<=1,arctan_pass=abs(ae)<=1))
    result['primary_all_pass']=all(row['primary_pass'] for row in result['rows'])
    result['arctan_all_pass']=all(row['arctan_pass'] for row in result['rows'])
    write_json(root/'results.json',result)
    return target,root,result


def failures(report):
    return [check for check in report['checks'] if not check['passed']]


@pytest.mark.parametrize('label', ['A','B'])
@pytest.mark.parametrize('opposite',[False,True])
def test_complete_independent_reconstruction_in_both_parameter_units(tmp_path,label,opposite):
    target,root,_=fixture(tmp_path,label,opposite)
    result=audit.audit_target(target,root)
    assert result['integrity_pass'], failures(result)
    assert result['primary_all_pass']
    assert len(result['rows'])==12
    assert all(row['crossing_margin']>0 for row in result['rows'])


def test_strict_crossing_and_no_silent_nan_drop():
    assert audit.first_exit([.5,1.,1.01,np.nan],1.)['count']==3
    assert audit.first_exit([.5,1.,1.01,np.nan],1.)['passed']
    assert not audit.first_exit([np.nan,.5,1.01,np.nan],1.)['passed']
    assert not audit.first_exit([.5,1.01,1.2],1.)['passed']
    assert audit.first_exit([.1,.2],1.)['count'] is None
    assert not audit.first_exit([.1,np.nan],1.)['passed']


def test_hand_computed_scalar_clock_and_integral():
    # u0=0: updates 0 -> .5 -> 1.5; strict >1 exits at the second step.
    n,arc=audit.scalar_prediction(0.,1.,2.,.5,1.)
    assert n==2
    assert arc==pytest.approx(np.arctan(2.))
    with pytest.raises(ValueError):
        audit.scalar_prediction(0.,1.,2.,0.,1.)


def test_corrupt_input_hash_fails_closed_with_all_rows_retained(tmp_path):
    target,root,_=fixture(tmp_path)
    with (root/'inputs.npz').open('ab') as stream:
        stream.write(b'altered')
    result=audit.audit_target(target,root)
    assert not result['integrity_pass']
    assert not result['primary_all_pass']
    assert len(result['rows'])==12


@pytest.mark.parametrize('change', ['count','clock','missing','prediction','parameter','duplicate'])
def test_mutated_runtime_metadata_is_detected(tmp_path,change):
    target,root,result=fixture(tmp_path)
    if change=='count': result['rows'][0]['measured_G']+=1
    if change=='clock': result['rows'][0]['measured_F']+=1
    if change=='missing': result['rows'].pop()
    if change=='prediction': result['rows'][0]['quadratic_G']+=2
    if change=='parameter': result['groups'][0]['alpha']+=1e-5
    if change=='duplicate': result['rows'][-1]=result['rows'][0]
    write_json(root/'results.json',result)
    checked=audit.audit_target(target,root)
    assert not checked['integrity_pass']
    assert len(checked['rows'])==12


def test_missing_trace_retains_missing_trial_ids(tmp_path):
    target,root,_=fixture(tmp_path)
    (root/'group-2.npz').unlink()
    result=audit.audit_target(target,root)
    assert not result['integrity_pass']
    assert not result['primary_all_pass']
    assert len(result['rows'])==12
    assert sum(r.get('status')=='missing_or_invalid_trace' for r in result['rows'])==3


def test_incorrect_full_crossing_state_is_detected_even_with_rehashed_npz(tmp_path):
    target,root,result=fixture(tmp_path)
    p=root/'group-0.npz'; data=audit.load_npz(p)
    data['crossing_states'][0]=data['initial_states'][0]
    np.savez(p,**data)
    result['groups'][0]['trace_sha256']=audit.digest(p)
    write_json(root/'results.json',result)
    report=audit.audit_target(target,root)
    assert not report['integrity_pass']
    assert any('strict_full_state_exit' in row['check'] for row in failures(report))


def test_setup_failed_target_is_retained_without_false_success(tmp_path):
    target,root,_=fixture(tmp_path)
    target['status']='setup_gate_failed'
    report=audit.audit_target(target)
    assert report['integrity_pass']
    assert not report['primary_all_pass']
    assert len(report['rows'])==12
    assert all(row['status']=='not_run_setup_gate' for row in report['rows'])


def test_censoring_preserves_integrity_but_cannot_pass_scientific_claim(tmp_path):
    target,root,result=fixture(tmp_path)
    path=root/'group-0.npz';data=audit.load_npz(path)
    cap=target['max_g'];old=len(data['u'])
    for key,shape in [('u',(cap,3)),('transverse',(cap,3)),('state_trace',(cap,3,2))]:
        enlarged=np.full(shape,np.nan)
        enlarged[:old]=data[key]
        data[key]=enlarged
    state=data['initial_states'][0].copy()
    data['u'][:,0]=(state-np.array([1.,0.]))@np.array([1.,0.])
    data['transverse'][:,0]=0.
    data['state_trace'][:,0]=state
    data['previous_states'][0]=np.nan;data['crossing_states'][0]=np.nan;data['states_at_stop'][0]=state
    np.savez(path,**data)
    result['groups'][0].update(trace_sha256=audit.digest(path),G_evaluations=cap,batch_model_steps=target['period']*cap)
    result['rows'][0].update(status='censored',measured_G=None,measured_F=None,recorded_G=cap,
                             quadratic_error=None,arctan_error=None,primary_pass=False,arctan_pass=False)
    result.update(primary_all_pass=False,arctan_all_pass=False)
    write_json(root/'results.json',result)
    report=audit.audit_target(target,root)
    assert report['integrity_pass'], failures(report)
    assert not report['primary_all_pass'] and not report['arctan_all_pass']
    assert report['rows'][0]['measured_G'] is None


def test_corruption_of_interior_full_state_is_detected(tmp_path):
    target,root,result=fixture(tmp_path)
    path=root/'group-0.npz';data=audit.load_npz(path)
    data['state_trace'][0,0,0]+=.001
    np.savez(path,**data)
    result['groups'][0]['trace_sha256']=audit.digest(path)
    write_json(root/'results.json',result)
    report=audit.audit_target(target,root)
    assert not report['integrity_pass']
    assert any(row['check'].endswith('/every_state_projection') for row in failures(report))


def test_full_study_freeze_tree_and_temporal_binding_without_git_mutation(tmp_path,monkeypatch):
    ta,ra,ma=fixture(tmp_path,'A');tb,rb,mb=fixture(tmp_path,'B')
    repo=tmp_path/'repo';plan_dir=repo/'codex/confirm-fold';plan_dir.mkdir(parents=True)
    source=plan_dir/'fake.py';source.write_text('# fixture source\n')
    plan=dict(schema=1,created_utc='2026-10-13T00:59:00+00:00',
              execution=dict(epsilon_order=list(audit.EPS),entrances=list(audit.ENTRANCES),device='cpu',dtype='float64'),
              source=[dict(path='codex/confirm-fold/fake.py',sha256=audit.digest(source))],targets=[ta,tb])
    path=plan_dir/'design.json';write_json(path,plan)
    for folder,meta in [(ra,ma),(rb,mb)]:
        meta['plan_sha256']=audit.digest(path);write_json(folder/'results.json',meta)
    calls=[]
    def fake_show(args,**kwargs):
        calls.append(args)
        assert args[:2]==['git','show']
        sha,rel=args[2].split(':',1)
        assert sha=='a'*40
        return (repo/rel).read_bytes()
    monkeypatch.setattr(audit.subprocess,'check_output',fake_show)
    result=audit.audit_study(path,{'A':ra,'B':rb},repo=repo,expected_freeze='a'*40)
    assert result['integrity_pass'], failures(result)
    assert result['primary_all_pass']
    assert len(calls)==4
    # An outcome before the claimed push is rejected even with matching bytes.
    ma['first_target_outcome_at_utc']='2026-10-13T00:59:59+00:00'
    write_json(ra/'results.json',ma)
    late=audit.audit_study(path,{'A':ra,'B':rb},repo=repo,expected_freeze='a'*40)
    assert not late['integrity_pass'] and not late['primary_all_pass']
    assert any(row['check']=='A/prospective_timestamps' for row in failures(late))
    # A matching runtime string is not enough if the Git tree bytes disagree.
    monkeypatch.setattr(audit.subprocess,'check_output',lambda *args,**kwargs:b'different committed bytes')
    changed=audit.audit_study(path,{'A':ra,'B':rb},repo=repo,expected_freeze='a'*40)
    assert not changed['integrity_pass']
    assert any(row['check'].endswith('/plan_in_freeze_tree') for row in failures(changed))
