import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'codex/cusp-crosscheck'
def load(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'))
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
N=load('numerics');E=load('endpoint_check');R=load('endpoint_runtime');I=load('inputs')


class Toy:
    d=2
    def __init__(self,critical=1.):self.critical=critical
    def prefix_cache(self,alpha):
        from types import SimpleNamespace
        return SimpleNamespace(alpha=torch.as_tensor(alpha),layers=())
    def forward(self,x,alpha,t,cache=None):
        u,v=x.unbind(-1)
        return torch.stack((self.critical*u+.2*u*u+.3*u**3+.1*u*v+alpha*(1+u),.25*v+.1*u*u),dim=-1)
    def period_map(self,x,alpha,t,period,cache=None):
        for _ in range(period):x=self.forward(x,alpha,t,cache)
        return x
    def jvp(self,x,v,alpha,t,period=1,cache=None):
        return torch.func.jvp(lambda q:self.period_map(q,alpha,t,period,cache),(x,),(v,))[1]


def test_complete_analytic_endpoint_and_every_ladder_row():
    records=[];ops=E.Operations(Toy(),0.,.2,1,lambda n,v:records.append((n,v)),lambda:None,batch=2)
    row=E.check_endpoint(ops,torch.zeros(2,dtype=torch.float64))
    assert row['numerical_qualified'] and row['certificate_passed'] is False and row['cusp_qualified'] is False
    assert row['coefficients']['b']==pytest.approx(.2)
    assert row['coefficients']['coefficient']['c']==pytest.approx(.3+.01/.75)
    assert row['coefficients']['a_physical_alpha']==pytest.approx(1.)
    assert len(row['fd_rows'])==7 and len(row['probe_fd_rows'])==7 and len(row['alpha_fd_rows'])==3
    assert all(row['precision_gates'].values())
    assert row['precision_gates']['b_projected'] and row['precision_gates']['physical_alpha_projected']
    assert all(row['numerical_sign_resolution'].values()) and row['orientation_bracket_ready'] and row['cubic_nonzero_supported']
    assert row['operation_counts']['dense-batch']==1
    assert row['operation_counts']['state-ad-2']==3 and row['operation_counts']['state-ad-3']==1
    assert next(i for i,(n,v) in enumerate(records) if n=='classified-fold-basis')<next(i for i,(n,v) in enumerate(records) if n.startswith('state-ad-2'))


def test_wrong_crossing_refuses_before_coefficients_with_spectrum():
    records=[];ops=E.Operations(Toy(-1.),0.,.2,1,lambda n,v:records.append((n,v)),lambda:None)
    with pytest.raises(E.GateFailure,match='fold-classification'):E.check_endpoint(ops,torch.zeros(2,dtype=torch.float64))
    assert any(n=='raw-full-spectrum' for n,v in records)
    assert not any(n.startswith(('state-ad','alpha-ad')) for n,v in records)


def test_failed_closure_refuses_before_dense_and_coefficients():
    records=[];ops=E.Operations(Toy(),.1,.2,1,lambda n,v:records.append((n,v)),lambda:None)
    with pytest.raises(E.GateFailure,match='orbit'):E.check_endpoint(ops,torch.zeros(2,dtype=torch.float64))
    assert not any(n.startswith(('dense-batch','state-ad')) for n,v in records)


def test_saved_matrix_reuse_and_corruption_gate():
    ops=E.Operations(Toy(),0.,.2,1,lambda *a:None,lambda:None)
    row=E.check_endpoint(ops,torch.zeros(2,dtype=torch.float64),saved_matrix=np.diag([1.,.25]))
    assert row['operation_counts'].get('dense-batch',0)==0
    ops=E.Operations(Toy(),0.,.2,1,lambda *a:None,lambda:None)
    with pytest.raises(E.GateFailure,match='dense-composed'):
        E.check_endpoint(ops,torch.zeros(2,dtype=torch.float64),saved_matrix=np.array([[1.,.01],[0.,.25]]))


def test_completed_nonfinite_return_and_no_selected_plateau():
    records=[];ops=E.Operations(Toy(),0.,0.,1,lambda n,v:records.append((n,v)),lambda:None)
    with pytest.raises(ValueError,match='nonfinite completed'):
        ops.call('native',{},lambda:torch.tensor([float('nan')]))
    assert [n for n,v in records]==['native-001-input','native-001-return']
    assert not E.ladder_gate([.1,.0001,.0002,.1])
    assert not E.ladder_gate([.1,.001,np.nan])
    assert E.ladder_gate([.1,.0002,.0001])


def test_actual_policy_prepared_refusal_and_literal_indices():
    p=dict(schema='codex-cusp-endpoints-v1',admission='prepared',envelope=dict(cpu_threads=2,nice=10,memory_gib=12,
        minimum_start_available_gib=28,minimum_runtime_available_gib=16,timeout_seconds=1,maximum_output_mib=64),
        source_sha256={s:'a'*64 for s in R.SOURCES},profile=dict(root=str(R.OWNED/'profile-001'),receipt_sha256='b'*64,source_commit='c'*40,plan_sha256='d'*64),
        indices=[13,14],tolerances=E.TOLERANCES,fd_steps=list(E.FD_STEPS),parameter_steps=list(E.PARAMETER_STEPS),scope='test')
    with pytest.raises(ValueError,match='reviewed'):R.validate_plan(p)
    p['admission']='reviewed';assert R.validate_plan(p)['timeout_seconds']==1
    p['indices']=[14,13]
    with pytest.raises(ValueError,match='two fixed'):R.validate_plan(p)


def profile_fixture(base,corrupt=False,*,timed_out=False,cusp_qualified=False):
    import hashlib,json
    run=base/'run';run.mkdir(parents=True);counter=0
    def put(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    def arr(value,label):
        nonlocal counter
        path=run/f'a-{counter:04d}-{label}.npy';counter+=1
        with path.open('wb') as f:np.save(f,value,allow_pickle=False)
        return dict(path=path.name,shape=list(value.shape),dtype=str(value.dtype))
    state=np.arange(896,dtype=np.float64)*1e-4;matrix=np.eye(896,dtype=np.float64);alpha=.01;t=.2
    put(run/'000-candidate-vector.json',dict(index=13,alpha=alpha,log10_temperature=t,state=arr(state,'state')))
    put(run/'500-dense896.json',dict(columns=896,matrix=arr(matrix,'matrix')))
    for start in range(0,896,16):
        directions=np.eye(896,dtype=np.float64)[start:start+16]
        if corrupt and start==16:directions=directions.copy();directions[0,0]=1
        put(run/f'100-basis-{start:03d}-input.json',dict(start=start,alpha=alpha,log10_temperature=t,period=7,
            state=arr(np.broadcast_to(state,(16,896)).copy(),'points'),directions=arr(directions,'directions')))
        put(run/f'101-basis-{start:03d}-return.json',dict(output=arr(matrix[:,start:start+16].T.copy(),'return')))
    old_sources=set(R.SOURCES)-{'codex/cusp-crosscheck/endpoint_check.py','codex/cusp-crosscheck/endpoint_runtime.py'}
    pins={}
    for name in old_sources:
        path=base/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('synthetic '+name)
        pins[name]=I.sha(path)
    backend=dict(torch='synthetic',numpy='synthetic',python='synthetic',threads=2)
    put(run/'original-plan.json',dict(profile_index=13,period=7,dimension=896,source_sha256=pins))
    put(run/'status.json',dict(status='resource_profile_completed',dense_columns_completed=896,computational_completed=True,
        backend=backend,resource_probe_only=True,fold_qualified=False,cusp_qualified=cusp_qualified,certificate_passed=False))
    put(run/'input-binding.json',dict(authority_sha256=I.AUTHORITY_SHA,ancestry_manifest_sha256=I.ANCESTRY_SHA))
    artifacts=[]
    for path in base.rglob('*'):
        if path.is_file():
            artifacts.append(dict(path=str(path.relative_to(base)),bytes=path.stat().st_size,sha256=I.sha(path)));path.chmod(0o444)
    rec=dict(source_commit='a'*40,plan_sha256=I.sha(run/'original-plan.json'),exit_code=0,timed_out=timed_out,worker_termination_verified=True,
        uncertain_ownership=[],status='resource_pilot_completed',artifacts=artifacts)
    put(base/'receipt.json',rec);(base/'receipt.json').chmod(0o444)
    authority=dict(root=str(base),receipt_sha256=I.sha(base/'receipt.json'),source_commit='a'*40,plan_sha256=rec['plan_sha256'])
    return authority,state,alpha,t,pins,backend


def test_full896_saved_basis_binding_and_refusals(tmp_path):
    args=profile_fixture(tmp_path/'valid')
    matrix,receipt=R.authenticated_matrix(*args)
    assert matrix.shape==(896,896) and receipt['all_896_column_inputs_returns_verified']
    changed=list(args);changed[1]=args[1]+1e-8
    with pytest.raises(ValueError,match='state/rate'):R.authenticated_matrix(*changed)
    changed=list(args);changed[4]=dict(args[4]);changed[4]['codex/crosscheck/qwen_map.py']='0'*64
    with pytest.raises(ValueError,match='same evaluator'):R.authenticated_matrix(*changed)
    changed=list(args);changed[5]=dict(args[5],torch='other')
    with pytest.raises(ValueError,match='same backend'):R.authenticated_matrix(*changed)
    bad=profile_fixture(tmp_path/'wrong-basis',corrupt=True)
    with pytest.raises(ValueError,match='coordinate-basis'):R.authenticated_matrix(*bad)


def test_projected_scalar_gate_detects_nonnormal_amplification():
    # Vector disagreement is tiny, but the left dual amplifies it after projection.
    left=np.array([1.,1e6]);reference=np.array([1.,0.]);estimate=np.array([1.,1e-7])
    assert E.relative(estimate,reference,floor=1.)<1e-5
    projected=E.scalar_error(left@estimate,left@reference)
    assert projected==pytest.approx(.1)
    assert not E.ladder_gate([projected]*3)
    assert E.sign_resolved(1.,[1.,1.,1.])
    assert not E.sign_resolved(1.,[1.,1.,-1.])
    assert not E.sign_resolved(0.,[0.,0.,0.])
    assert not E.sign_resolved(1.,[1.,1.,float('nan')])


@pytest.mark.parametrize('field,value,message',[('timed_out',True,'not completed/reaped'),('cusp_qualified',True,'resource-only')])
def test_completed_profile_contract_refuses_inconsistent_flags(tmp_path,field,value,message):
    args=profile_fixture(tmp_path/'profile',**{field:value})
    with pytest.raises(ValueError,match=message):R.authenticated_matrix(*args)
