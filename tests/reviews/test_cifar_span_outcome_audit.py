"""Synthetic-only independent auditor controls; never open fixed data paths."""
import copy
import importlib.util
import json
import math
from fractions import Fraction
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
def module(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
A=module('independent_span_auditor','tools/reviews/audit_cifar_input_span_outcome.py')
L=module('independent_span_launcher','tools/reviews/launch_cifar_input_span_audit.py')
def length(a):return math.hypot(*np.asarray(a).ravel())
def check(a,b,scale):
    error=length(a-b);limit=1e-12+1e-8*scale
    return dict(error=error,limit=limit,scale=scale,passed=error<=limit)
def vector_fixture(zero=False):
    x=np.array([[1.,0.,0.],[2.,0.,0.]]) # rank one, two retained orthonormal columns
    q=np.array([[1.,0.],[0.,1.],[0.,0.]])
    v=np.zeros(8) if zero else np.array([1.,2.,3.,4.,5.,6.,7.,8.])
    active=v.copy();active[4:6]=0.;perp=np.zeros(8);perp[4:6]=v[4:6]
    n,an,pn=map(length,(v,active,perp));w=v[:6].reshape(3,2);nw=perp[:6].reshape(3,2)
    checks=dict(reconstruction=check(active+perp,v,n),implicit_projector_idempotence=check(active,active,n),
      complement_orthogonality=check(q.T@nw,np.zeros((2,2)),n),x_complement_preactivation=check(x@nw,np.zeros((2,2)),length(x@w)))
    saved=dict(norm=n,active_norm=an,complement_norm=pn,pythagorean_error=abs(n*n-an*an-pn*pn),pythagorean_limit=1e-12+1e-8*n*n,
      complementary_squared_fraction=pn*pn/(n*n) if n else None,active_squared_fraction=an*an/(n*n) if n else None,checks=checks)
    return v,active,perp,x,q,2,saved

def test_rank_deficient_completed_basis_without_qr(monkeypatch):
    monkeypatch.setattr(np.linalg,'qr',lambda *_a,**_k:pytest.fail('QR forbidden'))
    x=np.array([[1.,0.,0.],[2.,0.,0.]]);q=np.array([[1.,0.],[0.,1.],[0.,0.]]);r=np.array([[1.,2.],[0.,0.]])
    saved=dict(orthogonality_frobenius=0.,representation_relative=0.,x_i_minus_p_relative=0.,implicit_idempotence_on_q=0.,qr_reconstruction_relative=0.,retained_columns=2,rank_truncation=False,passed=True)
    assert A.basis(x,q,r,saved)[1]
    with pytest.raises(ValueError):A.basis(x,q,r+1e-3,saved)
    with pytest.raises(ValueError):A.basis(x,q*2,r,saved)
    with pytest.raises(ValueError):A.basis(x,q[:,:1],r,saved)

@pytest.mark.parametrize('zero',[False,True])
def test_saved_decomposition_packing_and_reduction(zero):
    values,passed=A.decompose(*vector_fixture(zero));assert passed
    if zero:assert values['complementary_squared_fraction'] is None
    else:assert values['complement_norm']==pytest.approx(math.sqrt(61))

@pytest.mark.parametrize('mutation',['first_active','last_active','last_perp','metric','gate','nan'])
def test_saved_vector_mutations_fail(mutation):
    args=list(vector_fixture());saved=args[-1]
    if mutation=='first_active':args[1][0]+=.01
    elif mutation=='last_active':args[1][-1]+=.01
    elif mutation=='last_perp':args[2][-1]=-.0 # byte-level packing mismatch
    elif mutation=='metric':saved['norm']+=.01
    elif mutation=='gate':saved['checks']['reconstruction']['passed']=False
    else:args[1][0]=np.nan
    with pytest.raises(ValueError):A.decompose(*args)

@pytest.mark.parametrize('bad',[np.array([np.inf]),np.array([np.nan]),np.array([1],dtype=np.float32),np.array([1j])])
def test_nonfinite_and_precision_fail(bad):
    with pytest.raises(ValueError):A.finite(bad)

def test_finite_norm_overflow_and_shape_fail():
    with pytest.raises((ValueError,FloatingPointError)):A.nrm(np.array([1e308]))
    with pytest.raises(ValueError):A.finite(np.ones(2),(3,))
    assert not A.same(np.array([0.]),np.array([-0.]))

def decay_record(eta=.00827753,lam=.01):
    q=1-Fraction(eta)*Fraction(lam);rounded=1-eta*lam;factor=rounded**2-1
    return dict(eta=eta,lam=lam,period=2,eta_hex=eta.hex(),lam_hex=lam.hex(),eta_exact=str(Fraction(eta)),lam_exact=str(Fraction(lam)),exact_q=str(q),exact_factor=str(q*q-1),rounded_q=rounded,rounded_q_hex=rounded.hex(),rounded_factor=factor,rounded_factor_hex=factor.hex(),elimination_valid=q*q!=1,omitted_directions_attracting=abs(q*q)<1)

def test_exact_rational_and_rounded_factor_are_distinct():
    record=decay_record();q,f=A.decay_audit(record['eta'],record['lam'],record)
    assert q*q-1!=Fraction(f)
    corrupted=copy.deepcopy(record);corrupted['exact_factor']=str(Fraction(f))
    with pytest.raises(ValueError):A.decay_audit(record['eta'],record['lam'],corrupted)
    record=decay_record(.125,0.);q,f=A.decay_audit(.125,0.,record)
    assert q==1 and f==0 and not record['elimination_valid']

@pytest.mark.parametrize('passed,fraction,expected',[(True,.5,True),(True,float(np.nextafter(.5,0)),False),(False,.9,False),(True,None,False),(True,0.,False)])
def test_fixed_inclusive_half_triage(passed,fraction,expected):assert A.decision(passed,fraction) is expected

@pytest.mark.parametrize('fraction',[float('nan'),float('inf'),-.01])
def test_invalid_triage_fraction(fraction):
    with pytest.raises(ValueError):A.decision(True,fraction)

def save_manifest(folder):
    rows={p.name:A.entry(p) for p in folder.iterdir() if p.name!='manifest.json'}
    (folder/'manifest.json').write_bytes(A.canonical(rows));return A.entry(folder/'manifest.json')['sha256']

@pytest.mark.parametrize('mutation',['hash','extra','member_link','manifest_link'])
def test_complete_inventory_corruptions(tmp_path,mutation):
    folder=tmp_path/'data';folder.mkdir();(folder/'a').write_bytes(b'abc');pin=save_manifest(folder)
    assert A.inventory(folder,pin,{'a'})
    if mutation=='hash':(folder/'a').write_bytes(b'bad')
    elif mutation=='extra':(folder/'extra').write_bytes(b'1')
    elif mutation=='member_link':(folder/'a').unlink();(tmp_path/'other').write_bytes(b'abc');(folder/'a').symlink_to(tmp_path/'other')
    else:(tmp_path/'manifest').write_bytes((folder/'manifest.json').read_bytes());(folder/'manifest.json').unlink();(folder/'manifest.json').symlink_to(tmp_path/'manifest')
    with pytest.raises(ValueError):A.inventory(folder,pin)

def test_reduction_cancellation_tolerance_does_not_relax_scientific_gate():
    args=list(vector_fixture());args[-1]['pythagorean_error']+=1e-13
    assert A.decompose(*args)[1]
    args[-1]['pythagorean_error']+=.01
    with pytest.raises(ValueError):A.decompose(*args)

@pytest.mark.parametrize('mode',['profile','collect'])
def test_launcher_full_binding_and_tampered_command(tmp_path,monkeypatch,mode):
    monkeypatch.setattr(A,'BASE',tmp_path);label='profile-001' if mode=='profile' else 'attempt-001';key='profile' if mode=='profile' else 'target'
    folder=tmp_path/(label+'-launcher');folder.mkdir();wrapper=b'synthetic wrapper';pins={key+'_source_commit':'a'*40,'source_review_commit':'b'*40,'profile_manifest_sha256':'c'*64,'wrapper_sha256':A.digest(wrapper)}
    argv=['timeout','--signal=TERM','--kill-after='+('2s' if mode=='profile' else '5s'),'8s' if mode=='profile' else '45s','nice','-n','10','/data2/neural-shrimp/.venv/bin/python','tools/diagnose_cifar_input_span.py',mode,'--out',str(tmp_path/label),'--review-commit','b'*40]
    if mode=='collect':argv+=['--profile-manifest-sha256','c'*64]
    runtime=dict(mode=mode,source_commit='a'*40,review_commit='b'*40,argv=argv,cwd='/home/d7082791602/.codex/worktrees/post-edge-next-steps/neural-shrimp',out=str(tmp_path/label),thread_env={k:'1' for k in A.THREADS},log_limit=128*1024,launcher_byte_limit=256*1024,supervisor_seconds=10 if mode=='profile' else 50)
    rows={'fake.json':dict(bytes=3,sha256='d'*64)};receipt=dict(timeout_pid=100,raw_files=rows,raw_bytes=3,elapsed_seconds=.1)
    for name,value in [('runtime-start.json',runtime),('receipt.json',receipt),('process.json',dict(timeout_pid=100))]:(folder/name).write_bytes(A.canonical(value))
    (folder/'launcher-source.py').write_bytes(wrapper);(folder/'stdout.log').write_bytes(b'')
    pins[key+'_launcher_manifest_sha256']=save_manifest(folder)
    assert A.launcher(mode,pins,rows)['bytes']>0
    runtime['argv'][6]='0';(folder/'runtime-start.json').write_bytes(A.canonical(runtime));pins[key+'_launcher_manifest_sha256']=save_manifest(folder)
    with pytest.raises(ValueError):A.launcher(mode,pins,rows)

@pytest.mark.parametrize('failure',[True,False])
def test_audit_driver_retains_partial_or_completed_evidence(tmp_path,monkeypatch,capsys,failure):
    root=tmp_path/'source';root.mkdir();monkeypatch.setattr(A,'ROOT',root)
    for name in A.AUDIT_FILES:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'synthetic source')
    source='a'*40;pins={k:('b'*(40 if k.endswith('commit') else 64)) for k in A.PIN_KEYS};pp=tmp_path/'pins.json';pp.write_bytes(A.canonical(pins))
    def git(argv,**kwargs):return source+'\n' if argv[1]=='rev-parse' else (root/argv[2].split(':',1)[1]).read_bytes()
    monkeypatch.setattr(A.subprocess,'check_output',git);monkeypatch.setattr(A.os,'getpriority',lambda *_:10)
    def fake_audit(_pins,emit):
        emit('partial.json',{'saved':True})
        if failure:raise TimeoutError('synthetic interruption')
        return dict(verdict='saved_outcome_audit_pass')
    monkeypatch.setattr(A,'audit',fake_audit);out=tmp_path/'out'
    assert A.run(out,pp,A.entry(pp)['sha256'],source)==(2 if failure else 0)
    result=A.read(out/'result.json');assert result['status']==('failed' if failure else 'finished')
    assert (out/'partial.json').is_file() and A.inventory(out,A.entry(out/'manifest.json')['sha256'])
    assert not out.stat().st_mode&0o222
    with pytest.raises(ValueError):A.run(out,pp,A.entry(pp)['sha256'],source)
    out.chmod(0o755)
    for path in out.iterdir():path.chmod(0o644)

def test_ordinary_nice_rejected_before_any_output(tmp_path,monkeypatch):
    monkeypatch.setattr(A.os,'getpriority',lambda *_:0)
    with pytest.raises(ValueError,match='nice10'):A.run(tmp_path/'out',tmp_path/'pins','0'*64,'0'*40)
    assert not (tmp_path/'out').exists()

def test_launcher_clean_source_guard_prevents_launch(tmp_path,monkeypatch):
    pp=tmp_path/'pins';pp.write_bytes(b'{}');monkeypatch.setattr(L.sys,'argv',['launcher','a'*40,str(pp),L.entry(pp)['sha256']])
    monkeypatch.setattr(L.subprocess,'check_output',lambda args,**kwargs:'a'*40+'\n' if args[1]=='rev-parse' else ' M dirty')
    monkeypatch.setattr(L.subprocess,'Popen',lambda *_a,**_k:pytest.fail('must not launch'))
    with pytest.raises(AssertionError):L.main()
