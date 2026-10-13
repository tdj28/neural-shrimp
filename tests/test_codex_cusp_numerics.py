import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parents[1]/'codex/cusp-crosscheck'
def load(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'))
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
N=load('numerics');P=load('dense_profile');I=load('inputs')


def test_fold_cubic_exact_nonnormal_and_orientation():
    # S(u,v)=(u+v,v), |Sr|=1: exact c=5+3*2/(1-.25)=13.
    S=torch.tensor([[1.,1.],[0.,1.]],dtype=torch.float64);Sinv=torch.linalg.inv(S)
    def f(x):
        u,v=Sinv@x
        return S@torch.stack((u+7*u*u+5*u**3+3*u*v,.25*v+2*u*u))
    x=torch.zeros(2,dtype=torch.float64)
    M=torch.func.jacfwd(f)(x).numpy();basis=N.fold_basis(M)
    r=torch.tensor(basis['right']);l=basis['left']
    B=N.directional(f,x,(r,r)).numpy();C=N.directional(f,x,(r,r,r)).numpy()
    h=N.homological(M,r.numpy(),l,B)
    mixed=N.directional(f,x,(r,torch.tensor(h['h2']))).numpy();c=N.cubic(l,C,mixed)
    assert h['b']==pytest.approx(7) and c['c']==pytest.approx(13)
    assert h['bordered_relative_residual']<1e-14 and abs(h['orthogonality'])<1e-14
    hminus=N.homological(M,-r.numpy(),-l,B)
    np.testing.assert_allclose(hminus['h2'],h['h2'])
    assert hminus['b']==pytest.approx(-7)
    assert N.cubic(-l,-C,-mixed)['c']==pytest.approx(c['c'])


def test_near_multiplier_homological_and_normalization():
    mu=1.+1e-9;M=np.diag([mu,.25]);r=np.array([1.,0.]);l=r.copy();B=np.array([14.,4.])
    exact=N.homological(M,r,l,B);shift=N.homological(M,r,l,B,multiplier=mu)
    assert exact['h2'][1]==pytest.approx(4/.75)
    assert shift['h2'][1]==pytest.approx(4/(mu*mu-.25))
    assert shift['h2'][1]!=exact['h2'][1]
    with pytest.raises(ValueError,match='unit right'):N.homological(M,2*r,l,B)


def test_classify_before_coefficients_hyperbolic_not_attracting():
    for matrix in (np.diag([-.99,.2]),np.diag([1.,1.]),np.diag([1.,-.9999])):
        with pytest.raises(ValueError):N.fold_basis(matrix)
    row=N.fold_basis(np.diag([1.,1.5]))
    assert row['complement_attracting'] is False and row['nearest_other_unit_circle']==.5
    assert row['next_modulus']==1.5


def test_mixed_third_ad_matches_explicit_polynomial():
    def f(x):return torch.stack((x[0]*x[1]+x[0]**3,2*x[1]**3))
    x=torch.tensor([.2,-.3],dtype=torch.float64)
    r=torch.tensor([.4,.5],dtype=torch.float64);h=torch.tensor([.6,-.7],dtype=torch.float64)
    B=N.directional(f,x,(r,h))
    expected=torch.tensor([r[0]*h[1]+r[1]*h[0]+6*x[0]*r[0]*h[0],12*x[1]*r[1]*h[1]],dtype=torch.float64)
    torch.testing.assert_close(B,expected)
    C=N.directional(f,x,(r,r,r));torch.testing.assert_close(C,torch.tensor([6*r[0]**3,12*r[1]**3]))


class Fake:
    d=4
    def prefix_cache(self,alpha):
        from types import SimpleNamespace
        return SimpleNamespace(alpha=torch.tensor(alpha),layers=(torch.ones(1),))
    def period_map(self,x,alpha,t,period,cache=None):return x+x*x*.1+alpha*x+float(t)
    def jvp(self,x,v,alpha,t,period=7,cache=None):
        return torch.func.jvp(lambda y:self.period_map(y,alpha,t,period,cache),(x,),(v,))[1]


def test_profile_full_basis_order_and_cost_scope():
    events=[];x=torch.arange(4,dtype=torch.float64)/10
    row=P.profile(Fake(),x,.02,.18,lambda n,v:events.append((n,v)),lambda:None,dimension=4,batch=2)
    assert row['dense_columns_completed']==4 and row['operation_counts']['basis_batches']==2
    M=next(v['matrix'] for n,v in events if n=='dense896')
    torch.testing.assert_close(M,torch.diag(1.02+.2*x))
    assert row['computational_completed'] and not row['normal_form_coefficients_computed']
    assert all(events[i-1][0].endswith('-input') for i,(n,v) in enumerate(events) if n.startswith('basis-') and n.endswith('-return'))
    assert row['operation_counts']['third_ad']==1


def test_completed_nonfinite_native_return_is_retained():
    events=[]
    with pytest.raises(ValueError,match='nonfinite completed'):
        P.measured(lambda n,v:events.append((n,v)),lambda:None,'native',{},lambda:torch.tensor([float('nan')]),[])
    assert [n for n,v in events]==['native-input','native-return']
    assert torch.isnan(events[1][1]['output']).all()


def test_store_and_post_return_deadline(tmp_path):
    calls=[]
    def stop():calls.append(1);raise TimeoutError('stop')
    store=P.Store(tmp_path,stop)
    with pytest.raises(TimeoutError):store.emit('actual-return',dict(output=np.array([np.inf])))
    arrays=list(tmp_path.glob('*.npy'));assert len(arrays)==1
    assert np.isinf(np.load(arrays[0],allow_pickle=False))[0]
    assert len(list(tmp_path.glob('*.json')))==1


def test_literal_candidate_metadata_and_prepared_plan_refuse():
    import copy
    curve=dict(period=7,kind='fold',args=dict(alpha_scale=10.),points=[{} for _ in range(40)])
    for i,w in I.CANDIDATES.items():curve['points'][i]=dict(index=i,theta=[w['log10_temperature'],w['alpha']*10],**w)
    assert set(I.metadata(curve))=={'13','14'}
    bad=copy.deepcopy(curve);bad['points'][14]['alpha']+=1e-6
    with pytest.raises(ValueError):I.metadata(bad)
    plan=dict(schema='codex-cusp-profile-v1',envelope=P.CAPS.copy(),source_sha256={p:'a'*64 for p in P.SOURCES},
        candidate_authority_sha256=I.AUTHORITY_SHA,profile_index=13,period=7,dimension=896,batch_directions=16,
        vector_sha256='b'*64,scope='test',admission='prepared')
    with pytest.raises(ValueError,match='does not authorize'):P.validate_plan(plan)
    plan['admission']='reviewed';assert P.validate_plan(plan)==plan
    plan['period']=2
    with pytest.raises(ValueError):P.validate_plan(plan)


def test_nonfinite_algebra_outputs_refuse(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(np.linalg,'eig',lambda a:(np.array([1.,np.nan]),np.eye(2)))
        with pytest.raises(ValueError,match='nonfinite dense'):N.fold_basis(np.diag([1.,.2]))
    with monkeypatch.context() as m:
        m.setattr(np.linalg,'solve',lambda a,b:np.array([0.,np.inf,0.]))
        with np.errstate(all='ignore'),pytest.raises(ValueError,match='nonfinite homological'):
            N.homological(np.diag([1.,.2]),np.array([1.,0.]),np.array([1.,0.]),np.ones(2))
    with np.errstate(all='ignore'),pytest.raises(ValueError,match='nonfinite cubic'):
        N.cubic(np.array([1e308,1e308]),np.array([2.,2.]),np.array([2.,2.]))


def test_map_ancestry_recipe_rejects_parameter_or_prefix_change():
    import copy
    branch=dict(period=7,args=I.RECIPE.copy())
    report=dict(declared_map=I.RECIPE.copy(),period=7,alpha_scale=10.,prefix_decodes_to_expected_prompt=True,
        prefix_ids=[1,2],independent_checkpoint_sha256='a'*64,current_vector_sha256='b'*64,limits=['metadata-only'])
    assert I.ancestry_metadata(branch,report)['declared_map']==I.RECIPE
    for key,value in [('window',2),('layer',11),('vectors','other')]:
        bad=copy.deepcopy(branch);bad['args'][key]=value
        with pytest.raises(ValueError,match='map recipe'):I.ancestry_metadata(bad,report)
    report['prefix_decodes_to_expected_prompt']=False
    with pytest.raises(ValueError,match='map recipe'):I.ancestry_metadata(branch,report)


def test_actual_input_binding_without_array_decode(monkeypatch):
    monkeypatch.setattr(np,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('target array decode forbidden')))
    binding=I.authenticate()
    assert binding['array_values_decoded'] is False
    assert len(binding['ancestry']['prefix_ids'])==37 and binding['ancestry']['declared_map']==I.RECIPE


def test_isolated_source_closure_without_model_or_arrays(tmp_path):
    import os,subprocess
    repo=Path(__file__).resolve().parents[1]
    for name in P.SOURCES:
        target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((repo/name).read_bytes())
    script=tmp_path/'import_only.py'
    script.write_text("import pathlib,sys\nroot=pathlib.Path(__file__).parent\nsys.path[:0]=[str(root),str(root/'codex/cusp-crosscheck')]\nimport numpy as np\nnp.load=lambda *a,**k:(_ for _ in ()).throw(AssertionError('arrays forbidden'))\nimport inputs,numerics,dense_profile\nfrom codex.crosscheck.qwen_map import ManualQwenSoftMap\nManualQwenSoftMap.__init__=lambda *a,**k:(_ for _ in ()).throw(AssertionError('model construction forbidden'))\na=dense_profile.module(root/'codex/hostile-fold-crosscheck/assets.py','own_assets')\nprint('isolated import closure passed; no model or arrays')\n")
    env=os.environ.copy();env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    result=subprocess.run([sys.executable,'-I','-B',str(script)],cwd=tmp_path,env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert 'closure passed' in result.stdout
    assert not list(tmp_path.rglob('*.pyc'))
