import cmath,hashlib,json,pathlib,math,numpy as np
p=pathlib.Path('/data2/neural-shrimp-data/explore/curvature-mechanism/attempt-001-panel')
for name in ('eig','eigh','eigvals','eigvalsh','svd','solve','inv','pinv','norm','cond'):
 setattr(np.linalg,name,lambda *a,**k:(_ for _ in ()).throw(AssertionError('forbidden new decomposition/norm/solve')))
def fro(x): return math.sqrt(float(np.sum(np.abs(x)**2)))
def check(x,y):
 if not np.allclose(x,y,rtol=2e-11,atol=2e-12):raise AssertionError((x,y))
def load(f):
 r=json.loads((p/f).read_text()); ap=p/r['arrays_file']; assert hashlib.sha256(ap.read_bytes()).hexdigest()==r['arrays_sha256']
 with np.load(ap,allow_pickle=False) as z:
  def dec(v):
   if isinstance(v,dict) and 'array' in v:
    a=z[v['array']];assert a.dtype.str==v['dtype'] and list(a.shape)==v['shape']; assert hashlib.sha256(a.tobytes(order='C')).hexdigest()==v['sha256'];return a.copy()
   if isinstance(v,dict):return {k:dec(t) for k,t in v.items()}
   if isinstance(v,list):return [dec(t) for t in v]
   return v
  return dec(r['payload'])
def leading(w):return np.array(sorted(range(len(w)),key=lambda i:(-abs(w[i]),-w[i].real,-w[i].imag,i))[:2])
errs=[]; oracles=[]; checks=0
for line in (p/'journal.jsonl').read_text().splitlines():
 j=json.loads(line)
 if j['kind']!='analysis':continue
 row=load(j['file']);a=row['analysis']; f,g=a['phases']; n=len(f['u']);root=np.sqrt(a['inputs']['rates'])
 for phase,H in zip(a['phases'],a['inputs']['hessians']):
  K=root[:,None]*(H+H.T)/2*root[None,:]; w=phase['eigenvalues'];V=phase['eigenvectors'];u=V[:,-1]/fro(V[:,-1]);b=(np.trace(K)-w[-1])/(n-1); alpha=w[-1]-b; S=np.eye(n)-K; S0=(1-b)*np.eye(n)-alpha*np.outer(u,u)
  for x,y in [(K,phase['K']),(S,phase['S']),(S0,phase['S0']),(S-S0,phase['E']),(b,phase['bulk_curvature']),(alpha,phase['strength']),(fro(S-S0),phase['error_frobenius']),(fro(K@V-V*w)/max(1,fro(K)),phase['eigen_residual'])]:check(x,y)
 check(a['pair_product'],g['S']@f['S']);check(a['reverse_pair_product'],f['S']@g['S']);check(a['model_product'],g['S0']@f['S0'])
 target=a['inputs']['actual_monodromies'][0]*root[None,:]/root[:,None];check(target,a['target']); formation=fro(target-g['S']@f['S']);check(formation,a['formation_error_frobenius'])
 fit=g['model_norm2']*fro(f['E'])+fro(g['E'])*f['model_norm2']+fro(g['E'])*fro(f['E']);check(fit,a['delta_fit']);check(fit+formation,a['delta_op_total']);check(fro(target-a['model_product'])/max(1,fro(target)),a['relative_product_error'])
 tw=a['target_eigensystem']['values']; V=a['target_eigensystem']['right']; L=a['target_eigensystem']['left'];check(target@V-V*tw,a['spectral_diagnostic']['R_V']);check(target.T@L-L*np.conj(tw),a['spectral_diagnostic']['left_residual_matrix']);ti=leading(tw);check(ti,a['target_pair']['indices']);t=tw[ti];radius=max(abs(tw));check(t,a['target_pair']['values'])
 aa,bb=f['bulk_factor'],g['bulk_factor'];al,be=f['strength'],g['strength'];c=float(f['u']@g['u']);check(c,a['orientation_cosine']);model_errors={}
 for name,overlap in [('rotating',c),('aligned',1.),('orthogonal',0.)]:
  T=2*aa*bb-bb*al-aa*be+al*be*overlap**2;D=aa*bb*(aa-al)*(bb-be);d=T*T-4*D;m=a['models'][name]
  w=np.array([(aa-al)*(bb-be)]+[aa*bb]*(n-1),complex) if 1-overlap**2<=1e-10 else np.array([(T+cmath.sqrt(d))/2,(T-cmath.sqrt(d))/2]+[aa*bb]*(n-2),complex)
  for x,y in [(T,m['trace']),(D,m['determinant']),(d,m['discriminant']),(w,m['values'])]:check(x,y)
  pred=w[leading(w)];check(pred,m['leading']['values']);dist=[max(abs(pred-t)),max(abs(pred-t[::-1]))];ix=int(dist[1]<dist[0]);err=dist[ix]/max(1,radius);comp=a['comparisons'][name];check(err,comp['normalized']);check(max(abs(abs(pred)-abs(t[::-1] if ix else t))),comp['modulus_error']);model_errors[name]=err
  if comp['angle_error'] is not None:check(abs(cmath.phase(pred[np.argmax(pred.imag)])-cmath.phase(t[np.argmax(t.imag)])),comp['angle_error'])
 oracle=min(model_errors['aligned'],model_errors['orthogonal']);check(oracle,a['oracle_commuting_error']);errs.append(model_errors['rotating']);oracles.append(oracle)
 diag=a['spectral_diagnostic'];check(fro(diag['R_V'])/diag['sigma_min'],diag['delta_diag']);check(diag['sigma_max']/diag['sigma_min'],diag['kappa2']);check(diag['delta_diag']+diag['kappa2']*a['delta_op_total'],diag['rho_num'])
 checks+=1
 print(json.dumps({'slot':row['slot'],'target':[str(x) for x in t],'predicted':[str(x) for x in a['models']['rotating']['leading']['values']],'pair_error':model_errors['rotating'],'oracle':oracle,'rho_num':diag['rho_num'],'kappa2':diag['kappa2'],'phase_fit_errors':[f['error_frobenius'],g['error_frobenius']],'relative_product_error':a['relative_product_error'],'formation_error':formation}))
r=json.loads((p/'result.json').read_text());check(sum(errs)/6,r['aggregate']['mean_pair_error']);check(sum(oracles)/6,r['aggregate']['mean_oracle_commuting_error']);assert all(e>.02 for e in errs)
print(json.dumps({'verified_saved_analyses':checks,'mean_pair_error':sum(errs)/6,'mean_oracle':sum(oracles)/6,'relative_mean_improvement':1-sum(errs)/sum(oracles),'all_accuracy_fail':True,'no_model_derivative_or_decomposition_calls':True}))
