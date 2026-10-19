from pathlib import Path
import hashlib,json,math,statistics,re,sys
D=Path('/data2/neural-shrimp-data/explore'); records={}
def read(p):
 p=Path(p); raw=p.read_bytes(); records[str(p)]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}; return json.loads(raw)
C=[]
for family,name,label in [('fm5c','a1-up-fold-minus-19','R upper'),('fm5c','a1-up-fold-minus-30','R lower'),('fm5d','a1-down-fold-minus-14','L lower'),('fm5d','a1-down-fold-minus-39','L upper'),('fm5c','a2-up-fold-plus-12','R upper duplicate'),('fm5c','a2-up-fold-plus-23','R lower duplicate')]:
 p=D/'llm-soft-loop/pod64'/family/f'locate-curve-{name}'/'locate.json'; d=read(p); f=d['final']; m=complex(*f['critical_multiplier'])
 C.append(dict(label=label,path=str(p),kind=d.get('kind'),located=d.get('located'),checks=d.get('checks'),c=f['c'],c_plateau=f['c_plateau_relative_spread'],detJ=f['det_J'],detJ_recomputed=f['J'][0][0]*f['J'][1][1]-f['J'][0][1]*f['J'][1][0],mu=f['critical_multiplier'],mu_modulus=abs(m),mu_argument=math.atan2(m.imag,m.real),mu_error=abs(m-1),next_modulus=f['second_modulus'],f=f['f'],b=f['b'],bracket_max=d['initial_f_max'],alpha=f['alpha'],alpha2=f['alpha2'],closure_residual=f.get('residual','not recorded')))
S=read(D/'latent-loop/huginn-fixed-spectrum-gsm-claire/fixed-spectrum.json'); eigen=[complex(*z) for v in S['spectra_of_F'].values() for z in v]; candidates=sorted([(2*(1-z.real)/abs(1-z)**2,z) for z in eigen if z.real<1],key=lambda x:x[0]); lc,mu=candidates[0]; critical=1-lc+lc*mu
N=read(D/'latent-loop/huginn-ns-gsm-claire/ns-check.json'); rows=N['rows']; above=[r for r in rows if r['lambda']>1.0778]; x=[r['lambda'] for r in above]; y=[r['late_distance_median']**2 for r in above]; xm=statistics.mean(x);ym=statistics.mean(y); slope=sum((a-xm)*(b-ym) for a,b in zip(x,y))/sum((a-xm)**2 for a in x); intercept=ym-slope*xm
NS=dict(fixed_residual=S['fixed_point_residual_F'],arnoldi_convergence=S.get('arnoldi_converged','not recorded'),counts={k:len(v) for k,v in S['spectra_of_F'].items()},mu=[mu.real,mu.imag],mu_modulus=abs(mu),lambda_formula=lc,critical=[critical.real,critical.imag],critical_modulus=abs(critical),turns=abs(math.atan2(critical.imag,critical.real))/(2*math.pi),next_crossing=candidates[2][0],amplitude_fit_lambda=-intercept/slope,amplitude_slope=slope,rows=[{k:r[k] for k in ['lambda','late_distance_median','initial_log_rate_per_step','dominant_turns_per_step']} for r in rows],amplitude_ratios=[math.sqrt(b/(a+intercept/slope)) for a,b in zip(x,y)])
rot=read(D/'latent-loop/huginn-rotation-a1/rotation.json'); NS['rotation_rows']=rot['rows']
for name in ('a1','a2','b1','b2'):
 d=read(D/f'latent-loop/huginn-rotation-{name}/ns-check.json');NS[f'rotation_{name}_checks']={'expected':d['args']['lambdas'],'recorded':[r['lambda'] for r in d['rows']],'fixed_residual':d['fixed_point_residual_F'],'steps':d['args']['steps']}
A=[]
for i in range(30):
 p=D/(f'huginn-relax/arc-dev-spectrum-{i}/fixed-spectrum.json' if i<4 else f'huginn-relax/arc-dev-spectra/q{i:04d}/fixed-spectrum.json');d=read(p); vals=[complex(*z) for v in d['spectra_of_F'].values() for z in v]; z=max(vals,key=abs);r=d['fixed_point_residual_F']
 A.append(dict(index=i,path=str(p),fixed_residual=r,fixed_tol=d['args'].get('fixed_tol','not recorded'),is_fixed_point=d.get('is_fixed_point','not recorded'),transient_lams=d.get('transient_lams',[d['args']['lam']]),spectral_radius=abs(z),leading=[z.real,z.imag],unstable=abs(z)>1,arnoldi_converged=d.get('arnoldi_converged','not recorded'),spectrum_counts={k:len(v) for k,v in d['spectra_of_F'].items()},predicted_intervals=d.get('predicted_stable_lambda_intervals')))
old=read(D/'huginn-relax/arc-dev-spectrum-4/fixed-spectrum.json');old={'fixed_residual':old['fixed_point_residual_F'],'is_fixed_point':old.get('is_fixed_point','not recorded'),'arnoldi_converged':old.get('arnoldi_converged','not recorded')}
conv=[]
for i in range(5):
 d=read(D/f'huginn-relax/arc-dev-conv-{i}/convergence.json'); conv.append(dict(index=i,expected_lams=d['args']['lams'],traces={a:{'n':len(v['native_residual']),'status':v['status'],'rate_per_step':v['rate_per_step'],'late_residual_min':min(v['native_residual'][-10:]),'late_residual_max':max(v['native_residual'][-10:])} for a,v in d['traces'].items()}))
G={}; violations=[]; counts={}
for folder in ('partB-0','partB-96','partB-100'):
 p=D/'llm-soft-loop/pod64/huginn'/folder/'rows.jsonl';raw=p.read_bytes();records[str(p)]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()};rrows=[json.loads(line) for line in raw.splitlines()];counts[folder]=len(rrows)
 for row in rrows:
  key=(row['arm'],row['index']); bad=[]
  if key in G:bad.append('duplicate')
  if not(row['split']=='test' and row['steps']==32 and row['prompt_style']=='cot8' and row['seed']==row['index'] and float(row['arm'])==row['lam']):bad.append('settings')
  found=re.findall(r'(-?[$0-9.,]{2,})|(-?[0-9]+)',row['text']); pred=((found[-1][0] or found[-1][1]).replace(',','').replace('$','').rstrip('.')) if found else None
  if pred!=row['pred']:bad.append('prediction extraction')
  
  try:correct=pred is not None and abs(float(pred)-float(row['gold']))<1e-6
  except ValueError:correct=False
  if correct!=row['correct']:bad.append('scoring')
  if row['core_calls']!=32*row['new_tokens']:bad.append('core calls')
  if bad:violations.append({'key':key,'gates':bad})
  G[key]=row
GS={'n_rows':len(G),'by_file':counts,'violations':violations,'arms':{}}
for arm in ('1.0','1.05','1.15','1.20'):
 ids={i for a,i in G if a==arm}; m={'coverage':len(ids),'coverage_exact':ids==set(range(200))};base=[G[('1.0',i)]['correct'] for i in range(200)];other=[G[(arm,i)]['correct'] for i in range(200)];lost=sum(a and not b for a,b in zip(base,other));gained=sum(not a and b for a,b in zip(base,other));n=lost+gained;delta=(gained-lost)/200
 m.update(correct=sum(other),accuracy=sum(other)/200,lost=lost,gained=gained,difference=delta,mcnemar_less=sum(math.comb(n,k) for k in range(gained+1))/2**n if n else 1,capped=sum(G[(arm,i)]['new_tokens']>=256 for i in range(200)))
 # Exact empirical paired-bootstrap law, not producer's Monte Carlo approximation.
 probs={0:1.};pminus=lost/200;pplus=gained/200;pzero=1-pminus-pplus
 for _ in range(200):
  nxt={}
  for val,prob in probs.items():
   for delta0,p0 in [(-1,pminus),(0,pzero),(1,pplus)]:nxt[val+delta0]=nxt.get(val+delta0,0)+prob*p0
  probs=nxt
 mass=0
 for val,prob in sorted(probs.items()):
  mass+=prob
  if mass>=.05:m['exact_empirical_bootstrap_q05']=val/200;break
 GS['arms'][arm]=m
rest=[z for z in eigen if abs(z-mu)>1e-4 and abs(z-mu.conjugate())>1e-4]
NS['next_modulus_at_true_crossing']=max(abs(1-lc+lc*z) for z in rest)
NS['modulus_at_printed_1p0777']=abs(1-1.0777+1.0777*mu)
NS['modulus_crossing_derivative']=1-mu.real
NS['reported_interval_upper']=S['predicted_stable_lambda_intervals'][0][1]
producer_generation=read(D/'huginn-relax/test-partB/generation.json')
GS['producer_seeded_monte_carlo_lower95']=producer_generation['B3_noninferior']['lower_one_sided_95']
out={'python':sys.executable,'scope':'small original JSON/JSONL and scalar checks only; no model calls; uv snap-confine refused, direct Python used by root-approved fallback','cusps':C,'ns':NS,'arc_rows':A,'arc_old_q4_failed_attempt':old,'arc_convergence_rows':conv,'generation':GS,'sources':records}
Path('/tmp/task37-raw-check.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'arc_unstable_indices':[r['index'] for r in A if r['unstable']], 'arc_partial_indices':[r['index'] for r in A if not all(r['arnoldi_converged'].values())], 'generation':GS},indent=1))
