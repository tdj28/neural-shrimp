"""Independent, bounded reductions of named saved geometry records; no model imports."""
import hashlib,json,math,pathlib,resource,signal,sys,time
resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2)
resource.setrlimit(resource.RLIMIT_CPU,(100,)*2);signal.alarm(120)
import numpy as np
D=pathlib.Path('/data2/neural-shrimp-data/explore/llm-soft-loop');P=D/'pod64'
OUT={};PINS={}
def pin(p):
 p=pathlib.Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 PINS[str(p)]={'bytes':p.stat().st_size,'sha256':h.hexdigest()};return p
def js(p):return json.loads(pin(p).read_text())
def nz(p):return np.load(pin(p),allow_pickle=False)
def counts(a):return {str(k):int(v) for k,v in zip(*np.unique(a,return_counts=True))}
def basic(z):
 per=z['period'];ly=z['lyapunov'];chaos=(per<=0)&(ly>1e-3)
 axes=['alpha1','alpha2'] if 'alpha1' in z else ['alpha','log10_temperature']
 return {'shape':per.shape,'finite_exponents':int(np.isfinite(ly).sum()),'cells':per.size,'chaos':int(chaos.sum()),'chaos_share':float(chaos.mean()),'neither':int(((per<=0)&~chaos).sum()),'periods':counts(per),'axes':{k:{'first':float(z[k][0]),'last':float(z[k][-1]),'step_min':float(np.diff(z[k]).min()),'step_max':float(np.diff(z[k]).max())} for k in axes}}
for name in ['hr10','hr11','hr12','hp5']:
 base='map' if name=='hp5' else 'scan';z=nz(P/name/(base+'.npz'));meta=js(P/name/(base+'.json'));r=basic(z);r['bands']=[]
 for i in range(16):
  if name=='hp5':bp=P/'tb'/f'hp5-c{i:02d}'/'tb-t-0.100.npz';fields=['alpha1','alpha2','lyapunov','period','chaotic'];axis='alpha2'
  else:
   bn=meta['bands'][i] if name=='hr12' else f'{name}-c{i:02d}';bp=P/'bands'/bn/'scan.npz';fields=['alpha','log10_temperature','lyapunov','period'];axis='log10_temperature'
  b=nz(bp);rows=b[axis].size;lo=i*rows;checks={}
  for key in fields:
   wanted=z[key][lo:lo+rows] if z[key].ndim==2 or key==axis else z[key]
   checks[key]=bool(np.array_equal(wanted,b[key],equal_nan=True))
  r['bands'].append({'name':str(bp),'rows':rows,'equals_stitched':checks,'finite':int(np.isfinite(b['lyapunov']).sum())})
  if name=='hr12' and bn.endswith('m'):
   orig=nz(P/'bands'/bn[:-1]/'scan-partial.npz');redo=nz(P/'bands'/(bn[:-1]+'b')/'scan.npz')
   r['bands'][-1]['merge_inputs']={k:{'prefix_38_rows':bool(np.array_equal(b[k][:38],orig[k][:38],equal_nan=True)),'rerun_tail':bool(np.array_equal(b[k][38:],redo[k],equal_nan=True))} for k in ['lyapunov','period']}
  b.close()
 r['chaotic_flag_matches_rule']=bool(np.array_equal(z['chaotic'],(z['period']<=0)&(z['lyapunov']>1e-3))) if 'chaotic' in z else None
 OUT[name]=r;z.close()
cusps=[]
for name in ['p6s/locate-tu-minus-34','p6s/locate-wide2-a-64','p6s/locate-wide2-b-82','kite/locate-la-plus-14','kite/locate-la-plus-94','kite/locate-la-plus-112']:
 q=js(P/name/'locate.json');f=q['final'];J=f['J'];det=J[0][0]*J[1][1]-J[0][1]*J[1][0]
 cusps.append({'name':name,'located':q['located'],'checks':q['checks'],'alpha':f['alpha'],'log10_temperature':f['log10_temperature'],'c':f['c'],'det_J':f['det_J'],'det_recomputed':det,'f':f['f'],'unfolding_b_separate_estimator':f['b'],'c_plateau_relative_spread':f['c_plateau_relative_spread'],'critical':f['critical_multiplier'],'critical_gap':abs(complex(*f['critical_multiplier'])-1),'next':f['second_modulus'],'initial_f_max':q['initial_f_max'],'initial_bracket':q['brackets'][0]['f'],'all_brackets_sign_change':all(np.prod(v['f'])<0 for v in q['brackets']),'record_has_final_residual':'residual' in f})
OUT['cusps']=cusps
z=nz(P/'dss/p6-body/grid.npz');ok=z['ok'];m=z['multipliers'];v=m[...,0]+1j*m[...,1];lead=np.abs(v[...,0]);ij=np.unravel_index(np.nanargmin(np.where(ok,lead,np.nan)),ok.shape);complexmask=ok&(abs(v[...,0].imag)>1e-8);delta=v[...,0]*v[...,1]
changes=[]
for i,j in np.ndindex(ok.shape):
 for di,dj in [(0,1),(1,0)]:
  a,b=i+di,j+dj
  if a>=ok.shape[0] or b>=ok.shape[1] or not(ok[i,j] and ok[a,b]):continue
  if abs(v[i,j,0].imag)>1e-8 or abs(v[a,b,0].imag)>1e-8 or v[i,j,0].real*v[a,b,0].real>=0:continue
  ratio=[abs(v[t][0])/abs(v[t][1]) for t in [(i,j),(a,b)]]
  changes.append({'cells':[[i,j],[a,b]],'ratios':ratio,'within_factor3_both':max(ratio)<=3})
OUT['dss']={'shape':ok.shape,'ok':int(ok.sum()),'failed':int((~ok).sum()),'clean':int(z['clean'].sum()),'same_tokens_ok':int((z['same_tokens']&ok).sum()),'max_newton_ok':float(z['newton'][ok].max()),'minimum':{'ij':ij,'alpha':float(z['alpha'][ij[1]]),'log10_temperature':float(z['log10_temperature'][ij[0]]),'modulus':float(lead[ij]),'multipliers':m[ij].tolist()},'complex_cells':int(complexmask.sum()),'complex_modulus_range':[float(lead[complexmask].min()),float(lead[complexmask].max())],'complex_angle_degrees_range':[float(np.degrees(np.angle(v[...,0][complexmask])).min()),float(np.degrees(np.angle(v[...,0][complexmask])).max())],'delta_negative':int((ok&(delta.real<0)).sum()),'delta_max_abs_imag':float(abs(delta[ok].imag).max()),'leading_sign_changes':len(changes),'endpoint_factor3_both':sum(x['within_factor3_both'] for x in changes),'endpoint_other':sum(not x['within_factor3_both'] for x in changes),'endpoint_rows':changes}
z.close()
rawR=js(D/'kite/newton-at-R/newton_at_R.json');OUT['R_rows']=rawR
for al,lt in [(0.072912,-0.154104),(0.0730,-0.1539),(0.0731,-0.1537)]:
 orbs=[]
 for name in ['p6s-ad-curve-plus','p6s-td-curve-minus']:
  p=pin(D/'kite/newton-at-R'/f'orbit-{al:.6f}-{lt:.6f}-{name}.npy');orbs.append(np.load(p,allow_pickle=False).reshape(6,-1))
 A,B=orbs;norm=np.mean(np.linalg.norm(A,axis=1));gaps=[float(np.max(np.linalg.norm(A-np.roll(B,k,axis=0),axis=1))) for k in range(6)]
 OUT.setdefault('R_recomputed_gaps',[]).append({'alpha':al,'log10_temperature':lt,'shift':int(np.argmin(gaps)),'max_gap':min(gaps),'mean_A_norm':norm,'relative_max_gap':min(gaps)/norm,'all_shift_gaps':gaps,'A_divisors':{str(q):float(np.max(np.linalg.norm(A-np.roll(A,q,axis=0),axis=1))/norm) for q in [1,2,3]},'B_divisors':{str(q):float(np.max(np.linalg.norm(B-np.roll(B,q,axis=0),axis=1))/np.mean(np.linalg.norm(B,axis=1))) for q in [1,2,3]}})
OUT['branch_gates']=[]
for name in ['kite-la-branch','kite-ut-branch']:
 q=js(P/'kite'/name/'branch.json');pts=q['points'];turns=[i for i,x in enumerate(pts) if x.get('turned_since_previous')];OUT['branch_gates'].append({'name':name,'period':q['period'],'args':q['args'],'count':len(pts),'max_residual':max(x['residual'] for x in pts),'unconverged_multipliers':[i for i,x in enumerate(pts) if not x.get('multipliers_converged')],'turns':[{k:v for k,v in pts[i].items() if k in ['index','lambda','turn_lambda_estimate','spectral_radius','spectral_radius_refined','tangent_lambda']} for i in turns],'crossings':q['crossings'],'stable_runs':[[i,x['lambda'],x.get('spectral_radius_refined',x['spectral_radius'])] for i,x in enumerate(pts) if x.get('spectral_radius_refined',x['spectral_radius'])<1]})
bas=js(D/'kite/basin-check/basin-check.json');hr=nz(P/'hr11/scan.npz');offsets=[]
for key,row in bas['fixed_start'].items():
 i=int(np.argmin(abs(hr['log10_temperature']-float(key))));j=int(np.argmin(abs(hr['alpha']-bas['args']['alpha'])));mu=complex(*row['mu'][0][:2]);ly=float(hr['lyapunov'][i,j]);asym=math.log(abs(mu))/6
 offsets.append({'cell':[i,j],'alpha':float(hr['alpha'][j]),'log10_temperature':float(hr['log10_temperature'][i]),'period':int(hr['period'][i,j]),'map_lyapunov':ly,'mu':row['mu'][0],'log_mu_over6':asym,'offset':ly-asym})
OUT['basin']={'raw':bas,'offsets':offsets,'stipple_path_exists':(D/'kite/basin-check-stipple/basin-check.json').exists()}
OUT['pins']=PINS
def default(x):
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,np.ndarray):return x.tolist()
 raise TypeError(type(x).__name__)
print(json.dumps(OUT,indent=2,allow_nan=False,default=default))
