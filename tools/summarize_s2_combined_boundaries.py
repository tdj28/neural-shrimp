"""Fixed saved-only boundary overlay; no new model or derivative evaluations."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import hashlib,json,subprocess,time,resource,gzip
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/data2/neural-shrimp-data/explore/s2-period18-boundary-curves')
PINS={'attempt-003-target':'1a1b58e678f11db81ca7adb2821ca2ff84056768734073a11e70e004813a71f9','attempt-004-fold-plus':'231b1d448fd06b09a180861d4e67569d3ef0f42967557ba4e24dd7b4780284f6','attempt-005-ray51-flip':'b3ccbdafc68c8ce6981d286d7f7a4145624a9b8d6806e721471a06186b7135c9','attempt-006-ray32-fold':'5a407527b29bf0a94540e581b23967c5e956b175e2f6ecf1d7b51806e7688f7d','attempt-007-ray13-flip':'c889339446be01c3110303e311b06266e5a10b1bdacb1dd45f4ee06abd2e49a7','attempt-008-ray32-plus':'e1bd837122ebc25726f5a71a1928c220bb5f227a623c2f2825a0ff210c53e7ad','attempt-009-ray13-plus':'80906f161004f78f50c9f6496fec2e81c4dcd69af6be06debf370e459317eae5'}
PINS.update({'attempt-010-ray51-upper': 'dbfb2b3e07a2bde52653a44dd81b07672f9921a8ba790b93eb23ad62d4112b35', 'attempt-011-ray13-upper': '46d50714e954e8885d6efcf53a62ad93e1cb9e1eb029576e89cc362db6e2b640'})
MAP=Path('/data2/neural-shrimp-data/explore/s2-28011-w587-512/map.npz')
MAP_PIN='5d9fc07f06bbc36025d2ec0b315be1b4cd812141eb0a0a4f9d5464d6b71fa874'
def ent(p):b=p.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip()
 assert os.getpriority(os.PRIO_PROCESS,0)>=10
 resource.setrlimit(resource.RLIMIT_AS,(1024*1024**2,1024*1024**2));start=time.monotonic()
 out=BASE/'combined-summary005';out.mkdir(exist_ok=False);records=[];rows=[];native={};fold_details=[]
 for name,pin in PINS.items():
  base=BASE/name;assert ent(base/'manifest.json')['sha256']==pin
  inv=read(base/'manifest.json')['files'];assert set(inv)=={str(p.relative_to(base))for p in base.rglob('*')if p.is_file()}-{'manifest.json'}
  for f,e in inv.items():assert ent(base/f)==e and not (base/f).stat().st_mode&0o222
  launch=read(base/'launcher-result.json');assert launch['exit_code']==0 and not launch['owned_group_live_pids']
  for pid in (launch['launcher_pid'],launch['owned_group_pid']):assert not Path('/proc',str(pid)).exists()
  outcome=read(base/'run/outcome.json');assert outcome['status']=='numerical_boundary_panel_recorded'
  sigma=read(base/'run/inputs.json')['sigma'];nodes=[]
  for f in sorted((base/'run').glob('node-*.json')):
   a=read(f);q=a['qualification'];assert q['qualified'];point=np.array(a['point'],dtype=float)
   ev=np.array(q['multipliers_real'])+1j*np.array(q['multipliers_imag']);critical=q['critical_index'];other=np.delete(ev,critical)
   row={'run':name,'file':f.name,'sigma':sigma,'sign':a['sign'],'index':a['index'],'hidden':float(point[13]),'output':float(point[14]),'critical_gap':float(min(abs(other-sigma))),'nearest_other_real':float(other[np.argmin(abs(other-sigma))].real),'nearest_other_imag':float(other[np.argmin(abs(other-sigma))].imag),'complement_radius':q['complement_radius'],'closure':q['closure'],'proper_divisor_separations':q['proper_divisor_separations']}
   rows.append(row);nodes.append(row);native[(name,f.name)]=a
  assert len(nodes)==sum(d['accepted']for d in outcome['directions'])==64
  records.append({'root':str(base),'manifest_sha256':pin,'outcome':outcome,'launch':launch,'nodes':len(nodes),'bytes':sum(e['bytes']for e in inv.values())})
 # Reduce only saved derivative returns. No new map or derivative calls.
 for name in PINS:
  branch=[r for r in rows if r['run']==name]
  if branch[0]['sigma']!=1:continue
  lookup={tuple(native[(name,r['file'])]['point']):r for r in branch};found={}
  with gzip.open(BASE/name/'run/evaluations.jsonl.gz','rt') as f:
   for line in f:
    a=read_line=json.loads(line)
    if a['kind']=='evaluation_return' and tuple(a['z']) in lookup:found[tuple(a['z'])]=a
  assert len(found)==len(branch)
  for point,row in lookup.items():
   record=found[point];data=record['data'];z=np.array(point,dtype=float);q=z[15:];m=np.array(data['monodromy']);j=np.array(data['jacobian'])
   eigen,vec=np.linalg.eig(m.T);critical=int(np.argmin(abs(eigen-1)));left=vec[:,critical]
   assert abs(eigen[critical]-1)<1e-7 and max(abs(left.imag))<1e-8 and abs(left.real@q)>1e-8
   left=left.real/(left.real@q);others=np.delete(eigen,critical);near=others[np.argmin(abs(others-1))];minus=others[np.argmin(abs(others+1))]
   c=float(left@j[13:26,:13]@q);unfold=left@j[:13,13:15];t=np.array(native[(name,row['file'])]['tangent']);pt=t[13:15]*[10,.1]
   fold_details.append(dict(run=name,file=row['file'],sign=row['sign'],index=row['index'],c=c,quadratic_coefficient_c_over_2=c/2,critical_gap=float(min(abs(others-1))),nearest_other_plus_real=float(near.real),nearest_other_plus_imag=float(near.imag),nearest_other_minus_real=float(minus.real),nearest_other_minus_imag=float(minus.imag),left_norm=float(np.linalg.norm(left)),parameter_tangent_norm=float(np.linalg.norm(pt)),unfolding_tangent_dot=float(unfold@pt),left_residual=float(np.linalg.norm(m.T@left-left)/max(1,np.linalg.norm(m)*np.linalg.norm(left))),evaluation_id=record['id']))
 # Parameter proximity alone never establishes a common full-state cycle.
 pairs=[];flips=[r for r in rows if r['sigma']==-1]
 for fold in sorted(fold_details,key=lambda r:abs(complex(r['nearest_other_minus_real'],r['nearest_other_minus_imag'])+1))[:5]:
  frow=next(r for r in rows if (r['run'],r['file'])==(fold['run'],fold['file']))
  nearest=min(flips,key=lambda r:((r['hidden']-frow['hidden'])/10)**2+((r['output']-frow['output'])/.1)**2)
  fa=np.array(native[(frow['run'],frow['file'])]['orbit'])[:18];fb=np.array(native[(nearest['run'],nearest['file'])]['orbit'])[:18]
  scale=max(1,float(np.max(abs(fa))),float(np.max(abs(fb))))
  errors=[float(np.max(abs(fa-np.roll(fb,k,axis=0)))/scale) for k in range(18)]
  pairs.append({'fold':fold,'nearest_flip':nearest,'scaled_parameter_distance':float(np.hypot((nearest['hidden']-frow['hidden'])/10,(nearest['output']-frow['output'])/.1)),'best_cyclic_phase':int(np.argmin(errors)),'phasewise_state_distance':min(errors),'scope':'different parameter points; closeness is diagnostic only, not shared orbit or connected branch'})
 changes=[]
 for name in PINS:
  for sign in (-1,1):
   branch=sorted([r for r in fold_details if r['run']==name and r['sign']==sign],key=lambda r:r['index'])
   for a,b in zip(branch,branch[1:]):
    if a['c']*b['c']<0:changes.append({'from':a,'to':b,'cusp_established':False,'reason':'finite normalized coefficient sign change; must exclude critical-multiplier collision/pole and locate a nondegenerate zero'})
 # Saved polyline intersections are only proposals; neither common parameters
 # nor full-state identity are proved by interpolating samples.
 segments={51:[],13:[]}
 for family,names in [(51,['attempt-005-ray51-flip','attempt-010-ray51-upper']),
                      (13,['attempt-007-ray13-flip','attempt-009-ray13-plus','attempt-011-ray13-upper'])]:
  for name in names:
   for sign in (-1,1):
    part=sorted([r for r in rows if r['run']==name and r['sign']==sign],key=lambda r:r['index'])
    for a,b in zip(part,part[1:]):segments[family].append((a,b))
 intersections=[]
 for a,b in segments[51]:
  p=np.array([a['hidden']/10,a['output']/.1]);v=np.array([(b['hidden']-a['hidden'])/10,(b['output']-a['output'])/.1])
  for c,d in segments[13]:
   q=np.array([c['hidden']/10,c['output']/.1]);w=np.array([(d['hidden']-c['hidden'])/10,(d['output']-c['output'])/.1]);mat=np.column_stack([v,-w])
   if abs(np.linalg.det(mat))<1e-14:continue
   t,u=np.linalg.solve(mat,q-p)
   if not (0<=t<=1 and 0<=u<=1):continue
   params=(p+t*v)*[10,.1]
   aa=(1-t)*np.array(native[(a['run'],a['file'])]['orbit'])[:18]+t*np.array(native[(b['run'],b['file'])]['orbit'])[:18]
   cc=(1-u)*np.array(native[(c['run'],c['file'])]['orbit'])[:18]+u*np.array(native[(d['run'],d['file'])]['orbit'])[:18]
   scale=max(1,float(abs(aa).max()),float(abs(cc).max()));errors=[float(abs(aa-np.roll(cc,k,axis=0)).max()/scale)for k in range(18)]
   intersections.append({'segment51':[a,b],'segment13':[c,d],'fractions':[float(t),float(u)],'parameters':params.tolist(),'interpolated_best_phase':int(np.argmin(errors)),'interpolated_native_state_distance':min(errors),'scope':'floating polyline intersection proposal only; no evaluated common-parameter orbit or proof'})
 assert ent(MAP)['sha256']==MAP_PIN
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.colors import ListedColormap
 fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
 m=np.load(MAP);lh,lo=m['log10_hidden'],m['log10_output'];status,period=m['status'],m['period'];layer=np.zeros(status.shape);layer[(status==2)&(period!=18)]=1;layer[(status==2)&(period==36)]=2;layer[(status==2)&(period==18)]=3
 axs[0].imshow(layer,origin='lower',extent=[lo[0],lo[-1],lh[0],lh[-1]],aspect='auto',interpolation='nearest',cmap=ListedColormap(['white','#ddd','#aaa','#555']),vmin=-.5,vmax=3.5)
 for name in PINS:
  for sign in [-1,1]:
   part=sorted([r for r in rows if r['run']==name and r['sign']==sign],key=lambda r:r['index'])
   if not part:continue
   c={'attempt-003-target':'#2377bf','attempt-004-fold-plus':'#2377bf','attempt-005-ray51-flip':'#c8552e','attempt-006-ray32-fold':'#20a076','attempt-007-ray13-flip':'#9d37ad','attempt-008-ray32-plus':'#20a076','attempt-009-ray13-plus':'#9d37ad','attempt-010-ray51-upper':'#c8552e','attempt-011-ray13-upper':'#9d37ad'}[name];label=('fold' if part[0]['sigma']==1 else 'flip')+' '+name+' '+str(sign)
   x=np.log10([r['output']for r in part]);y=np.log10([r['hidden']for r in part]);axs[0].plot(x,y,'.-',color=c,lw=.8,markersize=2,label=label)
   unstable=np.array([r['complement_radius']>=1 for r in part]);axs[0].plot(x[unstable],y[unstable],'x',color='black',ms=4)
   axs[1].plot(range(len(part)),[r['complement_radius']for r in part],'.-',label=label,ms=2)
 axs[0].plot(np.log10(.8210596690584651),np.log10(71.337643833463),'r+',ms=8,label='certified interior center')
 axs[0].set(xlabel='log10 output learning rate',ylabel='log10 hidden learning rate',xlim=(-.105,-.07),ylim=(1.8,1.94),title='Carried numerical boundaries over historical period screen')
 axs[0].legend(fontsize=6);axs[1].axhline(1,color='black',lw=.7);axs[1].set(xlabel='Accepted node within segment',ylabel='Other-multiplier spectral radius',title='Crosses mark nonattracting complements on left');axs[1].legend(fontsize=6)
 fig.savefig(out/'combined-boundaries.png',dpi=160);fig.savefig(out/'combined-boundaries.pdf');axs[0].set_xlim(min(np.log10(r['output'])for r in rows)-.001,max(np.log10(r['output'])for r in rows)+.001);axs[0].set_ylim(min(np.log10(r['hidden'])for r in rows)-.005,max(np.log10(r['hidden'])for r in rows)+.005);fig.savefig(out/'full-boundaries.png',dpi=160);plt.close(fig)
 result={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source':ent(Path(__file__)),'runs':records,'map':{'path':str(MAP),**ent(MAP)},'rows':rows,'fold_coefficients':fold_details,'coefficient_sign_changes':changes,'nearest_fold_flip_pairs':pairs,'flip_polyline_intersections':intersections,'elapsed_seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'scope':'Numerical carried fold/flip nodes only; historical screen is context, no rigorous inter-node connection or shrimp proof.'}
 save(out/'resource-outcome.json',{'elapsed_seconds':result['elapsed_seconds'],'peak_rss_kib':result['peak_rss_kib'],'elapsed_gate':result['elapsed_seconds']<45,'rss_gate':result['peak_rss_kib']<512*1024})
 assert result['elapsed_seconds']<45 and result['peak_rss_kib']<512*1024
 save(out/'summary.json',result);(out/'source.py').write_bytes(Path(__file__).read_bytes());save(out/'manifest.json',{'files':{p.name:ent(p)for p in out.iterdir()}})
 for p in out.iterdir():p.chmod(0o444)
 out.chmod(0o555);print(json.dumps({'root':str(out),'manifest':ent(out/'manifest.json'),'rows':len(rows),'elapsed':result['elapsed_seconds']}))
if __name__=='__main__':main()
