"""Independent saved sheet-map arithmetic. No producer imports or model calls."""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np

def group(fam,lya,tokens):
 out=np.zeros(fam.shape,dtype=int)
 for f in np.unique(fam):
  if f>=0:
   assert int(f) in tokens,'missing family metadata'
   out[fam==f]=1 if '\n\n' in ''.join(tokens[int(f)]) else -1
 out[(fam<0)&(lya>1e-3)]=2
 return out

def edges(default,foot3,foot2):
 ans=np.zeros(default.shape+(2,),dtype=int);foot={1:foot3,-1:foot2}
 for axis,k in [(1,0),(0,1)]:
  for i in range(default.shape[0]):
   for j in range(default.shape[1]):
    ni,nj=i+(axis==0),j+(axis==1)
    if ni>=default.shape[0] or nj>=default.shape[1]:continue
    a,b=int(default[i,j]),int(default[ni,nj])
    if a not in (-1,1) or b not in (-1,1) or a==b:continue
    a_there=bool(foot[a][ni,nj]);b_here=bool(foot[b][i,j]);ans[i,j,k]=1 if a_there and b_here else (2 if not a_there and not b_here else 3)
 return ans

def controls():
 d=np.array([[1,-1,2],[-1,1,0]]);both=np.ones((2,3),bool);e=edges(d,both,both);assert int((e==1).sum())==4 and not e[:,-1,0].any() and not e[-1,:,1].any()
 e=edges(d,d==1,d==-1);assert int((e==2).sum())==4
 assert group(np.array([[0,1,-1,-1]]),np.array([[0.,0.,.002,-.1]]),{0:['\n\n'],1:[' is']}).tolist()==[[1,-1,2,0]]
 try:group(np.array([[2]]),np.array([[0.]]),{})
 except AssertionError:pass
 else:raise AssertionError('missing metadata silently classified')
 return {'passed':True,'groups':4}

def canonical(tokens):
 t=tuple(tokens);return min(t[i:]+t[:i] for i in range(len(t))) if t else ()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--card');ap.add_argument('--out');ap.add_argument('--controls',action='store_true');a=ap.parse_args()
 if a.controls:print(json.dumps(controls()));return
 c=json.loads(Path(a.card).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=False);tic=time.monotonic()
 for pin in c['inputs']:
  p=Path(pin['path']);assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['sha256']
 maps={};grid=None;grouping=[];first_keys=None
 for path in c['map_files']:
  path=Path(path);z=np.load(path,allow_pickle=False);meta=json.loads(path.with_name('sheets.json').read_text());tokens={int(x['family']):x['tokens'] for x in meta['families']};axes=(z['log10_temperature'],z['alpha']);keys=z['key_tokens'].astype(str).tolist();bi=keys.index('\n\n')
  if first_keys is None:first_keys=keys
  else:assert keys==first_keys,'key-token order differs across maps'
  assert all(np.isfinite(x).all() and np.all(np.diff(x)>0) for x in axes)
  if grid is None:grid=axes
  else:assert all(np.array_equal(np.round(x,12),np.round(y,12)) for x,y in zip(grid,axes))
  for name in sorted({k.split('__')[0] for k in z.files if '__' in k}):
   assert name not in maps,'duplicate map would overwrite evidence'
   v={key:z[name+'__'+key] for key in ['period','family','lyapunov','settle','keyprob','entropy']};assert v['period'].shape==(len(axes[0]),len(axes[1]));v.update(tokens=tokens,source=str(path),group=group(v['family'],v['lyapunov'],tokens));pb=v['keyprob'][...,bi];v['legacy']=np.where(np.isnan(pb),0,np.where(pb>=.05,1,-1))
   exact=np.zeros(v['family'].shape,int)
   for f in np.unique(v['family']):
    if f>=0:exact[v['family']==f]=1 if '\n\n' in tokens[int(f)] else -1
   grouping.append(dict(name=name,period_positive_family_negative=int(((v['period']>0)&(v['family']<0)).sum()),period_zero_family_positive=int(((v['period']==0)&(v['family']>=0)).sum()),legacy_vs_new_different=int((v['legacy']!=v['group']).sum()),periodic_legacy_vs_new_different=int(((v['family']>=0)&(v['legacy']!=v['group'])).sum()),joined_text_vs_exact_token_different=int(((v['family']>=0)&(exact!=v['group'])).sum())))
   maps[name]=v
 names=list(maps);G=np.stack([maps[n]['group'] for n in names]);f3=np.any(G==1,axis=0);f2=np.any(G==-1,axis=0);over=f3&f2;d=maps['default']['group'];fixed=d!=0;e=edges(d,f3,f2);stl=maps['default']['settle'].astype(float);on=np.any(e>0,axis=-1)
 # Reproduce the published roll convention, then retain a boundary-corrected sensitivity.
 near=on|np.roll(on,1,axis=0)|np.roll(on,1,axis=1);correct=on.copy();correct[1:]|=on[:-1];correct[:,1:]|=on[:,:-1]
 pb=maps['default']['keyprob'][...,bi];chaos=d==2;bins=[0,.01,.03,.05,.08,.12,.2,.3,1.]
 summary=dict(maps={n:v['source'] for n,v in maps.items()},grid={'log10_temperature':[float(round(grid[0][0],12)),float(round(grid[0][-1],12)),len(grid[0])],'alpha':[float(round(grid[1][0],12)),float(round(grid[1][-1],12)),len(grid[1])]},threshold_p_blank=.05,cells=int(fixed.sum()),footprint_3group=int((f3&fixed).sum()),footprint_2group=int((f2&fixed).sum()),overlap_cells=int((over&fixed).sum()),fixed_start_3group_cells=int((d==1).sum()),fixed_start_2group_cells=int((d==-1).sum()),fixed_start_3group_inside_overlap=int(((d==1)&over).sum()),fixed_start_2group_inside_overlap=int(((d==-1)&over).sum()),edges_of_fixed_start_map={'switch':int((e==1).sum()),'one_family_ends':int((e==3).sum()),'both_families_end':int((e==2).sum())},median_settle_near_edges=float(np.median(stl[near&fixed])) if (near&fixed).any() else None,median_settle_elsewhere=float(np.median(stl[~near&fixed])) if (~near&fixed).any() else None,per_map_group_cells={n:{'3group':int((v['group']==1).sum()),'2group':int((v['group']==-1).sum()),'chaos':int((v['group']==2).sum()),'no_period':int((v['group']==0).sum())} for n,v in maps.items()},chaos_in_fixed_start_map={'cells':int(chaos.sum()),'p_blank_histogram':{'bins':bins,'counts':np.histogram(pb[chaos],bins)[0].tolist()},'chaos_cells_inside_overlap':int((chaos&over).sum())})
 published=json.loads(Path(c['producer_summary']).read_text());comparisons=[]
 def compare(x,y,path=''):
  if isinstance(x,dict):
   assert isinstance(y,dict) and set(x)==set(y),path
   for k in x:compare(x[k],y[k],path+'/'+k)
  elif isinstance(x,list):
   assert len(x)==len(y),path
   for i,(xx,yy) in enumerate(zip(x,y)):compare(xx,yy,path+'/'+str(i))
  else:comparisons.append({'path':path,'own':x,'producer':y,'equal':x==y,'delta':x-y if isinstance(x,(int,float)) and isinstance(y,(int,float)) else None})
 compare(summary,published)
 fields=np.load(c['producer_fields'],allow_pickle=False);order=[names.index(str(n)) for n in fields['names']];fieldchecks={};fielddiffs={}
 for key,own in [('groups',G[order]),('foot3',f3),('foot2',f2),('overlap',over),('edge_class',e),('settle_default',stl),('log10_temperature',np.round(grid[0],12)),('alpha',np.round(grid[1],12))]:
  target=fields[key];assert own.shape==target.shape,key
  delta=own.astype(float)-target.astype(float);equal=(own==target)|(np.isnan(own)&np.isnan(target));fielddiffs[key]=delta
  fieldchecks[key]={'equal':bool(equal.all()),'unequal_count':int((~equal).sum()),'max_abs_finite_delta':float(np.max(abs(delta[np.isfinite(delta)]))) if np.isfinite(delta).any() else None}
 np.savez_compressed(out/'field-differences.npz',**fielddiffs)
 cross=np.load(c['cross_npz'],allow_pickle=False);cm=json.loads(Path(c['cross_json']).read_text());ct={int(v['family']):canonical(v['tokens']) for v in cm['families']};nt={f:canonical(t) for f,t in maps['default']['tokens'].items()};ai=np.argmin(abs(grid[1][:,None]-cross['alpha'][None,:]),axis=1);ti=np.argmin(abs(grid[0][:,None]-cross['log10_temperature'][None,:]),axis=1);ad=grid[1]-cross['alpha'][ai];td=grid[0]-cross['log10_temperature'][ti];assert np.max(abs(ad))<1e-10
 oldp=cross['period'][np.ix_(ti,ai)];oldf=cross['family'][np.ix_(ti,ai)];familyequal=np.zeros(d.shape,bool)
 for i in range(d.shape[0]):
  for j in range(d.shape[1]):
   f,g=int(oldf[i,j]),int(maps['default']['family'][i,j]);familyequal[i,j]=(f<0 and g<0) or (f>=0 and g>=0 and ct[f]==nt[g])
 period_equal=oldp==maps['default']['period'];np.savez_compressed(out/'fields.npz',groups=G,names=np.array(names),foot3=f3,foot2=f2,overlap=over,edge_class=e,legacy_groups=np.stack([maps[n]['legacy'] for n in names]),cross_period_equal=period_equal,cross_family_equal=familyequal,cross_temperature_indices=ti,cross_alpha_indices=ai,cross_temperature_offsets=td,cross_alpha_offsets=ad)
 hyst={}
 for axis in ['alpha','temp']:
  if axis+'+' in maps and axis+'-' in maps:
   u,v=maps[axis+'+'],maps[axis+'-'];hyst[axis]={'group_disagreement':int((u['group']!=v['group']).sum()),'period_disagreement':int((u['period']!=v['period']).sum()),'both_periodic_group_disagreement':int(((abs(u['group'])==1)&(abs(v['group'])==1)&(u['group']!=v['group'])).sum())}
 result={'completed':True,'summary':summary,'comparisons':comparisons,'field_checks':fieldchecks,'grouping_diagnostics':grouping,'unmasked_footprints':{'3group':int(f3.sum()),'2group':int(f2.sum()),'overlap':int(over.sum()),'grid_cells':d.size},'near_edge_wrap_sensitivity':{'changed_mask_cells':int((near!=correct).sum()),'median_near':float(np.median(stl[correct&fixed])) if (correct&fixed).any() else None,'median_elsewhere':float(np.median(stl[~correct&fixed])) if (~correct&fixed).any() else None},'cross512_nearest_row':{'cells':d.size,'period_equal':int(period_equal.sum()),'canonical_token_family_equal':int(familyequal.sum()),'both_equal':int((period_equal&familyequal).sum()),'max_abs_T_offset':float(np.max(abs(td))),'max_abs_alpha_offset':float(np.max(abs(ad))),'alpha_index_stride':np.unique(np.diff(ai)).tolist(),'exact_T_rows':int((td==0).sum())},'hysteresis':hyst,'elapsed':time.monotonic()-tic,'limitations':['Absent from finite seed/sweep outputs is not proof of branch disappearance or bifurcation.','Token groups do not independently prove orbit-family continuation or stability.','Positive finite exponents retain producer threshold labels, not independent asymptotic chaos verification.','Regrouping was after seeded outcomes but reported before sweep inspection; both original threshold and amended groups retained.','Nearest-temperature comparison is not an exact-cell reproducibility test.']}
 for pin in c['inputs']:
  p=Path(pin['path']);assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['sha256'],'input changed during audit'
 result['inputs_reverified']=len(c['inputs'])
 (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
