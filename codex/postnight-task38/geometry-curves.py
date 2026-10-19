import json,hashlib,pathlib,resource,signal
resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2);resource.setrlimit(resource.RLIMIT_CPU,(100,)*2);signal.alarm(120)
mem={k:int(v.split()[0])*1024 for k,v in (x.split(':',1) for x in pathlib.Path('/proc/meminfo').read_text().splitlines())}
if mem['MemAvailable']<16*1024**3:raise SystemExit('Below shared runtime memory floor')
import numpy as np
D=pathlib.Path('/data2/neural-shrimp-data/explore/llm-soft-loop');P=D/'pod64';pins={};out={}
def pin(p):
 p=pathlib.Path(p);pins[str(p)]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()};return p
def js(p):return json.loads(pin(p).read_text())
def nz(p):return np.load(pin(p),allow_pickle=False)
def dist(pts,curves,scale):
 seg=[(x[:-1]/scale,x[1:]/scale) for x in curves if len(x)>1];A=np.concatenate([x[0] for x in seg]);d=np.concatenate([x[1]-x[0] for x in seg]);d2=(d*d).sum(1);res=[]
 for p in np.asarray(pts):
  t=np.clip(((p/scale-A)*d).sum(1)/np.maximum(d2,1e-30),0,1);res.append(float(np.sqrt(((A+t[:,None]*d-p/scale)**2).sum(1)).min()))
 return np.asarray(res)
def gate_summary(d):return {'pairs':len(d),'within1':int((d<=1).sum()),'within2':int((d<=2).sum()),'median':float(np.median(d)),'max':float(d.max())}
def edges(z,coords,period,diags):
 x,y=z[coords[0]],z[coords[1]];lab=z['label'];per=z['period'];r={'flip':[],'fold':[]}
 offsets=[(-1,0),(1,0),(0,-1),(0,1)]+([(-1,-1),(-1,1),(1,-1),(1,1)] if diags else [])
 for i,j in zip(*np.where((lab==1)&(per==period))):
  for di,dj in offsets:
   ii,jj=i+di,j+dj
   if ii<0 or jj<0 or ii>=len(y) or jj>=len(x):continue
   kind='flip' if lab[ii,jj]==1 and per[ii,jj]==2*period else 'fold' if lab[ii,jj]==0 else None
   if kind:r[kind].append([(x[j]+x[jj])/2,(y[i]+y[ii])/2])
 return r
allcurves={};gates=[]
for run in ['p6s','kite','fm5','fm5c','fm5d','fm5x','fm5l-a']:
 globbed=set((P/run).glob('curve-*/curve.json')) if run.startswith('fm') else set((P/run).glob('*curve-*/curve.json'))|set((P/run).glob('*-fold-*/curve.json'))
 for p in sorted(globbed):
  q=js(p);key=run+'/'+p.parent.name;pts=q['points'];target=1 if q['kind']=='fold' else -1;bad=[];good=[]
  for j,x in enumerate(pts):
   mu=x.get('critical_multiplier_refined',x.get('critical_multiplier'));gap=abs(complex(*mu[:2])-target) if isinstance(mu,list) else abs(mu-target)
   if x.get('rejected') or x.get('residual',1)>1e-10 or gap>1e-7:bad.append({'index':j,'rejected':x.get('rejected',False),'residual':x.get('residual'),'critical_gap':gap})
   else:good.append(x)
  gates.append({'name':key,'kind':q['kind'],'points':len(pts),'passed_strict':len(good),'failed_or_rejected':bad,'max_all_residual':max(x.get('residual',1) for x in pts)})
  allcurves[key]=q
out['curve_gates']=gates
fa=nz(P/'fa/t-0.100/follow.npz');cur5={k:[np.array([[x['alpha'],x['alpha2']] for x in q['points']]) for n,q in allcurves.items() if n.startswith('fm') and q['kind']==k] for k in ['fold','flip']};pairs=edges(fa,['alpha1','alpha2'],5,True);scale=float(fa['alpha1'][1]-fa['alpha1'][0]);out['FA_boundary']={k:gate_summary(dist(p,cur5[k],scale)) for k,p in pairs.items()};out['FA_boundary']['fold_or_flip']=gate_summary(dist(pairs['fold'],cur5['fold']+cur5['flip'],scale))
hp=nz(P/'hp5/map.npz');ii=np.argmin(abs(hp['alpha2'][:,None]-fa['alpha2'][None,:]),axis=1);jj=np.argmin(abs(hp['alpha1'][:,None]-fa['alpha1'][None,:]),axis=1);inside=fa['label'][ii[:,None],jj[None,:]]>=1;per=hp['period'];ly=hp['lyapunov'];masks={'family':(per>0)&(per%5==0),'other_periodic':(per>0)&(per%5!=0),'chaos':(per<=0)&(ly>1e-3),'neither':(per<=0)&(ly<=1e-3)}
out['HP5_in_FA_nearest_cell']={'cells':int(inside.sum()),'counts':{k:int((inside&v).sum()) for k,v in masks.items()},'shares':{k:float((inside&v).sum()/inside.sum()) for k,v in masks.items()},'period4':int((inside&(per==4)).sum())}
fa6=nz(P/'p6s/follow/follow.npz');c6={k:[] for k in ['fold','flip']}
for n,q in allcurves.items():
 if not n.startswith(('p6s/','kite/')):continue
 pts=[]
 for x in q['points']:
  mu=x.get('critical_multiplier_refined',x.get('critical_multiplier'));target=1 if q['kind']=='fold' else -1
  if not x.get('rejected') and x.get('residual',1)<=1e-8 and abs(complex(*mu[:2])-target)<=1e-3:pts.append(x)
 c6[q['kind']].append(np.array([[x['alpha'],x['log10_temperature']] for x in pts]))
sc=np.array([fa6['alpha1'][1]-fa6['alpha1'][0],fa6['log10_temperature'][1]-fa6['log10_temperature'][0]])
out['P6_boundary']={}
for diag in [False,True]:
 pair=edges(fa6,['alpha1','log10_temperature'],6,diag);out['P6_boundary'][str(diag)]={k:gate_summary(dist(v,c6[k],sc)) for k,v in pair.items()}
def xy(q):return np.array([[x['alpha'],x['log10_temperature']] for x in q['points'] if not x.get('rejected')])
ad=np.vstack([xy(allcurves['p6s/p6s-ad-curve-plus'])[::-1],xy(allcurves['p6s/p6s-ad-curve-minus'])]);td=np.vstack([xy(allcurves['p6s/p6s-td-curve-plus'])[::-1],xy(allcurves['p6s/p6s-td-curve-minus'])]);kite=np.vstack([xy(allcurves['kite/kite-la-curve-plus'])[::-1],xy(allcurves['kite/kite-la-curve-minus'])])
def cross(A,B):
 rows=[]
 for a,b in zip(A[:-1],A[1:]):
  for c,d in zip(B[:-1],B[1:]):
   u,v=b-a,d-c;den=u[0]*v[1]-u[1]*v[0]
   if abs(den)<1e-30:continue
   t=((c-a)[0]*v[1]-(c-a)[1]*v[0])/den;s=((c-a)[0]*u[1]-(c-a)[1]*u[0])/den
   if 0<=t<=1 and 0<=s<=1:
    p=a+t*u
    if .0728<=p[0]<=.0731 and -.1544<=p[1]<=-.153:rows.append(p.tolist())
 return rows
R=cross(ad,td);out['corners']={'R':R,'A':cross(kite,td),'B':cross(kite,ad)}
step=np.array([(.0775-.0625)/511,(-.125+.17)/511]);out['R_nearest_saved_points_Fig1_steps']={n:float(np.min(np.linalg.norm((x-np.array([.072912,-.154104]))/step,axis=1))) for n,x in [('ad',ad),('td',td)]}
out['pins']=pins
print(json.dumps(out,indent=2,allow_nan=False))
