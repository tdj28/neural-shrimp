"""Fixed independent saved replay of two Clark51 tubes and selected generic fold."""
import gzip,hashlib,json,os,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from collections import Counter
from fractions import Fraction as F
import flint
import numpy as np
from flint import arb,arb_mat,ctx
import review_clark_correlated as D
import review_clark_extension_math as R
A,H,M,need=D.A,D.H,D.M,D.need
ROOT=D.ROOT
PARENT=Path('/data2/neural-shrimp-data/explore/clark-rnn/correlated-fold-extension/attempt001')
SOURCE='3616fca0b27528670e6211a74f5c3011a35b2608'
PINS=['49cb143ed9328833f6910d2122479519428d7fe2d95c2304b035b1b69ac5767e','08e483a208bc4685f6744cfbd7b35f164dbe6fa6e6e3e0c4dce8325fce32b544']
OUT=Path('/data2/neural-shrimp-data/audits/clark-correlated-extension-independent/attempt001/run')
THREADS=D.THREADS;read,entry,serial,covered=D.read,D.entry,D.serial,D.covered
EXPECTED={'native_even_odd_phase':45,'clark_gradient_path_input':30,'clark_gradient_path_return':30,'paired_native_phase':18,'explicit_family_attempt':12,'explicit_amplitude_fold_return':12,'validated_inverse':12,'path_attempt':10,'clark_correlated_path_return':10,'correlated_center_newton_proposal':4,'cached_residual_reuse':4,'context_attempt':3,'paired_native_result':3,'newton_proposal':3,'quadratic_predictor_proposal':2,'correlated_radius_proposal':2,'correlated_explicit_tube_return':2,'explicit_organizer_anchor':1,'explicit_fixed_gain_root':1,'paired_implicit_solve':1,'amplitude_fold_return':1,'explicit_endpoint_fold':1}
FILES=sorted(set(D.FILES+['tools/reviews/review_clark_extension_math.py','tools/reviews/review_clark_extension.py','tools/launch_clark_extension_review.py','tests/reviews/test_clark_extension_math.py','tests/reviews/test_clark_extension_driver.py','docs/plans/clark-extension-independent.md','docs/reviews/clark-correlated-extension-attempt001.json','docs/reviews/clark-correlated-prefix-independent.json']))
class Work(D.Work):pass

def metadata(w):
 for p,pin in zip((PARENT,PARENT/'run'),PINS,strict=True):D.inventory(p,pin,w)
 D.archive(PARENT,SOURCE,0,w);run=PARENT/'run';out=read(run/'outcome.json');need(out['passed']and out['new_charts']==2 and out['prior_independent_charts']==6 and out['calls']==dict(algebra=6,anchor=1,cache_reuses=4,context=3,events=207,explicit=12,implicit=1,native_phase_returns=93,path=10),'literal complete93phase result')
 paths=sorted((run/'events').iterdir());kinds=[p.name.split('-',1)[1].split('.json')[0]for p in paths];need(len(paths)==207 and [int(p.name[:4])for p in paths]==list(range(1,208))and Counter(kinds)==EXPECTED,'complete207 events/all93 actual phases')
 card=read(run/'source/docs/plans/clark-correlated-extension-inputs.json');p=Path(card['independent_prefix']['root']);spec=card['independent_prefix']
 for q,pin in zip((p,p/'run'),(spec['outer'],spec['worker']),strict=True):D.inventory(q,pin,w)
 D.archive(p,spec['source'],0,w);prior=read(p/'run/outcome.json');ep=read(p/'run/connected-endpoint.json');need(prior['passed']and prior['native_calls']==0 and prior['range']==['0','3/8192']and len(prior['accepted'])==6 and entry(p/'run/connected-endpoint.json')==spec['endpoint']and A.point(ep['parameter'])==F(3,8192)and len(ep['box'])==51,'exact independently known51 endpoint')
 pp=Path(card['producer_prefix']['root']);spec2=card['producer_prefix']
 for q,pin in zip((pp,pp/'run'),(spec2['outer'],spec2['worker']),strict=True):D.inventory(q,pin,w)
 D.archive(pp,spec2['source'],2,w);ib=read(run/'input-binding.json');oldib=read(pp/'run/input-binding.json');auditib=read(p/'run/input-binding.json');need(ib['map']==oldib['map']==auditib['map']and ib['organizer_box']==oldib['organizer_box'],'literal same native map/organizer phase')
 need(auditib['root']==str(pp)and auditib['source']==spec2['source']and [auditib['outer'],auditib['worker']]==[spec2['outer'],spec2['worker']],'same prior actual source proof')
 # Every inherited stage is authenticated again; no native reductions here.
 oldcard=read(pp/'run/source/docs/plans/clark-correlated-fold-inputs.json')
 for key in('profile','failed_correlated','failed_explicit','diagnosis'):
  s=oldcard[key];q=Path(s['root']);D.inventory(q,s['outer'],w);D.inventory(q/'run',s['worker'],w);D.archive(q,s['source'],0 if key in('profile','diagnosis')else 2,w)
 for key in('organizer','bridge'):
  s=auditib[key];q=Path(s['root']);D.inventory(q,s['outer'],w);D.inventory(q/'run',s['worker'],w);D.archive(q,s['source'],0,w)
 need(entry(Path(ib['card']['matrix']['path']))==ib['card']['matrix']['entry'],'literal original fixed matrix bytes')
 w.save('input-binding.json',dict(parent=str(PARENT),source=SOURCE,pins=PINS,independent_prefix=card['independent_prefix'],producer_prefix=spec2,map=ib['map'],events=EXPECTED,native_calls=0,scope='Native analytical returns and exact frozen caller input genealogy shared; independent full51/root/native/inverse/curvature reductions.'))
 return paths,kinds,[M.decode(x)for x in ep['box']],ib

def mixed_group(paths,start):
 req=read(paths[start]);rows=[read(paths[start+k])for k in(1,2,3)];got=read(paths[start+4]);need('explicit_family_attempt'in paths[start].name and all('native_even_odd_phase'in paths[start+k].name for k in(1,2,3))and 'explicit_amplitude_fold_return'in paths[start+4].name and req['input']==got['input']and req['squared_amplitude']==got['squared_amplitude'],'literal actual MixedJet base/direction caller association')
 f,J,h=R.explicit(req,rows)
 need(all(x.contains(0)for a,b in zip((f,J,h),[M.matrix(got[k])for k in('value','jacobian','parameter_derivative')],strict=True)for row in(a-b).tolist()for x in row),'separate full51 mixed assembly consistency');return req,f,J,h

def endpoint_inputs(proof,center_req,whole_req,gain):
 def raw_unknown(box):return box[:25]+[gain]+box[25:50]
 need(len(proof['box'])==len(proof['center'])==51 and center_req['input']==raw_unknown(proof['center'])and whole_req['input']==raw_unknown(proof['box'])and center_req['squared_amplitude']==proof['center'][-1]and whole_req['squared_amplitude']==proof['box'][-1],'literal original fixedgain center/whole acquired domains')
 need(all(A.point(v)==A.point(c)for v,c in zip(center_req['input'],raw_unknown(proof['center']),strict=True))and A.point(gain)>0,'exact actual center/gain')

def interval_join(known,b,t):
 c,s,a,r=[[M.decode(v)for v in t[k]]for k in('center','slope','curvature','radii')];p=M.decode(t['parameter']);h=M.decode(t['halfwidth']);d=b-p;need(len(known)==51,'full51 root');inside=bool(p-h<=b and b<=p+h);err=[known[i]-c[i]-s[i]*d-a[i]*d*d/2 for i in range(51)];checks=[bool(v.is_finite()and abs(v).upper()<r[i])for i,v in enumerate(err)];return dict(passed=bool(inside and all(checks)),parameter_inside=inside,coordinates=sum(checks),error=err)

def point_native(paths,start,unknown,b,w,tag):
 req=read(paths[start]);n=12;z=unknown[:26];v=unknown[26:];q=z[13:25]
 need(req['support']is None and len(req['input'])==26 and len(req['paired_tangent'])==25 and all(covered(x,y)for x,y in zip(req['input']+[x[0]for x in req['paired_tangent']],unknown,strict=True))and covered(req['squared_amplitude'],b),'same exact endpoint covered by actual native context')
 need(b>0 and z[25]>0 and 0<z[12]<1,'positive physical endpoint');amplitude=b.sqrt();bb=b;lo=b;hi=b;starts=[[z[i]+sgn*amplitude*q[i]for i in range(n)]for sgn in(1,-1)];modes=[[sgn*(q[i]+2*b*v[13+i])+2*amplitude*v[i]for i in range(n)]for sgn in(1,-1)];rows=[read(paths[start+k])for k in range(1,7)];orbits=[];blocks=[]
 for partner in(0,1):
  raw=rows[3*partner:3*partner+3];need([(x['partner'],x['phase'])for x in raw]==[(partner,k)for k in range(3)],'literal actual native partner/phase order');need(all(covered(x,y)for x,y in zip(raw[0]['input'],starts[partner],strict=True)),'actual phase0 acquisition covers same root partner')
  for k in(1,2):need(raw[k]['input']==raw[k-1]['state'],'literal native state chain')
  orbits.append([[M.decode(x)for x in raw[0]['input']]]+[[M.decode(x)for x in row['state']]for row in raw]);blocks.append([M.matrix(row['jacobian'])for row in raw])
 need(all((orbits[p][3][i]+starts[1-p][i]).contains(0)for p in(0,1)for i in range(n)),'literal native F3 partner state consistency')
 contexts=[]
 for partner in(0,1):
  need(all(x.contains(0)for x in M.vec(blocks[partner][2]*blocks[partner][1]*blocks[partner][0]*M.col(modes[partner])+M.col(modes[1-partner]))),'same exact paired mode transport consistency')
  need(sum((modes[partner][i]*(1 if partner==0 else-1)*q[i]for i in range(n)),arb(0))>0,'same original normalized orientation')
  cycle=orbits[partner][:3]+[[-x for x in row]for row in orbits[1-partner][:3]];js=blocks[partner]+blocks[1-partner]
  for shift in(0,1):
   mode=modes[partner]if shift==0 else M.vec(js[0]*M.col(modes[partner]));out=H.native_context(cycle[shift:]+cycle[:shift],js[shift:]+js[:shift],mode);need(out['passed'],'all native primitive6 and11stable quotient');contexts.append(dict(partner=partner,shift=shift,cycle=cycle[shift:]+cycle[:shift],**out))
 w.save(tag+'/native.json.gz',dict(passed=True,contexts=contexts,parameter=[lo,hi],positive_amplitude=bool(lo>0),scope='Exact paired identities inherited from authenticated full51 root; own native products, normalized line, primitive divisors and quotient power reductions. Shared actual native derivative returns.'));return contexts

def numerical_association(contexts,unknown,binding,w):
 spec=binding['card']['numerical'];p=Path(spec['root']);D.inventory(p,spec['outer'],w);D.inventory(p/'run',spec['worker'],w)
 selected=read(PARENT/'run/fold-input-binding.json')['card']['numerical_fold'];ap=p/'run'/selected['array'];mp=p/'run'/selected['point'];need(entry(ap)==selected['array_identity']and entry(mp)==selected['point_identity'],'literal actual previous two-partner fold arrays')
 md=read(mp);need(md['passed']and all(md['gates'].values())and md['gain']==selected['gain']==4.0756,'original selected qualified gain')
 with np.load(ap,allow_pickle=False)as z:arrays={k:z[k].copy()for k in z.files}
 need(all(x.dtype==np.float64 and np.isfinite(x).all()for x in arrays.values())and arrays['cycle_x'].shape==arrays['cycle_y'].shape==(6,12)and arrays['physical'].shape==arrays['paired_physical'].shape==(25,)and arrays['partner_mode'].shape==(12,),'full original native arrays')
 def exact(v):return arb(str(F.from_float(float(v))))
 need(unknown[25]==exact(selected['gain']),'literal physical fixed gain')
 rows=[]
 for partner in(0,1):
  context=next(v for v in contexts if v['partner']==partner and v['shift']==0);matches=[];errors=[]
  for original in(0,1):
   old=arrays['cycle_x'if original==0 else'cycle_y'];scale=exact(max(1.,float(np.max(np.abs(old)))));phase=[max(abs(context['cycle'][k][i]-exact(old[k,i])).upper()for i in range(12))/scale for k in range(6)];q=arrays['physical'][13:]if original==0 else arrays['partner_mode'];modes=[sum((abs(context['mode'][i]-sign*exact(q[i])).upper()*abs(context['mode'][i]-sign*exact(q[i])).upper()for i in range(12)),arb(0)).sqrt().upper()for sign in(1,-1)];signs=[v for v,e in zip((1,-1),modes)if e<arb('1e-6')];ok=all(v<arb('1e-6')for v in phase)and len(signs)==1
   if ok:matches.append(original)
   errors.append(dict(original_partner=original,phase=phase,mode=modes,signs=signs,passed=bool(ok)))
  need(len(matches)==1,'unique literal full6phase/native mode partner association');rows.append(dict(partner=partner,original_partner=matches[0],comparisons=errors))
 need(sorted(x['original_partner']for x in rows)==[0,1]and abs(unknown[12]-exact(arrays['paired_physical'][-1])).upper()<arb('1e-8'),'both distinct partners and same physical leak');return dict(passed=True,rows=rows,scope='Independent saved full6 phase/normalized mode arithmetic; no state sign relabeling or new native acquisition. Exact branch identity separately follows strict51 root/tube membership.')

def target(w):
 paths,kinds,known,ib=metadata(w);matrix=read(paths[kinds.index('path_attempt')])['matrix'];need([[A.point(x)for x in row]for row in matrix]==[[F(x)for x in row]for row in ib['map']['j0']],'same exact matrix');proofs=[]
 recent=[];left=arb(3)/8192;accepted=[];attempts=[]
 for i,kind in enumerate(kinds):
  if kind=='path_attempt':recent.append(i)
  if kind!='correlated_explicit_tube_return':continue
  attempt=len(attempts)+1;tag=f'charts/{attempt:02d}';t=read(paths[i]);need(t==read(PARENT/f'run/charts/{attempt:02d}/tube.json.gz')and len(recent)>=5,'literal complete retained tube');c,s,a,r=[[M.decode(v)for v in t[k]]for k in('center','slope','curvature','radii')];b=M.decode(t['parameter']);h=M.decode(t['halfwidth']);need(all(x.is_exact()for x in c+s+a+r+[b,h])and len(c)==len(s)==len(a)==len(r)==51,'complete exact curved geometry');found={}
  for k in recent[-5:]:
   req=read(paths[k])
   if any(req[x]!=t[y]for x,y in [('center','center'),('slope','slope'),('curvature','curvature'),('parameter','parameter')]):continue
   zero=all(A.point(v)==0 for v in req['error'])if all(A.interval(v)[0]==A.interval(v)[1]for v in req['error'])else False;center=A.interval(req['d'])==(F(0),F(0));key=(zero,center);need(key not in found,'unique actual proof acquisition')
   if not zero:need(all(A.covers(v,(-A.point(z),A.point(z)))for v,z in zip(req['error'],t['radii'],strict=True)),'same full actual error domain')
   if not center:need(A.covers(req['d'],(-A.point(t['halfwidth']),A.point(t['halfwidth']))),'whole acquired parameter covers original interval')
   found[key]=D.path_group(paths,k,matrix);w.tick()
  need(set(found)=={(True,True),(True,False),(False,True),(False,False)},'four complete residual/fullE acquisitions');need(found[False,True][0]['error']==found[False,False][0]['error']==t['actual_error'],'same literal fullE in center/whole Jacobian acquisitions');fc=found[True,True][1];fw=found[True,False][1];J=[found[False,True][2][0],found[False,True][2][1],found[False,False][2][2]];residual_coefficients=[M.col([fc[k,j]if j<3 else fw[k,3]for k in range(51)])for j in range(4)];C=[M.matrix(v)for v in t['inverse_coefficients']];data=H.reduction(C,J,residual_coefficients,r,h);w.save(tag+'/root.json.gz',data);decision=read(PARENT/f'run/charts/{attempt:02d}/decision.json');record=dict(attempt=attempt,root_passed=data['passed'],contraction=data['contraction'],original_accepted=decision['passed'])
  if decision['passed']:
   need(attempt in(1,2)and data['passed']and left==b-h,'ordered original accepted closed support');join=H.join(known,left,c,s,a,b,h,r);w.save(tag+'/join.json',join);need(join['passed']and join['coordinates']==51,'all51 exact known phase/root join');need(kinds[i+1]=='context_attempt','native context immediately after accepted root');native=D.qualify_native(paths,i+1,t,data,w,tag);known=H.endpoint(data,c,s,a,1);left=b+h;w.save(tag+'/endpoint.json',dict(parameter=left,box=known));accepted.append(dict(attempt=attempt,parameter=left,contraction=data['contraction'],native_contexts=len(native)))
  attempts.append(record);proofs.append(t)

 need(len(attempts)==2 and len(accepted)==2 and left==arb(1)/2048,'two complete independent successor tubes')
 w.save('connected-endpoint.json',dict(parameter=left,box=known))
 root=read(PARENT/'run/endpoint-root.json');r0,f0,_,_=mixed_group(paths,175);rw,fw,Jw,hb=mixed_group(paths,180);gain=read(PARENT/'run/outcome.json')['fixed_gain'];endpoint_inputs(root,r0,rw,gain)
 need(read(paths[185])==root,'literal fixedgain root record');J=R.fixed_gain_jacobian(Jw,hb);center=[M.decode(v)for v in root['center']];box=[M.decode(v)for v in root['box']];bounds=[tuple(arb(str(x))for x in A.interval(v))for v in root['box']]
 result=M.krawczyk(center,box,f0,J,exact_bounds=bounds);w.save('endpoint-root.json.gz',result);unique=[center[i]+result['image'][i,0]for i in range(51)];full=unique[:25]+[M.decode(gain)]+unique[25:50];b=unique[50]
 joins=[interval_join(full,b,t)for t in proofs];w.save('endpoint-joins.json',dict(rows=joins));need(any(x['passed']and x['coordinates']==51 for x in joins),'same exact endpoint belongs to independently connected branch');native=point_native(paths,186,full,b,w,'endpoint')
 implicit=read(paths[205]);solve=read(paths[201]);need('amplitude_fold_return'in paths[205].name and 'paired_implicit_solve'in paths[201].name and implicit['input']==rw['input'][:26]and implicit['squared_amplitude']==rw['squared_amplitude']and solve['solution']==implicit['paired_tangent'],'both implicit passes on exact original basez/b and actual saved direction')
 first=[read(paths[k])for k in(198,199,200)];second=[read(paths[k])for k in(202,203,204)];need(all('native_even_odd_phase'in paths[k].name for k in(198,199,200,202,203,204)),'all six actual implicit phases');z=[M.decode(v)for v in implicit['input']];bb=M.decode(implicit['squared_amplitude']);saved_v=[row[0]for row in solve['solution']]
 need(all(covered(x,y)for x,y in zip(implicit['input'],full[:26],strict=True))and covered(implicit['squared_amplitude'],b),'same independently proved endpoint inside both actual implicit base domains')
 curvature=R.paired_curvature(z,bb,first,second,saved_v);w.save('endpoint-curvature.json.gz',curvature)
 # Recompute original all-phase/mode numeric association from authenticated arrays;
 # exact identity separately follows independently strict root/tube membership.
 association=numerical_association(native,full,ib,w);w.save('numerical-fold-association.json',association)
 return dict(passed=True,status='independent_connected_organizer_to_generic_native6_fold',range=['0','1/2048'],prior_independent_tubes=6,new_independent_tubes=2,new_native_contexts=12,endpoint_coordinate_inclusions=51,endpoint_memberships=[i+1 for i,v in enumerate(joins)if v['passed']],gain=M.decode(gain),leak_bb=curvature['leak_bb'],leak_aa=curvature['leak_aa'],native_calls=0,scope='Separate saved raw coefficient/MixedJet/root/native/paired inverse/curvature reductions; source-bound native analytic derivatives and Arb shared. Exact known organizer through connected critical tubes to selected generic native6 fold; not uniform genericity/native12/24/full-window proof.')

def main():
 resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2);need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'clean CPU1nice10 source');OUT.mkdir(parents=True,exist_ok=False);D.OUT=OUT;w=Work();out=dict(passed=False,status='incomplete');signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,180)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,sources={n:entry(ROOT/n)for n in FILES},pid=os.getpid(),precision=512,python=platform.python_version(),numpy=np.__version__,python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,native_call_cap=0,seconds=180,rss_bytes=2*1024**3,raw_bytes=128*1024**2))
  for n in FILES:
   b=(ROOT/n).read_bytes();need(b==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed own source');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);w.used+=len(b)
  with ctx.workprec(512):D.synthetic(w);out=target(w)
 except BaseException as e:out.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);out.update(native_calls=0,elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.stopped=False;w.save('outcome.json',out,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(serial(out)));return 0 if out['passed']else 2
if __name__=='__main__':raise SystemExit(main())
