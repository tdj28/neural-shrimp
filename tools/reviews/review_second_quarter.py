"""Second-quarter adapter using unchanged separately authored independent reducers.

Adapter by producer agent; root source/domain review and execution required.
Zero native calls; all saved native acquisitions and Arb remain shared.
"""
import os
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gc,json,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'tools/reviews'),str(ROOT/'tools'),str(ROOT/'src')]
import flint
from flint import arb,ctx
import review_fresh_center as A
Q,H,R,M,need,read=A.Q,A.H,A.R,A.M,A.need,A.read
ball_json,vector_json,matrix_json=A.ball_json,A.vector_json,A.matrix_json
OUT=Path('/data2/neural-shrimp-data/audits/native705-second-quarter-independent/attempt001/run')
CARD='docs/plans/native705-second-quarter-independent-inputs.json'
FILES=sorted(set(A.FILES+['tools/reviews/review_second_quarter.py','tools/reviews/launch_second_quarter_review.py','tests/reviews/test_second_quarter_review.py','docs/plans/native705-second-quarter-independent.md',CARD]))


def domains():return [dict(index=i+1,endpoints=[str(F(a,2048)),str(F(b,2048))])for i,(a,b)in enumerate([(-17,-1),(-1,15)])]
def cell(domain,depth,index):
 need(domain in(1,2)and type(depth)is int and 0<=depth<=3 and type(index)is int and 0<=index<2**depth,'exact bounded binary cell');a,b=map(F,domains()[domain-1]['endpoints']);width=(b-a)/2**depth;a=a+index*width;b=a+width
 return dict(domain=domain,depth=depth,index=index,endpoints=[str(a),str(b)],center=str((a+b)/2),halfwidth=str((b-a)/2))

def union(rows,projection):
 need(rows and len(rows)<=16 and all(r['passed']and r['complement_dimension']==11 for r in rows),'complete independent full11 leaves');ranges=[]
 for r in rows:
  spec=r['cell'];need(spec==cell(spec['domain'],spec['depth'],spec['index']),'literal independent geometry');ranges.append(tuple(map(F,spec['endpoints'])))
 ranges.sort();need(ranges[0][0]==F(-17,2048)and ranges[-1][1]==F(15,2048)and all(a[1]==b[0]for a,b in zip(ranges,ranges[1:])),'exact closed secondquarter union')
 lo,hi=map(F,projection);need(F(-17,2048)<=lo<=hi<=F(15,2048),'entire stable-prefix endpoint inside proved union')
 return dict(passed=True,parameter_endpoints=['-17/2048','15/2048'],first_and_second_union=['-49/2048','15/2048'],prior_stable_projection=list(projection),complement_dimension=11,total_cells=len(rows))

def pairs(folder,start,count):
 need({p.name for p in(folder/'native').iterdir()if p.is_file()}=={f'{j:04d}-{k}.json'for j in range(start+1,start+count+1)for k in('input','return')},'complete actual input/return history')

def tree_history(out,run):
 attempts=out['completed_attempts'];need(2<=len(attempts)<=30,'finite complete producer tree');cursor=144;selected=[];idx=0;all_domains=[]
 for d in domains():
  folder=run/f"domains/{d['index']:03d}";dom=read(folder/'domain-outcome.json');need(dom==dict(passed=True,domain=d,dimension=705,native_calls=90,first_call=cursor+1,last_call=cursor+90),'exact90-phase derivative domain');pairs(folder,cursor,90);all_domains.append(dict(domain=d,start=cursor));cursor+=90;queue=[cell(d['index'],0,0)]
  while queue:
   need(idx<len(attempts),'complete binary subtree');spec=queue.pop(0);r=attempts[idx];idx+=1;f=run/f'cells/{idx:03d}'
   need(r['cell']==spec and r['completed']and r['native_calls']==126 and r['dimension']==705 and r['complement_dimension']==11 and r['first_call']==cursor+1 and r['last_call']==cursor+126,'complete literal bounded cell')
   need(read(f/'cell-input.json')==spec and read(f/'cell-outcome.json')==r and read(run/f'completed-attempt-{idx:03d}.json')==r,'actual ordered cell records');pairs(f,cursor,126)
   if r['passed']:selected.append(dict(ordinal=idx,cell=spec,start=cursor))
   else:
    need(r['passed']is False and spec['depth']<3 and read(f/'complement.json')['passed']is False,'only completed strict-metric failure split');queue=[cell(spec['domain'],spec['depth']+1,2*spec['index']+k)for k in(0,1)]+queue
   cursor+=126
 need(idx==len(attempts)and out['native_calls']==cursor==144+180+126*idx and out['profile_native_calls']==144 and out['target_native_calls']==cursor-144,'entire4104 inclusive acquisition history')
 need(out['accepted_cells']==[attempts[r['ordinal']-1]for r in selected],'all accepted leaves retained');return all_domains,selected


def metadata(w,card):
 need(card['status']=='reviewed'and set(card['producer'])=={'root','source','outer','worker'}and all(isinstance(v,str)and v for v in card['producer'].values()),'literal completed producer authority required')
 root,rt,out,iv=A.S.bundle(card['producer'],w.tick);first,frt,fo,fiv=A.S.bundle(card['first_quarter'],w.tick);run=root/'run';pc=read(run/'source/docs/plans/native705-second-quarter-inputs.json')
 need(out['passed']and out['status']=='second_quarter_complement_covered'and pc['status']=='reviewed'and pc['domains']==domains()and pc['parameter_endpoints']==['-17/2048','15/2048']and pc['maximum_centers']==30 and pc['maximum_depth']==3,'fixed complete producer card')
 need(pc['independent_first_quarter']==card['first_quarter']and fo['passed']and fo['native_calls']==0 and fo['total_cells']==17 and fo['parameter_endpoints']==['-49/2048','-17/2048']and fo['complement_dimension']==11,'adjacent independently proved entire firstquarter')
 caps=dict(seconds=3000,rss_bytes=4*1024**3,as_bytes=6*1024**3,raw_bytes=6*1024**3,profile_calls=144,maximum_target_calls=3960,maximum_total_calls=4104);need(pc['caps']==rt['caps']==caps,'full exact original cap')
 pairs(run,0,144);dd,selected=tree_history(out,run);prof=read(run/'batch-profile.json');need(prof['passed']and prof['maximum_centers']==30 and prof['maximum_total_calls']==4104 and prof['forecast_seconds']<3000 and prof['forecast_bytes']<6*1024**3,'actual worstcase resource admission')
 width,wrt,wo,wiv=A.S.bundle(pc['width_diagnosis'],w.tick);need(wo['passed']and wo['native_calls']==0 and[r['halfwidth']for r in wo['results']]==['1/2048','1/1024','1/512','1/256','1/128']and[r['passed']for r in wo['results']]==[True,True,False,False,False],'retained fixed width diagnosis')
 ab=read(run/'authority-binding.json');need(ab['first']==card['first_quarter']and ab['first_outcome']==fo and ab['width']==pc['width_diagnosis']and ab['width_outcome']==wo,'same actual prerequisite authorities');ov=ab['overlap'];ir,irt,io,iiv=A.S.bundle({k:ov[k]for k in('root','source','outer','worker')},w.tick)
 mf=ir/'run/known-root-membership.json';need(A.S.entry(mf)==ov['known_membership']and A.S.entry(mf)['sha256']=='cfbc127d130f098e31976db4647bff6ccb10e352a09a9406b2e24c0e0774c197','literal independently proved previous705 anchor');km=read(mf);need(km['passed']and km['parameter_inside']and km['coordinates']==705 and io['passed'],'complete previous known-root membership');a,b=M.interval(km['projection']);projection=[str(a),str(b)];need(projection==ov['exact_projection_endpoints']and F(-17,2048)<=a<=b<=F(15,2048),'exact full stable-prefix projection')
 fib=read(first/'run/authority-binding.json');need(fib['original_whole']['outer']==A.OLD['outer']and fib['pilot']==A.PARENT and ov['first_quarter']==fo['parameter_endpoints']and ov['second_quarter']==pc['parameter_endpoints']and F(fo['parameter_endpoints'][1])==F(pc['parameter_endpoints'][0]),'same root firstquarter and exact shared endpoint')
 for n in A.B.B.NATIVE+A.B.T.PATH_FILES:need(rt['sources'][n]==A.S.entry(ROOT/n),'unchanged actual native/path '+n)
 w.save('authority-binding.json',dict(producer=card['producer'],first_quarter=card['first_quarter'],width=pc['width_diagnosis'],projection=projection,domains=dd,cells=selected,inventories=iv+fiv+wiv+iiv,authorship='Producer agent writes adapter; independent root/Clark proof reducers unchanged and root reviewed.'))
 return run,dd,selected,projection


def context(w):
 model,c,s,a,normal,origin,sc,weights,original,image,C,ds,ff,rp,E,metrics,domain=A.B.bind(w);del model;gc.collect();return dict(c=c,s=s,a=a,normal=normal,origin=origin,sc=sc,weights=weights,original=original,image=image,C=C,ds=ds,ff=ff,rp=rp,E=E,metrics=metrics,domain=domain)


def shared_domain(w,run,spec,data):
 folder=run/f"domains/{spec['domain']['index']:03d}";td=read(folder/'tangent-domain-inputs.json');qd=read(folder/'quotient-domain-inputs.json');left,right=map(F,spec['domain']['endpoints']);center=(left+right)/2;half=(right-left)/2;shift=center-M.point(ball_json(data['sc']));lo,hi=shift-half,shift+half;normal,origin=data['normal'],data['origin'];c,s,a=[data[k]for k in('c','s','a')]
 need(td['domain']==qd['domain']==spec['domain']and td['scale']=='1/256'and td['original_offset']==[str(lo),str(hi)]and td['normal']==vector_json(normal)and td['origin']==vector_json(origin)and M.point(td['parameter'])==center,'actual exact tangent domain and native gauge');need(qd['complements']==[matrix_json(e)for e in data['E']]and qd['metrics']==[matrix_json(b)for b in data['metrics']],'fixed exact frames')
 err,proof=Q.root(data['ds'],data['ff'],data['weights'],data['original'],data['image'],lo,hi,w.tick);dd=arb(R.exact(shift),R.exact(half));pred=[c[i]+s[i]*dd+a[i]*dd*dd/2 for i in range(705)];pp=[s[i]+a[i]*dd for i in range(705)];whole=[pred[i]+err[i,0]for i in range(705)];segment=[pred[i]+arb(0,abs(err[i,0]).upper())for i in range(705)]
 Q.cover(data['domain']['whole_box'],whole,'original full root');Q.cover(data['domain']['whole_box'],segment,'original full segment');Q.cover(td['segment'],segment,'actual segment');Q.cover([v[0]for v in td['error']],[err[i,0]for i in range(705)],'actual correction');Q.cover(td['predictor_tangent'],pp,'actual predictor tangent');plus=[err[i,0]*256+pp[i]/256 for i in range(705)];minus=[err[i,0]*256-pp[i]/256 for i in range(705)];Q.cover(td['plus'],plus,'actual plus');Q.cover(td['minus'],minus,'actual minus')
 paths=[A.A.path_event(read(folder/f'native/{spec["start"]+j:04d}-input.json'),read(folder/f'native/{spec["start"]+j:04d}-return.json'),72+j)for j in range(1,73)]
 mixed=H.polarized_paths(segment,plus,minus,paths,normal,origin,R.exact(center),13,18,w.tick);tangent,tp=Q.tangent(data['C'],data['ds'],data['rp'],mixed,pp,data['weights'],lo,hi,w.tick);Q.cover(qd['full705']['whole'],whole,'actual705 whole');Q.cover(qd['full705']['tangent'],tangent,'actual705 tangent');w.save(f"domains/{spec['domain']['index']:03d}/root.json",proof);w.save(f"domains/{spec['domain']['index']:03d}/tangent.json",tp)
 derivatives=[]
 for j in range(18):
  k=spec['start']+73+j;inp=read(folder/f'native/{k:04d}-input.json');out=read(folder/f'native/{k:04d}-return.json');aw,ap=A.native(inp,out,whole,whole,tangent,'whole_tangent',j);derivatives.append((aw,ap))
 return dict(whole=whole,tangent=tangent,derivatives=derivatives,spec=spec)


def one(w,run,item,data,shared):
 spec=item['cell'];folder=run/f"cells/{item['ordinal']:03d}";shift=F(spec['center'])-M.point(ball_json(data['sc']));d=R.exact(shift);c,s,a=[data[k]for k in('c','s','a')];err,_=Q.root(data['ds'],data['ff'],data['weights'],data['original'],data['image'],shift,shift,w.tick);pc=[c[i]+s[i]*d+a[i]*d*d/2 for i in range(705)];known=[pc[i]+err[i,0]for i in range(705)];ca=read(folder/'center-authority.json');need(ca['cell']==spec and ca['offset']==str(shift)and ca['predictor']==vector_json(pc)and ca['original_error']==data['domain']['error'],'literal same predictor/root');Q.cover(ca['known'],known,'known center');Q.original(data['domain']['error'],pc,known)
 ri=read(folder/'residual-inputs/target2.json');point=[R.exact(M.point(v))for v in ri['point']];need(ri['normal']==vector_json(data['normal'])and ri['origin']==vector_json(data['origin'])and M.point(ri['parameter'])==F(spec['center']),'same actual point gauge');Q.original(data['domain']['error'],pc,point);events=[]
 for j in range(36):
  k=item['start']+73+j;aa=read(folder/f'native/{k:04d}-input.json');bb=read(folder/f'native/{k:04d}-return.json');need(aa['kind']==bb['kind']=='residual'and aa['stage']==bb['stage']=='target2','literal final residual group');events.append(A.J.mixed_direction(dict(aa,kind='full_jacobian',phase=j//2,direction='q'if j%2==0 else'r'),dict(bb,kind='full_jacobian')))
 value,jac=H.P.rebuild(point,events,13,18,data['normal'],data['origin'],R.exact(F(spec['center'])));del jac;center,proof=Q.point(data['C'],data['ds'],value,point,known,data['weights'],shift,w.tick);Q.original(data['domain']['error'],pc,center);w.save(f"cells/{item['ordinal']:03d}/center.json",proof);qr=read(folder/'quotient-domain-inputs.json');need(qr['cell']==spec and qr['halfwidth']==spec['halfwidth']and qr['parameter_endpoints']==spec['endpoints']and qr['complements']==[matrix_json(e)for e in data['E']]and qr['metrics']==[matrix_json(b)for b in data['metrics']],'literal cell width/frame proposal');whole,tangent=shared['whole'],shared['tangent']
 for key,v in [('center',center),('whole',whole),('tangent',tangent)]:Q.cover(qr['full705'][key],v,'actual705 '+key)
 native=[]
 for j,(aw,ap)in enumerate(shared['derivatives']):
  k=item['start']+109+j;ac,_=A.native(read(folder/f'native/{k:04d}-input.json'),read(folder/f'native/{k:04d}-return.json'),center,whole,tangent,'center',j);native.append((ac,aw,ap))
 retained=read(folder/'retained-derivatives.json');need(retained['domain']==shared['spec']['domain']and retained['calls']==list(range(shared['spec']['start']+73,shared['spec']['start']+91)),'exact retained derivative identities')
 records=[read(folder/f'events/{j+2:03d}-phase.json')for j in range(18)];metric=read(folder/'events/020-metric.json');proof=H.reduction(center[:-1],whole[:-1],tangent[:-1],data['E'],data['metrics'],native,records,metric,R.exact(F(spec['halfwidth'])),whole[-1],w.tick);w.save(f"cells/{item['ordinal']:03d}/complement.json",proof);return dict(passed=True,cell=spec,ordinal=item['ordinal'],complement_dimension=11,modulus=proof['complementary_modulus_upper'],penalty=proof['remainder_penalty'])


class Work:
 def __init__(self):self.start=time.monotonic();self.used=0;self.rows=[]
 def tick(self):need(time.monotonic()-self.start<1200 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'1200s4GiB saved audit')
 def save(self,name,value,closing=False):
  b=(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode();need(self.used+len(b)<(256 if closing else 252)*1024**2,'256MiB4MiB reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
  self.used+=len(b)


def main():
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean exact nice10 source');OUT.mkdir(parents=True,exist_ok=False);w=Work();result=dict(passed=False,native_calls=0)
 def stop(*_):raise TimeoutError('owned1200s saved audit')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,1200)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:A.S.entry(ROOT/n)for n in FILES},precision=480,native_calls=0,python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,threads=dict.fromkeys(THREADS,'1'),authorship='Producer agent adapter; unchanged separately authored independent proof reducers and root source review.'))
  for n in FILES:
   b=(ROOT/n).read_bytes();need(b==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed source');need(w.used+len(b)<252*1024**2,'bounded source retention');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);w.used+=len(b)
  with ctx.workprec(480):
   card=read(ROOT/CARD);before=time.monotonic();trial=A.A.synthetic(705,w.tick);seconds=time.monotonic()-before;prior=Path(card['first_quarter']['root']);need(A.S.entry(prior/'run/manifest.json')['sha256']==card['first_quarter']['worker'],'literal measured full firstquarter');pm=read(prior/'run/manifest.json');need(A.S.entry(prior/'run/outcome.json')==pm['files']['outcome.json'],'literal runtime floor');cost=read(prior/'run/outcome.json');forecast=8*seconds+4*cost['elapsed_seconds']+2*61.466+60;raw=w.used+4*sum(v['bytes']for v in pm['files'].values())+64*1024**2;ok=trial['passed']and forecast<1200 and raw<256*1024**2 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+512*1024<4*1024**2;w.save('profile.json',dict(passed=ok,seconds=seconds,forecast_seconds=forecast,forecast_bytes=raw,target_arrays_unopened=True));need(ok,'full705 measured same-size replay admission');gc.collect()
   run,dd,selected,projection=metadata(w,card);data=context(w)
   for ds in dd:
    shared=shared_domain(w,run,ds,data)
    for item in selected:
     if item['cell']['domain']!=ds['domain']['index']:continue
     row=one(w,run,item,data,shared);w.rows.append(row);w.save(f'prefix-{len(w.rows):02d}.json',dict(rows=w.rows));gc.collect()
   joined=union(w.rows,projection);w.save('union.json',joined);result.update(**joined,scope='Independent all11 complementary contraction on exact secondquarter, joined to independently proved firstquarter and whole chart30 stable-prefix endpoint projection. Neutral critical pair retained; no wholechart/window or generic NS assertion.')
 except BaseException as e:result.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(completed_cells=w.rows,elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):A.S.entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
