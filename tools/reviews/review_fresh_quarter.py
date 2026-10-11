"""Saved first-quarter complementary cover replay with separately authored math.

This adapter is written by the quarter producer's agent. Its polynomial,
known-root, raw native assembly and quotient proof reducers are the unchanged
independent root/Clark implementations, with root source review before launch.
No native derivatives, producer proof reducers or new cells are evaluated.
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
OUT=Path('/data2/neural-shrimp-data/audits/native705-fresh-quarter-independent/attempt001/run')
QUARTER=dict(root='/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/unit-pair/pal/complement/fresh-quarter001',source='0cdfc549b27dc743804e308eaa3662bf7b7a2456',outer='31e64ad5230a2807d4351b3bff8fb36806433c4b076e0e1b5f65c62619654aac',worker='36e64ea223090efd6db4e5b7f697d02e942f69602a3af6e40bf279d3d0bbef83')
PRIOR=dict(root='/data2/neural-shrimp-data/audits/native705-fresh-center-independent/attempt001',source='9c83b5ec0c2d4dfc78839313ffa6e4ed54dac116',outer='57b364cd44e47f018550a7bd45c7f54463b6f0f43faa6883c300173e52124206',worker='c7fe8421ab5ab2b939e759e18da40d8ac7cf43bf051bee37d9c6cc5861ddf926')
CARD='docs/plans/native705-fresh-center-quarter-inputs.json'
FILES=sorted(set(A.FILES+['tools/reviews/review_fresh_quarter.py','tools/reviews/launch_fresh_quarter_review.py','tests/reviews/test_fresh_quarter_review.py','docs/plans/native705-fresh-quarter-independent.md']))


def cells():
 edges=[(-49,-47),(-47,-45),(-45,-43),(-43,-41),(-41,-39),(-39,-37),(-37,-35),(-35,-34),(-32,-30),(-30,-28),(-28,-26),(-26,-24),(-24,-22),(-22,-20),(-20,-18),(-18,-17)]
 return [dict(index=i+1,endpoints=[str(F(a,2048)),str(F(b,2048))],center=str(F(a+b,4096)),halfwidth=str(F(b-a,4096)))for i,(a,b)in enumerate(edges)]


def closed_union(rows,prior):
 expected=cells();need(len(rows)==16 and prior['passed']and prior['native_calls']==0 and prior['parameter_endpoints']==['-17/1024','-1/64']and prior['complement_dimension']==11,'literal independently qualified central cell')
 for row,spec in zip(rows,expected,strict=True):need(row['passed']and row['cell']==spec and row['complement_dimension']==11,'all16 independent cell proofs')
 ranges=sorted([tuple(F(x)for x in r['cell']['endpoints'])for r in rows]+[(F(-17,1024),F(-1,64))]);need(ranges[0][0]==F(-49,2048)and ranges[-1][1]==F(-17,2048)and all(a<b for a,b in ranges)and all(x[1]==y[0]for x,y in zip(ranges,ranges[1:])),'exact closed17-cell union with disjoint interiors')
 return dict(passed=True,parameter_endpoints=['-49/2048','-17/2048'],new_cells=16,reused_cells=1,total_cells=17,complement_dimension=11)


def native_histories(run):
 names={p.name for p in(run/'native').iterdir()if p.is_file()};expected={f'{i:03d}-{kind}.json'for i in range(1,73)for kind in('input','return')}|{f'{i:04d}-{kind}.json'for i in range(73,145)for kind in('input','return')};need(names==expected,'all actual144 synthetic pairs')
 for cell in cells():
  start=144+126*(cell['index']-1);folder=run/f"cells/{cell['index']:03d}";actual={p.name for p in(folder/'native').iterdir()if p.is_file()};need(actual=={f'{i:04d}-{kind}.json'for i in range(start+1,start+127)for kind in('input','return')},'all126 actual input/return pairs per new cell')
  need(read(folder/'cell-input.json')==cell,'exact actual selected center/domain');out=read(folder/'cell-outcome.json');need(out['index']==cell['index']and out['endpoints']==cell['endpoints']and out['passed']and out['dimension']==705 and out['complement_dimension']==11 and out['native_calls']==126,'complete producer selected-cell receipt')


def metadata(w):
 root,rt,out,inv=A.S.bundle(QUARTER,w.tick);ir,irt,prior,pinv=A.S.bundle(PRIOR,w.tick);pr,prt,po,priv=A.S.bundle(A.PARENT,w.tick);old,ort,oo,oinv=A.S.bundle(A.OLD,w.tick,2);run=root/'run';card=read(run/'source'/CARD)
 need(card['status']=='reviewed'and card['cells']==cells()and card['original_quarter']==['-49/2048','-17/2048']and card['reused_cell']==['-17/1024','-1/64']and card['pilot']==A.PARENT and card['independent_pilot']==PRIOR,'exact source-bound quarter card and prerequisite')
 caps=dict(seconds=1500,rss_bytes=4*1024**3,as_bytes=6*1024**3,raw_bytes=3*1024**3,profile_calls=144,target_calls=2016,total_calls=2160);need(card['caps']==rt['caps']==caps,'actual complete original cap')
 need(out['passed']and out['status']=='first_quarter_complement_covered'and out['native_calls']==2160 and out['profile_native_calls']==144 and out['target_native_calls']==2016 and len(out['completed_cells'])==16,'complete original16-cell packet')
 need(prior['passed']and prior['native_calls']==0 and prior['parameter_endpoints']==card['reused_cell']and prior['complement_dimension']==11 and read(ir/'run/source-binding.json')['pilot']==A.PARENT,'same completed independent prior cell')
 need(po['passed']and po['native_calls']==270 and po['parameter_endpoints']==card['reused_cell']and not oo['passed']and oo['native_calls']==216 and oo['status']=='completed_correlated_bound_unresolved','complete pilot and retained failed whole-quarter metric')
 native_histories(run);A.pairs(pr/'run',270);A.pairs(old/'run',216)
 for n in A.B.B.NATIVE+A.B.T.PATH_FILES:need(rt['sources'][n]==prt['sources'][n]==ort['sources'][n]==A.S.entry(ROOT/n),'same actual native/path source '+n)
 for spec in cells():need(out['completed_cells'][spec['index']-1]==read(run/f"cells/{spec['index']:03d}/cell-outcome.json"),'ordered full completed prefix')
 need(read(run/'completed-prefix-16.json')['rows']==out['completed_cells']and read(run/'union.json')==out['union']==dict(passed=True,endpoints=card['original_quarter'],new_cells=16,reused_cells=1,total_cells=17),'exact retained union and completed prefix')
 profile=read(run/'batch-profile.json');need(profile['passed']and profile['total_native_calls']==2160 and profile['forecast_seconds']<1500 and profile['forecast_bytes']<3*1024**3,'actual full-resource admission')
 w.save('authority-binding.json',dict(producer=QUARTER,prior=PRIOR,original_whole=A.OLD,pilot=A.PARENT,cells=cells(),inventories=inv+pinv+priv+oinv,authorship='Adapter by producer agent; math/reductions authored separately by root and Clark, unchanged. Root source review required.'))
 return run,old/'run',prior


def quarter_context(w,prev):
 model,c,s,a,normal,origin,sc,weights,original,image,C,ds,ff,rp,E,metrics,domain=A.B.bind(w);del model;gc.collect();td=read(prev/'tangent-domain-inputs.json');dom=read(prev/'quotient-domain-inputs.json')
 need(td['parameter_endpoints']==dom['parameter_endpoints']==['-49/2048','-17/2048']and td['scale']=='1/256'and dom['halfwidth']=='1/128','literal original whole quarter')
 need(dom['complements']==[matrix_json(e)for e in E]and dom['metrics']==[matrix_json(b)for b in metrics],'same fixed exact E/B')
 shift,half=F(-3,128),F(1,128);lo,hi=shift-half,shift+half;err,ep=Q.root(ds,ff,weights,original,image,lo,hi,w.tick);dd=arb(R.exact(shift),R.exact(half));d=R.exact(shift);pred=[c[i]+s[i]*dd+a[i]*dd*dd/2 for i in range(705)];pprime=[s[i]+a[i]*dd for i in range(705)];whole=[pred[i]+err[i,0]for i in range(705)];segment=[pred[i]+arb(0,abs(err[i,0]).upper())for i in range(705)]
 Q.cover(domain['whole_box'],whole,'original full known-root domain');Q.cover(domain['whole_box'],segment,'original full predictor-root segment');Q.cover(td['segment'],segment,'actual polarized segment');Q.cover([v[0]for v in td['error']],[err[i,0]for i in range(705)],'actual root error');Q.cover(td['predictor_tangent'],pprime,'actual predictor tangent')
 plus=[err[i,0]*256+pprime[i]/256 for i in range(705)];minus=[err[i,0]*256-pprime[i]/256 for i in range(705)];Q.cover(td['plus'],plus,'actual plus');Q.cover(td['minus'],minus,'actual minus');need(td['normal']==vector_json(normal)and td['origin']==vector_json(origin)and M.point(td['parameter'])==M.point(ball_json(sc+d)),'actual inherited tangent gauge')
 paths=[A.A.path_event(read(prev/f'native/{i:03d}-input.json'),read(prev/f'native/{i:03d}-return.json'),i-36)for i in range(109,181)];mix=H.polarized_paths(segment,plus,minus,paths,normal,origin,sc+d,13,18,w.tick);tangent,tp=Q.tangent(C,ds,rp,mix,pprime,weights,lo,hi,w.tick);Q.cover(dom['full705']['whole'],whole,'original acquired whole root');Q.cover(dom['full705']['tangent'],tangent,'original acquired true tangent')
 w.save('quarter-root.json',ep);w.save('quarter-tangent.json',tp);del paths,mix,rp;gc.collect()
 return dict(c=c,s=s,a=a,normal=normal,origin=origin,sc=sc,weights=weights,original=original,image=image,C=C,ds=ds,ff=ff,E=E,metrics=metrics,domain=domain,whole=whole,tangent=tangent)


def residual(folder,spec,normal,origin):
 start=144+126*(spec['index']-1);inp=read(folder/'residual-inputs/target2.json');point=[R.exact(M.point(x))for x in inp['point']];parameter=R.exact(F(spec['center']));need(inp['normal']==vector_json(normal)and inp['origin']==vector_json(origin)and M.point(inp['parameter'])==F(spec['center']),'literal actual selected residual gauge');events=[]
 for i in range(36):
  aa=read(folder/f'native/{start+73+i:04d}-input.json');bb=read(folder/f'native/{start+73+i:04d}-return.json');need(aa['kind']==bb['kind']=='residual'and aa['stage']==bb['stage']=='target2','literal final36 residual group');events.append(A.J.mixed_direction(dict(aa,kind='full_jacobian',phase=i//2,direction='q'if i%2==0 else'r'),dict(bb,kind='full_jacobian')))
 value,jac=H.P.rebuild(point,events,13,18,normal,origin,parameter);del jac;return point,value


def one(w,run,prev,spec,data):
 folder=run/f"cells/{spec['index']:03d}";c,s,a=[data[k]for k in('c','s','a')];normal,origin,sc=[data[k]for k in('normal','origin','sc')];shift=F(spec['center'])-M.point(ball_json(sc));half=F(spec['halfwidth']);need(F(-1,32)<=shift-half<shift+half<=F(-1,64),'cell exactly inside original first quarter');d=R.exact(shift)
 err,ep=Q.root(data['ds'],data['ff'],data['weights'],data['original'],data['image'],shift,shift,w.tick);pc=[c[i]+s[i]*d+a[i]*d*d/2 for i in range(705)];known=[pc[i]+err[i,0]for i in range(705)];authority=read(folder/'center-authority.json');need(authority['cell']==spec and authority['offset']==str(shift)and authority['predictor']==vector_json(pc)and authority['original_error']==data['domain']['error'],'same exact predictor and inherited root domain');Q.cover(authority['known'],known,'actual new inherited center');Q.original(data['domain']['error'],pc,known)
 point,value=residual(folder,spec,normal,origin);Q.original(data['domain']['error'],pc,point);center,proof=Q.point(data['C'],data['ds'],value,point,known,data['weights'],shift,w.tick);Q.original(data['domain']['error'],pc,center);w.save(f"cells/{spec['index']:03d}/center.json",proof)
 qr=read(folder/'quotient-domain-inputs.json');need(qr['cell']==spec and qr['parameter_endpoints']==spec['endpoints']and qr['halfwidth']==spec['halfwidth']and qr['complements']==[matrix_json(e)for e in data['E']]and qr['metrics']==[matrix_json(b)for b in data['metrics']],'actual full native domain/finite frame/width');whole,tangent=data['whole'],data['tangent']
 for key,v in [('center',center),('whole',whole),('tangent',tangent)]:Q.cover(qr['full705'][key],v,'actual full705 '+key)
 derivatives=[];start=144+126*(spec['index']-1)
 for j in range(18):
  ac,_=A.native(read(folder/f'native/{start+109+j:04d}-input.json'),read(folder/f'native/{start+109+j:04d}-return.json'),center,whole,tangent,'center',j);aw,ap=A.native(read(prev/f'native/{182+2*j:03d}-input.json'),read(prev/f'native/{182+2*j:03d}-return.json'),center,whole,tangent,'whole_tangent',j);derivatives.append((ac,aw,ap));w.tick()
 records=[read(folder/f'events/{j+2:03d}-phase.json')for j in range(18)];metric=read(folder/'events/020-metric.json');out=H.reduction(center[:-1],whole[:-1],tangent[:-1],data['E'],data['metrics'],derivatives,records,metric,R.exact(half),whole[-1],w.tick);w.save(f"cells/{spec['index']:03d}/complement.json",out)
 return dict(passed=True,cell=spec,complement_dimension=11,modulus=out['complementary_modulus_upper'],remainder=out['remainder_norm_upper'],penalty=out['remainder_penalty'])


class Work:
 def __init__(self):self.start=time.monotonic();self.used=0;self.rows=[]
 def tick(self):need(time.monotonic()-self.start<1200 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'1200s4GiB saved cap')
 def save(self,name,value,closing=False):
  raw=(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode();need(self.used+len(raw)<(256 if closing else 252)*1024**2,'256MiB4MiB reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)


def main():
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean exact nice10 source');OUT.mkdir(parents=True,exist_ok=False);w=Work();result=dict(passed=False,native_calls=0)
 def stop(*_):raise TimeoutError('owned1200s saved audit')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,1200)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:A.S.entry(ROOT/n)for n in FILES},precision=480,native_calls=0,python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,threads=dict.fromkeys(THREADS,'1'),authorship='Producer agent writes adapter; unchanged separately authored independent reducers and root source review.'))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'complete exact source archive');need(w.used+len(raw)<252*1024**2,'bounded complete source retention');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(480):
   before=time.monotonic();trial=A.A.synthetic(705,w.tick);seconds=time.monotonic()-before;prior=Path(PRIOR['root']);need(A.S.entry(prior/'run/manifest.json')['sha256']==PRIOR['worker'],'literal full prior measured byte floor');pm=read(prior/'run/manifest.json');need(A.S.entry(prior/'run/outcome.json')==pm['files']['outcome.json'],'exact prior measured runtime');cost=read(prior/'run/outcome.json');forecast=8*seconds+2*16*cost['elapsed_seconds']+60;raw=w.used+2*16*sum(v['bytes']for v in pm['files'].values())+64*1024**2;ok=trial['passed']and forecast<1200 and raw<256*1024**2 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+512*1024<4*1024**2;w.save('profile.json',dict(passed=ok,seconds=seconds,forecast_seconds=forecast,forecast_bytes=raw,target_arrays_unopened=True));need(ok,'full705 measured independent16-cell admission');gc.collect()
   run,prev,prior=metadata(w);data=quarter_context(w,prev)
   for spec in cells():
    row=one(w,run,prev,spec,data);w.rows.append(row);w.save(f"prefix-{len(w.rows):02d}.json",dict(rows=w.rows));gc.collect()
   union=closed_union(w.rows,prior);w.save('union.json',union);result.update(passed=True,**{k:v for k,v in union.items()if k!='passed'},scope='Independent arithmetic on16 new cells plus prior independent centered cell proves all11 complementary multipliers strictly interior throughout chart31 first quarter. Shared authenticated native derivatives/Arb and original coefficient decoder. Adapter by producer agent, reducers separately authored and root source reviewed; neutral critical pair retained, no fullchart/window or generic NS theorem.')
 except BaseException as e:result.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(completed_cells=w.rows,elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):A.S.entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
