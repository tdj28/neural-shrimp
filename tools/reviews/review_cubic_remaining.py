"""Cubic remaining-portion adapter using unchanged separately authored independent reducers.

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
import review_second_quarter as S
import review_cubic_center as C
import bind_native705_cubic_complement as G
import probe_native705_cubic_remaining as P
from flint import arb_mat
Q,H,R,M,need,read=A.Q,A.H,A.R,A.M,A.need,A.read
ball_json,vector_json,matrix_json=A.ball_json,A.vector_json,A.matrix_json
OUT=Path('/data2/neural-shrimp-data/audits/native705-cubic32-remaining-independent/attempt004/run')
CARD='docs/plans/native705-cubic32-remaining-independent-inputs.json'
FILES=sorted(set(S.FILES+P.FILES+['docs/reviews/native705-cubic-polynomial-screen.json','docs/plans/native705-cubic-polynomial-screen.md','tools/reviews/review_cubic_polynomial_domain.py','tests/reviews/test_cubic_domain_repair.py','tests/reviews/test_cubic_polynomial_domain.py','src/neural_shrimp/rigorous_arclength_connection.py','src/neural_shrimp/rigorous_arclength_slices.py','src/neural_shrimp/rigorous_connection.py','src/neural_shrimp/rigorous_connection_frames.py','src/neural_shrimp/rigorous_correlated_cocycle.py','src/neural_shrimp/rigorous_flips.py','src/neural_shrimp/rigorous_norm_cocycle.py','src/neural_shrimp/rigorous_periodic_lyapunov.py','src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py','tools/reviews/review_cubic_center.py','tools/reviews/review_cubic_remaining.py','tools/reviews/launch_cubic_remaining_review.py','tools/bind_native705_cubic_complement.py','tools/probe_native705_cubic_complement.py','tools/launch_native705_cubic_complement.py','src/neural_shrimp/cubic_complement_geometry.py','tests/test_cubic_complement_geometry.py','tests/test_native705_cubic_complement.py','docs/plans/native705-cubic-complement.md','docs/plans/native705-cubic-complement-inputs.json','tools/reviews/check_native705_chart31_boundary_hypotheses.py','docs/reviews/native705-chart31-boundary-hypotheses.json','tests/reviews/test_cubic_remaining_review.py','docs/plans/native705-cubic32-remaining-independent.md',CARD]))

def domains():return [dict(index=i+1,endpoints=[str(F(a,1024)),str(F(b,1024))])for i,(a,b)in enumerate([(55+8*i,63+8*i)for i in range(6)])]
def cell(domain,depth,index):
 need(domain in range(1,7)and type(depth)is int and 0<=depth<=3 and type(index)is int and 0<=index<2**depth,'exact bounded binary cell');a,b=map(F,domains()[domain-1]['endpoints']);width=(b-a)/2**depth;a=a+index*width;b=a+width
 return dict(domain=domain,depth=depth,index=index,endpoints=[str(a),str(b)],center=str((a+b)/2),halfwidth=str((b-a)/2))

def union(rows,projection):
 need(rows and len(rows)<=48 and all(r['passed']and r['complement_dimension']==11 for r in rows),'complete independent full11 leaves');ranges=[]
 for r in rows:
  spec=r['cell'];need(spec==cell(spec['domain'],spec['depth'],spec['index']),'literal independent geometry');ranges.append(tuple(map(F,spec['endpoints'])))
 ranges.sort();need(ranges[0][0]==F(55,1024)and ranges[-1][1]==F(103,1024)and all(a[1]==b[0]for a,b in zip(ranges,ranges[1:])),'exact closed cubic remaining portion union')
 lo,hi=map(F,projection);need(F(39,1024)<=lo<=hi<=F(55,1024),'entire stable-prefix endpoint in independently covered adjacent first portion')
 return dict(passed=True,parameter_endpoints=['55/1024','103/1024'],whole_chart_endpoints=['39/1024','103/1024'],prior_stable_projection=list(projection),complement_dimension=11,total_cells=len(rows))

def pairs(folder,start,count):
 need({p.name for p in(folder/'native').iterdir()if p.is_file()}=={f'{j:04d}-{k}.json'for j in range(start+1,start+count+1)for k in('input','return')},'complete actual input/return history')

def tree_history(out,run):
 attempts=out['completed_attempts'];need(6<=len(attempts)<=90,'finite complete producer tree');cursor=144;selected=[];idx=0;all_domains=[]
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
 need(idx==len(attempts)and out['native_calls']==cursor==144+540+126*idx and out['profile_native_calls']==144 and out['target_native_calls']==cursor-144,'entire12024 inclusive acquisition history')
 need(out['accepted_cells']==[attempts[r['ordinal']-1]for r in selected],'all accepted leaves retained');return all_domains,selected


def metadata(w,card):
 need(card['status']=='reviewed'and set(card['producer'])=={'root','source','outer','worker'}and all(isinstance(v,str)and v for v in card['producer'].values()),'literal completed remaining producer required before arrays')
 root,rt,out,inventory=A.S.bundle(card['producer'],w.tick);run=root/'run';pc=read(run/'source/docs/plans/native705-cubic32-remaining-inputs.json')
 need(out['passed']and out['status']=='cubic32_remaining_complement_covered'and pc['status']=='reviewed'and pc['ordinal']==32 and pc['domains']==domains()and pc['parameter_endpoints']==['55/1024','103/1024']and pc['maximum_centers']==90 and pc['maximum_depth']==3,'fixed entire producer remaining portion')
 P.card_check(pc);need(pc['caps']==rt['caps']==P.CAPS,'fixed exact source caps');pairs(run,0,144);dd,selected=tree_history(out,run);prof=read(run/'batch-profile.json');need(prof['passed']and prof['maximum_total_calls']==12024 and prof['forecast_seconds']<9000 and prof['forecast_bytes']<18*1024**3,'complete measured original admission')
 original,prior=P.metadata(w,pc);need(pc['first_independent']==card['measured_prior'],'literal independent whole first-portion authority');data=G.metadata(original,w.tick);ib=read(run/'context/input-binding.json')
 need(ib['index']==data['index']and ib['prior_card']==data['independent_binding']['card']and ib['prior_stable_authority']==data['prior']and ib['prior_projection']==data['anchor_projection']and ib['predictor']==original['predictor']and ib['coefficient_artifacts']==original['coefficients'],'full same root/source/gauge/prior spectrum binding')
 need(prior['parameter_endpoints']==['39/1024','55/1024']and pc['parameter_endpoints'][0]==prior['parameter_endpoints'][1]and prior['prior_stable_projection']==data['anchor_projection'],'closed first/new union and stable prior projection')
 for n in G.B.NATIVE+G.M.T.PATH_FILES:need(rt['sources'][n]==A.S.entry(ROOT/n),'identical actual native/path source '+n)
 w.save('authority-binding.json',dict(producer=card['producer'],first_independent=pc['first_independent'],original_root=data['index'],prior_stable=data['prior'],projection=data['anchor_projection'],domains=dd,cells=selected,inventories=inventory,authorship='Producer agent adapter reviewed/executed by root; shared P.metadata/G.metadata and coefficient decoding; unchanged separately authored independent root/tangent/point/quotient reductions.'))
 return run,dd,selected,data['anchor_projection'],data,original


def context(w,data,pc):
 domain,proof=data['domain'],data['proof'];proposal=read(data['predictor_path']);ip=data['coefficient_dir'];cfg=data['index']['producer']['settings']
 for k in ('center','slope','curvature','cubic'):need(proposal[k]==domain[k],'same original cubic coefficient')
 c,s,a,b=[[R.exact(M.point(v))for v in domain[k]]for k in('center','slope','curvature','cubic')];normal=[R.exact(F(x))for x in proof['chart']['normal']];origin=[R.exact(F(x))for x in proof['chart']['origin']]
 radii=[R.exact(M.point(v))for v in proof['root']['radii']];weights=[v/arb((1,-10))for v in radii];original=arb_mat([[arb(0,v)]for v in radii]);image=arb_mat([[R.original_ball(v)]for v in proof['root']['image_correction']]);Q.cover(domain['error'],[original[i,0]for i in range(705)],'original fullE')
 C0=G.exact_inverse(proposal['preconditioner_exact']);C1=G.P.small_matrix(ip/'inverse_linear_proposal.json.gz',705);need(all(v.is_exact()for rr in C1.tolist()for v in rr),'exact independent C1')
 ds=[G.P.small_matrix(ip/f'defect_coefficient_{j}.json.gz',705)for j in range(4)];ff=[G.P.small_matrix(ip/f'forcing_coefficient_{j}.json.gz',1)for j in range(6)];rr=[G.P.small_matrix(ip/f'residual_coefficient_{j}.json.gz',1)for j in range(5)];rp=[(j+1)*rr[j+1]for j in range(4)]
 E,metrics=G.M.OLD.fixed_frames(data['inputs']['map_inputs'],read(ROOT/G.M.CARD));w.save('context-binding.json',dict(predictor=pc['predictor'],coefficients=pc['coefficients'],scope='Shared exact finite coefficient decoding only. Independent root/tangent/point/quotient reducers called separately; no model constructor or native calls.'))
 return dict(c=c,s=s,a=a,b=b,normal=normal,origin=origin,sc=R.exact(F(cfg['parameter'])),weights=weights,original=original,image=image,C=[C0,C1],ds=ds,ff=ff,rp=rp,E=E,metrics=metrics,domain=domain)

class Work:
 def __init__(self):self.start=time.monotonic();self.used=0;self.rows=[]
 def tick(self):need(time.monotonic()-self.start<1800 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'1800s4GiB saved audit')
 def save(self,name,value,closing=False):
  b=(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode();need(self.used+len(b)<(256 if closing else 252)*1024**2,'256MiB4MiB reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
  self.used+=len(b)


def main():
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean exact nice10 source');OUT.mkdir(parents=True,exist_ok=False);w=Work();result=dict(passed=False,native_calls=0)
 def stop(*_):raise TimeoutError('owned1800s saved audit')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,1800)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:A.S.entry(ROOT/n)for n in FILES},precision=480,native_calls=0,python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,threads=dict.fromkeys(THREADS,'1'),authorship='Producer agent adapter; unchanged separately authored independent proof reducers and root source review.'))
  for n in FILES:
   b=(ROOT/n).read_bytes();need(b==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed source');need(w.used+len(b)<252*1024**2,'bounded source retention');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);w.used+=len(b)
  with ctx.workprec(480):
   card=read(ROOT/CARD);before=time.monotonic();trial=A.A.synthetic(705,w.tick);seconds=time.monotonic()-before;prior=Path(card['measured_prior']['root']);need(A.S.entry(prior/'run/manifest.json')['sha256']==card['measured_prior']['worker'],'literal measured full firstquarter');pm=read(prior/'run/manifest.json');need(A.S.entry(prior/'run/outcome.json')==pm['files']['outcome.json'],'literal runtime floor');cost=read(prior/'run/outcome.json');forecast=8*seconds+24*cost['elapsed_seconds']+2*61.466+60;raw=w.used+24*sum(v['bytes']for v in pm['files'].values())+64*1024**2;ok=trial['passed']and forecast<1800 and raw<256*1024**2 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+512*1024<4*1024**2;w.save('profile.json',dict(passed=ok,seconds=seconds,forecast_seconds=forecast,forecast_bytes=raw,target_arrays_unopened=True));need(ok,'full705 measured same-size replay admission');gc.collect()
   run,dd,selected,projection,original,pc=metadata(w,card);data=context(w,original,pc)
   for ds in dd:
    shared=C.shared_domain(w,run,ds,data)
    for item in selected:
     if item['cell']['domain']!=ds['domain']['index']:continue
     row=C.one(w,run,item,data,shared);w.rows.append(row);w.save(f'prefix-{len(w.rows):02d}.json',dict(rows=w.rows));gc.collect()
   joined=union(w.rows,projection);w.save('union.json',joined);w.tick();result.update(**joined,scope='Independent all11 complementary contraction on remaining chart32 with exact closed union to independently qualified first portion and prior chart31 stable projection. Neutral critical pair retained. Wholechart complementary stability also uses the literal independent first portion; no entire window or generic NS assertion.')
 except BaseException as e:result.update(passed=False,error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(completed_cells=w.rows,elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):A.S.entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
