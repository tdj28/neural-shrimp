"""Remaining-quarter binding adapter; unchanged independent secondquarter reducers.

Producer agent authors this adapter. Root independently reviews/runs it; all proof
reducers remain separately authored. Saved acquisitions/Arb are shared explicitly.
"""
import os
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gc,json,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'tools/reviews'),str(ROOT/'tools'),str(ROOT/'src')]
import review_second_quarter as S
A,Q,H,R,M,need,read=S.A,S.Q,S.H,S.R,S.M,S.need,S.read
ctx=S.ctx
OUT=Path('/data2/neural-shrimp-data/audits/native705-remaining-quarters-independent/attempt001/run')
CARD='docs/plans/native705-remaining-quarters-independent-inputs.json'
FILES=sorted(set(S.FILES+['tools/reviews/review_remaining_quarters.py','tools/reviews/launch_remaining_quarters_review.py','tests/reviews/test_remaining_quarters_review.py','docs/plans/native705-remaining-quarters-independent.md',CARD]))
SECOND=dict(root='/data2/neural-shrimp-data/audits/native705-second-quarter-independent/attempt001',source='c59e0957054eded4fd38dfef27463b5daac94cb8',outer='5a815a2533d67a1fd2b51b162357f6e79f6abb87f5475c4a95a41b29cc38d849',worker='7eaf5976b6a9caa362755b1d64c5a047daf37cdb5492f5bf1659e5b44c7579b4')


def domains(quarter):
 need(quarter in(3,4),'literal remaining quarter');off=32*(quarter-2)
 return [dict(index=i+1,endpoints=[str(F(a+off,2048)),str(F(b+off,2048))])for i,(a,b)in enumerate([(-17,-1),(-1,15)])]


def cell(quarter,domain,depth,index):
 need(domain in(1,2)and type(depth)is int and 0<=depth<=3 and type(index)is int and 0<=index<2**depth,'exact bounded binary cell');a,b=map(F,domains(quarter)[domain-1]['endpoints']);width=(b-a)/2**depth;a+=index*width;b=a+width
 return dict(domain=domain,depth=depth,index=index,endpoints=[str(a),str(b)],center=str((a+b)/2),halfwidth=str((b-a)/2))


def union(rows,prior):
 need(prior['passed']and prior['parameter_endpoints']==['-17/2048','15/2048']and prior['first_and_second_union']==['-49/2048','15/2048']and prior['complement_dimension']==11,'exact independently proved adjacent prefix');unions=[]
 for q in(3,4):
  selected=[r for r in rows if r['quarter']==q];need(selected and len(selected)<=16 and all(r['passed']and r['complement_dimension']==11 for r in selected),'complete independent all11 leaves');ranges=[]
  for r in selected:
   spec=r['cell'];need(spec==cell(q,spec['domain'],spec['depth'],spec['index']),'literal quarter cell');ranges.append(tuple(map(F,spec['endpoints'])))
  ranges.sort();expected=[domains(q)[0]['endpoints'][0],domains(q)[1]['endpoints'][1]];need([str(ranges[0][0]),str(ranges[-1][1])]==expected and all(a[1]==b[0]for a,b in zip(ranges,ranges[1:])),'exact closed nonoverlapping quarter union');unions.append(dict(quarter=q,parameter_endpoints=expected,total_cells=len(selected)))
 need(len(rows)==sum(u['total_cells']for u in unions)and unions[0]['parameter_endpoints'][0]==prior['parameter_endpoints'][1]and unions[0]['parameter_endpoints'][1]==unions[1]['parameter_endpoints'][0],'all rows and exact preceding adjacency');projection=prior['prior_stable_projection'];lo,hi=map(F,projection);need(F(-17,2048)<=lo<=hi<=F(15,2048),'entire original stable projection retained')
 return dict(passed=True,parameter_endpoints=['15/2048','79/2048'],whole_chart_endpoints=['-49/2048','79/2048'],prior_stable_projection=projection,complement_dimension=11,total_cells=len(rows),quarters=unions)


def tree_history(out,run):
 attempts=out['completed_attempts'];need(4<=len(attempts)<=60,'finite complete producer tree');cursor=144;idx=0;selected=[];all_domains=[]
 for q in(3,4):
  qstart=idx
  for d in domains(q):
   folder=run/f"quarters/{q}/domains/{d['index']:03d}";dom=read(folder/'domain-outcome.json');need(dom==dict(passed=True,domain=d,dimension=705,native_calls=90,first_call=cursor+1,last_call=cursor+90),'exact90-phase domain');S.pairs(folder,cursor,90);all_domains.append(dict(quarter=q,domain=d,start=cursor));cursor+=90;queue=[cell(q,d['index'],0,0)]
   while queue:
    need(idx<len(attempts),'complete binary subtree');spec=queue.pop(0);r=attempts[idx];idx+=1;f=run/f'cells/{idx:03d}'
    need(r['quarter']==q and r['cell']==spec and r['completed']and r['native_calls']==126 and r['dimension']==705 and r['complement_dimension']==11 and r['first_call']==cursor+1 and r['last_call']==cursor+126,'complete literal bounded cell');need(read(f/'cell-input.json')==dict(quarter=q,cell=spec)and read(f/'cell-outcome.json')==r and read(run/f'completed-attempt-{idx:03d}.json')==r,'actual ordered cell records');S.pairs(f,cursor,126)
    if r['passed']:selected.append(dict(quarter=q,ordinal=idx,cell=spec,start=cursor))
    else:
     need(r['passed']is False and spec['depth']<3 and read(f/'complement.json')['passed']is False,'only completed strict-metric failure split');queue=[cell(q,spec['domain'],spec['depth']+1,2*spec['index']+k)for k in(0,1)]+queue
    cursor+=126
  need(idx-qstart<=30,'perquarter attempt cap')
 need(idx==len(attempts)and out['native_calls']==cursor==144+360+126*idx and out['profile_native_calls']==144 and out['target_native_calls']==cursor-144,'all8064 inclusive calls');need(out['accepted_cells']==[attempts[r['ordinal']-1]for r in selected],'complete accepted list');return all_domains,selected


def metadata(w,card):
 need(card['status']=='reviewed'and card['second_quarter']==SECOND and isinstance(card['producer'],dict)and set(card['producer'])=={'root','source','outer','worker'}and all(isinstance(v,str)and v for v in card['producer'].values()),'completed exact producer and independent prefix required')
 root,rt,out,iv=A.S.bundle(card['producer'],w.tick);second,srt,so,siv=A.S.bundle(SECOND,w.tick);run=root/'run';pc=read(run/'source/docs/plans/native705-remaining-quarters-inputs.json');sib=read(second/'run/authority-binding.json')
 need(out['passed']and out['status']=='remaining_chart31_quarters_complement_covered'and pc['status']=='reviewed'and pc['quarters']==[3,4]and pc['domains']=={str(q):domains(q)for q in(3,4)}and pc['second_quarter_independent']==SECOND,'literal finaltwoquarter producer card');need(so['passed']and so['native_calls']==0 and so['parameter_endpoints']==['-17/2048','15/2048']and so['first_and_second_union']==['-49/2048','15/2048']and so['complement_dimension']==11,'completed independent joined prefix');need(sib['producer']==pc['second_quarter_producer'],'same secondquarter producer')
 sb=read(run/'second-quarter-binding.json');need(sb['producer']==sib['producer']and sb['independent']==SECOND and sb['independent_outcome']==so and sb['independent_binding']==sib,'actual prior independent authority binding')
 first,frt,fo,fiv=A.S.bundle(sib['first_quarter'],w.tick);fib=read(first/'run/authority-binding.json');need(fo['passed']and fo['parameter_endpoints']==['-49/2048','-17/2048']and fo['complement_dimension']==11 and fib['original_whole']['outer']==A.OLD['outer']and fib['pilot']==A.PARENT,'same original independently proved firstquarter/root');need(read(run/'authority-binding.json')['first']==sib['first_quarter'],'same producer root ancestry')
 caps=dict(seconds=6000,rss_bytes=4*1024**3,as_bytes=6*1024**3,raw_bytes=12*1024**3,profile_calls=144,maximum_target_calls=7920,maximum_total_calls=8064);need(pc['caps']==rt['caps']==caps,'literal original caps');S.pairs(run,0,144);dd,selected=tree_history(out,run);prof=read(run/'batch-profile.json');need(prof['passed']and prof['maximum_centers']==60 and prof['maximum_total_calls']==8064 and prof['forecast_seconds']<6000 and prof['forecast_bytes']<12*1024**3,'actual measured worstcase admission')
 for n in A.B.B.NATIVE+A.B.T.PATH_FILES:need(rt['sources'][n]==A.S.entry(ROOT/n),'unchanged actual native/path '+n)
 w.save('authority-binding.json',dict(producer=card['producer'],second_quarter=SECOND,first_quarter=sib['first_quarter'],prior_outcome=so,domains=dd,cells=selected,inventories=iv+siv+fiv,authorship='Producer agent binding adapter; unchanged separate independent reducers, root source review and execution.'));return run,dd,selected,so


class Work(S.Work):
 def __init__(self):super().__init__();self.prefix=''
 def tick(self):need(time.monotonic()-self.start<2400 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'2400s4GiB saved replay')
 def save(self,name,value,closing=False):
  if self.prefix:name=self.prefix+'/'+name
  b=(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode();need(self.used+len(b)<(512 if closing else 508)*1024**2,'512MiB4MiBreserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
  self.used+=len(b)


def main():
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10 source');OUT.mkdir(parents=True,exist_ok=False);w=Work();result=dict(passed=False,native_calls=0)
 def stop(*_):raise TimeoutError('owned2400s saved replay')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,2400)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:A.S.entry(ROOT/n)for n in FILES},precision=480,native_calls=0,python=platform.python_version(),python_flint=S.flint.__version__,flint=S.flint.__FLINT_VERSION__,threads=dict.fromkeys(THREADS,'1'),authorship='Producer agent adapter; unchanged separately authored independent reducers, root source review.'))
  for n in FILES:
   b=(ROOT/n).read_bytes();need(b==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact source');need(w.used+len(b)<508*1024**2,'bounded source');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);w.used+=len(b)
  with ctx.workprec(480):
   card=read(ROOT/CARD);before=time.monotonic();trial=A.A.synthetic(705,w.tick);seconds=time.monotonic()-before;prior=Path(SECOND['root']);need(A.S.entry(prior/'run/manifest.json')['sha256']==SECOND['worker'],'actual prior cost source');pm=read(prior/'run/manifest.json');need(A.S.entry(prior/'run/outcome.json')==pm['files']['outcome.json'],'actual outcome cost');cost=read(prior/'run/outcome.json');forecast=8*seconds+16*cost['elapsed_seconds']+2*61.466+60;raw=w.used+8*sum(v['bytes']for v in pm['files'].values())+64*1024**2;ok=trial['passed']and forecast<2400 and raw<512*1024**2 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+512*1024<4*1024**2;w.save('profile.json',dict(passed=ok,seconds=seconds,forecast_seconds=forecast,forecast_bytes=raw,target_arrays_unopened=True));need(ok,'full705 measured replay admission');gc.collect()
   run,dd,selected,prior=metadata(w,card);data=S.context(w)
   for ds in dd:
    w.prefix=f"quarters/{ds['quarter']}";shared=S.shared_domain(w,run/f"quarters/{ds['quarter']}",ds,data);w.prefix=''
    for item in selected:
     if item['quarter']!=ds['quarter']or item['cell']['domain']!=ds['domain']['index']:continue
     row=S.one(w,run,item,data,shared);row['quarter']=item['quarter'];w.rows.append(row);w.save(f'prefix-{len(w.rows):02d}.json',dict(rows=w.rows));gc.collect()
   joined=union(w.rows,prior);w.save('union.json',joined);result.update(**joined,scope='Independent uniform all11 complementary contraction on entire chart31, joined to prior30 via independently qualified secondquarter projection. Critical unit pair neutral. No completewindow or generic NS assertion.')
 except BaseException as e:result.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  w.prefix='';signal.setitimer(signal.ITIMER_REAL,0);result.update(completed_cells=w.rows,elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):A.S.entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
