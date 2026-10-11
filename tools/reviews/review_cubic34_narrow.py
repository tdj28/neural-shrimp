"""Accepted chart34 prefix saved adapter: producer-authored binding, independent proof reducers."""
import os
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gc,json,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'tools/reviews'),str(ROOT/'tools'),str(ROOT/'src')]
import flint
from flint import ctx
import review_cubic_panel as B
import probe_native705_cubic34_narrow as N
C,A,need,read=B.C,B.B,B.need,B.read
OUT=Path('/data2/neural-shrimp-data/audits/native705-cubic34-narrow-independent/attempt001/run');CARD='docs/plans/native705-cubic34-narrow-independent-inputs.json'
LIMITS=dict(seconds=3600,rss_bytes=4*1024**3,as_bytes=6*1024**3,raw_bytes=1024*1024**2,native_calls=0)
PREFIX=['83/512','57/256']
FAILED=[1,4,7,66,69,72,75,76]
FILES=sorted(set(B.FILES+N.FILES+['tools/reviews/review_cubic34_narrow.py','tools/reviews/launch_cubic34_narrow_review.py','tests/reviews/test_cubic34_narrow_review.py','docs/plans/native705-cubic34-narrow-independent.md',CARD]))

def card_check(c):
 need(c['status']=='reviewed'and c['limits']==LIMITS and c['ordinal']==34 and c['maximum_domains']==64 and c['maximum_centers']==192,'literal reviewed saved whole34')
 need(c['target_endpoints']==PREFIX and c['expected']==dict(domains=63,attempts=76,accepted=68,native_calls=15390,failed=FAILED),'literal earned prefix scope')
 need(c['previous']==N.INDEPENDENT,'exact completed whole33 authority')
 need(set(c['producer'])=={'root','source','outer','worker'}and all(type(v)is str and v for v in c['producer'].values())and c['producer']['root']==str(N.OUT.parent),'complete literal producer pins')

def history(out,run,pc):
 need(out['passed']is False and out['status']=='failed_or_incomplete'and out['error']=='ValueError: fixed maximum cell depth1'and out['domains_done']==63,'literal preserved terminal depth1 failure')
 domains=pc['domains'];need(domains==N.H.partition(),'exact full34 source partition');rows=out['completed_attempts'];need(len(rows)==76,'complete literal76 attempts')
 A.pairs(run,0,144);cursor=144;index=0;stages=[];selected=[]
 for domain in domains[:63]:
  part=run/f"domains/{domain['index']:03d}";dom=read(part/'domain-outcome.json');need(dom==dict(passed=True,domain=domain,dimension=705,native_calls=90,first_call=cursor+1,last_call=cursor+90),'literal successive90-call domain');A.pairs(part,cursor,90);stages.append(dict(domain=domain,start=cursor));cursor+=90;queue=[N.C.H.cell(domains,domain['index'])]
  while queue:
   need(index<len(rows),'complete depth1 subtree');cell=queue.pop(0);ordinal=index+1;row=rows[index];index+=1;part=run/f'cells/{ordinal:03d}'
   need(row['cell']==cell and row['completed']and row['native_calls']==126 and row['dimension']==705 and row['complement_dimension']==11 and row['first_call']==cursor+1 and row['last_call']==cursor+126,'ordered full705 fresh center')
   need(read(part/'cell-input.json')==cell and read(part/'cell-outcome.json')==row and read(run/f'completed-attempt-{ordinal:03d}.json')==row,'literal complete center records');A.pairs(part,cursor,126);cursor+=126
   if row['passed']:
    need(domain['index']<=62,'no terminal domain63 promotion');selected.append(dict(ordinal=ordinal,cell=cell,start=cursor-126))
   else:
    need(row['passed']is False and read(part/'complement.json')['passed']is False,'only complete negative11-mode metric')
    if cell['depth']==1:
     need(domain['index']==63 and ordinal==len(rows)==76 and cell['index']==0 and out['pending_cells']==[cell]+queue,'exact terminal failed child and retained right sibling');break
    queue=N.H.children(domains,cell,row)+queue
 need(index==len(rows)and out['accepted_cells']==[rows[x['ordinal']-1]for x in selected],'every accepted/rejected center retained')
 need(len(selected)==68 and [i+1 for i,r in enumerate(rows)if r['passed']is False]==FAILED,'literal all68 positives and eight negatives')
 need(out['native_calls']==cursor==144+63*90+126*len(rows)==15390 and out['profile_native_calls']==144 and out['target_native_calls']==cursor-144,'inclusive complete30096 acquisition grammar')
 B.exact_union(out['accepted_cells'],PREFIX);need('union'not in out and not(run/'union.json').exists()and not(run/'domains/064').exists(),'no invented whole34 union or domain64 acquisition')
 return stages,selected

def prefix_projection(projection):
 lo,hi=map(B.F,projection);a,b=map(B.F,PREFIX)
 need(a<=lo<=hi<=b,'entire independently joined whole33 projection inside earned prefix')
 return dict(projection=projection,endpoints=PREFIX,passed=True)


def metadata(w,c):
 card_check(c);root,rt,out,inventory=A.A.S.bundle(c['producer'],w.tick,code=2);run=root/'run';pc=read(run/'source'/N.CARD);N.card_check(pc)
 need(pc['independent_previous']==c['previous']and rt['caps']==N.CAPS,'exact prerequisite and producer resources')
 for n in N.P.G.B.NATIVE+N.P.G.M.T.PATH_FILES:need(rt['sources'][n]==A.A.S.entry(ROOT/n),'same acquired native/path source')
 for n in ('tools/probe_native705_cubic34_narrow.py','src/neural_shrimp/cubic_complement_narrow.py'):
  need(rt['sources'][n]==A.A.S.entry(ROOT/n),'literal producer history adapter '+n)
 profile=read(run/'batch-profile.json');need(profile['passed']and profile['maximum_total_calls']==30096,'full worst-case admission before arrays')
 stages,cells=history(out,run,pc);w.prefix='';previous=N.bind(w,pc);original=N.G.metadata(pc['chart'],previous,w.tick);saved=read(run/'context/input-binding.json')
 need(saved['index']==original['index']and saved['prior_card']==original['independent_binding']['card']and saved['prior_stable_authority']==previous and saved['prior_projection']==original['anchor_projection']and saved['predictor']==pc['chart']['predictor']and saved['coefficient_artifacts']==pc['chart']['coefficients'],'same original full705 root/predictor/phase/gauge genealogy')
 joined=prefix_projection(original['anchor_projection'])
 w.prefix='';w.save('authority-binding.json',dict(producer=c['producer'],previous=c['previous'],chart=pc['chart'],inventories=inventory,domains=stages,cells=cells,retained_failed_attempts=FAILED,pending_cells=out['pending_cells'],prefix_endpoints=PREFIX,stable_prefix_join=joined,whole_chart_endpoints=pc['chart']['parameter_endpoints'],authorship='Producer-agent adapter; independently authored exact-rational/root/tangent/point/quotient reducers unchanged and root-reviewed. Zero new native evaluations.'))
 return run,stages,cells,pc['chart'],original

class Work(B.Work):
 def tick(self):need(time.monotonic()-self.start<3600 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'3600s4GiB independent whole34')
 def save(self,name,value,closing=False):
  raw=(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode();need(self.used+len(raw)<(1024 if closing else 1020)*1024**2,'1GiB4MiB reserve');p=OUT/self.prefix/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)

def main():
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean CPU1nice10');c=read(ROOT/CARD);card_check(c);OUT.mkdir(parents=True,exist_ok=False);w=Work();result=dict(passed=False,native_calls=0)
 def stop(*_):raise TimeoutError('owned3600s saved audit')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,3600)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:A.A.S.entry(ROOT/n)for n in FILES},limits=LIMITS,precision=480,native_calls=0,python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,threads=dict.fromkeys(THREADS,'1')))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed source');need(w.used+len(raw)<1020*1024**2,'source reserve');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(480):
   before=time.monotonic();trial=A.A.A.synthetic(705,w.tick);seconds=time.monotonic()-before;spec=c['previous'];pr=Path(spec['root'])/'run';need(A.A.S.entry(pr/'manifest.json')['sha256']==spec['worker'],'literal whole33 cost inventory');pm=read(pr/'manifest.json')['files'];need(A.A.S.entry(pr/'outcome.json')==pm['outcome.json'],'complete prior cost');prior=read(pr/'outcome.json');need(prior['passed']and len(prior['completed_cells'])==39 and prior['native_calls']==0,'independent complete39-cell reference')
   ratio=max(64/8,192/39);wall=time.monotonic()-w.start+8*seconds+2*ratio*prior['elapsed_seconds']+2*61.466+60;raw=w.used+2*ratio*sum(v['bytes']for v in pm.values())+64*1024**2;ok=trial['passed']and wall<3600 and raw<1024*1024**2;w.save('profile.json',dict(passed=ok,seconds=seconds,forecast_seconds=wall,forecast_bytes=raw,maximum_domains=64,maximum_centers=192,target_arrays_unopened=True));need(ok,'full worst-case192-center independent admission');gc.collect()
   run,stages,cells,spec,original=metadata(w,c);w.prefix='context';data=A.context(w,original,spec);w.prefix=''
   for domain in stages:
    if domain['domain']['index']==63:continue
    shared=C.shared_domain(w,run,domain,data)
    for item in cells:
     if item['cell']['domain']!=domain['domain']['index']:continue
     row=C.one(w,run,item,data,shared);w.rows.append(row);w.prefix='';w.save(f'prefix-{len(w.rows):03d}.json',dict(completed_cells=w.rows));gc.collect()
   union=B.exact_union(w.rows,PREFIX);w.prefix='';w.save('union.json',union);need(len(w.rows)==len(cells)==68 and union['endpoints']==PREFIX,'entire independently replayed earned34 prefix');w.tick();result.update(passed=True,complement_dimension=11,union=union,scope='Independent full11 complementary stability only on exact existing cubic34 prefix [83/512,57/256], joined by original full705 uniqueness to independently qualified whole33. All eight negative metrics and unacquired tail remain unqualified. Critical pair remains neutral; no whole34, whole-window or final-junction claim.')
 except BaseException as exc:result.update(passed=False,error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);w.prefix='';elapsed=time.monotonic()-w.start;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;result['passed']=bool(result['passed']and elapsed<LIMITS['seconds']and peak*1024<LIMITS['rss_bytes']);result.update(completed_cells=w.rows,elapsed_seconds=elapsed,peak_rss_kib=peak);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):A.A.S.entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
