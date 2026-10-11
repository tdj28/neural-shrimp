"""Saved-only independently reduced complete signed470 march prefix."""
import os,resource
if __name__=='__main__':resource.setrlimit(resource.RLIMIT_AS,(10*1024**3,)*2)
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gc,gzip,json,platform,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
import flint
from flint import ctx,arb_mat
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import review_signed_correlated as S
import review_signed_march_math as T
from neural_shrimp.rigorous_orbits import canonical_bytes
need,read,entry,inventory,archive=S.need,S.read,S.entry,S.inventory,S.archive
OUT=Path('/data2/neural-shrimp-data/audits/signed470-march-independent/attempt001/run');CARD='docs/plans/native470-march-independent-inputs.json'
LIMITS=dict(seconds=4800,rss_bytes=6*1024**3,as_bytes=10*1024**3,raw_bytes=12*1024**3,precision=512,native_calls=0,unit_seconds=300,unit_raw_bytes=768*1024**2)
FILES=sorted(set(S.FILES+[CARD,'docs/plans/native470-march-independent.md','tools/reviews/review_signed_march.py','tools/reviews/review_signed_march_math.py','tools/launch_signed_march_review.py','tests/reviews/test_signed_march_review.py']))
class Work:
 def __init__(self):self.start=time.monotonic();self.unit_start=self.start;self.used=0;self.unit_used=0;self.stage='profile';self.prefix='';self.stopped=False
 def tick(self):need(not self.stopped and time.monotonic()-self.start<4800 and time.monotonic()-self.unit_start<300 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<6*1024**2,'4800s/300s perunit/6GiB saved audit')
 def check_bytes(self,added,closing=False):need(self.used+added<12*1024**3-(0 if closing else 16*1024**2)and self.unit_used+added<768*1024**2-(0 if closing else 4*1024**2),'complete and perunit audit storage with closing reserve')
 def save(self,n,v,closing=False):
  raw=canonical_bytes(v);self.check_bytes(len(raw),closing);p=OUT/self.prefix/n;p.parent.mkdir(parents=True,exist_ok=True)
  temp=p.with_name(p.name+'.part')if n=='unit-start.json'else p
  with temp.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  if temp!=p:os.link(temp,p);temp.unlink()
  self.used+=len(raw);self.unit_used+=len(raw)
 def matrix(self,n,a):
  need(S.H.finite(a),'finite derived matrix');p=OUT/self.prefix/n;p.parent.mkdir(parents=True,exist_ok=True)
  try:
   with p.open('xb')as raw:
    with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0)as f:
     f.write(b'{"encoding":"mid-rad-dyadic-pairs","rows":[')
     for i in range(a.nrows()):
      row=[[[int(t)for t in a[i,j].mid().man_exp()],[int(t)for t in a[i,j].rad().man_exp()]]for j in range(a.ncols())];f.write(((','if i else'')+json.dumps(row,separators=(',',':'))).encode());self.check_bytes(raw.tell());self.tick()
     f.write(b']}')
    raw.flush();os.fsync(raw.fileno())
  finally:
   size=p.stat().st_size if p.exists()else 0;self.used+=size;self.unit_used+=size
  self.check_bytes(0)
 def emit(self,k,v):
  if isinstance(v,arb_mat):self.matrix(self.stage+'/'+k+'.json.gz',v)
  else:self.save(self.stage+'/'+k+'.json',v)
  self.tick()
 def unit(self,i):
  self.prefix=f'chart{i:03d}';self.unit_start=time.monotonic();self.unit_used=0;self.stage='target';self.save('unit-start.json',dict(index=i,started_monotonic=self.unit_start,maximum_seconds=300))

class Prefix:
 def __init__(self,w,p):self.w=w;self.p=p
 def tick(self):self.w.tick()
 def save(self,n,v,closing=False):self.w.save(self.p+n,v,closing)

def check_history(out,rows,inputs,start):
 need(len(rows)==len(inputs)==len(out['attempts'])and 1<=len(rows)<=24,'entire bounded attempts')
 accepted=[];current=F(start);width=-3;strong=0;previous=None;terminal=False
 for i,(row,inp,record)in enumerate(zip(rows,inputs,out['attempts'],strict=True),1):
  need(not terminal and row==record and row['index']==i and inp['previous_accepted']==previous,'ordered complete attempted history')
  h=F(2)**width;p=current-h;ends=[str(current-2*h),str(current)]
  need(inp['parameter']==str(p)and inp['parameter_endpoints']==ends and inp['known']['exact_parameter']==str(current)and inp['settings']==dict(halfwidth_exponent=width,radius_exponent=-12),'literal current known scalar and proposed domain')
  if row.get('status')=='incomplete_or_guard_failure':terminal=True;continue
  need(row['native_calls']==144 and row['ticks']==150 and row['parameter_endpoints']==ends,'complete mathematical attempt')
  g=row['gates'];goodroot=g['uniform_normalized_eigenpair_locus'];joined=g['anchor_identified'];primitive=g['primitive_period']
  expected='advance'if goodroot and joined and primitive else 'retry'if not goodroot and joined and primitive and width>-8 else 'stop'
  need(row['decision']['action']==expected,'only complete associated root-bound failure can retry')
  if expected=='advance':
   need(row['passed']and row['strict_inclusions']==row['anchor_inclusions']==470,'entire original accepted proof/connection')
   accepted.append(i);previous=i;current-=2*h;lo,hi=S.M.interval(row['contraction']);good=hi<F(1,5)
   if good and strong:width=min(-1,width+1);strong=0
   else:strong=int(good)
  elif expected=='retry':width-=1;strong=0
  else:terminal=True;strong=0
  need(row['decision']['width']==width and row['decision']['strong']==strong,'actual disclosed adaptive policy')
 need(accepted==out['accepted_indices']and accepted and out['connected']and out['last_known_parameter']==str(current),'exact promoted finite prefix only')
 need(out['native_calls']==180+sum(r['native_calls']for r in rows)and out['ticks']==186+sum(r['ticks']for r in rows),'inclusive fullprofile plus all retained attempts')
 return accepted

def bind(card,w,metadata_only=False):
 need(card['status']=='reviewed'and card['limits']==LIMITS,'fixed reviewed saved-prefix audit')
 prior=card['known'];ap=Path(prior['root']);spec=card['producer'];p=Path(spec['root'])
 for root,s,expected in((ap,prior,0),(p,spec,spec['exit_code'])):
  inventory(root,s['outer']['sha256'],w);inventory(root/'run',s['worker']['sha256'],w);archive(root,s['source'],w);la=read(root/'launcher-result.json');need(la['exit_code']==expected and la['failure']is None and la['worker_pid_absent'],'sealed owned prior run')
 ai=read(ap/'run/input-binding.json');ao=read(ap/'run/outcome.json');need(ao==prior['outcome']and ao['passed']and ao['root_coordinates']==ao['anchor_coordinates']==470 and ao['native_calls']==0,'entire independently known signed470 root')
 S.bind(ai['card'],Prefix(w,'ancestry-'),True)
 for n,k in [('connected-left-endpoint.json','endpoint'),('independent512.json','proof')]:need(entry(ap/'run'/n)==prior[k],'literal exact independent endpoint '+n)
 rt=read(p/'run/runtime.json');ib=read(p/'run/input-binding.json');out=read(p/'run/outcome.json');pr=read(p/'run/profile.json');ad=read(p/'run/whole-packet-admission.json')
 need(out==spec['outcome']and ib['card']['independent']==prior and pr['passed']and pr['native_calls']==180 and pr['ticks']==186 and ad['passed'],'actual full ancestry/profile/wholepacket')
 for n,e in ai['native_source'].items():need(rt['sources'][n]==e==ib['native_source'][n],'literal unchanged actual native map '+n)
 need(ib['model']==ai['model'],'complete same model/data');need(rt['policy']==ib['card']['policy']==dict(max_attempts=24,initial_width=-3,minimum_width=-8,maximum_width=-1,strong_contraction='1/5',strong_passes=2,radius_exponent=-12,newton_steps=2,precision=256,period=18,sigma=-1,hidden_stop='76'),'literal complete actual execution policy');rows=[];inputs=[];raw_known=ib['known']
 dirs=sorted((p/'run').glob('attempt*'));need([d.name for d in dirs]==[f'attempt{i:03d}'for i in range(1,len(out['attempts'])+1)],'no skipped retained attempt directories')
 for d in dirs:
  path=d/('outcome.json'if(d/'outcome.json').exists()else'incomplete.json');row=read(path);inp=read(d/'inputs.json');rows.append(row);inputs.append(inp)
  need(inp['source_card']==rt['sources']['docs/plans/native470-correlated-march-inputs.json']and inp['model']==ib['model']and inp['normal']==ib['known']['normal']and inp['origin']==ib['known']['origin']and inp['weights']==ib['weights'],'exact perattempt source/map/gauge/weights')
  need(inp['known']==raw_known,'literal full470 producer predecessor after each attempted unit')
  if 'certificate'in row:need(entry(d/'target-certificate.json.gz')==row['certificate'],'literal complete attempted certificate')
  if row.get('decision',{}).get('action')=='advance':raw_known=read(d/'connected-left-endpoint.json')
 accepted=check_history(out,rows,inputs,'-81/256');need(accepted==card['accepted_indices'],'fixed whole earned prefix')
 w.save('input-binding.json',dict(card=card,model=ib['model'],native_source=ai['native_source'],accepted_indices=accepted,all_attempts=len(rows),scope='Same exact hidden gauge; actual derivative associations and independent full470 roots/joins replayed for each accepted unit.'))
 if metadata_only:return p,ap
 endpoint=read(ap/'run/connected-left-endpoint.json');proof=read(ap/'run/independent512.json');need(endpoint['box']==proof['recomputed_endpoints'][0]and endpoint['exact_parameter']=='-81/256'and endpoint['normal']==ib['known']['normal']and endpoint['origin']==ib['known']['origin'],'actual independently reconstructed seed');return p,endpoint

def admission(card,w):
 pr=read(OUT/'profile.json');count=len(card['accepted_indices']);old=card['known']['outcome'];oldbytes=sum(e['bytes']for e in read(Path(card['known']['root'])/'run/manifest.json')['files'].values());unit_seconds=max(pr['forecast']['seconds'],2*old['seconds']+20);unit_bytes=max(pr['forecast']['bytes'],2*oldbytes+32*1024**2);forecast=dict(seconds=time.monotonic()-w.start+count*unit_seconds,bytes=w.used+count*unit_bytes+16*1024**2,rss_kib=max(pr['forecast']['rss_kib'],2*old['peak_rss_kib']+256*1024),unit_seconds=unit_seconds,unit_bytes=unit_bytes);ok=forecast['seconds']<4800 and forecast['bytes']<12*1024**3 and forecast['rss_kib']<6*1024**2 and unit_seconds<300 and unit_bytes<768*1024**2;w.save('whole-prefix-admission.json',dict(passed=ok,forecast=forecast,count=count,prior_actual=old,target_arrays_unopened=True));need(ok,'entire actual earned-prefix forecast')

def main():
 need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10 source');OUT.mkdir(parents=True,exist_ok=False);w=Work();answer=dict(passed=False,native_calls=0,accepted_indices=[])
 def stop(*_):w.stopped=True
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,4800)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),limits=LIMITS,sources={n:entry(ROOT/n)for n in FILES},python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,threads=dict.fromkeys(THREADS,'1')))
  for n in FILES:
   data=(ROOT/n).read_bytes();need(data==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact source archive');q=OUT/'source'/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(data);w.used+=len(data);w.unit_used+=len(data)
  card=read(ROOT/CARD);need(card['status']=='reviewed'and card['limits']==LIMITS,'fixed reviewed card before profile')
  with ctx.workprec(512):
   S.synthetic(w);admission(card,w);p,known=bind(card,w);rows=[]
   for i in card['accepted_indices']:
    w.unit(i);known,row=T.replay_one(p/'run'/f'attempt{i:03d}',known,w);rows.append(dict(index=i,**row));w.save('outcome.json',rows[-1]);need(row['passed']and row['root_coordinates']==row['anchor_coordinates']==470,'complete independently earned root and join');answer['accepted_indices'].append(i);gc.collect()
   w.prefix='';w.unit_used=0;w.unit_start=time.monotonic();w.save('connected-left-endpoint.json',known);answer.update(passed=True,rows=rows,exact_parameter=known['exact_parameter'],scope='Independent saved512 fullE arithmetic and ordered native470 joins; shared native derivative implementation/Arb. No complementary stability/genericity/whole-window claim.')
 except BaseException as e:answer.update(passed=False,error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);elapsed=time.monotonic()-w.start;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;answer['passed']=bool(answer['passed']and elapsed<4800 and peak<6*1024**2);answer.update(seconds=elapsed,peak_rss_kib=peak);w.prefix='';w.unit_used=0;w.save('outcome.json',answer,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(answer));return 0 if answer['passed']else 2
if __name__=='__main__':raise SystemExit(main())
