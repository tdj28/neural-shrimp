"""Saved independent replay of one advancing signed470 chart; no native calls."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import review_signed470 as C
from review_signed470 import (need,read,entry,inventory,archive,profile,Work,ctx,os,resource,subprocess,signal,time,platform,flint,hashlib,traceback,json,THREADS)
OUT=Path('/data2/neural-shrimp-data/audits/signed470-advancing-independent/attempt001/run');CARD='docs/plans/native470-advancing-independent-inputs.json'
FILES=sorted(set(C.FILES+[CARD,'docs/plans/native470-advancing-independent.md','tools/reviews/review_signed470_advancing.py','tools/launch_signed470_advancing_review.py','tests/reviews/test_signed470_advancing_review.py']))

def same_gauge(new,old):
 need(new['normal']==old['normal']==['0']*468+['1','0']and new['origin']==old['origin'],'literal identical native hidden gauge')
 need(new['parameter']=='-9/256'and old['parameter']=='0'and new['weights']==old['weights'],'literal parameters and weights')
 return '-1/256'

def binding(card,w):
 need(card['status']=='reviewed'and card['schema']=='independent-advancing-signed470-v1'and card['precision']==512,'fixed reviewed advancing audit')
 known=card['known'];ap=Path(known['root'])
 for b,k in ((ap,'outer'),(ap/'run','worker')):inventory(b,known[k]['sha256'],w)
 archive(ap,known['source'],w);out=read(ap/'run/outcome.json');ib=read(ap/'run/input-binding.json');need(out==known['outcome']and out['passed']and out['root_coordinates']==out['anchor_coordinates']==470 and out['native_calls']==0 and ib==known['binding']and entry(ap/'run/independent512.json')==known['proof'],'complete exact independent known endpoint authority')
 class Prior:
  def tick(self):w.tick()
  def save(self,name,row,closing=False):w.save('prior-'+name,row,closing)
 oldp,parent,oldinp=C.binding(ib['card'],Prior())
 spec=card['producer'];p=Path(spec['root'])
 for b,k in ((p,'outer'),(p/'run','worker')):inventory(b,spec[k]['sha256'],w)
 rt=archive(p,spec['source'],w);la=read(p/'launcher-result.json');o=read(p/'run/outcome.json');need(la['exit_code']==0 and la['failure']is None and o==spec['outcome']and o['passed']and o['strict_inclusions']==o['anchor_inclusions']==470 and o['native_calls']==276 and o['ticks']==278 and o['events']==842,'complete fresh advancing native acquisition')
 for name,key in [('target-certificate.json.gz','certificate'),('target-inputs.json','inputs'),('continuation-binding.json','binding')]:need(entry(p/'run'/name)==spec[key],'exact producer acquired array/input '+name)
 inp=read(p/'run/target-inputs.json');need(inp['settings']==dict(period=18,sigma=-1,halfwidth_exponent=-5,radius_exponent=-14,newton_steps=2,precision=256)and inp['map_inputs']==oldinp['map_inputs'],'actual new domain and unchanged full native map')
 oldrt=read(oldp/'run/runtime.json')
 for n,v in ib['native_source'].items():need(rt['sources'][n]==oldrt['sources'][n]==v,'same authenticated native derivative equations')
 scalar=same_gauge(inp,oldinp);cb=read(p/'run/continuation-binding.json');need(cb['authority']['proof']==ib['card']['producer']and cb['authority']['independent']==known,'producer joined exact independent predecessor')
 w.save('input-binding.json',dict(card=card,model=inp['map_inputs'],native_source=ib['native_source'],exact_old_parameter=scalar,scope='Fresh independently reduced full470 chart; previous independent complete left endpoint retained with literal identical gauge.'))
 return p,ap,inp,scalar

def replay(card,w):
 p,ap,inp,scalar=binding(card,w);old=read(ap/'run/independent512.json');need(old['passed']and old['inclusions']==old['anchor_coordinates']==470 and all(old['gates'].values())and len(old['recomputed_endpoints'])==2 and all(len(v)==470 for v in old['recomputed_endpoints']),'entire independent previous root/anchor/endpoints')
 anchor=dict(box=old['recomputed_endpoints'][0],exact_parameter=scalar);c=read(p/'run/target-certificate.json.gz');need(c['sigma']==-1 and c['weights']==inp['weights']and c['normal']==inp['normal']and c['origin']==inp['origin']and c['parameter_center']==inp['parameter'],'same actual signed whole-domain inputs')
 events=[]
 for path in sorted((p/'run/events').glob('*-target-*.json.gz')):
  kind=path.name.split('-target-',1)[1].removesuffix('.json.gz')
  if kind in ('phase_return','path_phase_return','path_assembly_return'):events.append((kind,read(path)))
 proof=C.E.check_chart(c,anchor,events,512,expected_settings={k:inp['settings'][k]for k in ('precision','newton_steps','halfwidth_exponent','radius_exponent')},expected_sigma=-1);w.save('independent512.json',proof);need(proof['passed']and proof['inclusions']==proof['anchor_coordinates']==470,'all470 fresh independent root/known endpoint join');w.save('connected-left-endpoint.json',dict(box=proof['recomputed_endpoints'][0],normal=c['normal'],origin=c['origin'],exact_parameter=c['parameter_endpoints'][0],period=18,sigma=-1,dimension=13))
 return dict(passed=True,native_calls=0,contraction=proof['contraction'],root_coordinates=470,anchor_coordinates=470,scope='Independent full470 negative-one root, primitive18 and same-gauge known-left-root join. Shared native derivatives/Arb; no complementary stability or global window closure.')

def main():
 C.OUT=OUT;resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();ans=dict(passed=False,native_calls=0)
 def stop(*_):raise TimeoutError('300s owned')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,300)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),precision=512,python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,sources={n:entry(ROOT/n)for n in FILES},threads=dict.fromkeys(THREADS,'1')))
  for n in FILES:
   b=(ROOT/n).read_bytes();need(hashlib.sha256(b).digest()==hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).digest(),'own exact source');q=OUT/'source'/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b);w.used+=len(b)
  with ctx.workprec(512):
   card=read(ROOT/CARD);profile(w,card);ans=replay(card,w)
 except BaseException as e:ans.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);ans.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',ans,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(ans));return 0 if ans['passed']else 2
if __name__=='__main__':raise SystemExit(main())
