"""Source-bound independent saved-data audit of the two quadratic470D charts."""
import argparse,gzip,hashlib,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
from fractions import Fraction
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import review_arclength_connection as O
import review_eigenpair_path_math as E
from neural_shrimp.rigorous_orbits import ball_json,canonical_bytes
need=E.need
RAW=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/attempt-003-quadratic-target')
OUTER='cd181db39a626724dc6307974f97a99efdb311358523101778c58ba3e5a0059c';WORKER='cbb8445ec92f4d91369fa66827ef6ea81ced600ac4c33519d9dafd8aec0b6db9';EXEC='8bb19e4abc2b02aaff91fc11f13c26853fbb3071'
PARENT=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-fold-2026-10-09/attempt-002-target/run')
FILES=list(dict.fromkeys(O.FILES+['tools/reviews/review_eigenpair_path_math.py','tools/reviews/review_eigenpair_path.py','tests/reviews/test_eigenpair_path_review.py','docs/plans/eigenpair-path-independent-audit.md']))
def read(p):
 if p.suffix=='.gz':
  with gzip.open(p,'rb') as f:b=f.read(256*1024**2+1)
  need(len(b)<=256*1024**2,'bounded uncompressed input');return json.loads(b)
 return json.loads(p.read_text())
def bind():
 O.verify(RAW,OUTER);O.verify(RAW/'run',WORKER)
 run=RAW/'run';rt=read(run/'runtime.json');la=read(RAW/'launcher-result.json');done=read(run/'outcome.json')
 need(rt['commit']==la['commit']==EXEC and la['exit_code']==0 and not la['owned_live_pids'] and la['stop_reason'] is None and done['passed'],'successful exact producer')
 need(rt['settings']=={'period':18,'sigma':1,'halfwidth_exponent':-15,'radius_exponent':-18,'newton_steps':2,'precision':256},'fixed producer settings')
 need(rt['threads']==dict.fromkeys(O.THREADS,'1') and rt['nice']>=10,'local bounded producer')
 for name,v in rt['sources'].items():
  need(O.entry(run/'source'/name)==v and hashlib.sha256(subprocess.check_output(['git','show',EXEC+':'+name],cwd=ROOT)).hexdigest()==v['sha256'],'exact committed archived source '+name)
 for pid in [rt['pid']]:need(not Path('/proc',str(pid)).exists(),'owned worker terminated')
 rp=ROOT/'docs/reviews/s2-period18-fold-certificate-independent-review.json'
 need(O.entry(rp)['sha256']=='87cfca558719d4e240703b5a59f027b27ab1aa7a9f7a0f8a70341d20e30062e0','independent original fold review')
 review=read(rp);need(review['status']=='passed','original fold independently reviewed')
 need(O.entry(PARENT/'certificate.json')['sha256']=='e894dedb29bafa2c054ff6eb5dfd6040cdc10711960c5321c341a32af92d8067' and O.entry(PARENT/'inputs.json')['sha256']=='4a5a8c53b51cd9d79c2f5ed760a53b3092c0003978033ab290486d75fcaed470','original exact proof/input')
 proof=read(PARENT/'certificate.json')['result'];need(read(run/'parent-proof.json.gz')==proof and proof['passed'] and all(proof['gates'].values()),'whole authenticated parent proof')
 need(read(run/'exact-map-inputs.json')==read(PARENT/'inputs.json')['map_inputs'],'identical native objective/data')
 return run,proof

def worker(out):
 need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed auditor')
 need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1' for k in O.THREADS),'CPU1 nice10')
 resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stop=[False];result={'passed':False,'charts':[]}
 def save(n,v):
  b=canonical_bytes(v);need(len(b)+sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<24*1024**2,'24MiB auditor output');p=out/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 def tick():need(not stop[0] and time.monotonic()-start<180 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<3*1024**2,'180s/3GiB audit cap')
 signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGALRM,lambda *_:stop.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,180)
 save('runtime.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':{n:O.entry(ROOT/n) for n in FILES},'pid':os.getpid(),'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':dict.fromkeys(O.THREADS,'1'),'scope':'saved-data audit only; no model calls'})
 for n in FILES:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 try:
  run,proof=bind();tick();anchor=read(run/'anchor.json.gz');inp=read(run/'chart-inputs.json');result0=read(run/'result.json.gz');need(result0['passed'] and len(result0['charts'])==2,'two original charts')
  events=[]
  for p in sorted((run/'events').glob('*.json.gz')):
   kind=p.name.split('-',1)[1].removesuffix('.json.gz')
   if kind=='chart_return':events.append((kind,{}))
   elif kind in ('anchor_phase_return','phase_return','path_phase_return','path_assembly_return'):events.append((kind,read(p)))
  ae=[v for k,v in events if k=='anchor_phase_return'];need(len(ae)==18,'all anchor native propagation')
  with ctx.workprec(256):
   saved=E.vector(proof['box']);state=saved[:13];hidden=saved[13];mode=saved[14:];output=E.R.exact('1848862009810323/2251799813685248');full=[]
   for k,e in enumerate(ae):
    need(e['phase']==k,'anchor order');E.identity(state+mode+[hidden,output],e['input'],'anchor entire input')
    full+=state+mode;value=E.R.matrix(e['value']);need(value.nrows()==26 and value.ncols()==1,'anchor native dimensions');state=[value[i,0] for i in range(13)];mode=[value[i+13,0] for i in range(13)]
   full+=[hidden,output];E.identity(full,anchor['full_root_box'],'full propagated fold anchor');need(anchor['source_box']==proof['box'],'original correlated fold box');E.identity(state,anchor['terminal_state'],'terminal state');E.identity(mode,anchor['terminal_mode'],'terminal mode')
   # Closure is supplied by the prior validated fold proof, never by overlap.
   need(inp['normal']==['0']*469+['1'] and inp['origin']==inp['initial'],'original exact output gauge')
   need(all(E.M.point(ball_json(v.mid()))==Fraction(x) for v,x in zip(full,inp['initial'],strict=True)),'midpoint is gauge origin only')
  grouped=[];current=[]
  for kind,row in events:
   if kind=='anchor_phase_return':continue
   if kind=='chart_return':grouped.append(current);current=[]
   else:current.append((kind,row))
  need(len(grouped)==2 and not current,'complete chart event groups')
  known={'box':anchor['full_root_box'],'exact_parameter':'0'};fresh=known
  for i,c in enumerate(result0['charts']):
   need(c['normal']==inp['normal'] and c['origin']==inp['origin'] and c['weights']==inp['weights'] and Fraction(c['parameter_center'])==Fraction(i,16384),'native gauge/source/center')
   original=E.check_chart(c,known,grouped[i],256);save(f'chart-{i}-256.json',original);tick()
   higher=E.check_chart(c,fresh,grouped[i],384);save(f'chart-{i}-384.json',higher);tick()
   need(original['passed'] and higher['passed'],'independent strict root/primitive/native join')
   result['charts'].append({'index':i,'passed':True,'contraction384':higher['contraction'],'coordinate_inclusions':higher['inclusions']})
   known={'box':c['endpoint_root_boxes'][1],'exact_parameter':c['parameter_endpoints'][1]};fresh={'box':higher['recomputed_endpoints'][1],'exact_parameter':c['parameter_endpoints'][1]}
  save('endpoint.json',fresh);result.update(passed=True,exact_gauge=['-1/32768','3/32768'],scope='Independent native derivative-block/path-coefficient assembly,256 replay,384 Krawczyk/primitive and full470D endpoint joins; original fold proof supplies root identity. Analytic derivative/path-jet acquisitions and Arb remain shared. No simplicity, generic fold, cusp, attraction or full shrimp theorem.')
 except BaseException as e:result['error']=type(e).__name__+': '+str(e)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(out)):O.entry(p) for p in sorted(out.rglob('*')) if p.is_file()}})
 print(json.dumps(result));return 0 if result['passed'] else 2
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();raise SystemExit(worker(a.out))
