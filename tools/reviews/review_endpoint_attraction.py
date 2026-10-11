"""Independent direct384-bit metric reduction of saved physical endpoint data."""
import gzip,hashlib,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import review_eigenpair_path_math as E
import review_arclength_connection as O
from neural_shrimp.rigorous_orbits import canonical_bytes,ball_json,matrix_json
need=E.need
RAW=Path('/data2/neural-shrimp-data/explore/certificates/s2-near-fold-endpoint-attraction-2026-10-10/attempt-003');PIN='dc4b966c5179cf928bfff941f9e5b805b6cde7bd1716dcc917bfd9492b3adcdc';EXEC='d0922dc6'
PARENT=Path('/data2/neural-shrimp-data/audits/s2-period18-near-fold-endpoint-localization/attempt-001');PARENTPIN='220b77b44a053a598bdbdbc264458f33a77887d27517256e60ab3c65f6789cb9'
FILES=O.FILES+['tools/reviews/review_eigenpair_path_math.py','tools/reviews/review_endpoint_attraction.py','tests/reviews/test_endpoint_attraction_review.py','docs/plans/s2-endpoint-attraction-independent.md']
def read(p):
 if p.suffix=='.gz':return json.load(gzip.open(p,'rt'))
 return json.loads(p.read_text())
def contains(a,b):
 x,y=E.M.interval(a);u,v=E.M.interval(b);return x<=u<=v<=y

def positive(A,basis):
 n=A.nrows();need(A.ncols()==n==basis.nrows()==basis.ncols(),'square congruence')
 need(all(basis[i,j].is_exact() for i in range(n) for j in range(n)),'exact saved basis')
 inverse=basis.inv();C=arb_mat([[inverse[i,j].mid() for j in range(n)] for i in range(n)]);defect=E.M.ident(n)-C*basis
 bound=max(sum((abs(defect[i,j]).upper() for j in range(n)),arb(0)).upper() for i in range(n));need(bound<1,'exact congruence matrix is nonsingular')
 transformed=C*A*C.transpose();margins=[transformed[i,i].lower()-sum((abs(transformed[i,j]).upper() for j in range(n) if j!=i),arb(0)) for i in range(n)]
 return {'passed':all(v.is_finite() and v>0 for v in margins),'inverse_defect':ball_json(bound),'congruence':matrix_json(transformed),'row_margins':[ball_json(v) for v in margins]}

def worker(out):
 need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1' for k in O.THREADS),'CPU1nice10');out.mkdir(parents=True,exist_ok=False);start=time.monotonic();result={'passed':False};stop=[False];resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2)
 def save(n,v):
  b=canonical_bytes(v);need(sum(p.stat().st_size for p in out.rglob('*') if p.is_file())+len(b)<16*1024**2,'16MiBraw');p=out/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 def tick():need(not stop[0] and time.monotonic()-start<20 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<512*1024,'20s/512MiB')
 signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGALRM,lambda *_:stop.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,20)
 save('runtime.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':{n:O.entry(ROOT/n) for n in FILES},'pid':os.getpid(),'nice':os.getpriority(os.PRIO_PROCESS,0),'precision':384,'model_calls':0})
 for n in FILES:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 try:
  O.verify(RAW,PIN);O.verify(PARENT,PARENTPIN);need(read(PARENT/'run/outcome.json')['passed'],'independent inherited primitive root');known=read(PARENT/'run/reduction-256.json')['final_state_box'];inp=read(RAW/'run/inputs.json');rt=read(RAW/'run/runtime.json');done=read(RAW/'run/outcome.json');proof=read(RAW/'run/proof.json.gz')
  need(rt['source_commit'].startswith(EXEC) and done['passed'] and done['completed'] and done['interval_phases_passed']==18 and not Path('/proc',str(rt['pid'])).exists(),'completed correct source/worker')
  need(inp['root_box']==known and inp['exact_gauge']=='45/128' and inp['independent_parent']==PARENTPIN,'entire exact same primitive root')
  for n,d in rt['sources'].items():need(O.entry(RAW/'run/source'/n)==d and hashlib.sha256(subprocess.check_output(['git','show',rt['source_commit']+':'+n],cwd=ROOT)).hexdigest()==d['sha256'],'source '+n)
  need(proof['passed'] and proof['phase_count']==18 and proof['dimension']==13 and proof['cyclic_P_final_index']==0,'cyclic physical dimensions')
  with ctx.workprec(384):
   metrics=[E.R.matrix(v['P']) for v in proof['phases']];rows=[]
   for j,P in enumerate(metrics):
    need(all(P[a,b].is_exact() and P[a,b]==P[b,a] for a in range(13) for b in range(13)),'exact symmetric metric')
    phase=read(RAW/f'run/phase-{j:02d}.json');need(phase['phase']==j and all(contains(a,b) for a,b in zip(phase['state'],known[j*13:(j+1)*13],strict=True)) and contains(phase['rate'][0],known[-1]),'whole physical root derivative domain')
    need(all(contains(a,b) for ar,br in zip(proof['interval_J'][j],phase['jacobian'],strict=True) for a,b in zip(ar,br,strict=True)),'saved witness covers physical J')
    J=E.R.matrix(proof['interval_J'][j]);D=P-J.transpose()*metrics[(j+1)%18]*J
    pc=positive(P,E.R.matrix(proof['phases'][j]['P_positive']['basis']));dc=positive(D,E.R.matrix(proof['phases'][j]['interval_D_positive']['basis']))
    row={'phase':j,'P_positive':pc,'direct_interval_D_positive':dc,'direct_D':matrix_json(D)};save(f'phase-{j:02d}.json',row);need(pc['passed'] and dc['passed'],'strict independent P and direct D');rows.append({'phase':j,'minimum_P_margin':min(pc['row_margins'],key=lambda x:float(E.R.original_ball(x))),'minimum_D_margin':min(dc['row_margins'],key=lambda x:float(E.R.original_ball(x)))});tick()
  result.update(passed=True,phases=rows,scope='Independent384 direct P−JᵀPnextJ and exact-congruence Gershgorin reduction, full native derivative domains and cyclic metric identity. Local attraction of same independently localized primitive18 endpoint only. Analytic J acquisitions and Arb remain shared; no surrounding branch stability.')
 except BaseException as exc:result['error']=type(exc).__name__+': '+str(exc)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(out)):O.entry(p) for p in sorted(out.rglob('*')) if p.is_file()}})
 print(json.dumps(result));return 0 if result['passed'] else 2
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();raise SystemExit(worker(a.out))
