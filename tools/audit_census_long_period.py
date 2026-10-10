"""Fixed row86 period58/72D outcome audit and one fresh768-bit shared-kernel replay."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import hashlib,json,resource,signal,subprocess,sys,time
from dataclasses import asdict
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import flint
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
from neural_shrimp import rigorous_orbits as C
from neural_shrimp.trainability import SYSTEMS,array_sha256
CARD='docs/plans/census-long-period-independent-inputs.json'
BASE=Path('/data2/neural-shrimp-data/audits/census-period58-independent/attempt001')
SOURCES=['tools/audit_census_long_period.py','tools/launch_census_long_period_independent.py','tests/test_census_long_period_independent.py',CARD,'docs/plans/census-long-period-independent.md','src/neural_shrimp/rigorous_orbits.py','src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py','src/neural_shrimp/trainability.py','src/neural_shrimp/gdmaps.py']
CAP=512*1024**2
OLD=Path('/data2/neural-shrimp-data/explore/certificates/census-period58/attempt001-target')
OLD_PIN='c0f7488ef7836f9c3f631e1636b9f7ca28a85191e7f1834d3060451fb9f45fc8'
OLD_SOURCE='3488be385d98b6c38a2280adca7e509ea8ff72da'

def need(v,s):
 if not v:raise ValueError(s)
def read(p):return json.loads(p.read_bytes())
def entry(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  while b:=f.read(1024**2):h.update(b)
 return dict(bytes=p.stat().st_size,sha256=h.hexdigest())
def rat(p):
 need(isinstance(p,list)and len(p)==2 and all(type(v)is int for v in p)and abs(p[1])<=100000,'finite bounded dyadic pair');return F(p[0])*F(2)**p[1]
def bounds(v):
 need(v.get('finite',True),'finite dyadic ball');m,r=rat(v['mid']),rat(v['rad']);need(r>=0,'nonnegative radius');return m-r,m+r
def decode(v):
 bounds(v);return arb(arb(tuple(v['mid'])),arb(tuple(v['rad'])))
def mat(v):return arb_mat([[decode(x)for x in row]for row in v])
def identity(n):return arb_mat([[int(i==j)for j in range(n)]for i in range(n)])
def norm(a):return max(sum((abs(a[i,j]).upper()for j in range(a.ncols())),arb(0)).upper()for i in range(a.nrows()))
def exact(v):return arb(v.numerator)/v.denominator

def saved_arithmetic(r,period=58,n=72):
 need(len(r['box'])==len(r['center'])==n and len(r['jacobians'])==period and len(r['orbit_enclosures'])==period+1,'complete native certificate arrays')
 X=[decode(v)for v in r['box']];center=[decode(v)for v in r['center']];radii=[]
 for b,c in zip(r['box'],r['center']):
  need(rat(c['rad'])==0 and rat(b['mid'])==rat(c['mid']),'exact centered correction box');radii.append(rat(b['rad']))
 requested=F(2)**r['settings']['radius_exponent'];need(len(set(radii))==1 and all(requested<=v<requested*(1+F(2)**-20)for v in radii),'fixed uniform outward-rounded radius')
 need(all(rat(v['rad'])==0 for row in r['preconditioner']for v in row),'exact fixed preconditioner');inverse=mat(r['preconditioner']);need(inverse.nrows()==inverse.ncols()==n,'complete square preconditioner')
 need(r['orbit_enclosures'][0]==r['box'],'identical initial full box')
 M=identity(n)
 for a in r['jacobians']:
  J=mat(a);need(J.nrows()==J.ncols()==n,'complete native phase Jacobian');M=J*M
 D=identity(n)-inverse*(M-identity(n));h=mat(r['residual']);need(h.nrows()==n and h.ncols()==1,'full saved center residual')
 correction=-inverse*h+D*arb_mat([[arb(0,exact(v))]for v in radii]);checks=[bool(abs(correction[i,0]).upper()<exact(radii[i]))for i in range(n)];contraction=norm(D)
 need(all(checks)and contraction.is_finite()and contraction<1,'recomputed strict root inclusion/contraction')
 proper=[d for d in range(1,period)if period%d==0];need(proper==[1,2,29]if period==58 else True,'complete native divisor list');divisors=[]
 for d in proper:
  diff=[decode(r['orbit_enclosures'][d][i])-X[i]for i in range(n)];witness=[i for i,v in enumerate(diff)if v.is_finite()and not v.contains(0)];need(witness,'recomputed proper divisor exclusion');divisors.append(dict(divisor=d,coordinates=witness))
 k=r['attraction_witness_power'];need(type(k)is int and 1<=k<=4096 and k&(k-1)==0,'bounded power-of-two attraction witness');A=M;power=1
 while power<k:A=A*A;power*=2
 bound=norm(A);need(bound.is_finite()and bound<1,'recomputed strict attraction power')
 return dict(strict_inclusions=sum(checks),contraction=C.ball_json(contraction),divisors=divisors,attraction_power=k,attraction_bound=C.ball_json(bound),phase_jacobians=period,scope='Independent interval matrix products/Krawczyk/divisor/power arithmetic from authenticated shared-kernel acquisitions.')

def contains(outer,inner):
 need(len(outer)==len(inner)==72,'all72 identity domains');checks=[]
 for a,b in zip(outer,inner):
  al,ah=bounds(a);bl,bh=bounds(b);checks.append(al<bl<=bh<ah)
 need(all(checks),'original exact root box inside fresh unique box');return checks

def inventory(root,pin):
 need(entry(root/'manifest.json')['sha256']==pin,'exact manifest');m=read(root/'manifest.json')['files'];need(set(m)=={str(p.relative_to(root))for p in root.rglob('*')if p.is_file()and p!=root/'manifest.json'},'complete immutable inventory')
 for name,item in m.items():
  p=root/name;need(not Path(name).is_absolute()and '..'not in Path(name).parts and not p.is_symlink()and entry(p)==item and not p.stat().st_mode&0o222,'immutable file identity '+name)
 return m

def binding(card):
 parent=Path(card['producer_root']);inventory(parent,card['producer_outer']);inventory(parent/'run',card['producer_worker']);run=parent/'run';done=read(run/'outcome.json');la=read(parent/'launcher-result.json');rt=read(run/'runtime.json')
 need(la['exit_code']==0 and la['owned_live_pids']==[]and done['passed']and done['status']=='certificate_complete'and rt['source_commit']==la['source_commit']==card['producer_source'],'completed exact producer')
 need(rt['settings']==card['producer_settings']and rt['mode']=='target'and rt['nice']>=10 and rt['threads']==dict.fromkeys(THREADS,'1')and done['completed_native_map_calls']==174 and rt['explicit_newton_proposals']==1,'original arithmetic contract')
 for name,item in rt['sources'].items():need(entry(run/'source'/name)==item and hashlib.sha256(subprocess.check_output(['git','show',card['producer_source']+':'+name],cwd=ROOT)).hexdigest()==item['sha256'],'historical source '+name)
 for name in ['src/neural_shrimp/rigorous_orbits.py','src/neural_shrimp/models.py','src/neural_shrimp/trainability.py','src/neural_shrimp/gdmaps.py']:need(entry(ROOT/name)==rt['sources'][name],'same shared mathematical/generator kernel')
 need(not Path('/proc',str(rt['pid'])).exists()and not Path('/proc',str(read(parent/'owned-process.json')['pid'])).exists(),'original producer ended')
 certroot=run/'certificate';receipt=read(certroot/'receipt.json');need(done['certificate_receipt']==receipt and receipt['external_root']==str(certroot),'nested receipt association')
 for name,item in receipt['files'].items():need(entry(certroot/name)==item,'nested receipt inventory')
 need(entry(certroot/'certificate.json')==card['certificate'],'literal native certificate bytes')
 actual=read(run/'proof-inputs.json');need((certroot/'inputs.json').read_bytes()==C.canonical_bytes(actual),'exact nested inputs');cert=read(certroot/'certificate.json');r=cert['result'];need(cert['inputs_sha256']==hashlib.sha256(C.canonical_bytes(actual)).hexdigest()and r['settings']==card['producer_settings'],'exact certificate input/settings')
 need(r['passed']and r['exact_statement']==dict(primitive_period=58,dimension=72,existence_and_local_uniqueness=True,primitive_period_excluded_divisors=True,attraction=True,model='exact-real GD with supplied exact dyadic constants'),'exact full native producer statement')
 need(r['attraction_witness_power']==8,'literal successful native witness power')
 need([rat(v['mid'])for v in r['center']]==[F(v)for v in actual['approximation']]and all(rat(v['rad'])==0 for v in r['center']),'exact zero-Newton original center')
 proposal=read(run/'newton-proposal.json');need([rat(v['mid'])for v in proposal['proposal']]==[F(v)for v in actual['approximation']]and all(rat(v['rad'])==0 for v in proposal['proposal']),'recorded original proposal identity')
 inventory(OLD,OLD_PIN);oldrt=read(OLD/'run/runtime-start.json');oldla=read(OLD/'launcher-result.json');oldout=read(OLD/'run/result.json')
 need(oldrt['source_commit']==oldla['source_commit']==OLD_SOURCE and oldla['exit_code']==2 and not oldout['passed']and oldla['owned_live_pids']==[],'retain original failed512 attempt')
 for name,digest in oldrt['source_sha256'].items():need(entry(OLD/'run/source'/name)['sha256']==digest==hashlib.sha256(subprocess.check_output(['git','show',OLD_SOURCE+':'+name],cwd=ROOT)).hexdigest(),'original historical source')
 inputs=read(OLD/'run/inputs.json');need(all(inputs[k]==actual[k]for k in ['schema','network','x','y','lr_hidden','lr_output','period']),'unchanged full native map and physical rates')
 frozen=read(OLD/'run/source/docs/plans/census-long-period-inputs.json');paths={k:ROOT/v['path'][5:]if v['path'].startswith('repo:')else Path(v['path'])for k,v in frozen['pins'].items()}
 for k,p in paths.items():need(entry(p)['sha256']==frozen['pins'][k]['sha256'],'exact census input '+k)
 rows=read(paths['certificates']);agreed={(v['map_id'],v['component'])for v in read(paths['summary'])['gate']['qualifying']}&{(v['map_id'],v['component'])for v in read(paths['audit'])['audit_gate']['qualifying']};eligible=[(i,v)for i,v in enumerate(rows)if(v['map_id'],v['component'])in agreed];need(len(agreed)==len(eligible)==61 and [i for i,v in eligible if v['period']==max(w['period']for _,w in eligible)]==[86],'unique maximum among61 agreed representatives')
 row=rows[86];need((row['map_id'],row['component'],row['period'])==('S1R-28002-zoom',262,58),'exact row86');spec=next(v for v in read(paths['plan'])['maps']if v['map_id']=='S1R-28002');need(spec['seed']==28002 and spec['system']=='S1R','unprotected frozen dataset')
 net,x,y,theta=SYSTEMS['S1R'](28002);network=asdict(net);realized={k:array_sha256(v)for k,v in [('x',x),('y',y),('theta0',theta)]};need(network==spec['network']and realized==spec['data_sha256'],'exact regenerated data/model')
 for k in ('in_scale','out_scale','loss_scale','ridge'):network[k]=str(F(network[k]))
 def fractions(v):return [F(float(a))for a in v]
 need(inputs['network']==network and inputs['period']==58 and len(inputs['approximation'])==72,'full exact native model')
 need([[F(v)for v in r]for r in inputs['x']]==[fractions(r)for r in x]and[F(v)for v in inputs['y']]==fractions(y),'exact data dyadics')
 need([F(v)for v in inputs['approximation']]==fractions(row['certificate']['orbit'][0])and[F(inputs[k])for k in ['lr_hidden','lr_output']]==[F(float(row[k]))for k in ['lr_hidden','lr_output']]and all(F(inputs[k])>0 for k in ['lr_hidden','lr_output']),'exact selected phase/positive physical rates')
 return actual,r,np.asarray(row['certificate']['orbit']),dict(producer=card,exact_regenerated_hashes=realized,actual_positive_rates=[inputs['lr_hidden'],inputs['lr_output']],scope='One selected72D period58 orbit, no census/window geometry claim.')

def forecast(times):
 need(len(times)==3 and all(type(v)is float and np.isfinite(v)and v>0 for v in times),'three measured finite native costs')
 return 2*116*max(times)+30

def fresh_inputs(inputs,center):
 value=json.loads(json.dumps(inputs));need(len(center)==72 and all(rat(v['rad'])==0 for v in center),'exact original72D center')
 value['approximation']=[str(rat(v['mid']))for v in center]
 return value

def main():
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2);need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean committed source');out=BASE/'run';out.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=0
 def stop(*_):raise TimeoutError('600s bounded audit')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,600)
 def save(name,v):
  nonlocal used
  b=C.canonical_bytes(v);need(used+len(b)<CAP,'512MiB audit cap');used+=len(b);p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
 source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();identity={n:entry(ROOT/n)for n in SOURCES};save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources=identity,threads=dict.fromkeys(THREADS,'1'),nice=os.getpriority(os.PRIO_PROCESS,0),precision=768,numpy=np.__version__,python=sys.version,python_flint=flint.__version__,flint=str(getattr(flint,'__FLINT_VERSION__','unavailable'))))
 for n in SOURCES:need(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==identity[n]['sha256'],'committed dependency');p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used+=p.stat().st_size
 result={'passed':False,'native_calls':0}
 try:
  inputs,old,orbit,bind=binding(read(ROOT/CARD));save('input-binding.json',bind)
  with ctx.workprec(768):arithmetic=saved_arithmetic(old)
  save('saved-arithmetic.json',arithmetic);original_box=old['box'];original_center=old['center'];del old
  import gc;gc.collect()
  actual=fresh_inputs(inputs,original_center);save('fresh-inputs.json',actual)
  with ctx.workprec(768):
   times=[];model=C.BallTanhMap(actual)
   for phase in [0,19,38]:
    t=time.monotonic();values,J=model.step([exact(F(float(v)))for v in orbit[phase]]);times.append(time.monotonic()-t);result['native_calls']+=1
    need(all(v.is_finite()for v in values)and all(J[i,j].is_finite()for i in range(72)for j in range(72)),'finite full native measured profile');save(f'profile-{phase}.json',dict(value=C.vector_json(values),jacobian=C.matrix_json(J)))
  projection=forecast(times);save('profile.json',dict(times=times,forecast_seconds=projection,remaining_seconds=600-(time.monotonic()-start),passed=projection<600-(time.monotonic()-start)));need(projection<600-(time.monotonic()-start),'fresh768 measured remaining-budget forecast')
  # Zero proposals: the exact authenticated640-bit dyadic center is retained.
  fresh=C.check_orbit(actual,precision=768,radius_exponent=-286,newton_steps=0,max_power=4096);result['native_calls']+=116
  save('fresh-certificate.json',fresh);need(fresh['passed'],'fresh768 complete native proof')
  with ctx.workprec(768):fresh_arithmetic=saved_arithmetic(fresh);same=contains(fresh['box'],original_box)
  need(fresh['settings']==dict(precision=768,radius_exponent=-286,newton_steps=0,max_power=4096),'literal fresh settings')
  save('fresh-arithmetic.json',fresh_arithmetic);save('root-identity.json',dict(original_box_strictly_contained_in_fresh=same,original_center=original_center,original_box=original_box))
  result.update(passed=True,primitive_period=58,dimension=72,saved=arithmetic,fresh=fresh_arithmetic,same_unique_root=True,scope='Independent saved interval arithmetic and fresh768-bit exact-real C1 proof using shared native Arb derivative kernel. Original640-bit root box lies strictly in fresh unique box. No independent derivative implementation, window geometry, whole census or utility claim.')
 except BaseException as exc:
  import traceback;result.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);result['passed']=result['passed']and result['elapsed_seconds']<600 and result['peak_rss_kib']<4*1024**2;save('outcome.json',result);save('manifest.json',dict(files={str(p.relative_to(out)):entry(p)for p in out.rglob('*')if p.is_file()}))
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
