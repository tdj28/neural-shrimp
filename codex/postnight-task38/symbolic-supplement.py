"""Bounded independent quadratic MSS check and retained c13 discrepancy detail; no model."""
import os, sys, json, hashlib, resource, signal, itertools, functools, time
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
resource.setrlimit(resource.RLIMIT_CPU,(120,120)); signal.alarm(120)
import numpy as np
from scipy.optimize import brentq
out=Path(sys.argv[1]);out.mkdir(exist_ok=False); start=time.monotonic();man={}
def pin(p):
 p=Path(p);man[str(p)]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size};return p
def order(a,b):
 parity=1
 for x,y in zip(a,b):
  if x!=y:return (1 if 'LCR'.index(x)>'LCR'.index(y) else -1)*parity
  if x=='C':return 0
  if x=='R':parity=-parity
 return 0
def value(a,k):
 x=a*0
 for _ in range(k):x=a-x*x
 return x
try:
 available=int([s for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')][0].split()[1])*1024
 if available<16*1024**3:raise RuntimeError('RAM floor')
 pin(__file__); mesh=np.linspace(0,2,131073); rows=[];counts={}
 for k in range(1,10):
  expected=[]
  for bits in itertools.product('LR',repeat=k-1):
   w=''.join(bits);s=w+'C'
   if all(order((s[j:]+s[:j])*20,s*20)<0 for j in range(1,k)):expected.append(w)
  vals=value(mesh,k); roots=[float(mesh[i]) for i in np.flatnonzero(vals==0)]
  for i in np.flatnonzero(vals[:-1]*vals[1:]<0):roots.append(brentq(lambda a:value(a,k),mesh[i],mesh[i+1],xtol=5e-324,rtol=4*np.finfo(float).eps))
  primitive=[]
  for a in sorted(set(roots)):
   if any(abs(value(a,d))<1e-10 for d in range(1,k) if k%d==0):continue
   x=a;w=''
   for _ in range(k-1):w+='R' if x>0 else 'L';x=a-x*x
   primitive.append({'word':w,'period':k,'a':a,'closure':abs(value(a,k))})
  assert sorted(expected)==sorted(r['word'] for r in primitive),(k,expected,primitive)
  rows.extend(primitive);counts[k]=len(primitive)
 byword=sorted(rows,key=functools.cmp_to_key(lambda a,b:order((a['word']+'C')*20,(b['word']+'C')*20)))
 assert all(a['a']<b['a'] for a,b in zip(byword,byword[1:]))
 base=Path('/data2/neural-shrimp-data/explore/llm-soft-loop/pod64/bands')
 z=np.load(pin(base/'sdw-c13/scan.npz'),allow_pickle=False);h=np.load(pin(base/'hr10-c13/scan.npz'),allow_pickle=False)
 m=(h['period']>0)&(h['period']<=32)&(z['lyapunov']!=h['lyapunov']);detail=[]
 for r,c in zip(*np.nonzero(m)):
  detail.append({'row':int(r),'col':int(c),'period':int(h['period'][r,c]),'alpha':float(z['alpha'][c]),'log10_temperature':float(z['log10_temperature'][r]),'sdw':float(z['lyapunov'][r,c]),'hr10':float(h['lyapunov'][r,c]),'difference':float(z['lyapunov'][r,c]-h['lyapunov'][r,c])})
 result={'MSS_counts':counts,'quadratic_rows':rows,'max_quadratic_closure':max(r['closure'] for r in rows),'matches_all_enumerated_words':True,'kneading_order_matches_parameter':True,'parameter_search':'131073 uniform nodes in [0,2], sign-change bracketing and Brent; proper-divisor roots removed; finite numerical check, not completeness proof','c13_differences':detail,'c13_max_abs_exponent_difference':max(abs(r['difference']) for r in detail),'c13_all_grid_bits_equal':bool(np.array_equal(z['alpha'],h['alpha']) and np.array_equal(z['log10_temperature'],h['log10_temperature'])),'elapsed_seconds':time.monotonic()-start,'source_sha256':man[str(Path(__file__))]['sha256'],'limits':{'AS_GiB':8,'cpu_seconds':120,'alarm_seconds':120,'threads':1,'nice':os.getpriority(os.PRIO_PROCESS,0)},'available_bytes':available}
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ('quadratic_rows','c13_differences')}))
except BaseException as e:
 (out/'failure.json').write_text(json.dumps({'type':type(e).__name__,'error':str(e)},indent=2)+'\n');raise
finally:(out/'input-manifest.json').write_text(json.dumps(man,indent=2)+'\n')
