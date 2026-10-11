"""Independent coefficient reduction from the previously proved exact Jordan root.

Use one primal bordered inverse and transpose it, unlike the producer's
separate adjoint inversion. Shared Arb and saved native MixedJet enclosures.
"""
import json,os,platform,resource,signal,subprocess,sys,time,flint
from pathlib import Path
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools/reviews')]
import review_arclength_connection as O
from neural_shrimp.rigorous_orbits import ball_json,matrix_json,vector_json

def need(c,msg):
 if not c:raise ValueError(msg)
def dec(v):
 m,e=v['mid'];r,f=v['rad'];need(r>=0,'nonnegative radius');return arb(arb((m,e)),arb((r,f)))
def mat(v):return arb_mat([[dec(x)for x in row]for row in v])
def vec(v):return [dec(x)for x in v]
def dot(a,b):return sum((x*y for x,y in zip(a,b)),arb(0))
def mul(a,x):return [dot([a[i,j]for j in range(a.ncols())],x)for i in range(a.nrows())]

def reduce_coefficients(z,j):
 n=(len(z)-2)//3;need(n>=2 and len(z)==3*n+2 and j.nrows()==j.ncols()==len(z),'native dimensions')
 need(all(x.is_finite()for x in z)and all(j[i,k].is_finite()for i in range(len(z))for k in range(len(z))),'finite inputs')
 q=z[n+2:2*n+2];r=z[2*n+2:];K=arb_mat(n+1,n+1)
 for i in range(n):
  for k in range(n):K[i,k]=j[i,k]
  K[i,n]=r[i];K[n,i]=q[i]
 Q=K.inv();need(all(Q[i,k].is_finite()for i in range(n+1)for k in range(n+1)),'rigorous finite primal inverse')
 # The transpose is the adjoint inverse exactly, including the interval domains.
 Qt=Q.transpose();p0=mul(Qt,[arb(0)]*n+[arb(1)]);p1=mul(Qt,p0[:n]+[arb(0)])
 need(p0[-1].contains(0)and p1[-1].contains(0),'exact root auxiliary zeros');p0=p0[:n];p1=p1[:n]
 need(dot(p0,q).contains(0)and dot(p0,r).contains(1)and dot(p1,q).contains(1)and dot(p1,r).contains(0),'exact chain dual identities')
 Bq=arb_mat([[j[n+i,k]for k in range(n)]for i in range(n)]);Br=arb_mat([[j[2*n+i,k]for k in range(n)]for i in range(n)])
 qq=mul(Bq,q);qr=mul(Bq,r);a=dot(p0,qq);b=dot(p0,qr)+dot(p1,qq)
 out={'a':ball_json(a),'b':ball_json(b),'b_minus_a':ball_json(b-a),'s':ball_json(a*(b-a)),'quadratic_nondegenerate':not a.contains(0),'resonant_nondegenerate':not(b-a).contains(0),'unfolding_transverse':False,'primal_border':matrix_json(K),'primal_inverse':matrix_json(Q),'p0':vector_json(p0),'p1':vector_json(p1)}
 if a.contains(0):return out
 obstructions=[];traces=[];states=[]
 for k in range(2):
  Rk=[j[i,n+k]for i in range(n)];Aqk=[j[n+i,n+k]for i in range(n)];Ark=[j[2*n+i,n+k]for i in range(n)]
  h=mul(Q,[-x for x in Rk]+[arb(0)]);c=dot(p0,Rk);need((h[-1]+c).contains(0),'range obstruction sign');h=h[:n]
  d=dot(p0,[u+v for u,v in zip(mul(Bq,h),Aqk)]);H=[u-v*d/a for u,v in zip(h,q)]
  trace=dot(p1,[u+v for u,v in zip(mul(Bq,H),Aqk)])+dot(p0,[u+v for u,v in zip(mul(Br,H),Ark)])
  obstructions.append(c);traces.append(trace);states.append(vector_json(H))
 det=obstructions[0]*traces[1]-obstructions[1]*traces[0]
 out.update(unfolding_determinant=ball_json(det),unfolding_transverse=not det.contains(0),fold_state_extensions=states,unfolding_matrix=matrix_json(arb_mat([obstructions,traces])))
 return out

NATIVE=Path('/data2/neural-shrimp-data/audits/generalized-chain-independent/attempt002')
TARGET=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/unfolding/attempt001-target')
def main(out):
 need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed source')
 need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');resource.setrlimit(resource.RLIMIT_AS,(1024**3,)*2)
 need(all(os.environ.get(k)=='1'for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']),'one-thread settings')
 start=time.monotonic();out.mkdir(parents=True,exist_ok=False)
 def save(n,v):
  data=(json.dumps(v,indent=2,allow_nan=False)+'\n').encode();need(sum(p.stat().st_size for p in out.iterdir()if p.is_file())+len(data)<10*1024**2,'10MiB raw cap')
  with (out/n).open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
 def stop(*_):raise TimeoutError('30s saved audit budget')
 signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,30)
 result={'passed':False,'native_calls':0,'scope':'Independent coefficient and actual-rate-rank assembly using primal inverse transpose. Conditional root/spectrum freshly proved in pinned prior audit. Shared native derivatives and Arb; boundary connection is separately audited.'}
 try:
  for name,p,pin in [('native',NATIVE,'ccc9147597490b78bcfc92f87a13093097694fb402c0ef1303ef7c4a52b64d91'),('coefficients',TARGET,'e9cb93feaf321edaafd8878e6fc0cdca5942c3d4b106d8eaf75bf8acc13a7f68')]:
   result[name+'_inventory']=O.verify(p,pin)
  prior=O.read(NATIVE/'run/outcome.json');fresh=O.read(NATIVE/'run/fresh-box.json');rt=O.read(NATIVE/'run/runtime.json')
  need(rt['commit']=='4e8e0ec0844ed806d7a9c6452da9e7c603e63f02'and prior['passed']and prior['root_passed']and prior['complement_attracting']and prior['exact_algebraic_multiplicity_two']and prior['positive_actual_rates'],'proved same native Jordan orbit')
  need(prior['strict_inclusions']==41 and prior['primitive_period']==18 and fresh['root']['coordinate_inclusions']==[True]*41,'strict primitive native root')
  need([x['divisor']for x in fresh['divisors']]==[1,2,3,6,9]and all(x['passed']for x in fresh['divisors']),'native divisors')
  launch=O.read(TARGET/'launcher-result.json');runtime=O.read(TARGET/'run/runtime.json');producer=O.read(TARGET/'run/outcome.json');inp=O.read(TARGET/'run/inputs.json');reference=O.read(TARGET/'run/coefficients.json')
  need(launch['commit']==runtime['commit']=='9e0bdb55ab0774353b6a8302c98d2741b9c4e1aa'and launch['exit_code']==0 and not launch['owned_live_pids']and producer['passed']and producer['ticks']==0,'completed bounded saved coefficient unit')
  need(inp['bindings']['fresh_identity']==O.entry(NATIVE/'run/fresh-box.json')and inp['bindings']['spectrum_identity']==O.entry(NATIVE/'run/quotient-spectrum.json'),'same exact proved root/Jacobian/spectrum')
  with ctx.workprec(512):
   z=vec(fresh['box']);j=mat(fresh['whole_jacobian']);zz=vec(inp['box']);jj=mat(inp['jacobian'])
   need(len(z)==len(zz)==41 and all(zz[i].contains(z[i])for i in range(41))and all(jj[i,k].contains(j[i,k])for i in range(41)for k in range(41)),'producer intervals enclose independently proved domains')
   result['reduction']=r=reduce_coefficients(z,j)
   need(r['quadratic_nondegenerate']and r['resonant_nondegenerate']and r['unfolding_transverse'],'nonzero generic coefficients and actual-rate rank')
   need(dec(r['a'])<0 and dec(r['b_minus_a'])<0 and dec(r['s'])>0 and dec(r['unfolding_determinant'])<0,'all claimed signs strictly enclosed')
   for k in ['a','b','b_minus_a','s','unfolding_determinant']:need(dec(r[k]).overlaps(dec(reference[k])),'independent coefficient consistency '+k)
  result['passed']=True
 except BaseException as exc:result['error']=type(exc).__name__+': '+str(exc)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),precision=512,address_space_cap_bytes=1024**3)
  save('outcome.json',result);(out/'source.py').write_bytes(Path(__file__).read_bytes());save('manifest.json',{'files':{p.name:O.entry(p)for p in out.iterdir()if p.is_file()}})
  for p in out.iterdir():p.chmod(0o444)
  out.chmod(0o555)
 print(json.dumps({k:v for k,v in result.items()if k!='reduction'}));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main(Path(sys.argv[1])))
