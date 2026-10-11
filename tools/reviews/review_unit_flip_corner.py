"""Fixed saved-only independent complex-gradient audit of one physical corner."""
import hashlib,json,os,platform,resource,signal,subprocess,sys,time,traceback
from fractions import Fraction
from pathlib import Path
import flint
from flint import arb,arb_mat,acb,acb_mat,ctx
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'tools/reviews'),str(ROOT/'tests/reviews')]
import review_unit_flip_corner_math as A
from test_unit_flip_corner_math import fixture
CARD=ROOT/'docs/plans/unit-flip-corner-independent-inputs.json'
OUT=Path('/data2/neural-shrimp-data/audits/unit-flip-corner-independent/attempt001/run')
FILES=['tools/reviews/review_unit_flip_corner_math.py','tools/reviews/review_unit_flip_corner.py','tools/launch_unit_flip_corner_review.py','tests/reviews/test_unit_flip_corner_math.py','tests/reviews/test_unit_flip_corner_driver.py','docs/plans/unit-flip-corner-independent.md','docs/plans/unit-flip-corner-independent-inputs.json']
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
def read(p):return json.loads(p.read_bytes())
def entry(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  while b:=f.read(1024**2):h.update(b)
 return dict(bytes=p.stat().st_size,sha256=h.hexdigest())
def rat(v):
 A.need(len(v)==2 and all(type(x)is int for x in v)and abs(v[1])<=10000,'bounded dyadic');return Fraction(v[0])*Fraction(2)**v[1]
def decode(v):
 A.need(rat(v['rad'])>=0,'nonnegative encoded radius');return arb(arb(tuple(v['mid'])),arb(tuple(v['rad'])))
def mat(v):return arb_mat([[decode(x)for x in row]for row in v])
def pack(v):
 if isinstance(v,arb):
  A.need(v.is_finite(),'finite saved enclosure');return dict(mid=[int(x)for x in v.mid().man_exp()],rad=[int(x)for x in v.rad().man_exp()],display=str(v))
 if isinstance(v,arb_mat):return [[pack(x)for x in row]for row in v.tolist()]
 if isinstance(v,dict):return {k:pack(x)for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [pack(x)for x in v]
 return v
class Work:
 def __init__(self):self.start=time.monotonic();self.used=0
 def tick(self):A.need(time.monotonic()-self.start<30 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<512*1024,'30s512MiB saved cap')
 def save(self,name,value,closing=False):
  raw=(json.dumps(pack(value),indent=2,sort_keys=True,allow_nan=False)+'\n').encode();A.need(self.used+len(raw)<(16 if closing else 14)*1024**2,'16MiB with2MiB closing reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)
def inventory(root,pin,w):
 A.need(entry(root/'manifest.json')['sha256']==pin,'literal immutable inventory');m=read(root/'manifest.json')['files'];files={str(p.relative_to(root)):p for p in root.rglob('*')if p.is_file()and p!=root/'manifest.json'};A.need(set(files)==set(m),'complete files')
 for n,p in files.items():A.need(not p.is_symlink()and not p.stat().st_mode&0o222 and entry(p)==m[n],'immutable bytes '+n);w.tick()
def history(spec,w):
 root=Path(spec['root'])
 for path,key in[(root,'outer'),(root/'run','worker')]:inventory(path,spec[key],w)
 rt=read(root/'run/runtime.json');la=read(root/'launcher-result.json');key='source_commit'if'source_commit'in rt else'commit';A.need(rt[key]==la[key]==spec['source']and la['exit_code']==0 and not la['owned_live_pids'],'exact completed execution')
 owned=read(root/'owned-process.json');ids=[rt['pid']]+[v for k,v in owned.items()if k in('pid','group_pid')]+[la[k]for k in('launcher_pid','owned_group_pid')if k in la]
 A.need(all(type(p)is int and p>0 and not Path('/proc',str(p)).exists()for p in ids),'historical owned processes gone')
 files={str(p.relative_to(root/'run/source')):p for p in(root/'run/source').rglob('*')if p.is_file()};A.need(set(files)==set(rt['sources']),'full archived source')
 for n,p in files.items():A.need(entry(p)==rt['sources'][n]and hashlib.sha256(subprocess.check_output(['git','show',spec['source']+':'+n],cwd=ROOT)).hexdigest()==rt['sources'][n]['sha256'],'exact historical source '+n);w.tick()
 return root,rt

def strict_same_root(old,fresh,identity):
 A.need(len(old['box'])==len(fresh['box'])==55 and identity['original_box']==old['box']and identity['fresh_box']==fresh['box']and len(identity['fresh_inside_original'])==55 and all(identity['fresh_inside_original']),'complete actual same-root authority')
 for o,f in zip(old['box'],fresh['box']):
  om,orr,fm,fr=rat(o['mid']),rat(o['rad']),rat(f['mid']),rat(f['rad']);A.need(om-orr<fm-fr<=fm+fr<om+orr,'all55 strict original containment')

def root_check(root):
 A.need(root['dimension']==13 and root['augmented_dimension']==55 and root['period']==18 and root['settings']==dict(newton_steps=0,radius_exponent=-84),'literal independent root geometry')
 box=root['box'];center=root['center'];A.need(len(box)==len(center)==55,'full55 root box');radii=[]
 for b,c in zip(box,center):
  A.need(rat(c['rad'])==0 and rat(c['mid'])==rat(b['mid'])and rat(b['rad'])>0,'exact finite shared center/radius');radii.append(arb(str(rat(b['rad']))))
 C=mat(root['preconditioner']);J=mat(root['whole_jacobian']);F=mat(root['value']);A.need(C.nrows()==C.ncols()==J.nrows()==J.ncols()==55 and F.nrows()==55 and F.ncols()==1 and all(rat(v['rad'])==0 for row in root['preconditioner']for v in row),'full exact preconditioner proposal')
 D=A.eye(55)-C*J;image=-C*F+D*arb_mat([[arb(0,r)]for r in radii]);A.need(A.norm(D)<1 and all(abs(image[i,0]).upper()<radii[i]for i in range(55)),'own strict55 Krawczyk')
 orbit=root['orbit_enclosures'];A.need(len(orbit)==19 and orbit[0]==box[:13]and all(len(row)==13 for row in orbit),'actual complete phase domains')
 for d in(1,2,3,6,9):A.need(any(not(decode(orbit[d][i])-decode(box[i])).contains(0)for i in range(13)),'proper divisor '+str(d))
 return dict(inclusions=55,contraction=A.norm(D),proper_divisors=[1,2,3,6,9],image=image)

def complement(M,box,proposal):
 n=13;k=41;cols=proposal['coordinate_columns'];A.need(len(cols)==len(set(cols))==10 and all(type(i)is int and 0<=i<n for i in cols),'ten fixed coordinate columns')
 q,r,w=box[15:28],box[28:41],box[42:55];S=arb_mat([[q[i],r[i],w[i]]+[arb(int(i==j))for j in cols]for i in range(n)]);T=A.inverse(S)*M*S
 A.need(all(T[i,j].contains(0)for i in range(3,n)for j in range(3)),'conditional same-root invariant plane')
 p=proposal['spectral_exclusion']['proposal_basis'];A.need(len(p)==10 and all(len(row)==10 for row in p)and all(rat(z[c]['rad'])==0 for row in p for z in row for c in('real','imag')),'complete exact spectral proposal')
 V=acb_mat([[acb(decode(z['real']),decode(z['imag']))for z in row]for row in p]);Q=acb_mat([[acb(T[i,j])for j in range(3,n)]for i in range(3,n)]);B=V.inv()*Q*V;bounds=[sum((abs(B[i,j])for j in range(10)),arb(0)).upper()for i in range(10)];A.need(all(v.is_finite()and v<1 for v in bounds),'own ten strict complementary bounds');return bounds

def main():
 resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2);A.need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'clean committed CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();result=dict(passed=False,native_calls=0)
 def stop(*_):raise TimeoutError('30s corner audit cap')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,30)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),precision=512,native_calls=0,sources={n:entry(ROOT/n)for n in FILES},python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__))
  for n in FILES:
   raw=(ROOT/n).read_bytes();A.need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'committed source');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(512):
   start=time.monotonic();control=A.physical_gradients(*fixture(13));control55=A.inverse(A.eye(55));w.save('profile-algebra.json',dict(gradient=control,inverse55=control55));elapsed=time.monotonic()-start;forecast=20*elapsed+6;raw=4*w.used+2*1024**2;w.save('profile.json',dict(passed=bool(control['passed']and forecast<30 and raw<14*1024**2),seconds=elapsed,forecast_seconds=forecast,forecast_bytes=raw,target_arrays_unopened=True));A.need(control['passed']and forecast<30 and raw<14*1024**2,'full13/55 measured admission')
   card=read(CARD);corner,crt=history(card['corner'],w);ind,irt=history(card['independent'],w);orig,ort=history(card['original'],w);A.need(irt['precision']==480 and crt['precision']==512,'exact prior precisions')
   binding=read(corner/'run/input-binding.json');ib=read(ind/'run/binding.json');A.need(binding['independent_root']==str(ind)and binding['source']==card['independent']['source']and binding['outer']==card['independent']['outer']and binding['worker']==card['independent']['worker']and binding['original']==ib and ib['root']==str(orig)and ib['source']==card['original']['source']and ib['outer']==card['original']['outer']and ib['worker']==card['original']['worker'],'literal complete same-map/root chain')
   for fn,pin in card['independent']['files'].items():A.need(entry(ind/fn)==pin,'literal independent artifact')
   for fn,pin in card['corner']['files'].items():A.need(entry(corner/fn)==pin,'literal comparison artifact')
   root=read(ind/'run/fresh-root.json');old=read(orig/'run/root.json');identity=read(ind/'run/root-identity.json');out=read(ind/'run/outcome.json');A.need(out['passed']and out['same_unique_root']and out['coordinates']==55 and out['fresh']['inclusions']==55,'authenticated exact independent authority');strict_same_root(old,root,identity);check=root_check(root);w.save('root-reduction.json',check)
   box=[decode(v)for v in root['box']];M=mat(root['monodromy']);J=mat(root['whole_jacobian']);bounds=complement(M,box,read(orig/'run/same-root-complement.json'));w.save('complement-reduction.json',dict(bounds=bounds,scope='Same-root conditional invariant plane, own validated frame and transformed ten-mode bounds.'))
   z=A.physical_gradients(M,J,box);w.save('physical-corner-independent.json',z);A.need(z['passed']and z['attracting_direction'],'strict physical transversality and explicitly chosen attracting direction')
   producer=read(corner/'run/physical-corner.json');G=mat(producer['physical_gradient']);A.need(all(z['physical_gradient'][i,j].overlaps(G[i,j])for i in range(2)for j in range(2))and z['determinant'].overlaps(decode(producer['determinant']))and z['flip_derivative_along_unit_hidden'].overlaps(decode(producer['flip_derivative_along_unit_hidden'])),'independent gradient/determinant/crossing agreement')
   w.save('input-binding.json',dict(card=card,root_box=root['box'],source_commit=source,shared='Authenticated prior native augmented derivatives and Arb. No native reevaluation; new complex bordered reduction, own Krawczyk/complement arithmetic.'))
   result.update(passed=True,status='independent_same_root_physical_corner',physical_gradient=z['physical_gradient'],determinant=z['determinant'],flip_derivative_along_unit_hidden=z['flip_derivative_along_unit_hidden'],direction=z['direction'],directional_derivatives=z['directional_derivatives'],complement_bounds=bounds,root_contraction=check['contraction'],same_unique_root=True,scope=z['scope'])
 except BaseException as e:result.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(pack(result)));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
