"""Independent saved705 polynomial center and11-mode subinterval replay.

Original independent coefficient authority and metadata/decoding binder shared.
All root/tangent/point and quotient reductions use independent review routines.
No native model is evaluated; saved MixedJet/PathJet returns and Arb are shared.
"""
import os
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gc,json,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src'),str(Path(__file__).parent)]
import flint
from flint import arb,arb_mat,ctx
import probe_native705_polynomial_complement as B
import review_polynomial_center_math as Q
import review_native705_complement as A
import review_native705_halves_math as J
import review_native705_march as S
H,R,M,need,read=Q.H,Q.R,Q.M,Q.need,S.read
ball_json,vector_json,matrix_json=Q.ball_json,Q.vector_json,Q.matrix_json
OUT=Path('/data2/neural-shrimp-data/audits/native705-fresh-center-independent/attempt001/run')
PARENT=dict(root='/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/unit-pair/pal/complement/fresh-center001',source='f7db2cf7d9cac864b5f62e1522befc49fe50946f',outer='08b6319ba099074535ac60f071dc62013cb76fbb7781a78c75632f867d91edc1',worker='c4ffa9c7b92f170c0ce77b2e820455034c505163a7f3653de1bf98f0138e00ce')
OLD=dict(root=str(B.OUT.parent),source='2d932da23fddf602a90ee747da88dce9085a5ba9',outer='fd297b95b22708c5c33b0284fca27f55c095a2fd2d3d0cfc223c42a9e840e031',worker='d213344ab2edc294add003ae8a6f8dd9bb2a106c765fbf2d6c67bb0f85ce04c8')
FILES=sorted(set(B.FILES+['tools/reviews/review_fresh_center.py','tools/reviews/review_polynomial_center_math.py','tools/reviews/launch_fresh_center_review.py','tests/reviews/test_fresh_center_review.py','docs/plans/native705-fresh-center-independent.md']+[str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_file()and Path(m.__file__).resolve().is_relative_to(ROOT)]))


def pairs(run,count):
 names={p.name for p in(run/'native').iterdir()if p.is_file()};need(names=={f'{i:03d}-{kind}.json'for i in range(1,count+1)for kind in('input','return')},'all actual native input/return histories retained')


def residual(run,normal,origin,parameter):
 inp=read(run/'residual-inputs/target2.json');point=[R.exact(M.point(x))for x in inp['point']];need(inp['normal']==vector_json(normal)and inp['origin']==vector_json(origin)and M.point(inp['parameter'])==M.point(ball_json(parameter)),'exact inherited residual gauge');events=[]
 for i in range(36):
  a=read(run/f'native/{217+i:03d}-input.json');b=read(run/f'native/{217+i:03d}-return.json');need(a['kind']==b['kind']=='residual'and a['stage']==b['stage']=='target2','literal final residual acquisition group')
  # Phase/direction come from the source-authenticated sequential assemble calls;
  # actual native inputs/returns, all seeds, and state/mode/rates are checked.
  aa=dict(a,kind='full_jacobian',phase=i//2,direction='q'if i%2==0 else'r');bb=dict(b,kind='full_jacobian');events.append(J.mixed_direction(aa,bb))
 value,jac=H.P.rebuild(point,events,13,18,normal,origin,parameter);del jac;return point,value


def native(inp,out,center,whole,tangent,kind,phase):
 need(inp['kind']==out['kind']==kind and inp['phase']==out['phase']==phase and 0<=phase<18,'literal actual native phase/kind');off=39*phase;v=center if kind=='center'else whole
 Q.cover(inp['state'],v[off:off+13],'actual quotient state');Q.cover(inp['rates'],v[-3:-1],'actual both rates')
 if kind=='center':need(len(inp['state_tangent'])==13 and len(inp['rate_tangent'])==2 and all(M.point(x)==0 for x in inp['state_tangent']+inp['rate_tangent']),'zero native center tangents')
 else:Q.cover(inp['state_tangent'],tangent[off:off+13],'actual state tangent');Q.cover(inp['rate_tangent'],tangent[-3:-1],'actual both-rate tangent')
 a,ap=R.matrix(out['A']),R.matrix(out['Aprime']);need(a.nrows()==a.ncols()==ap.nrows()==ap.ncols()==13 and H.finite(a)and H.finite(ap),'full finite native quotient derivatives')
 if kind=='center':need(all(v.is_zero()for row in ap.tolist()for v in row),'actual zero center derivative')
 return a,ap


def audit(w):
 root,rt,out,ir=S.bundle(PARENT,w.tick);old,ort,oo,io=S.bundle(OLD,w.tick,2);run=root/'run';prev=old/'run';need(out['passed']and out['native_calls']==270 and out['profile_native_calls']==144 and out['target_native_calls']==126 and out['parameter_endpoints']==['-17/1024','-1/64'],'literal complete centered pilot')
 need(not oo['passed']and oo['status']=='completed_correlated_bound_unresolved'and oo['native_calls']==216,'retained full-quarter failed metric')
 pairs(run,270);pairs(prev,216)
 for n in B.B.NATIVE+B.T.PATH_FILES:need(rt['sources'][n]==ort['sources'][n]==S.entry(ROOT/n),'same actual native/path source '+n)
 model,c,s,a,normal,origin,sc,weights,original,image,C,ds,ff,rp,E,metrics,domain=B.bind(w);del model;gc.collect()
 td=read(prev/'tangent-domain-inputs.json');dom=read(prev/'quotient-domain-inputs.json');fresh=read(run/'fresh-center-inputs.json');qr=read(run/'quotient-domain-inputs.json')
 need(td['parameter_endpoints']==dom['parameter_endpoints']==['-49/2048','-17/2048']and td['scale']=='1/256'and dom['halfwidth']=='1/128'and qr['halfwidth']=='1/2048'and qr['parameter_endpoints']==out['parameter_endpoints'],'exact parent quarter and centered subset')
 need(dom['complements']==qr['complements']==[matrix_json(e)for e in E]and dom['metrics']==qr['metrics']==[matrix_json(b)for b in metrics],'same exact finite E/B proposals')
 shift=F(-3,128);half=F(1,128);lo,hi=shift-half,shift+half
 err,ep=Q.root(ds,ff,weights,original,image,lo,hi,w.tick);ce,cp=Q.root(ds,ff,weights,original,image,shift,shift,w.tick)
 dd=arb(R.exact(shift),R.exact(half));d=R.exact(shift);pred=[c[i]+s[i]*dd+a[i]*dd*dd/2 for i in range(705)];pprime=[s[i]+a[i]*dd for i in range(705)];pc=[c[i]+s[i]*d+a[i]*d*d/2 for i in range(705)];known=[pc[i]+ce[i,0]for i in range(705)];whole=[pred[i]+err[i,0]for i in range(705)];segment=[pred[i]+arb(0,abs(err[i,0]).upper())for i in range(705)]
 for label,values in [('whole',whole),('center',known),('segment',segment)]:Q.cover(domain['whole_box'],values,'original '+label)
 Q.cover(td['segment'],segment,'actual polarized segment');Q.cover([v[0]for v in td['error']],[err[i,0]for i in range(705)],'actual root error');Q.cover(td['predictor_tangent'],pprime,'actual predictor tangent')
 plus=[err[i,0]*256+pprime[i]/256 for i in range(705)];minus=[err[i,0]*256-pprime[i]/256 for i in range(705)];Q.cover(td['plus'],plus,'actual plus');Q.cover(td['minus'],minus,'actual minus')
 need(td['normal']==vector_json(normal)and td['origin']==vector_json(origin)and M.point(td['parameter'])==M.point(ball_json(sc+d)),'whole tangent inherited gauge')
 paths=[A.path_event(read(prev/f'native/{i:03d}-input.json'),read(prev/f'native/{i:03d}-return.json'),i-36)for i in range(109,181)]
 mix=H.polarized_paths(segment,plus,minus,paths,normal,origin,sc+d,13,18,w.tick);tangent,tp=Q.tangent(C,ds,rp,mix,pprime,weights,lo,hi,w.tick)
 Q.cover(dom['full705']['whole'],whole,'actual whole quotient root');Q.cover(dom['full705']['tangent'],tangent,'actual whole quotient tangent');w.save('whole-root.json',ep);w.save('whole-tangent.json',tp);del paths,mix,ff,rp,original,image;gc.collect()
 point,value=residual(run,normal,origin,sc+d);Q.original(domain['error'],pc,known);Q.original(domain['error'],pc,point);Q.cover(fresh['known'],known,'actual inherited center');center,proof=Q.point(C,ds,value,point,known,weights,shift,w.tick);Q.original(domain['error'],pc,center);Q.cover(qr['full705']['center'],center,'actual new center root');w.save('fresh-center.json',proof);del C,ds;gc.collect()
 Q.cover(qr['full705']['whole'],whole,'pilot whole root');Q.cover(qr['full705']['tangent'],tangent,'pilot whole tangent');derivatives=[]
 for j in range(18):
  ci=read(run/f'native/{253+j:03d}-input.json');co=read(run/f'native/{253+j:03d}-return.json');wi=read(prev/f'native/{182+2*j:03d}-input.json');wo=read(prev/f'native/{182+2*j:03d}-return.json')
  ac,_=native(ci,co,center,whole,tangent,'center',j);aw,ap=native(wi,wo,center,whole,tangent,'whole_tangent',j);derivatives.append((ac,aw,ap));w.tick()
 records=[read(run/f'events/{j+2:03d}-phase.json')for j in range(18)];metric=read(run/'events/020-metric.json');proof=H.reduction(center[:-1],whole[:-1],tangent[:-1],E,metrics,derivatives,records,metric,R.exact(F(1,2048)),whole[-1],w.tick);w.save('complement.json',proof);w.save('source-binding.json',dict(pilot=PARENT,whole_parent=OLD,inventories=ir+io,coefficient_authority='independent original480 chart31',shared='Original metadata/decoding binder and known-root coefficients; native derivative returns and Arb. No producer localization/tangent/point/quotient reducer called.'))
 return dict(passed=True,native_calls=0,subinterval_only=True,parameter_endpoints=out['parameter_endpoints'],complement_dimension=11,complementary_modulus_upper=proof['complementary_modulus_upper'],remainder=proof['remainder_norm_upper'],penalty=proof['remainder_penalty'],scope=proof['scope'])


class Work:
 def __init__(self):self.start=time.monotonic();self.used=0
 def tick(self):need(time.monotonic()-self.start<400 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'400s4GiB cap')
 def save(self,name,value,closing=False):
  b=(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode();need(self.used+len(b)<(64 if closing else 60)*1024**2,'64MiB4MiBreserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
  self.used+=len(b)


def main():
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2);need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10source');OUT.mkdir(parents=True,exist_ok=False);w=Work();result=dict(passed=False,native_calls=0)
 def stop(*_):raise TimeoutError('owned saved400s audit')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,400)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:S.entry(ROOT/n)for n in FILES},precision=480,native_calls=0,python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,threads=dict.fromkeys(THREADS,'1')))
  for n in FILES:
   data=(ROOT/n).read_bytes();need(data==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'complete committed source');q=OUT/'source'/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(data);w.used+=len(data)
  with ctx.workprec(480):
   before=time.monotonic();trial=A.synthetic(705,w.tick);seconds=time.monotonic()-before;profile=dict(passed=trial['passed']and 8*seconds+2*61.466+60<400,seconds=seconds,forecast_seconds=8*seconds+2*61.466+60,target_arrays_unopened=True);w.save('profile.json',profile);need(profile['passed'],'full705 measured arithmetic admission');gc.collect();result=audit(w)
 except BaseException as e:result.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):S.entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
