#!/usr/bin/env python3
"""Fixed nine same-phase signed-flip roots, quintic and physical unfolding."""
import argparse,hashlib,importlib.util,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('organizer',ROOT/'tools/continue_clark_rnn_n12_organizer.py');O=importlib.util.module_from_spec(sp);sp.loader.exec_module(O)
J,B,S,R,M=O.J,O.B,O.S,O.R,O.M;need=O.need
from neural_shrimp import clark_rnn_flip_normal_form as N
PARENT=('n12-organizer-connection-001','4eda3681c066743f014e451c85c0cbb2745c2d3a6bc5d6c8ab9bf3be5ea8e247','030b0c38596a690ceb4512147181750b5d3aba518290023f792ca429a26a2717','7c28210a8a41aa5b7aab5fab0a4ffd1e99eefb1a')
H=2.**-12
CAPS=dict(native=64,half=256,third=24,path5=64,normal=11,solve=320)
EXTRA=O.EXTRA+['src/neural_shrimp/clark_rnn_flip_normal_form.py','tests/test_clark_rnn_flip_normal_form.py','tools/explore_clark_rnn_n12_quintic.py','tools/launch_clark_rnn_n12_quintic.py','tests/test_clark_rnn_n12_quintic.py','docs/plans/clark-rnn-n12-quintic-unfolding.md','docs/reviews/clark-rnn-n12-organizer-connection-001.json']

def targets(left,right):
 need(left<right and right-left<=3e-9 and right<4.0756<4.08,'fixed inherited cubic bracket and fold gains');center=(left+right)/2
 return [('center',center),('left',left),('right',right),('minus_h',center-H),('minus_half',center-H/2),('plus_half',center+H/2),('plus_h',center+H),('fold_near',4.0756),('fold_far',4.08)]

def unfolding(points):
 by={r['tag']:r for r in points};need(len(points)==len(by)==9,'all nine fixed roots');a=by['center']['alpha_delta'];slope=(by['plus_h']['normal_cubic']-by['minus_h']['normal_cubic'])/(2*H);half=(by['plus_half']['normal_cubic']-by['minus_half']['normal_cubic'])/H;disagreement=abs(slope-half)/max(abs(half),1e-12)
 c5=[by[k]['quintic']for k in ('left','center','right')];consistent=all(v*c5[1]>0 and abs(v)>1e-5 for v in c5);variation=max(abs(v-c5[1])for v in c5)/max(abs(c5[1]),1e-12);bracket=by['left']['normal_cubic']>0>by['right']['normal_cubic'];det=-a*half
 return dict(cubic_gain_derivative_h=slope,cubic_gain_derivative_half_h=half,step_disagreement=disagreement,transverse_multiplier_leak_derivative=a,physical_determinant=det,bracket_quintic_relative_variation=variation,strict_cubic_bracket=bool(bracket),candidate=bool(bracket and consistent and variation<1e-6 and abs(a)>1e-5 and abs(half)>1e-5 and disagreement<.01),scope='Numerical generalized-flip evidence near the inherited cubic bracket. det D(lambda+1,cubic)/D(gain,leak)=-lambda_leak*d(cubic)/dgain along the simple flip curve. Finite step-size agreement is not an interval derivative or an exact cubic-zero root.')

def fold_prediction(row,pair):
 p=row['left'];x,y=pair[:12],pair[12:24];c,d,alpha=row['normal_cubic'],row['quintic'],row['alpha_delta'];actual=float(pair[-1]-row['delta']);u=float(p@(x-y)/2);u2=-c/(2*d)if d else None;gap=c*c/(4*d*alpha)if d*alpha else None
 return dict(actual_leak_gap=actual,projected_half_amplitude=u,actual_squared_half_amplitude=u*u,predicted_squared_half_amplitude=u2,predicted_leak_gap=gap,leak_ratio=None if not gap else actual/gap,amplitude_squared_ratio=None if not u2 else u*u/u2,scope='Leading local asymptotic comparison at a nearby signed flip: mu=-lambda_leak*(delta_fold-delta_flip), mu=c3*u²+c5*u⁴. Actual native fold is separately qualified. No finite-amplitude identity or proof from the approximation.')

def audit_binding(audit):
 need(audit['passed']and audit['outer']['manifest_sha256']==PARENT[1]and audit['worker']['manifest_sha256']==PARENT[2],'literal independent organizer audit')

def verify_input():
 j,*_,oldbinding=O.verify_input();O.PARENTS['organizer']=PARENT;parent,out=O.bound_parent('organizer');need(out['status']=='completed_native_organizer_phase_connection'and len(out['points'])==17 and len(out['associations'])==16 and len(out['phase_bridges'])==2,'complete audited organizer phase bridge')
 audit=json.loads((ROOT/'docs/reviews/clark-rnn-n12-organizer-connection-001.json').read_text())['independent_saved_audit'];audit_binding(audit)
 need(all(p['fold']['passed']and p['signed']['passed']for p in out['points'])and all(a['all_reverse_native_passed']and all(v['passed']for v in a['reverse'])and max(a['reverse_q_errors'])<=1e-7 for a in out['associations']),'all actual joint curve and reverse qualifications')
 need(all(p['unique_matching_phase']==4 and len(p['rows'])==6 for p in out['phase_bridges']),'same explicit actual native phase of both old cubic endpoints')
 saved={};pairs={}
 for tag,kind,row in [('left','branch_point',out['signed_extension'][1]),('right','branch_point',out['signed_extension'][0]),('fold_near','branch_point',out['points'][-1]['signed']),('fold_far','branch_point',out['points'][0]['signed'])]:
  md,a=O.record(parent,kind,row['tag']);need(all(md[k]==v for k,v in row.items()if k!='association')and md['passed'],'exact already qualified signed anchor');saved[tag]=(md,a['physical'],a['cycle'])
 for tag,index in [('fold_near',16),('fold_far',0)]:
  row=out['points'][index]['fold'];md,a=O.record(parent,'fold_curve_point',row['tag']);need(md['passed']and all(md[k]==v for k,v in row.items()),'exact already qualified actual native fold');pairs[tag]=a['paired_physical']
 need(saved['left'][0]['cubic']>0>saved['right'][0]['cubic'],'old strict cubic signs');return j,saved,pairs,dict(prior=oldbinding,organizer=PARENT,organizer_audit=S.sha(ROOT/'docs/reviews/clark-rnn-n12-organizer-connection-001.json'),fixed_targets=targets(saved['left'][0]['gain'],saved['right'][0]['gain']),old_global_phase_failure_preserved=True)

class Work(J.Work):
 def __init__(self,out):super().__init__(out);self.path5_calls=0;self.normal_calls=0;self.left={}
 def check(self):
  if self.interrupted:raise InterruptedError('owned signal')
  if time.monotonic()-self.start>60:raise TimeoutError('60s unit')
  if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>768*1024:raise MemoryError('768MiB RSS')
  need(self.calls<=384,'combined64native+256half+64path ceiling')
 def emit(self,kind,value):
  if kind=='branch_point':self.left[value['tag']]=value['left_vector'].copy()
  return super().emit(kind,value)
 def charge_solve(self):self.check();need(self.solve_calls<CAPS['solve'],'320 solve ceiling');self.solve_calls+=1
 def charge(self,units):self.check();need(self.augmented_calls+units<=CAPS['half'],'256 derivative half-units');self.augmented_calls+=units;self.calls+=units
 def it(self,x,j,g,d,tag):self.check();need(self.native_calls<CAPS['native'],'64 native6 calls');return super().it(x,j,g,d,tag)
 def third(self,x,q,j,g,d,tag):self.check();need(self.third_calls<CAPS['third'],'24 third calls');return super().third(x,q,j,g,d,tag)
 def path(self,h,j,g,d,tag):
  self.check();need(self.path5_calls<CAPS['path5'],'64 signed order-five paths');self.path5_calls+=1;self.calls+=1;self.emit('path5_attempt',dict(tag=tag,coefficients=h,gain=g,delta=d))
  v=N.signed_half_series(h,j,g,d,observe=lambda kind,row:self.emit(kind,dict(row,tag=tag)));self.emit('path5_return',dict(tag=tag,input=h,output=v,gain=g,delta=d));return v
 def normal(self,z,a,p,j,g,tag):
  self.check();need(self.normal_calls<CAPS['normal'],'eleven full normal forms inclusive profile');self.normal_calls+=1
  return N.normal_form5(z[:12],a['monodromy'],z[13:],p,lambda h:self.path(h,j,g,float(z[12]),tag),solve=lambda m,r,t:self.solve(m,r),observe=lambda k,v:self.emit(k,dict(v,tag=tag,linear_stage=v.get('tag'))))
 def point(self,z,h,b,a,j,g,tag):
  row,native=self.signed_point(z,h,b,a,j,g,tag);p=self.left[tag];normal=self.normal(z,a,p,j,g,tag)
  need(normal['invariance_residual']<1e-10 and max(normal['odd_gauge_errors'])<1e-8 and abs(normal['cubic']-row['cubic'])<=1e-10*(1+abs(row['cubic'])),'complete order-five invariance and independent old cubic agreement')
  row=dict(row,normal_cubic=normal['cubic'],quintic=normal['quintic'],invariance_residual=normal['invariance_residual'],cubic_crosscheck_error=abs(normal['cubic']-row['cubic']));self.emit('quintic_point',dict(row,physical=z,left=p,cycle=native['cycle']));return row,native

def profile(w,booked,out):
 j=np.eye(12)*.2+np.ones((12,12))*.025;x=np.arange(1,13)/20;q=np.eye(12)[0];z=np.r_[x,.8,q];times={k:[]for k in ['native','half','eigen','solve','normal']};sizes={k:[]for k in times}
 def measured(k,f):
  old=booked[0];t=time.monotonic();v=f();times[k].append(time.monotonic()-t);sizes[k].append(booked[0]-old);return v
 for i in range(10):
  native=measured('native',lambda:w.it(x,j,2.,.8,'synthetic-profile'));ret=measured('half',lambda:w.half(z,j,2.,-1,'synthetic-profile'));measured('half',lambda:w.third(x,q,j,2.,.8,'synthetic-profile'))
  def eig():e,v=np.linalg.eig(native[1]);w.emit('profile_eigen',dict(matrix=native[1],eigenvalues=e,eigenvectors=v))
  def sol():v=w.solve(ret[1],-ret[0]);w.emit('profile_solve',dict(matrix=ret[1],rhs=-ret[0],solution=v))
  measured('eigen',eig);measured('solve',sol)
 pj=np.diag([1.]+[.5]*11);pz=np.r_[np.zeros(12),1.,q];pa={'monodromy':-pj@pj@pj}
 for k in range(2):
  result=measured('normal',lambda:w.normal(pz,pa,q,pj,1.,'synthetic-normal'));need(abs(result['cubic']-1)<1e-12 and abs(result['quintic']+1.4)<1e-12 and result['invariance_residual']<1e-12,'full12D synthetic normal-form controls')
 t={k:max(v)for k,v in times.items()};b={k:max(v)for k,v in sizes.items()};seconds=2*(256*t['half']+64*(t['native']+4*t['eigen'])+320*t['solve']+11*t['normal'])+20;raw=256*(b['half']+4096)+64*(2*b['native']+4*b['eigen']+4096)+320*(b['solve']+4096)+11*(b['normal']+4096)+8*1024**2;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+256*1024**2
 gate=seconds<=60 and raw<=64*1024**2 and peak<=768*1024**2;S.save(out/'profile.json',dict(times=times,bytes=sizes,projected_seconds=seconds,projected_raw_bytes=raw,projected_peak_bytes=peak,gate=bool(gate),target_unopened=True,caps=CAPS,scope='Full12D actual path5/normalform and existing native/analytic/solve categories; all profile work consumes inclusive quotas. Normalform solves are conservatively also included in solve category.'));need(gate,'fixed measured quintic admission')

def run(w,j,saved,pairs,result):
 schedule=targets(saved['left'][0]['gain'],saved['right'][0]['gain']);gc=schedule[0][1];seed=saved['left'][1];cz,ch,cb,ca=w.correct(seed,seed,j,gc,'center',False);cr,cn=w.point(cz,ch,cb,ca,j,gc,'center');result.update(points=[cr],associations=[],fold_predictions=[]);S.save(w.out/'progress.json',result)
 for tag,g in schedule[1:]:
  z,h,b,a=w.correct(w.predict_gain(cz,cb,ca,g-gc,tag+'-predict'),cz,j,g,tag,False);row,nn=w.point(z,h,b,a,j,g,tag);forward=M.phase_match(cn['cycle'],nn['cycle']);result['pending']=dict(point=row,forward=forward);S.save(w.out/'progress.json',result);need(forward['passed'],'same literal native forward phase')
  rz,rh,rb,ra=w.correct(w.predict_gain(z,b,a,gc-g,tag+'-reverse-predict'),z,j,gc,tag+'-reverse',False);rn=w.native(rz[:12],j,gc,float(rz[12]),tag+'-reverse-native');reverse=B.native_join(cn['cycle'],rn['cycle'],cz[12],rz[12]);qe=float(np.linalg.norm(cz[13:]-rz[13:]));join=dict(tag=tag,forward=forward,reverse=reverse,reverse_native_passed=rn['passed'],reverse_q_error=qe)
  w.emit('quintic_association',dict(join,center_physical=cz,physical=z,reverse_physical=rz,center_cycle=cn['cycle'],cycle=nn['cycle'],reverse_cycle=rn['cycle']));need(rn['passed']and reverse['passed']and qe<1e-7,'full native state/leak/mode reverse join')
  if tag in saved:
   md,old,cycle=saved[tag];need(g==md['gain'],'literal saved gain');oldjoin=B.native_join(cycle,nn['cycle'],old[12],z[12]);oldq=float(np.linalg.norm(old[13:]-z[13:]));need(oldjoin['passed']and oldq<1e-7,'all exact saved phase/mode anchors');join.update(saved_join=oldjoin,saved_mode_error=oldq)
   if tag in('left','right'):need(row['cubic']*md['cubic']>0 and row['normal_cubic']*md['cubic']>0,'strict inherited cubic bracket signs')
  if tag in pairs:
   prediction=fold_prediction(dict(row,left=w.left[tag]),pairs[tag]);result['fold_predictions'].append(dict(tag=tag,gain=g,**prediction));w.emit('quintic_fold_asymptotic',dict(prediction,tag=tag,gain=g,signed_physical=z,paired_fold=pairs[tag],left=w.left[tag]))
  result['points'].append(row);result['associations'].append(join);result.pop('pending');S.save(w.out/'progress.json',result)
 need(len(result['points'])==9 and len(result['associations'])==8 and len(result['fold_predictions'])==2,'complete fixed nine-root/two-fold unit');result['unfolding']=unfolding(result['points'])

def main(out):
 identity=S.source();identity['sha256'].update({p:S.sha(ROOT/p)for p in EXTRA});need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in R.M.THREADS),'CPU1nice10');resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2);out.mkdir(parents=True,exist_ok=False);S.ACTIVE_ROOT,S.RAW_CAP=out,64*1024**2;w=Work(out);booked=[0];signal.signal(signal.SIGTERM,lambda *_:setattr(w,'interrupted',True));signal.signal(signal.SIGALRM,lambda *_:setattr(w,'interrupted',True));signal.setitimer(signal.ITIMER_REAL,60)
 def reserve(path,size,final=False):
  old=Path(path).stat().st_size if Path(path).exists()else 0;new=booked[0]-old+size;need(new<=(S.RAW_CAP if final else S.RAW_CAP-S.FINAL_RESERVE),'prewrite64MiB with closing reserve');booked[0]=new
 S.reserve=reserve;S.save(out/'runtime-start.json',dict(source=identity,pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),threads={k:os.environ[k]for k in R.M.THREADS},started_unix=time.time(),python=sys.version,numpy=np.__version__));result={'status':'incomplete','points':[]}
 try:
  need(int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024>2*1024**3,'2GiB shared headroom');profile(w,booked,out);j,saved,pairs,binding=verify_input();S.save(out/'input-binding.json',dict(binding,caps=CAPS));run(w,j,saved,pairs,result);w.check();result['status']='completed_fixed_native_quintic_unfolding'
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e))
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,native_calls=w.native_calls,derivative_half_units=w.augmented_calls,third_calls=w.third_calls,path5_calls=w.path5_calls,normal_form_calls=w.normal_calls,linear_solve_calls=w.solve_calls,total_calls=w.calls,scope='Fixed numerical same-native signed flip coefficients through order5, step-size checked physical unfolding and two local asymptotic fold comparisons. No exact cubic-zero root, interval quintic/unfolding proof or complete shrimp window.');S.save(out/'result.json',result,final=True);S.manifest(out)
 return 0 if result['status']=='completed_fixed_native_quintic_unfolding'else 2
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);raise SystemExit(main(p.parse_args().out))
