#!/usr/bin/env python3
"""One fixed exact local organizer-to-saved-critical-curve connection."""
import hashlib,json,os,platform,resource,signal,subprocess,sys,time,traceback
from fractions import Fraction as F
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
import numpy as np
import flint
from flint import arb,arb_mat,ctx
from neural_shrimp import rigorous_clark_critical_curve as C
from neural_shrimp import rigorous_clark_generalized_flip_proof as P
from neural_shrimp.rigorous_orbits import ball_json,vector_json,matrix_json,canonical_bytes,midpoint_matrix
need=C.require
OUT=Path('/data2/neural-shrimp-data/explore/clark-rnn/critical-organizer-bridge/attempt002/run')
BASE=Path('/data2/neural-shrimp-data/explore/clark-rnn')
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
CAPS=dict(family=12,context=2,transport=2,native_phase_returns=60,events=256)
CARD='docs/plans/clark-critical-organizer-bridge-inputs.json'
RADIUS_EXPONENTS=[-14, -15, -14, -14, -13, -13, -14, -15, -13, -15, -14, -14, -17, -15, -13, -14, -14, -13, -14, -14, -14, -14, -14, -14, -14, -39]
SETTINGS=dict(precision=384,halfwidth_exponent=-11,tube_radius_exponents=RADIUS_EXPONENTS,endpoint_radius_exponent=-80,newton_steps=2,seconds=90,rss_bytes=2*1024**3,as_bytes=4*1024**3,raw_bytes=128*1024**2)
FILES=['docs/reviews/clark-critical-bridge-diagnosis.json','docs/reviews/clark-critical-bridge-attempt001.json',CARD,'docs/plans/clark-critical-organizer-bridge.md','docs/reviews/clark-rnn-generalized-flip-independent.json','docs/reviews/clark-rnn-generalized-flip-outcome.json','docs/reviews/clark-rnn-n12-organizer-connection-001.json','tools/certify_clark_critical_bridge.py','tools/launch_clark_critical_bridge.py','tests/test_clark_critical_bridge_driver.py','tests/test_rigorous_clark_critical_curve.py']
FILES=sorted(set(FILES+[str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(ROOT)]))

def entry(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  while b:=f.read(1024**2):h.update(b)
 return dict(bytes=p.stat().st_size,sha256=h.hexdigest())
def read(p):return json.loads(p.read_bytes())
def exact_float(v):return arb(str(F.from_float(float(v))))
def decode(v):return arb(arb(tuple(v['mid'])),arb(tuple(v['rad'])))
def inventory(root,pin,tick):
 need(entry(root/'manifest.json')['sha256']==pin,'literal immutable inventory');files=read(root/'manifest.json')['files'];actual={str(p.relative_to(root)):p for p in root.rglob('*')if p.is_file()and p!=root/'manifest.json'};need(set(files)==set(actual),'complete immutable inventory')
 for n,p in actual.items():need(not p.is_symlink()and not p.stat().st_mode&0o222 and entry(p)==files[n],'actual immutable bytes '+n);tick()
 return dict(root=str(root),manifest=pin,files=len(files),bytes=sum(v['bytes']for v in files.values()))
def historical(root,source):
 rt=read(root/'run/runtime.json');la=read(root/'launcher-result.json');need(rt['source_commit']==la['source_commit']==source and la['exit_code']==0 and not la['owned_live_pids'],'completed exact execution')
 for pid in(rt['pid'],la['launcher_pid'],la['owned_group_pid']):need(not Path('/proc',str(pid)).exists(),'old owned PID absent')
 actual={str(p.relative_to(root/'run/source')):p for p in(root/'run/source').rglob('*')if p.is_file()};need(set(actual)==set(rt['sources']),'complete historical source archive')
 for n,p in actual.items():need(entry(p)==rt['sources'][n]and hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest()==rt['sources'][n]['sha256'],'exact historical source '+n)
 return rt

def weighted_radii(diagnosis,outcome):
 need(outcome['passed']is True and outcome['native_calls']==0 and outcome['status']=='saved_floating_diagnosis_complete'and outcome['original_strict_inclusions']==5,'complete retained diagnosis only')
 need(diagnosis['dimension']==26 and diagnosis['proposed_exponents']==RADIUS_EXPONENTS and len(diagnosis['proposed_radii'])==26,'literal precommitted26 weighted coordinates')
 need(all(type(e)is int and -40<=e<=-13 for e in RADIUS_EXPONENTS),'fixed dyadic physical scales')
 r=[arb((1,e))for e in RADIUS_EXPONENTS]
 need(all(exact_float(x)==v for x,v in zip(diagnosis['proposed_radii'],r)),'exact dyadic proposal equality')
 return r

def bind_weighted(w):
 card=read(ROOT/CARD)['weighted_amendment'];root=Path(card['diagnosis']['root']);pin=card['diagnosis']
 inventories=[inventory(root,pin['outer'],w.tick),inventory(root/'run',pin['worker'],w.tick)];historical(root,pin['source']);need(entry(root/'run/diagnosis.json')==pin['artifact'],'literal diagnostic proposal bytes')
 old=card['failure'];prior=Path(old['root']);inventories.extend([inventory(prior,old['outer'],w.tick),inventory(prior/'run',old['worker'],w.tick)])
 original=read(prior/'run/outcome.json');rt=read(prior/'run/runtime.json');la=read(prior/'launcher-result.json');need(original['error']=='ValueError: uniform full26 root tube'and not original['passed']and rt['source_commit']==la['source_commit']==old['source']and la['exit_code']==2 and not la['owned_live_pids'],'original failed scientific tube retained')
 binding=read(root/'run/input-binding.json');need(binding['root']==str(prior)and binding['source']==old['source']and binding['outer']==old['outer']and binding['worker']==old['worker']and binding['tube']==entry(prior/'run/tube.json'),'same diagnosed original bound')
 for n,p in rt['sources'].items():need(entry(prior/'run/source'/n)==p and hashlib.sha256(subprocess.check_output(['git','show',old['source']+':'+n],cwd=ROOT)).hexdigest()==p['sha256'],'historic failed source '+n)
 d=read(root/'run/diagnosis.json');out=read(root/'run/outcome.json');r=weighted_radii(d,out);w.save('weighted-radius-binding.json',dict(card=card,inventories=inventories,radii=vector_json(r),scope='Fixed saved floating proposals only. Complete fresh center/predictor/whole acquisitions establish every new mathematical bound; no expanded-domain inference from old derivatives.'));return r

def exact_authority(original,audit,fresh,same):
 need(original['passed']and audit['passed']and fresh['passed']and len(fresh['box'])==len(fresh['coordinate_inclusions'])==26 and all(fresh['coordinate_inclusions'])and same['all26_original_boxes_strictly_inside'],'complete exact same organizer authority')
 need(original['primitive_period']==audit['primitive_period']==6 and original['critical_signed_multiplier']==-1 and original['critical_native_multiplier']==1 and audit['complement_dimension']==11,'exact primitive signed organizer identity')

def numerical_gate(out):
 need(out['status']=='completed_native_organizer_phase_connection'and len(out['points'])==17 and len(out['associations'])==16 and len(out['phase_bridges'])==2,'complete original numerical connection')
 need(all(p['fold']['passed']and p['signed']['passed']for p in out['points'])and all(a['all_reverse_native_passed']and all(v['passed']for v in a['reverse'])and max(a['reverse_q_errors'])<=1e-7 for a in out['associations']),'all prior native and reverse qualifications')
 need(all(v['unique_matching_phase']==4 and len(v['rows'])==6 for v in out['phase_bridges']),'original distinct cubic phase already transported')
 need(out['points'][-1]['signed']['tag']=='signed16'and out['points'][-1]['signed']['gain']==4.0756 and all(out['points'][-1]['signed']['gates'].values()),'fixed same-phase endpoint')

def bind(w):
 card=read(ROOT/CARD);rows=[];roots={};rts={}
 for kind in('organizer','independent'):
  spec=card[kind];root=Path(spec['root']);roots[kind]=root
  rows.extend([inventory(root,spec['outer'],w.tick),inventory(root/'run',spec['worker'],w.tick)]);rts[kind]=historical(root,spec['source'])
 op,ip=roots['organizer']/'run',roots['independent']/'run';oo,io=read(op/'outcome.json'),read(ip/'outcome.json');fresh=read(ip/'fresh-root.json');same=read(ip/'same-root.json');exact_authority(oo,io,fresh,same)
 old=read(op/'inputs.json');ib=read(ip/'input-binding.json');need(old['map']==ib['map']and old['matrix']==ib['matrix']and ib['source']==card['organizer']['source'],'same independently bound original matrix/root')
 for name in('rigorous_clark_generalized_flip','rigorous_orbits'):
  n='src/neural_shrimp/'+name+'.py';need(entry(ROOT/n)==rts['organizer']['sources'][n]==rts['independent']['sources'][n],'unchanged exact native implementation '+n)
 for name,key in [('clark-rnn-generalized-flip-outcome.json','organizer'),('clark-rnn-generalized-flip-independent.json','independent')]:
  rec=read(ROOT/'docs/reviews'/name);need(rec['root']==card[key]['root']and rec['source']==card[key]['source']and rec['outcome']==read(roots[key]/'run/outcome.json'),'literal exact root receipt')
 mr=Path(card['matrix']['path']);need(entry(mr)==card['matrix']['entry']==old['matrix'],'exact original matrix file')
 with np.load(mr,allow_pickle=False)as f:jj=f['j'].copy()
 need(jj.dtype==np.float64 and jj.shape==(12,12)and np.isfinite(jj).all(),'original binary64 matrix');j=[[exact_float(v)for v in row]for row in jj];need([[str(F.from_float(float(v)))for v in row]for row in jj]==old['map']['j0'],'full exact map equality')
 num=card['numerical'];root=Path(num['root']);rows.extend([inventory(root,num['outer'],w.tick),inventory(root/'run',num['worker'],w.tick)]);nr=read(root/'run/runtime-start.json');la=read(root/'launch-end.json');out=read(root/'run/result.json');numerical_gate(out)
 need(nr['source']['commit']==la['source_commit']==num['source']and la['exit_code']==0 and la['owned_group_empty'],'complete numerical execution')
 for pid in(nr['pid'],la['launcher_pid'],la['child_pid']):need(not Path('/proc',str(pid)).exists(),'old numerical owned PID absent')
 for n,pin in nr['source']['sha256'].items():need(hashlib.sha256(subprocess.check_output(['git','show',num['source']+':'+n],cwd=ROOT)).hexdigest()==pin,'historical numerical source '+n)
 binding=read(root/'run/input-binding.json');need(binding['old_failed_phase_zero_join_retained']and binding['prior']['retained_failed_original_global_join']and binding['prior']['matrix_input_sha256']==card['matrix']['entry']['sha256'],'same matrix and retained old failed global join')
 audit=read(ROOT/'docs/reviews/clark-rnn-n12-organizer-connection-001.json')['independent_saved_audit'];need(audit['passed']and audit['raw_root']==str(root)and audit['execution_commit']==num['source']and audit['outer']['manifest_sha256']==num['outer']and audit['worker']['manifest_sha256']==num['worker'],'independent numerical source arithmetic receipt')
 # Exact seed genealogy is read from the original quintic acquisition binding,
 # already fully authenticated by the organizer's original consumed inventory.
 quintic=Path(old['inventories'][0]['root']);rows.extend([inventory(quintic,old['inventories'][0]['manifest'],w.tick),inventory(quintic/'run',old['inventories'][1]['manifest'],w.tick)])
 qb=read(quintic/'run/input-binding.json');need(qb['organizer']==[root.name,num['outer'],num['worker'],num['source']]and qb['old_global_phase_failure_preserved'],'actual quintic same continued-phase genealogy')
 p=root/'run'/num['point'];md=read(p);need(all(md[k]==v for k,v in out['points'][-1]['signed'].items())and md['array_file']==num['array']and md['array_sha256']==num['array_sha256'],'exact selected full endpoint record');ap=p.with_name(md['array_file']);need(entry(ap)['sha256']==num['array_sha256'],'literal endpoint arrays')
 with np.load(ap,allow_pickle=False)as f:arr={n:f[n].copy()for n in f.files}
 need(arr['physical'].shape==(25,)and arr['cycle'].shape==(6,12)and all(v.dtype==np.float64 and np.isfinite(v).all()for v in arr.values())and np.array_equal(arr['physical'][:12],arr['cycle'][0])and arr['physical'][12]==md['delta'],'complete unchanged native phase and mode')
 known=[decode(v)for v in fresh['box']];endpoint=[exact_float(v)for v in arr['physical']]+[exact_float(md['gain'])]
 w.save('input-binding.json',dict(card=card,inventories=rows,organizer_box=vector_json(known),exact_authority=dict(original=entry(op/'outcome.json'),independent=entry(ip/'outcome.json'),fresh=entry(ip/'fresh-root.json'),same_root=entry(ip/'same-root.json')),map=old['map'],seed=vector_json(endpoint),saved_point=entry(p),saved_array=entry(ap),phase_genealogy='Both exact organizer numerical seed and signed16 use the already continued phase; original phase4 bridge is separate and retained.'))
 return j,known,endpoint,arr

class Work:
 def __init__(self):self.start=time.monotonic();self.used=0;self.stopped=False;self.calls=dict.fromkeys(CAPS,0)
 def tick(self):need(not self.stopped and time.monotonic()-self.start<90 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2*1024**2,'90s/2GiB cap')
 def save(self,name,row,closing=False):
  raw=canonical_bytes(row);need(self.used+len(raw)<(128 if closing else 124)*1024**2,'128MiB with4MiB failure reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)
 def charge(self,key):self.tick();need(self.calls[key]<CAPS[key],'inclusive fixed '+key+' ceiling');self.calls[key]+=1
 def emit(self,kind,row):
  self.charge('events')
  if kind in('native_polynomial','native_context_phase','native_transport_phase'):self.charge('native_phase_returns')
  self.save(f'events/{self.calls["events"]:03d}-{kind}.json',row);self.tick()
 def family(self,j,z,s):self.charge('family');self.emit('family_attempt',dict(input=vector_json(z),parameter=ball_json(s)));return C.critical(j,z,s,self.emit)
 def context(self,j,z,mat):self.charge('context');return P.conditional_native_context(j,z,mat,z[13:25],max_power=512,observe=self.emit)
 def transport(self,j,z):self.charge('transport');return C.native_transport(j,z,observe=self.emit)

def synthetic(w):
 n=12;q=[arb(int(i==0))for i in range(n)];j=[[arb(int(i==k))/5+arb(1)/200 for k in range(n)]for i in range(n)];z=[arb(i+1)/50 for i in range(n)]+[arb(3)/5]+q+[arb(7)/5];times={};sizes={}
 def measure(k,f):
  t=time.monotonic();old=w.used;ret=f();times.setdefault(k,[]).append(time.monotonic()-t);sizes.setdefault(k,[]).append(w.used-old);return ret
 for k in range(3):measure('family',lambda:w.family(j,[arb(v,arb((1,-16)))for v in z]if k else z,z[-1]))
 jj=[[arb(int(i==k==0))for k in range(n)]for i in range(n)];zz=[arb(0)]*n+[arb(1)/2]+q+[arb(1)];mat=arb_mat([[(-1 if i==0 else -arb(1)/8)*int(i==k)for k in range(n)]for i in range(n)])
 context=measure('context',lambda:w.context(jj,zz,mat));need(not context['passed']and context['witness_power']==1,'primitive fixedpoint negative profile')
 measure('transport',lambda:w.transport(jj,zz))
 def quad(z,s):
  N=len(z);g=z[-1];return arb_mat([[v-g*g]for v in z[:-1]]+[[g-s]]),arb_mat([[int(i==k)-2*g*int(k==N-1)for k in range(N)]for i in range(N-1)]+[[int(k==N-1)for k in range(N)]]),{}
 _,_,_,report=measure('algebra',lambda:C.tube(quad,[arb(1)]*26,arb(1),arb(1)/16,[arb(1)/64]*26,observe=w.emit,tick=w.tick));need(report['passed'],'full26 synthetic curved exact family')
 t={k:max(v)for k,v in times.items()};b={k:max(v)for k,v in sizes.items()};forecast=time.monotonic()-w.start+2*(9*t['family']+t['context']+t['transport']+7*t['algebra'])+20;raw=w.used+2*(9*b['family']+b['context']+b['transport']+7*b['algebra'])+8*1024**2;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+256*1024**2
 report=dict(passed=bool(forecast<90 and raw<128*1024**2 and peak<2*1024**3),times=times,bytes=sizes,forecast_seconds=forecast,forecast_bytes=raw,forecast_peak_bytes=peak,target_arrays_unopened=True,calls=dict(w.calls));w.save('profile.json',report);need(report['passed'],'measured full12D/26D admission')

def newton_center(w,j,known,s):
 c=[v.mid()for v in known];c[-1]=s
 for k in range(2):
  h,J,_=w.family(j,c,s);step=midpoint_matrix(J).inv()*h;new=[(c[i]-step[i,0]).mid()for i in range(26)];new[-1]=s;w.emit('center_newton_proposal',dict(index=k,before=vector_json(c),value=matrix_json(h),jacobian=matrix_json(J),step=matrix_json(step),after=vector_json(new),exact_gain_fixed=ball_json(s)));c=new
 return c

def orientations(box,c,v,s,h,r):
 rows=[]
 for sign in(1,-1):
  z=list(box);z[13:25]=[sign*x for x in z[13:25]];rows.append(dict(sign=sign,join=C.anchor(z,c,v,s,h,r)))
 return rows

def choose_orientation(direct,returned):
 d=[r['sign']for r in direct if r['join']['passed']and r['join']['coordinates']==26];r=[r['sign']for r in returned if r['join']['passed']and r['join']['coordinates']==26]
 need(len(d)==len(r)==1 and d==r,'exactly one identical normalized mode orientation for direct and native return');return d[0]

def main():
 resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2);need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,90);result=dict(passed=False,status='incomplete')
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:entry(ROOT/n)for n in FILES},precision=384,nice=os.getpriority(os.PRIO_PROCESS,0),threads=dict.fromkeys(THREADS,'1'),python=platform.python_version(),numpy=np.__version__,python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,caps=CAPS))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest()==hashlib.sha256(raw).hexdigest(),'exact source bytes');need(w.used+len(raw)<124*1024**2,'source raw cap');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(384):
   need(read(ROOT/CARD)['settings']==SETTINGS,'literal fixed settings');synthetic(w);r=bind_weighted(w);j,known,seed,arrays=bind(w);s=known[-1].mid();h=arb((1,-11));need(s-h<seed[-1]<s+h,'fixed endpoint within predeclared gain interval');c=newton_center(w,j,known,s)
   whole,v,data,tube=C.tube(lambda z,g:w.family(j,z,g),c,s,h,r,observe=w.emit,tick=w.tick);w.save('tube.json',tube);need(tube['passed']and len(tube['coordinate_inclusions'])==26 and all(tube['coordinate_inclusions']),'uniform full26 root tube')
   join=C.anchor(known,c,v,s,h,r);w.save('organizer-join.json',join);need(join['passed']and join['coordinates']==26,'exact known organizer identity')
   context=w.context(j,whole,data['monodromy']);w.save('tube-native-context.json',context);need(context['passed'],'whole native primitive6/simple signed-minus/stable11 complement')
   sg=seed[-1];box,J,ed,root=P.root_enclosure(lambda z:w.family(j,z,sg),seed,newton_steps=2,radius_exponent=-80,observe=w.emit,tick=w.tick);w.save('endpoint-root.json',root);need(root['passed']and len(root['coordinate_inclusions'])==26 and all(root['coordinate_inclusions']),'separate exact saved critical root')
   direct=orientations(box,c,v,s,h,r);returned,transport=w.transport(j,box);w.save('endpoint-native-transport.json',transport);returns=orientations(returned,c,v,s,h,r);w.save('endpoint-joins.json',dict(direct=direct,full_native_return=returns));sign=choose_orientation(direct,returns)
   orbit=[[decode(x)for x in row]for row in transport['orbit']];scale=max([arb(1)]+[abs(exact_float(t)).upper()for t in arrays['physical'][:12]]);errors=[max(abs(orbit[k][i]-exact_float(arrays['cycle'][k,i])).upper()for i in range(12))/scale for k in range(6)];mode=sum(((box[13+i]-exact_float(arrays['physical'][13+i]))**2 for i in range(12)),arb(0)).sqrt().upper();leak=abs(box[12]-exact_float(arrays['physical'][12])).upper();association=dict(passed=bool(all(x<arb('1e-6')for x in errors)and mode<arb('1e-6')and leak<arb('1e-8')),native_phase_errors=vector_json(errors),mode_error=ball_json(mode),leak_error=ball_json(leak),scope='Literal full native phase/mode proximity to the original numerical endpoint, separate from exact26 root and uniqueness joins.');w.save('saved-native-association.json',association);need(association['passed'],'unchanged full native numerical endpoint identity');w.tick()
   result.update(passed=True,status='exact_local_organizer_critical_curve_connection',strict_inclusions=26,organizer_join=26,endpoint_join=26,native_return_join=26,mode_orientation=sign,gain_center=ball_json(s),gain_halfwidth=ball_json(h),endpoint_gain=ball_json(sg),contraction=tube['contraction'],primitive_period=6,critical_signed_multiplier=-1,critical_native_multiplier=1,complement_power_bounds=context['power_bounds'],scope='Uniform local26D signed-critical curve connected to the independently proved exact organizer and newly certified unchanged savedg4.0756 critical root. Full native6/mode return and strict uniqueness joins. Numerical remote native6 fold/native12/native24 sequence is not thereby a rigorously connected window; independent outcome audit remains pending.')
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,calls=w.calls);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
