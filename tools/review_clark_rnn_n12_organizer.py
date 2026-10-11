"""Saved-only independent arithmetic for fixed native6 organizer connection001."""
from pathlib import Path
import argparse,hashlib,json,math,os,resource,signal,subprocess,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RAW=Path('/data2/neural-shrimp-data/explore/clark-rnn/n12-organizer-connection-001')
EXECUTION='7c28210a8a41aa5b7aab5fab0a4ffd1e99eefb1a'
OUTER_PIN='4eda3681c066743f014e451c85c0cbb2745c2d3a6bc5d6c8ab9bf3be5ea8e247'
WORKER_PIN='030b0c38596a690ceb4512147181750b5d3aba518290023f792ca429a26a2717'

def norm(x):return math.sqrt(math.fsum(float(v)*float(v)for v in np.ravel(x)))
def dot(a,b):return math.fsum(float(x)*float(y)for x,y in zip(a,b))
def distance(a,b):return float(np.max(abs(a-b))/max(1.,np.max(abs(a)),np.max(abs(b))))
def forward(a,b):
 e=distance(a,b);best=min(distance(a,s*np.roll(b,k,axis=0))for s in(1,-1)for k in range(6));assert e<.05 and e-best<=64*np.finfo(float).eps;return e

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def inventory(root):
 p=root/'manifest.json';v=json.loads(p.read_text())['files'];assert set(v)=={str(f.relative_to(root))for f in root.rglob('*')if f.is_file()and f!=p}
 for name,row in v.items():
  f=root/name;assert not Path(name).is_absolute()and '..'not in Path(name).parts and not f.is_symlink()and not f.stat().st_mode&0o222 and f.stat().st_size==row['bytes']and sha(f)==row['sha256']
 return dict(manifest_sha256=sha(p),manifest_bytes=p.stat().st_size,files_including_manifest=len(v)+1,bytes_including_manifest=sum(x['bytes']for x in v.values())+p.stat().st_size)

def phase_arithmetic(a,k):
 assert 0<=k<6 and a['original_cycle'].shape==(6,12)and a['phase_jacobians'].shape==(6,12,12)
 original=a['original_physical'];physical=a['physical'];target=a['continued_physical'];product=np.eye(12)
 for jac in a['phase_jacobians'][:k]:product=jac@product
 transported=product@original[13:];length=norm(transported);assert length>1e-8;transported/=length
 assert np.array_equal(physical[:12],a['original_cycle'][k])and physical[12]==original[12]and norm(physical[13:]-transported)<1e-12
 assert np.max(abs(product-a['transport']))<1e-12 and abs(length-float(a['transport_norm']))<1e-12
 cyclic=distance(np.roll(a['original_cycle'],-k,axis=0),a['cycle']);assert cyclic<=1e-7
 reverse=np.eye(12)
 for jac in a['phase_jacobians'][k:]:reverse=jac@reverse
 back=reverse@transported;assert norm(back)>1e-8;back/=norm(back)
 reverse_error=min(norm(back-original[13:]),norm(back+original[13:]));assert reverse_error<=1e-7
 assert np.max(abs(reverse-a['reverse_transport']))<1e-12 and norm(back-a['reverse_q'])<1e-12
 state=distance(a['continued_cycle'],a['cycle']);leak=abs(target[12]-physical[12]);mode=min(norm(target[13:]-transported),norm(target[13:]+transported))
 joined=bool(state<=1e-7 and leak<=1e-9)
 if joined:assert mode<=1e-7
 return dict(phase=k,cyclic_error=cyclic,reverse_mode_error=reverse_error,native_error=state,leak_error=float(leak),mode_error=mode,joined=joined)

def audit():
 start=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in()).throw(TimeoutError('30s saved audit')));signal.alarm(30)
 assert os.getpriority(os.PRIO_PROCESS,0)>=10
 assert sha(RAW/'manifest.json')==OUTER_PIN and sha(RAW/'run/manifest.json')==WORKER_PIN
 outer,worker=inventory(RAW),inventory(RAW/'run');run=RAW/'run';read=lambda n:json.loads((run/n).read_text());result=read('result.json');launch=json.loads((RAW/'launch-end.json').read_text());rt=read('runtime-start.json');profile=read('profile.json')
 assert launch['source_commit']==rt['source']['commit']==EXECUTION and launch['exit_code']==0 and launch['owned_group_empty']and all(not Path('/proc/'+str(launch[k])).exists()for k in ['child_pid','launcher_pid'])
 for name,pin in rt['source']['sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',EXECUTION+':'+name],cwd=ROOT)).hexdigest()==pin
 records=[];counts={};contexts={};aug={};halves={};pairs={};critical={};folds={};signed={};third={};assocs=[];native=None;native_records={};phase_bridges={};signed_assocs=[];recomputed_cubics={}
 metrics=dict(max_closure=0.,min_divisor=math.inf,max_eigen_residual=0.,max_augmented_recompute=0.,max_fold_coefficient_error=0.,max_signed_coefficient_error=0.,max_pair_residual=0.,max_partner_identity=0.,max_forward=0.,max_reverse=0.,max_reverse_mode=0.,max_critical_error=0.,max_complement=0.)
 def maximum(k,v):metrics[k]=max(metrics[k],float(v))
 for path in sorted(run.glob('[0-9]*.json')):
  d=json.loads(path.read_text());assert sha(run/d['array_file'])==d['array_sha256']
  with np.load(run/d['array_file'],allow_pickle=False)as f:a={k:f[k].copy()for k in f.files}
  kind=d['kind'];tag=d.get('tag');counts[kind]=counts.get(kind,0)+1
  if kind=='native':native=(d,a);native_records[tag]=a
  if tag=='synthetic_profile':continue
  if kind=='native_context':
   nd,na=native;assert nd['tag']==tag and np.array_equal(a['theta'],na['theta'])and np.array_equal(a['monodromy'],na['monodromy']);c=a['cycle'];sc=max(1.,np.max(abs(c)));cl=float(max(abs(na['end']-a['theta']))/sc);sep=min(float(max(abs(c[k]-c[0]))/sc)for k in [1,2,3]);eig=float(np.max(abs(a['monodromy']@a['eigenvectors']-a['eigenvectors']*a['eigenvalues']))/max(1,np.max(abs(a['monodromy']))));assert cl<1e-10 and sep>1e-6 and eig<1e-10
   prod=np.eye(12)
   for jac in na['phase_jacobians']:prod=jac@prod
   assert np.max(abs(prod-a['monodromy']))<1e-12
   maximum('max_closure',cl);metrics['min_divisor']=min(metrics['min_divisor'],sep);maximum('max_eigen_residual',eig);contexts[tag]=(d,a)
  elif kind in ['native_fold_augmented','signed_half']:
   q=a['right_vector'];sgn=1 if kind=='native_fold_augmented'else-1;h=np.r_[a['endpoint']-a['theta'],(a['monodromy']-sgn*np.eye(12))@q,(dot(q,q)-1)/2];err=max(abs(h-a['h']));assert err<1e-12;maximum('max_augmented_recompute',err)
   (aug if kind=='native_fold_augmented'else halves)[tag]=(d,a)
  elif kind=='paired_half':pairs[tag]=(d,a)
  elif kind=='signed_third':third[tag]=(d,a)
  elif kind=='native_fold_criticality':critical[tag]=(d,a)
  elif kind=='fold_curve_point':
   z=a['physical'];p=a['paired_physical'];projection=a['projection'];ad,aa=aug[tag];pd,pa=pairs[tag+'-pair'];assert max(abs(aa['h']))<1e-13 and max(abs(aug[tag+'-partner'][1]['h']))<1e-11;assert np.array_equal(z[:12],aa['theta'])and np.array_equal(z[13:],aa['right_vector'])and z[12]==ad['delta'];assert np.array_equal(p[12:24],-aa['cycle'][3])
   h=np.r_[pa['Gx']-p[12:24],pa['Gy']-p[:12],dot(projection,p[:12]-p[12:24])-2*d['amplitude']];assert max(abs(h))<1e-13;maximum('max_pair_residual',max(abs(h)))
   qy=pa['first_x'][:,:12]@z[13:];qy/=np.linalg.norm(qy);assert np.array_equal(qy,a['partner_mode'])
   cy=[a['cycle_x'],a['cycle_y']];cross=float(np.max(abs(cy[0]+np.roll(cy[1],3,axis=0)))/max(1,np.max(np.abs(cy))));assert cross<1e-8;maximum('max_partner_identity',cross)
   for partner in ['x','y']:
    cd,ca=critical[tag+'-'+partner];zz=ca['physical'];ax=aa if partner=='x'else aug[tag+'-partner'][1];assert np.array_equal(zz[13:],z[13:]if partner=='x'else qy)
    w=ca['left'];q=zz[13:];bqq=np.einsum('ijk,j,k->i',ax['second'][:,:12,:12],q,q);par=dot(w,ax['first'][:,13]);quad=dot(w,bqq);err=max(abs(par-cd['parameter']),abs(quad-cd['quadratic']));assert err<1e-9 and abs(par)>1e-5 and abs(quad)>1e-5 and max(abs(w@ax['monodromy']-w))<1e-7 and abs(dot(w,q)-1)<1e-10;maximum('max_fold_coefficient_error',err)
    for suffix in ['', '-adjacent']:
     _,cn=contexts[tag+'-'+partner+suffix];ev=np.linalg.eigvals(cn['monodromy']);ix=int(np.argmin(abs(ev-1)));other=np.delete(ev,ix);assert abs(ev[ix]-1)<1e-7 and min(abs(other-1))>1e-4 and max(abs(other))<1-1e-5;maximum('max_critical_error',abs(ev[ix]-1));maximum('max_complement',max(abs(other)))
   folds[tag]=(d,a)
  elif kind=='branch_point':
   z=a['physical'];hd,ha=halves[tag];assert max(abs(ha['h']))<(1e-11 if '-phase'in tag else 1e-13);assert np.array_equal(z[:12],ha['theta'])and np.array_equal(z[13:],ha['right_vector'])and z[12]==hd['delta'];q=z[13:];w=a['left_vector'];gm=ha['monodromy'];BB=ha['second'][:,:12,:12];bqq=np.einsum('ijk,j,k->i',BB,q,q);sol=np.linalg.solve(gm-np.eye(12),np.column_stack([-ha['first'][:,13],bqq]));c3=third[tag+'-cubic'][1]['third'];alpha=dot(w,np.einsum('ijk,j,k->i',BB,q,sol[:,0])+ha['second'][:,:12,13]@q);cubic=dot(w,c3)/6-dot(w,np.einsum('ijk,j,k->i',BB,q,sol[:,1]))/2
   err=max(abs(alpha-d['alpha_delta']),abs(cubic-d['cubic']));assert err<1e-9 and abs(alpha)>1e-5 and max(abs(w@gm+w))<1e-7;maximum('max_signed_coefficient_error',err);assert cubic*d['cubic']>0;recomputed_cubics[tag]=cubic
   e=np.linalg.eigvals(gm);ix=int(np.argmin(abs(e+1)));assert abs(e[ix]+1)<1e-7 and min(abs(np.delete(e,ix)+1))>1e-4 and max(abs(np.delete(e,ix)))<1-1e-5
   for suffix in ['-native','-adjacent']:
    cn=contexts[tag+suffix][1];ev=np.linalg.eigvals(cn['monodromy']);ix=int(np.argmin(abs(ev-1)));other=np.delete(ev,ix);assert abs(ev[ix]-1)<1e-7 and min(abs(other-1))>1e-4 and max(abs(other))<1-1e-5
   c=a['cycle'];assert np.max(abs(c+np.roll(c,3,axis=0)))/max(1,np.max(abs(c)))<1e-8 and np.max(abs(a['native_monodromy']-gm@gm))/max(1,np.max(abs(a['native_monodromy'])))<1e-7;signed[tag]=(d,a)
  elif kind=='joint_curve_association':
   for prev,cur,rev in [('previous_cycle_x','current_cycle_x','reverse_cycle_x'),('previous_cycle_y','current_cycle_y','reverse_cycle_y'),('previous_signed_cycle','current_signed_cycle','reverse_signed_cycle')]:
    maximum('max_forward',forward(a[prev],a[cur]));error=distance(a[prev],a[rev]);assert error<=1e-7;maximum('max_reverse',error)
   for name in ['fold','signed']:
    old,reverse=a['previous_'+name],a['reverse_'+name];error=norm(old[13:]-reverse[13:]);assert error<=1e-7 and abs(old[12]-reverse[12])<=1e-9;maximum('max_reverse_mode',error)
   assocs.append(d)
  elif kind=='organizer_signed_association':
   old,current,reverse=(a[k]for k in ['previous_physical','current_physical','reverse_physical']);maximum('max_forward',forward(a['previous_cycle'],a['current_cycle']));error=distance(a['previous_cycle'],a['reverse_cycle']);qe=norm(old[13:]-reverse[13:]);assert error<=1e-7 and qe<=1e-7 and abs(old[12]-reverse[12])<=1e-9
   maximum('max_reverse',error);maximum('max_reverse_mode',qe);signed_assocs.append(d)
  elif kind=='literal_phase_bridge':
   k=d['phase'];old=signed[tag+'-original'][1];target=signed['organizer-'+tag.rsplit('-',1)[1]][1];phase=signed[tag+'-phase'+str(k)][1]
   for name,expected in [('original_physical',old['physical']),('original_cycle',old['cycle']),('physical',phase['physical']),('cycle',phase['cycle']),('continued_physical',target['physical']),('continued_cycle',target['cycle']),('phase_jacobians',native_records[tag+'-original-native']['phase_jacobians'])]:assert np.array_equal(a[name],expected)
   report=phase_arithmetic(a,k);assert report['joined']==d['continued_native_join']['passed'];assert recomputed_cubics[tag+'-original']*recomputed_cubics[tag+'-phase'+str(k)]>0
   phase_bridges.setdefault(tag,[]).append(report)
 assert result['status']=='completed_native_organizer_phase_connection'and len(result['points'])==len(folds)==17 and len(signed)==33 and len(assocs)==len(result['associations'])==16
 assert len(signed_assocs)==len(result['signed_extension'])==2 and len(phase_bridges)==len(result['phase_bridges'])==2
 assert recomputed_cubics['organizer-right']<0<recomputed_cubics['organizer-left']
 for bridge in result['phase_bridges']:
  rows=phase_bridges[bridge['tag']];assert [v['phase']for v in rows]==list(range(6));assert [v['phase']for v in rows if v['joined']]==[bridge['unique_matching_phase']]
 comparisons=[]
 for point in result['points']:
  fd,fa=folds[point['fold']['tag']];sd,sa=signed[point['signed']['tag']];p=fa['paired_physical'];z=sa['physical'];sc=max(1.,max(abs(p[:24])),max(abs(z[:12])));co=dict(signed_amplitude=dot(fa['projection'],p[:12]-p[12:24])/2,partner_distance=float(max(abs(p[:12]-p[12:24]))/sc),midpoint_to_signed_root=float(max(abs((p[:12]+p[12:24])/2-z[:12]))/sc),x_to_signed_root=float(max(abs(p[:12]-z[:12]))/sc),y_to_signed_root=float(max(abs(p[12:24]-z[:12]))/sc),fold_minus_signed_leak=float(p[-1]-z[12]))
  assert fd['gain']==sd['gain']==point['gain']and all(abs(v-point['comparison'][k])<1e-12 for k,v in co.items());comparisons.append(dict(gain=point['gain'],fold_delta=fd['delta'],signed_delta=sd['delta'],signed_cubic=sd['cubic'],**co))
 assert counts['native']==194==result['native_calls']and 2*(counts['native_fold_augmented']+counts['paired_half'])+counts['signed_half']+counts['signed_third']==466==result['derivative_half_units']and counts['signed_third']==43==result['third_calls']
 assert sum(counts[k]for k in ['profile_solve','critical_solve','critical_predictor','criticality_solves'])==242==result['linear_solve_calls']
 b={k:max(v)for k,v in profile['bytes'].items()};t={k:max(v)for k,v in profile['times'].items()};raw=2560*(b['derivative']+4096)+1024*(b['solve']+4096)+512*(2*b['native']+4*b['eigen']+4096)+16*1024**2;seconds=2*(2560*t['derivative']+1024*t['solve']+512*(t['native']+4*t['eigen']))+20
 assert raw==profile['projected_raw_bytes']<=256*1024**2 and seconds==profile['projected_seconds']<=180 and profile['projected_peak_bytes']<=768*1024**2
 signal.alarm(0);assert time.monotonic()-start<30 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1024**2
 return dict(schema='clark-native6-organizer-saved-audit-v1',passed=True,raw_root=str(RAW),execution_commit=EXECUTION,outer=outer,worker=worker,launch=launch,profile={k:v for k,v in profile.items()if k not in ['times','bytes']},native_contexts=len(contexts),points=17,joins=16,phase_bridges=phase_bridges,recomputed_cubics=recomputed_cubics,metrics=metrics,counts=counts,comparisons=comparisons,seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Independent saved-array native primitive6/eigensystem/fold and signed-flip coefficient, literal partner and raw forward/reverse arithmetic. Analytic derivative acquisitions shared. Numerical literal-phase connection to both signed-half cubic-bracket endpoints; all12 actual phase transports and independently recomputed strict cubic signs. No exact degenerate root, quintic, interval global connection or whole-shrimp claim.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
 try:out=audit()
 except BaseException as exc:
  import traceback
  out=dict(schema='clark-native6-organizer-saved-audit-v1',passed=False,error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc(),raw_root=str(RAW),execution_commit=EXECUTION,outer_manifest=OUTER_PIN,worker_manifest=WORKER_PIN)
 finally:signal.alarm(0)
 out['audit_source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();out['audit_source_sha256']=sha(Path(__file__))
 data=(json.dumps(out,indent=2)+'\n').encode();assert len(data)<4*1024**2
 with args.out.open('xb')as f:f.write(data)
 print(json.dumps({k:v for k,v in out.items()if k not in ['comparisons','counts','launch']}))
 raise SystemExit(0 if out['passed']else 2)
