#!/usr/bin/env python3
"""One full705 quadratic continuation chart from the independent704 endpoint."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,gc,gzip,hashlib,json,math,resource,signal,subprocess,sys,time
from pathlib import Path
import flint,numpy as np
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import profile_native_unit_pair_pal as P
import probe_native_unit_pair as OLD
from neural_shrimp.rigorous_unit_pair_pal import quadratic_tube
from neural_shrimp.rigorous_eigenpair_locus import require,exact_mid
from neural_shrimp.rigorous_eigenpair_march import decode
from neural_shrimp.rigorous_orbits import exact,dyadic,vector_json
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
BASE=P.BASE;TARGET=BASE/'attempt002-target/run';CAP=768*1024**2
SETTINGS={'period':18,'precision':256,'parameter':'63/4096','halfwidth_exponent':-6,'radius_exponent':-10,'newton_steps':2}
SOURCES=list(dict.fromkeys(OLD.SOURCES+P.SOURCES+['tools/probe_native_unit_pair_pal.py','tools/launch_native_unit_pair_pal.py','tests/test_native_unit_pair_pal_driver.py','docs/plans/native-unit-pair-pal-target.md']))
read=OLD.read;entry=P.entry;inventory=OLD.inventory
PARENT=BASE.parent/'attempt002-target'
PPINS={'source':'a7b5a169032facb39e6476b3aae7e58654750fbf','outer':'73c02be51125e80db4707709e0cdaec8d39f69d453e35e2ef5a4738ed1405447','worker':'9ee39499eb309e8283f2591857b4bcecb43aedfbfe05136d480899d2a39d92b6','certificate':'47ba97a17cd9883e6617dae3ebb40deb53091ad035e1694812bb3522d1ea30b6'}
AUDIT=Path('/data2/neural-shrimp-data/audits/native-unit-pair-independent/attempt001')
APINS={'source':'85617c281b847bc96cdf32f1d7eeb9f70eb6c49d','outer':'e2226227e7f1658406b14a038170d4b8cd6186fa76f6308fb3985d28539cf92a','worker':'e1d4797ddd31f9774c64bef9a8293ff7457cf38c7a3a007463e97d1864c31098'}

def sources():
 require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed source')
 identity={n:entry(ROOT/n)for n in SOURCES}
 for n,row in identity.items():require(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==row['sha256'],'committed source '+n)
 return identity

def admit_profile(out):
 require(out['passed']is True and out['target_acquired']is False and out['ticks']==590,'completed full synthetic705 profile')
 require(out['events']['direction_return']==348 and out['events']['pal_path_phase_return']==216 and out['events']['pal_profile_path_return']==24 and out['events']['pal_chart_return']==2,'two full705 and588 native profile passes')
 require(type(out['elapsed_seconds'])in(int,float)and math.isfinite(out['elapsed_seconds'])and 0<out['elapsed_seconds']<240,'profile time cap')
 require(type(out['peak_rss_kib'])is int and 0<out['peak_rss_kib']<4*1024**2,'profile RSS cap')
 forecast=P.projection(out['chart_times'],out['phase_times'],out['path_times'],out['peak_rss_kib'],out['stored_bytes_before_final'])
 require(forecast==out['projection']and forecast['resource_admitted'],'recomputed full705 measured forecast');return forecast

def bind_profile(args,identity):
 root=BASE/'attempt001-profile';inventory(root,args.profile_outer_manifest);inventory(root/'run',args.profile_manifest)
 rt=read(root/'run/runtime.json');la=read(root/'launcher-result.json');out=read(root/'run/outcome.json')
 require(rt['commit']==la['commit']and rt['mode']==la['mode']=='synthetic_only','exact synthetic profile execution')
 require(la['exit_code']==0 and la['owned_live_pids']==[],'completed profile ownership')
 require(set(rt['sources'])==set(P.SOURCES),'full profile source closure')
 for n in P.SOURCES:
  require(identity[n]==rt['sources'][n]==entry(root/'run/source'/n),'same profiled source '+n)
  require(hashlib.sha256(subprocess.check_output(['git','show',rt['commit']+':'+n],cwd=ROOT)).hexdigest()==identity[n]['sha256'],'historical profile source '+n)
 for k,v in {'settings':P.SETTINGS,'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1')}.items():require(rt[k]==v,'same profiled '+k)
 require(rt['nice']>=10,'profile priority')
 return {'root':str(root),'source':rt['commit'],'outer':args.profile_outer_manifest,'worker':args.profile_manifest,'forecast':admit_profile(out)}

def archive_bound(root,source):
 rt=read(root/'run/runtime.json');require(rt.get('commit',rt.get('source_commit'))==source,'exact historical execution')
 for name,row in rt['sources'].items():
  require(entry(root/'run/source'/name)==row,'immutable archived source '+name)
  require(hashlib.sha256(subprocess.check_output(['git','show',source+':'+name],cwd=ROOT)).hexdigest()==row['sha256'],'historical source bytes '+name)
 return rt

def admit_endpoint(proof,audit,out):
 require(proof['passed']is True and proof['shooting_dimension']==704 and proof['dimension']==13 and proof['period']==18,'original native704 theorem')
 require(proof['parameter_endpoints']==out['parameter_endpoints']==['0','1/32768'],'exact original finite parameter interval')
 require(out['passed']is True and out['strict_inclusions']==out['anchor_coordinates']==704 and out['primitive_period']==18 and out['same_independently_connected_root']is True,'independently proved connected primitive root tube')
 require(audit['passed']is True and audit['inclusions']==audit['anchor_coordinates']==704 and audit['precision']==384 and all(audit['gates'].values()),'all independent full705 precursor gates')
 require(len(audit['recomputed_endpoints'])==2 and all(len(v)==704 for v in audit['recomputed_endpoints']),'full independent native endpoints')
 return [decode(v)for v in audit['recomputed_endpoints'][1]]+[exact('1/32768')]

def bind_root():
 inventory(PARENT,PPINS['outer']);inventory(PARENT/'run',PPINS['worker']);archive_bound(PARENT,PPINS['source'])
 la=read(PARENT/'launcher-result.json');out=read(PARENT/'run/outcome.json')
 require(la['commit']==PPINS['source']and la['exit_code']==0 and not la['owned_live_pids'],'completed original704 owner')
 require(out['passed']is True and out['ticks']==217 and out['events']['direction_return']==216,'complete original704 acquisition')
 cf=PARENT/'run/certificate.json.gz';require(entry(cf)['sha256']==PPINS['certificate'],'exact complete704 certificate')
 inventory(AUDIT,APINS['outer']);inventory(AUDIT/'run',APINS['worker']);archive_bound(AUDIT,APINS['source'])
 al=read(AUDIT/'launcher-result.json');ao=read(AUDIT/'run/outcome.json');ab=read(AUDIT/'run/input-bindings.json')
 require(al['source_commit']==APINS['source']and al['exit_code']==0 and not al['owned_live_pids'],'completed independent704 owner')
 require(ab['card']['producer']==dict(root=str(PARENT),**PPINS)and ab['certificate']==entry(cf),'independent audit identical producer/source/array')
 proof=read(cf);audit=read(AUDIT/'run/native704-384.json');anchor=admit_endpoint(proof,audit,ao)
 inp=read(PARENT/'run/inputs.json')['map_inputs'];require(inp==read(OLD.U.L.PARENT/'run/inputs.json')['map_inputs'],'same historical original physical map')
 return inp,anchor,proof,{'producer':dict(root=str(PARENT),**PPINS),'independent':dict(root=str(AUDIT),**APINS),'independent_proof':entry(AUDIT/'run/native704-384.json'),'independent_bindings':entry(AUDIT/'run/input-bindings.json'),'certificate':entry(cf),'native_parameter':'1/32768'}

def chart_seed(anchor,proof):
 require(len(anchor)==705 and len(proof['slope'])==704 and len(proof['weights'])==704,'full inherited slope/weights')
 tangent=np.array([float(decode(v[0]).mid())for v in proof['slope']]+[1.])
 length=float(np.linalg.norm(tangent));require(np.isfinite(tangent).all()and np.isfinite(length)and length>0,'finite nonzero gauge proposal')
 normal=[dyadic(v/length)for v in tangent];origin=[exact_mid(v+arb(0))for v in anchor]
 nn=[exact(v)for v in normal];slope=[exact(dyadic(v/length))for v in tangent]
 # Exact dyadic gauge normalization removes floating dot-product error from
 # the initial predictor only; full freshly acquired equations remain proof.
 factor=sum((nn[i]*slope[i]for i in range(705)),arb(0));require(factor>0,'positive gauge orientation')
 slope=[v/factor for v in slope];sc=exact(SETTINGS['parameter'])
 initial=[exact_mid(anchor[i].mid()+slope[i]*sc)for i in range(705)]
 weights=proof['weights']+['1/512'];require(all(0<exact(v)<=1 for v in weights),'inherited coordinate weights')
 return initial,normal,origin,weights,{'known_endpoint':vector_json(anchor),'gauge_orientation':'increasing original t at the endpoint; unit Euclidean tangent proposal','old_tangent_norm_float':length,'weights_origin':'unchanged original704 weights plus fixed angle1/512','initial_arc_slope':vector_json(slope)}

def main(args):
 identity=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
 require(args.out==TARGET,'exclusive target path');out=args.out;out.mkdir(parents=True,exist_ok=False)
 start=time.monotonic();stop=[False];ticks=[0];serial=[0];used=[0];counts={}
 signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));resource.setrlimit(resource.RLIMIT_AS,(10*1024**3,)*2)
 def check():
  if stop[0]or time.monotonic()-start>=240:raise TimeoutError('240s target cap')
  if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=6*1024**2:raise MemoryError('6GiB sampled target RSS')
 def save(name,value,final=False):
  raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode();data=gzip.compress(raw,mtime=0)if name.endswith('.gz')else raw+b'\n'
  require(used[0]+len(data)<CAP-(0 if final else 4*1024**2),'aggregate target raw cap');p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
  used[0]+=len(data)
 def tick():ticks[0]+=1;require(ticks[0]<=253,'252 native passes and1preconditioner maximum');check()
 def observe(kind,row):
  counts[kind]=counts.get(kind,0)+1;save(f'events/{serial[0]:04d}-{kind}.json.gz',row);serial[0]+=1;check()
 save('runtime.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':identity,'mode':'target','settings':SETTINGS,'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'pid':os.getpid(),'worker_seconds':240,'sampled_rss_kib':6*1024**2,'hard_as_bytes':10*1024**3,'tick_cap':253})
 for n in SOURCES:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used[0]+=p.stat().st_size
 result={'passed':False,'status':'unresolved','scope':'One connected full705 native primitive18 positive unit-pair chart from the independent finite704 endpoint. Complement, signed criticality and junction connection remain separate.'}
 try:
  profile=bind_profile(args,identity);save('profile-binding.json',profile);check()
  with ctx.workprec(256):
   inp,anchor,old,binding=bind_root();initial,normal,origin,weights,proposal=chart_seed(anchor,old);del old;gc.collect()
   save('inputs.json',{'map_inputs':inp,'bindings':binding,'settings':SETTINGS});save('chart-inputs.json',{'initial':initial,'normal':normal,'origin':origin,'weights':weights,'anchor':vector_json(anchor),'proposal':proposal,'settings':SETTINGS});check()
   proof,_=quadratic_tube(TwoRateCuspMap(inp),initial,**SETTINGS,normal=normal,origin=origin,weights=weights,anchor={'box':anchor,'period':18,'dimension':13},tick=tick,observe=observe);save('certificate.json.gz',proof);check()
   require(ticks[0]==253 and counts['direction_return']==144 and counts['pal_path_phase_return']==108,'complete fixed full705 acquisition')
   positive=bool(decode(proof['whole_box'][-1])>0)
   result.update(passed=proof['passed']and positive,status='qualified_positive_unit_pair_pal_tube'if proof['passed']and positive else'completed_bounds_failed',gates=dict(proof['gates'],whole_angle_positive=positive),contraction=proof['contraction'],strict_inclusions=sum(proof['coordinate_inclusions']),parameter_endpoints=proof['parameter_endpoints'],endpoint_physical=[v[-3:]for v in proof['endpoint_root_boxes']])
 except BaseException as exc:result.update(passed=False,status='failed_or_interrupted',failure={'type':type(exc).__name__,'message':str(exc)})
 result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks[0],events=counts,stored_bytes_before_final=used[0])
 if result['elapsed_seconds']>=240 or result['peak_rss_kib']>=6*1024**2:result['passed']=False
 save('outcome.json',result,True);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p)for p in sorted(out.rglob('*'))if p.is_file()}},True)
 for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
 out.chmod(0o555);print(json.dumps(result));return 0 if result['passed']else 2

def parser():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--profile-manifest',required=True);p.add_argument('--profile-outer-manifest',required=True);return p

if __name__=='__main__':raise SystemExit(main(parser().parse_args()))
