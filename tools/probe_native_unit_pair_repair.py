#!/usr/bin/env python3
"""One fresh quarter-width weighted704 tube from the same audited Jordan root."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,gzip,hashlib,json,math,resource,signal,subprocess,sys,time
from fractions import Fraction
from pathlib import Path
import flint,numpy as np
from flint import ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import profile_native_unit_pair as P
import probe_eigenpair_chain_unfolding as U
from neural_shrimp.rigorous_unit_pair import native_lift,unit_pair_tube
from neural_shrimp.rigorous_eigenpair_locus import require,exact_mid
from neural_shrimp.rigorous_orbits import vector_json
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
BASE=P.BASE;SETTINGS=dict(P.SETTINGS,parameter='1/65536',halfwidth_exponent=-16,radius_exponent=-11)
SOURCES=list(dict.fromkeys(U.SOURCES+P.SOURCES+['tools/probe_native_unit_pair.py','tools/launch_native_unit_pair.py','tests/test_native_unit_pair_driver.py','docs/plans/native-unit-pair-target.md','src/neural_shrimp/unit_pair_conditioning.py','tools/diagnose_native_unit_pair.py','tools/launch_native_unit_pair_diagnosis.py','tests/test_unit_pair_conditioning.py','docs/plans/native-unit-pair-diagnosis.md','tools/probe_native_unit_pair_repair.py','tools/launch_native_unit_pair_repair.py','tests/test_native_unit_pair_repair.py','docs/plans/native-unit-pair-repair.md']))
BRIDGE=BASE.parent/'native-lift/membership-bridge/unit003'
BPINS={'source':'1bf9dfd31f29e302cc4aa6c947c4df80cba71227','outer':'b9cf0cf29584a5f2330a92f183ba0335d5184b95b90b32a2601d39432424e112','worker':'056f4acf1b1ae6e07dd3d6a2c474829cf7f2cd8408ee979100bb29b8f05caeb9'}
CAP=768*1024**2
read=U.read;entry=P.entry;inventory=U.inventory

def sources():
 require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed source')
 identity={n:entry(ROOT/n)for n in SOURCES}
 for n,row in identity.items():require(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==row['sha256'],'committed source '+n)
 return identity

def admit_profile(out):
 require(out['passed']is True and out['target_acquired']is False and out['ticks']==422,'completed full synthetic profile')
 require(out['events']['direction_return']==420 and out['events']['chart_return']==2,'two full704 and420 native profile passes')
 require(type(out['elapsed_seconds'])in(int,float)and math.isfinite(out['elapsed_seconds'])and 0<out['elapsed_seconds']<240,'profile time cap')
 require(type(out['peak_rss_kib'])is int and 0<out['peak_rss_kib']<4*1024**2,'profile RSS cap')
 forecast=P.projection(out['chart_times'],out['phase_times'],out['peak_rss_kib'],out['stored_bytes_before_final'])
 require(forecast==out['projection']and forecast['resource_admitted'],'strict recomputed measured target forecast')
 return forecast

def bind_profile(args,identity):
 root=BASE/'attempt001-profile';inventory(root,args.profile_outer_manifest);inventory(root/'run',args.profile_manifest)
 rt=read(root/'run/runtime.json');la=read(root/'launcher-result.json');out=read(root/'run/outcome.json')
 require(rt['commit']==la['commit']and rt['mode']==la['mode']=='synthetic_only','exact synthetic profile execution')
 require(la['exit_code']==0 and la['owned_live_pids']==[],'completed profile ownership')
 require(set(rt['sources'])==set(P.SOURCES),'full profile source list')
 for n in P.SOURCES:
  require(identity[n]==rt['sources'][n]==entry(root/'run/source'/n),'unchanged profiled math, model and packet '+n)
  require(hashlib.sha256(subprocess.check_output(['git','show',rt['commit']+':'+n],cwd=ROOT)).hexdigest()==identity[n]['sha256'],'historical profile source '+n)
 for k,v in {'settings':P.SETTINGS,'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1')}.items():require(rt[k]==v,'same profiled '+k)
 require(rt['nice']>=10,'profile priority')
 return {'root':str(root),'worker':args.profile_manifest,'outer':args.profile_outer_manifest,'source':rt['commit'],'forecast':admit_profile(out)}

def admit_bridge(out,member):
 require(out['passed']is True and out['status']=='rigorous_chain_membership'and out['accepted_count']==2,'completed exact bridge event')
 require(member['passed']is True and member['parameter_inside']is True and member['coordinate_inclusions']==[True]*470,'full470 root membership')
 require(out['membership']==member,'same membership outcome')

def bind_root():
 box,_,audit=U.bind_audit();model=read(U.L.PARENT/'run/inputs.json')['map_inputs']
 inventory(BRIDGE,BPINS['outer']);inventory(BRIDGE/'run',BPINS['worker'])
 rt=read(BRIDGE/'run/runtime.json');la=read(BRIDGE/'launcher-result.json');out=read(BRIDGE/'run/outcome.json');stop=read(BRIDGE/'run/membership-stop.json');inp=read(BRIDGE/'run/inputs.json')
 require(rt['commit']==la['commit']==BPINS['source']and rt['unit']==3,'exact final native bridge source')
 require(la['exit_code']==0 and la['owned_live_pids']==[]and la['stop_reason']is None,'completed bounded bridge ownership')
 require(inp['map_inputs']==model,'identical actual physical map')
 require([v['right']for v in stop['accepted']]==['293/128','295/128'],'exact accepted final bridge pair')
 admit_bridge(out,stop['membership'])
 # This producer connection is distinct from the peer's later complete-prefix
 # independent replay. Its immutable source fixes the original same-root lift.
 return model,box,{'audited_root':audit,'bridge':{'root':str(BRIDGE),'pins':BPINS,'membership':entry(BRIDGE/'run/membership-stop.json')},'connection_evidence':'producer full470 membership in107-chart locus; independent full-prefix audit separately reported'}

DIAG=BASE/'attempt001-diagnosis'
DPINS={'source':'8ee0c5bb1a77a63faf5b32a95da609967bf14ce7','outer':'c06a47b85832285783f0147141f8b2c88791ecbb3b5665ddae8235f2c2352315','worker':'8a8d2add8267d6fc7445c216d950139053962b46c1e1495442d8e9b4bf5a53f5'}

def repair_weights(diagnostic):
    require(diagnostic['scaling']['iterations']==512 and diagnostic['scaling']['collatz_max']<.95,'fixed numerical scaling proposal')
    proposal=diagnostic['resolvent_proposal'];radii=proposal['radii']
    require(proposal['nonnegative']is True and len(radii)==704,'full nonnegative resolvent proposal')
    require(all(type(v)in(int,float)and math.isfinite(v)and v>0 for v in radii),'strict positive finite proposed radii')
    # New physical radii=r_old_resolvent/8; base radius2^-11 gives weights256*r.
    weights=[Fraction.from_float(float(v))*256 for v in radii]
    require(all(v>0 and v<=1 for v in weights),'bounded exact dyadic weights')
    return [str(v)for v in weights]

def bind_repair():
    inventory(DIAG,DPINS['outer']);inventory(DIAG/'run',DPINS['worker'])
    rt=read(DIAG/'run/runtime.json');la=read(DIAG/'launcher-result.json');out=read(DIAG/'run/outcome.json');diagnostic=read(DIAG/'run/diagnostic.json')
    require(rt['commit']==la['commit']==DPINS['source']and la['exit_code']==0 and la['owned_live_pids']==[],'completed fixed saved diagnosis')
    require(out['passed']is True and out['native_calls']==rt['model_calls']==0 and out['resolvent_available']is True,'saved-only proposal admission')
    for n,row in rt['sources'].items():
        require(entry(ROOT/n)==entry(DIAG/'run/source'/n)==row,'unchanged diagnostic source '+n)
        require(hashlib.sha256(subprocess.check_output(['git','show',DPINS['source']+':'+n],cwd=ROOT)).hexdigest()==row['sha256'],'historical diagnostic source '+n)
    weights=repair_weights(diagnostic)
    return weights,{'root':str(DIAG),'pins':DPINS,'diagnostic':entry(DIAG/'run/diagnostic.json'),'exact_weights':weights,'settings':SETTINGS,
        'amendment':'Quarter the first parameter halfwidth and divide saved positive-resolvent radii by8. Exact dyadic proposals only; fresh full704 proof and original-root anchor remain mandatory. Same profiled algorithm/dimensions/call count, changed interval geometry.'}

def main(args):
 identity=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
 require(args.out==BASE/'attempt002-target/run','exclusive target path');out=args.out;out.mkdir(parents=True,exist_ok=False)
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
 def tick():ticks[0]+=1;require(ticks[0]<=217,'216 native passes and1preconditioner maximum');check()
 def observe(kind,row):
  counts[kind]=counts.get(kind,0)+1;save(f'events/{serial[0]:04d}-{kind}.json.gz',row);serial[0]+=1;check()
 save('runtime.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':identity,'mode':'target','settings':SETTINGS,'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'pid':os.getpid(),'worker_seconds':240,'sampled_rss_kib':6*1024**2,'hard_as_bytes':10*1024**3,'tick_cap':217})
 for n in SOURCES:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used[0]+=p.stat().st_size
 result={'passed':False,'status':'unresolved','scope':'One connected native primitive18 unit-pair tube only; complementary spectrum on the finite interval and complete enclosing-window geometry remain separate.'}
 try:
  profile=bind_profile(args,identity);save('profile-binding.json',profile);weights,repair=bind_repair();save('repair-binding.json',repair);check()
  with ctx.workprec(256):
   inp,box,binding=bind_root();save('inputs.json',{'map_inputs':inp,'original_41d_box':vector_json(box),'bindings':binding,'settings':SETTINGS});check()
   model=TwoRateCuspMap(inp);flat,lift=native_lift(model,box,18,tick,observe);save('native-lift.json',lift)
   require(lift['closure_contains_zero'],'conditional normalized Jordan lift must enclose closure')
   initial=[exact_mid(v)for v in flat];anchor={'box':flat,'period':18,'dimension':13,'parameter':'0'}
   save('chart-inputs.json',{'initial':initial,'anchor':lift,'weights':weights,'settings':SETTINGS})
   proof,_=unit_pair_tube(model,initial,**SETTINGS,weights=weights,anchor=anchor,tick=tick,observe=observe);save('certificate.json.gz',proof);check()
   require(ticks[0]==217 and counts['direction_return']==216,'complete fixed full704 acquisition')
   result.update(passed=proof['passed'],status='qualified_unit_pair_tube'if proof['passed']else'completed_bounds_failed',gates=proof['gates'],contraction=proof['contraction'],strict_inclusions=sum(proof['coordinate_inclusions']),parameter_endpoints=proof['parameter_endpoints'],endpoint_actual_rates=[v[-2:]for v in proof['endpoint_root_boxes']])
 except BaseException as exc:result.update(passed=False,status='failed_or_interrupted',failure={'type':type(exc).__name__,'message':str(exc)})
 result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks[0],events=counts,stored_bytes_before_final=used[0])
 if result['elapsed_seconds']>=240 or result['peak_rss_kib']>=6*1024**2:result['passed']=False
 save('outcome.json',result,True);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p)for p in sorted(out.rglob('*'))if p.is_file()}},True)
 for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
 out.chmod(0o555);print(json.dumps(result));return 0 if result['passed']else 2

def parser():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--profile-manifest',required=True);p.add_argument('--profile-outer-manifest',required=True);return p
if __name__=='__main__':raise SystemExit(main(parser().parse_args()))
