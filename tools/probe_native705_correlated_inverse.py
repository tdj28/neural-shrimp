#!/usr/bin/env python3
"""One wider fresh correlated705 tube from the independently audited24 prefix."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,gc,gzip,hashlib,json,resource,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import flint,numpy as np
from flint import arb,arb_mat,ctx
import profile_native705_correlated_inverse as P
import probe_native705_rotated as W
from neural_shrimp.rigorous_correlated_pal import correlated_tube
from neural_shrimp.rigorous_pal_rechart import localize_endpoint
from neural_shrimp.rigorous_eigenpair_march import decode
from neural_shrimp.rigorous_orbits import exact,vector_json
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
BASE=P.BASE;TARGET=BASE/'attempt001-target/run';CAP=1024**3
SETTINGS={'period':18,'precision':256,'parameter':'15/2048','halfwidth_exponent':-5,'radius_exponent':-10,'newton_steps':0}
AUDIT=Path('/data2/neural-shrimp-data/audits/native705-forward-independent/attempt004')
APINS={'source':'cfc5bc32d7c02142f47073b02d25b9dc60cc9c7b','outer':'640bb20df7c75308e74bb8c5a40636f1dac7071dd459b584c2067f549a2dc38b','worker':'3874b35953b0f1b63f3751b64093cf54d9108c7aa88603960ff5655e103c7750','endpoint':'d56ed2992697924b767e16b76e7cd1271384899b039e52e8d416f790318ac9be'}
OUTER='1071d42d45a686b4d30869d13931a29fa4609db685bce24d17fb4070029e2643';OLD_SOURCE='2c9fc56dd9e8fd75434970487d4a0fafb21fbe6d'
SOURCES=list(dict.fromkeys(P.SOURCES+W.SOURCES+['src/neural_shrimp/rigorous_parameter_localization.py','src/neural_shrimp/rigorous_connection.py','src/neural_shrimp/rigorous_pal_rechart.py','tools/probe_native705_correlated_inverse.py','tools/launch_native705_correlated_inverse.py','tests/test_native705_correlated_inverse_driver.py','docs/plans/native705-correlated-inverse-target.md']))
read=W.read;entry=W.entry;inventory=W.inventory;require=W.require
NATIVE_SOURCES=tuple('src/neural_shrimp/'+n+'.py'for n in ('__init__','models','controls','rigorous_orbits','rigorous_folds','rigorous_cusps','rigorous_eigenpair_locus','rigorous_eigenpair_path','rigorous_unit_pair','rigorous_unit_pair_pal'))

def same_native_source(identity,parent):
 for n in NATIVE_SOURCES:require(n in identity and n in parent and identity[n]==parent[n],'identical native map/gradient/closure source '+n)

def admit_profile(out):
 require(out['passed']is True and out['target_acquired']is False and out['ticks']==629,'complete sole synthetic629 profile')
 e=out['events'];require(e['direction_return']==180 and e['pal_path_phase_return']==252 and e['pal_chart_return']==1 and e['gradient_path_return']==e['gradient_path_input']==144 and e['gradient_path_assembly_return']==4 and e['correlated_chart_return']==e['correlated_domain']==2 and e['profile_gradient_return']==e['profile_gradient_input']==e['pal_profile_path_return']==24,'complete actual acquisitions and two full705 proof controls')
 require(0<out['elapsed_seconds']<240 and 0<out['peak_rss_kib']<6*1024**2,'profile actual limits')
 forecast=P.projection(out['chart_times'],out['gradient_times'],out['path_times'],out['peak_rss_kib'],out['stored_bytes_before_final'])
 require(forecast==out['projection']and forecast['resource_admitted'],'strict recomputed measured forecast');return forecast

def bind_profile(args,identity):
 root=BASE/'attempt001-profile';inventory(root,args.profile_outer);inventory(root/'run',args.profile_worker)
 rt=read(root/'run/runtime.json');la=read(root/'launcher-result.json');out=read(root/'run/outcome.json')
 require(rt['commit']==la['commit']and rt['mode']==la['mode']=='synthetic_only'and la['exit_code']==0 and not la['owned_live_pids'],'complete exact synthetic execution')
 require(set(rt['sources'])==set(P.SOURCES),'full measured source closure')
 for n in P.SOURCES:
  require(identity[n]==rt['sources'][n]==entry(root/'run/source'/n),'same profile source '+n)
  require(hashlib.sha256(subprocess.check_output(['git','show',rt['commit']+':'+n],cwd=ROOT)).hexdigest()==identity[n]['sha256'],'historical measured source '+n)
 for k,v in {'settings':P.SETTINGS,'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1')}.items():require(rt[k]==v,'same measured '+k)
 return {'root':str(root),'source':rt['commit'],'outer':args.profile_outer,'worker':args.profile_worker,'forecast':admit_profile(out)}

def bind_parent(tick,save,identity):
 inventory(W.BASE,OUTER);packet=read(W.BASE/'outcome.json')
 require(packet['source']==OLD_SOURCE and packet['connected_charts']==24 and len(packet['batches'])==4 and not packet['passed']and not packet['owned_live_pids'],'complete immutable24 plus failed8 packet')
 b=packet['batches'][2];require(b['passed']and b['connected_count']==8 and b['manifest']['sha256']=='fe94e9326f6bd7b71919f501a0a1689625926bda9d2af612c9b1d0f97ebd9607','exact final passing batch')
 parent=W.BASE/'batch003/chart008';native_runtime=W.C.archive_bound(parent,OLD_SOURCE);same_native_source(identity,native_runtime['sources']);done=read(parent/'run/outcome.json');old=read(parent/'run/certificate.json.gz')
 require(done['passed']and done['ticks']==253 and done['strict_inclusions']==705 and W.uniform_admission(old,decode(old['whole_box'][-1])>0),'complete final known curve')
 inventory(AUDIT,APINS['outer']);inventory(AUDIT/'run',APINS['worker']);W.C.archive_bound(AUDIT,APINS['source'])
 ao=read(AUDIT/'run/outcome.json');al=read(AUDIT/'launcher-result.json');ab=read(AUDIT/'run/input-binding.json');ep=read(AUDIT/'run/connected-endpoint.json')
 require(ao['passed']and ao['connected']and ao['connected_count']==8 and ao['connected_parameter_endpoints']==['-1/2048','255/2048']and al['exit_code']==0 and not al['owned_live_pids'],'complete independent24 endpoint')
 require(ab['card']['root']==str(parent.parent)and ab['card']['outer']==b['manifest']['sha256']and ab['card']['source']==OLD_SOURCE and ab['card']['qualified_prefix']==8 and ab['card']['charts'][-1]['worker']==entry(parent/'run/manifest.json')['sha256'],'same audited native final curve')
 require(entry(AUDIT/'run/connected-endpoint.json')['sha256']==APINS['endpoint']and len(ep['endpoint'])==705,'literal full independent endpoint')
 for k in ['normal','origin','parameter_center','parameter_endpoints','settings']:require(ep['chart'][k]==old[k],'same independent endpoint '+k)
 c,s,a=[[decode(v)for v in old[k]]for k in ['center','slope','curvature']];mat=lambda k:arb_mat([[decode(v)for v in row]for row in old[k]])
 localized,proof=localize_endpoint(mat('preconditioner'),mat('whole_jacobian'),mat('quadratic_center_coefficients'),mat('quadratic_whole_coefficients'),c,s,a,[exact(v)for v in old['weights']],arb((1,old['settings']['radius_exponent'])),arb((1,old['settings']['halfwidth_exponent'])),tick=tick)
 localized=[x.intersection(decode(y))for x,y in zip(localized,ep['endpoint'],strict=True)];proof['retained_independent_endpoint']=vector_json(localized);save('known-endpoint-localization.json',proof)
 candidate=W.BASE/'batch004/chart001';W.C.archive_bound(candidate,OLD_SOURCE);fail=read(candidate/'run/outcome.json');proposal=read(candidate/'run/certificate.json.gz');card=read(candidate.parent/'input-card.json')
 require(not fail['passed']and fail['status']=='completed_bounds_failed'and fail['ticks']==253 and fail['strict_inclusions']==704 and not proposal['gates']['uniform_full_native_unit_pair_locus'],'retained exact failed proposal is not evidence')
 require(card['source']==OLD_SOURCE and card['batch']==4 and card['parent']['root']==str(parent)and card['parent']['manifest']==entry(parent/'run/manifest.json'),'same actual preceding branch')
 for k in ['normal','origin','weights']:require(card[k]==proposal[k],'source-bound failed proposal '+k)
 require(proposal['parameter_center']==SETTINGS['parameter']and proposal['parameter_endpoints']==['-1/2048','31/2048']and proposal['settings']['halfwidth_exponent']==-7 and proposal['settings']['radius_exponent']==-10,'exact changed-domain proposal')
 inp=read(candidate/'run/inputs.json')['map_inputs'];require(inp==card['map_inputs']==read(parent/'run/inputs.json')['map_inputs'],'literal same physical map/data/rates')
 save('input-binding.json',{'packet':{'root':str(W.BASE),'outer':OUTER,'source':OLD_SOURCE},'parent':{'root':str(parent),'worker':entry(parent/'run/manifest.json')},'independent':dict(root=str(AUDIT),**APINS),'proposal':{'root':str(candidate),'worker':entry(candidate/'run/manifest.json')},'settings':SETTINGS,'prior_failure_preserved':True})
 return inp,proposal,{'box':localized,'dimension':13,'period':18}

def main(args):
 require(args.out==TARGET,'exclusive fixed target path');require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source');require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
 out=args.out;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();ticks=[0];serial=[0];used=[0];counts={};resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,)*2)
 def stop(*_):raise TimeoutError('240s target cap')
 signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,240)
 def check():
  if time.monotonic()-start>=240:raise TimeoutError('240s target cap')
  if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=6*1024**2:raise MemoryError('6GiB RSS cap')
 def save(name,row,final=False):
  raw=json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode();data=gzip.compress(raw,mtime=0)if name.endswith('.gz')else raw+b'\n';require(used[0]+len(data)<CAP-(0 if final else 4*1024**2),'1GiB raw cap');p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
  used[0]+=len(data)
 def tick():ticks[0]+=1;require(ticks[0]<=148,'144 native and4 algebra calls');check()
 def observe(k,v):counts[k]=counts.get(k,0)+1;save(f'events/{serial[0]:04d}-{k}.json.gz',v);serial[0]+=1;check()
 source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();identity={n:entry(ROOT/n)for n in SOURCES}
 for n,v in identity.items():require(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==v['sha256'],'committed source '+n)
 save('runtime.json',{'commit':source,'sources':identity,'settings':SETTINGS,'pid':os.getpid(),'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'worker_seconds':240,'rss_kib':6*1024**2,'hard_as_bytes':12*1024**3})
 for n in SOURCES:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used[0]+=p.stat().st_size
 result={'passed':False,'connected':False,'status':'unresolved'}
 try:
  save('profile-binding.json',bind_profile(args,identity));check()
  with ctx.workprec(256):
   inp,p,anchor=bind_parent(tick,save,identity);save('inputs.json',{'map_inputs':inp,'anchor':vector_json(anchor['box']),'settings':SETTINGS})
   c,s,a=[[decode(v)for v in p[k]]for k in ['center','slope','curvature']];C0=arb_mat([[decode(v)for v in row]for row in p['preconditioner']]);normal=[exact(v)for v in p['normal']];origin=[exact(v)for v in p['origin']];weights=[exact(v)for v in p['weights']];del p;gc.collect()
   proof=correlated_tube(TwoRateCuspMap(inp),c,s,a,C0,period=18,normal=normal,origin=origin,parameter=SETTINGS['parameter'],halfwidth_exponent=-5,radius_exponent=-10,weights=weights,anchor=anchor,tick=tick,observe=observe);save('certificate.json.gz',proof)
   require(ticks[0]==148 and counts['gradient_path_input']==counts['gradient_path_return']==72 and counts['gradient_path_assembly_return']==2 and counts['pal_path_phase_return']==72 and counts['pal_path_assembly_return']==2,'complete fresh changed-domain acquisitions')
   result.update(passed=proof['passed'],connected=proof['passed'],status='connected_correlated705'if proof['passed']else'completed_bounds_failed',gates=proof['gates'],contraction=proof['bound']['contraction'],strict_inclusions=sum(proof['bound']['coordinate_inclusions']),anchor_inclusions=sum(proof['anchor_check']['coordinate_inclusions']),parameter_endpoints=proof['parameter_endpoints'],physical_endpoint=proof['endpoint_root_boxes'][1][-3:])
 except BaseException as exc:result.update(status='failed_or_interrupted',error=type(exc).__name__+': '+str(exc))
 finally:signal.setitimer(signal.ITIMER_REAL,0)
 result.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks[0],events=counts,stored_bytes_before_final=used[0])
 if result['seconds']>=240 or result['peak_rss_kib']>=6*1024**2:result.update(passed=False,connected=False)
 save('outcome.json',result,True);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p)for p in sorted(out.rglob('*'))if p.is_file()}},True)
 for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
 out.chmod(0o555);print(json.dumps(result));return 0 if result['passed']else 2

def parser():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--profile-outer',required=True);p.add_argument('--profile-worker',required=True);return p
if __name__=='__main__':raise SystemExit(main(parser().parse_args()))
