#!/usr/bin/env python3
"""One source-bound known endpoint refinement inside maintained continuation."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,gc,hashlib,json,resource,signal,subprocess,sys,time
from pathlib import Path
from flint import ctx
import probe_native705_connected as W
import refine_native705_rechart_endpoint as F
ROOT=W.ROOT;need=W.need;entry=W.entry;read=W.read
POLICY_PATH=ROOT/'docs/plans/native705-connected-policy.json'

def source_archive(run,source,identity):
 rt=read(run/'runtime.json');need(rt['commit']==source,'exact proof source')
 for n,v in rt['sources'].items():need(entry(run/'source'/n)==v and hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'same original runtime source '+n)
 for n in W.W.C.NATIVE_SOURCES:need(identity[n]==rt['sources'][n],'same actual native equations '+n)
 return rt

def binding(card,identity,check):
 need(card['kind']=='native705-complete-root-projected-fallback-v1'and card['source']==subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'same fresh fallback source')
 root=W.BASE/f"attempt{card['attempt']:03d}";need(card['current_root']==str(root)and card['mode']=='rotated_gauge','one exact current rotated root')
 W.inventory(root/'run',card['current_worker']['sha256']);rt=source_archive(root/'run',card['source'],identity);ic=read(root/'input-card.json');done=read(root/'run/outcome.json');new=read(root/'run/certificate.json.gz')
 need(ic['source']==card['source']and ic['map_inputs']==card['model']and ic['parent']==card['parent']and ic['mode']=='rotated_gauge'and ic['anchor']=={'box':card['known']}and done['ticks']==331 and done['events']==W.EXPECTED and done['passed']and W.W.uniform(new),'unchanged successful complete current root, native model and projected known box')
 oldroot=Path(card['parent']['root']);need(oldroot!=root,'separate already accepted predecessor');W.inventory(oldroot,card['parent']['outer']);W.inventory(oldroot/'run',card['parent']['worker']);ort=source_archive(oldroot/'run',card['parent']['source'],identity);oi=read(oldroot/'input-card.json');od=read(oldroot/'run/outcome.json');old=read(oldroot/'run/certificate.json.gz')
 need(oi['map_inputs']==card['model']and od['ticks']==331 and od['events']==W.EXPECTED and od['passed']and W.W.uniform(old),'same complete old primitive18 unit-pair root')
 need(card['known']==old['endpoint_root_boxes'][1]and old['parameter_endpoints'][1]==card['old_parameter']and len(card['known'])==705,'literal known right endpoint of original root')
 paths=list((oldroot/'run/events').glob('*-cubic_correlated_domain.json.gz'));need(len(paths)==1,'unique original full error acquisition');domain=read(paths[0]);check()
 return oldroot/'run',old,new,domain

def main(args):
 need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed continuation source');need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');need(1<=args.attempt<=24,'fixed bounded root index')
 folder=W.BASE/f'attempt{args.attempt:03d}'/'fallback';out=folder/'run';need(args.out==out and entry(folder/'input-card.json')['sha256']==args.card_sha,'exact fallback card and exclusive path');card=read(folder/'input-card.json');need(card['attempt']==args.attempt,'same indexed fallback');out.mkdir(parents=True,exist_ok=False);F.OUT=out
 resource.setrlimit(resource.RLIMIT_AS,(10*1024**3,)*2);w=F.Work();signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,300);result=dict(passed=False,status='incomplete')
 try:
  identity={n:entry(ROOT/n)for n in W.SOURCES};source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(commit=source,sources=identity,precision=480,pid=os.getpid(),python=sys.version,python_flint=F.flint.__version__,flint=F.flint.__FLINT_VERSION__,threads=dict.fromkeys(F.THREADS,'1'),nice=os.getpriority(os.PRIO_PROCESS,0)))
  for n,v in identity.items():need(hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'committed exact source');p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());w.used+=p.stat().st_size
  # No duplicate synthetic work: the exact complete pilot profile and unchanged
  # entire source closure are authenticated by the parent before this card.
  prof=card['profile'];need(prof==read(POLICY_PATH)['fallback_forecast']and prof==read(W.BASE/'fallback-profile-binding.json'),'same completed parent-bound measured endpoint unit');need(prof['forecast_seconds']<300 and prof['forecast_bytes']<F.CAP and prof['forecast_peak_bytes']<6*1024**3,'unchanged per-fallback measured limits');w.save('profile-binding.json',prof);w.save('inputs.json',card)
  with ctx.workprec(480):
   oldrun,old,new,domain=binding(card,identity,w.check);localized,join,q=F.refine_pair(card['model'],oldrun,old,new,card['known'],domain,w);need(w.native==36 and w.saved==72,'one36-phase point residual and72 saved native replays');passed=bool(join['passed']and len(join['coordinate_inclusions'])==705 and all(join['coordinate_inclusions']));w.save('known-endpoint.json',dict(box=F.vector_json(localized),old_parameter=card['old_parameter']))
   result.update(passed=passed,status='same_known_endpoint_joined'if passed else'completed_projected_join_unresolved',strict_inclusions=sum(join['coordinate_inclusions']),weighted_bound=F.ball_json(q),scope='Conditional old known endpoint enclosure plus strict projected membership in already completed new uniform tube. No uniform root retry.')
 except BaseException as exc:result.update(status='failed_or_interrupted',failure=type(exc).__name__+': '+str(exc))
 finally:signal.setitimer(signal.ITIMER_REAL,0)
 result.update(seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,native_calls=w.native,saved_replay_calls=w.saved,events=w.events);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(out)):entry(p)for p in sorted(out.rglob('*'))if p.is_file()}),True)
 for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
 out.chmod(0o555);print(json.dumps(result));return 0 if result['passed']else 2

def parser():
 p=argparse.ArgumentParser();p.add_argument('--attempt',type=int,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--card-sha',required=True);return p
if __name__=='__main__':raise SystemExit(main(parser().parse_args()))
