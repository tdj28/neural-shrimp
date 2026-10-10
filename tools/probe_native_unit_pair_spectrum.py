#!/usr/bin/env python3
"""Saved-only complementary spectrum on the exact passed finite704 tube."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import gc,gzip,json,resource,signal,subprocess,sys,time
from pathlib import Path
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import probe_native_unit_pair_repair as R
from neural_shrimp.rigorous_unit_pair_spectrum import native_complement
from neural_shrimp.rigorous_eigenpair_march import decode
from neural_shrimp.rigorous_orbits import exact,vector_json,ball_json
require=R.require;entry=R.entry;inventory=R.inventory
BASE=R.BASE;PARENT=BASE/'attempt002-target';OUT=BASE/'attempt001-spectrum/run'
PINS={'source':'a7b5a169032facb39e6476b3aae7e58654750fbf','outer':'73c02be51125e80db4707709e0cdaec8d39f69d453e35e2ef5a4738ed1405447','worker':'9ee39499eb309e8283f2591857b4bcecb43aedfbfe05136d480899d2a39d92b6'}
SOURCES=list(dict.fromkeys(R.SOURCES+['src/neural_shrimp/rigorous_unit_pair_spectrum.py','tests/test_rigorous_unit_pair_spectrum.py','docs/plans/native-unit-pair-spectrum.md','tools/probe_native_unit_pair_spectrum.py','tools/launch_native_unit_pair_spectrum.py','tests/test_native_unit_pair_spectrum_driver.py','docs/plans/native-unit-pair-spectrum-attempt.md']))

def read(path):
 with gzip.open(path,'rt')if path.suffix=='.gz'else path.open()as f:return json.load(f)

def admit(proof,out):
 require(out['passed']is True and out['status']=='qualified_unit_pair_tube'and out['ticks']==217 and out['events']['direction_return']==216,'complete qualified704 outcome')
 require(proof['schema']=='neural-shrimp-native-unit-pair-tube-v1'and proof['passed']is True,'passed finite tube')
 require(proof['gates']==out['gates']==dict.fromkeys(('anchor_identified','primitive_period','realified_frame_independent','uniform_normalized_unit_pair'),True),'all actual finite tube gates')
 require((proof['dimension'],proof['period'],proof['shooting_dimension'])==(13,18,704),'exact native dimensions')
 require(proof['parameter_endpoints']==['0','1/32768']and proof['settings']=={k:R.SETTINGS[k]for k in ('precision','newton_steps','halfwidth_exponent','radius_exponent')},'exact changed-domain settings')
 require(proof['coordinate_inclusions']==[True]*704 and proof['anchor_check']['coordinate_inclusions']==[True]*704 and proof['anchor_check']['passed']is True and decode(proof['contraction'])<1,'strict root and full original-root anchor')
 require([v['divisor']for v in proof['divisors']]==[1,2,3,6,9]and all(v['excluded']is True for v in proof['divisors']),'all proper native divisors')

def native_blocks(proof,rows):
 n=proof['dimension'];p=proof['period'];flat=proof['whole_box'];J=proof['whole_jacobian'];require(len(rows)==2*p and len(flat)==3*n*p+2,'full whole-domain native phase count')
 blocks=[]
 for k in range(p):
  q,r=rows[2*k:2*k+2];off=3*n*k
  require(q['phase']==r['phase']==k and q['direction']=='q'and r['direction']=='r','native direction and phase order')
  require(q['input']==flat[off:off+2*n]+flat[-2:]and r['input']==flat[off:off+n]+flat[off+2*n:off+3*n]+flat[-2:],'exact same full proof-domain native acquisition')
  for row in(q,r):require(len(row['block'])==2*n and all(len(v)==2*n for v in row['block']),'full local derivative block')
  A=[v[:n]for v in q['block'][:n]]
  require(A==[v[:n]for v in r['block'][:n]],'same state derivative from both directions')
  require(A==[v[off:off+n]for v in J[off:off+n]],'same whole proof Jacobian state block')
  blocks.append(A)
 return blocks

def bind():
 inventory(PARENT,PINS['outer']);inventory(PARENT/'run',PINS['worker'])
 rt=read(PARENT/'run/runtime.json');la=read(PARENT/'launcher-result.json');out=read(PARENT/'run/outcome.json');inp=read(PARENT/'run/inputs.json')
 require(rt['commit']==la['commit']==PINS['source']and la['exit_code']==0 and la['owned_live_pids']==[]and la['stop_reason']is None,'completed exact finite tube source')
 require(rt['settings']==R.SETTINGS and rt['threads']==dict.fromkeys(R.THREADS,'1')and rt['nice']>=10,'actual target settings')
 require(set(rt['sources'])==set(R.SOURCES),'entire target source inventory')
 for n,row in rt['sources'].items():
  require(entry(ROOT/n)==entry(PARENT/'run/source'/n)==row,'unchanged archived target source '+n)
  import hashlib
  require(hashlib.sha256(subprocess.check_output(['git','show',PINS['source']+':'+n],cwd=ROOT)).hexdigest()==row['sha256'],'historical target source '+n)
 require(inp['bindings']['audited_root']['pins']==R.U.APINS and inp['bindings']['bridge']['pins']==R.BPINS,'same independent Jordan root and actual470 branch binding')
 proof=read(PARENT/'run/certificate.json.gz');admit(proof,out)
 files=sorted((PARENT/'run/events').glob('*-direction_return.json.gz'));require(len(files)==216,'complete acquired native directions')
 wholefiles=files[-36:];rows=[read(p)for p in wholefiles];blocks=native_blocks(proof,rows)
 flat=proof['whole_box'];del proof;gc.collect()
 return flat,blocks,{'parent':str(PARENT),'pins':PINS,'proof':entry(PARENT/'run/certificate.json.gz'),'whole_direction_files':{str(p.relative_to(PARENT/'run')):entry(p)for p in wholefiles},'root_and_bridge':inp['bindings']}

def main():
 require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed source');require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=[0];resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
 def stop(*_):raise TimeoutError('60s saved quotient cap')
 signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,60)
 def save(name,row):
  raw=(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode();require(used[0]+len(raw)<32*1024**2,'32MiB saved quotient output cap')
  with(OUT/name).open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  used[0]+=len(raw)
 save('runtime.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'pid':os.getpid(),'sources':{n:entry(ROOT/n)for n in SOURCES},'threads':dict.fromkeys(R.THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'precision':384,'native_calls':0})
 for n in SOURCES:p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used[0]+=p.stat().st_size
 result={'passed':False,'native_calls':0,'status':'unresolved','scope':'Conditional uniform complementary spectrum on the passed finite primitive18 unit-pair tube. No signed finite-interval NS coefficient, stability sides or complete window.'}
 try:
  flat,blocks,bindings=bind();save('bindings.json',bindings)
  with ctx.workprec(384):
   box=[decode(v)for v in flat];A=[arb_mat([[decode(v)for v in row]for row in mat])for mat in blocks];t=exact(R.SETTINGS['parameter'])+arb(0,arb((1,R.SETTINGS['halfwidth_exponent'])))
   save('inputs.json',{'whole_box':flat,'native_blocks':blocks,'parameter_interval':ball_json(t),'exact_parameter_endpoints':['0','1/32768']})
   proof=native_complement(A,box,18,13,t);save('quotient.json',proof)
   result.update(passed=proof['passed']and proof['complement_attracting'],status='qualified_finite_attracting_complement'if proof['passed']and proof['complement_attracting']else'completed_complement_unresolved',unit_circle_excluded=proof['passed'],complement_attracting=proof['complement_attracting'],complementary_discs=len(proof['spectral_exclusion']['discs']))
 except BaseException as exc:result.update(error=type(exc).__name__+': '+str(exc))
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  if result['seconds']>=60 or result['peak_rss_kib']>=4*1024**2:result['passed']=False
  save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(OUT)):entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}})
  for p in sorted(OUT.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
  OUT.chmod(0o555)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
