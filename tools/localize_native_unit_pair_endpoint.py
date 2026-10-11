#!/usr/bin/env python3
"""One fixed native704 endpoint localization and existing705 curve join."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import gc,gzip,json,resource,signal,subprocess,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import probe_native_unit_pair_pal as C
import profile_native_unit_pair_pal as P
import flint,numpy as np
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_unit_pair import assemble,phase
from neural_shrimp.known_endpoint_localization import localize
from neural_shrimp.rigorous_unit_pair_pal import anchor_test
from neural_shrimp.rigorous_orbits import exact,eye,vector_json,matrix_json,ball_json
from neural_shrimp.rigorous_eigenpair_locus import exact_mid,require
from neural_shrimp.rigorous_eigenpair_march import decode
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
BASE=P.BASE/'endpoint-localization';TARGET=BASE/'attempt001/run';PARENT=P.BASE/'attempt003-target'
PINS={'source': 'b03d990f5d5801db5addd6cd2c18e718bd4ba8ad', 'outer': 'a9af91fe221aa4f9dfaadd9b1fbb8d29ef50f84832a404d8780655e2e84f62aa', 'worker': '2b5344c9a2f1a699abadceb837ed9c75674f24ac14292c3c2493d7779e49734d'}
SOURCES=list(dict.fromkeys(C.SOURCES+['src/neural_shrimp/rigorous_connection.py','src/neural_shrimp/rigorous_parameter_localization.py','src/neural_shrimp/known_endpoint_localization.py','tools/localize_native_unit_pair_endpoint.py','tools/launch_native_unit_pair_endpoint.py','tests/test_known_endpoint_localization.py','docs/plans/native-unit-pair-endpoint.md','docs/reviews/known-endpoint-localization-provenance.json']))
entry=C.entry;read=C.read;inventory=C.inventory;CAP=128*1024**2;THREADS=C.THREADS

def matrix(rows):return arb_mat([[decode(v)for v in row]for row in rows])

def profile(tick,observe):
 start=time.monotonic();n=704;e=arb((1,-16));pre=arb_mat([[arb(int(i==j))+e for j in range(n)]for i in range(n)]);jac=arb_mat([[arb(int(i==j))-e for j in range(n)]for i in range(n)])
 c=[arb(1)]*n;known=[arb(1,arb((1,-12)))]*n;whole=[arb(1,arb((1,-10)))]*n
 test=localize(pre,jac,arb_mat(n,1),c,known,whole,[arb(1)]*n);require(test['passed'],'full704 unrelated localization')
 inp,x,q,r,rates=P.gd_fixture();model=TwoRateCuspMap(inp)
 for k in range(18):phase(model,x,q,r,rates,tick,observe,k)
 elapsed=time.monotonic()-start;return {'passed':True,'seconds':elapsed,'forecast_seconds':4*elapsed+4*15.713+20,'dimension':704,'native_profile_calls':36,'target_arrays_unopened':True}

def join(chart,box):
 require(chart['schema']=='neural-shrimp-native-unit-pair-pal-v1'and chart['shooting_dimension']==705,'exact full705 root family')
 require(chart['gates']=={'uniform_full_native_unit_pair_locus':True,'primitive_period':True,'realified_frame_independent':True,'anchor_identified':False},'retained unjoined uniform proof only')
 require(chart['parameter_center']=='31/4096'and chart['parameter_endpoints']==['-1/4096','63/4096']and chart['settings']=={'precision':256,'newton_steps':2,'halfwidth_exponent':-7,'radius_exponent':-10},'exact passed uniform domain')
 require(len(box)==705 and decode(chart['whole_box'][-1])>0,'full known root and positive angle')
 return anchor_test({'box':box,'dimension':13,'period':18},[decode(v)for v in chart['center']],[decode(v)for v in chart['slope']],[decode(v)for v in chart['curvature']],[exact(v)for v in chart['normal']],[exact(v)for v in chart['origin']],exact(chart['parameter_center']),arb((1,-7)),[decode(v)for v in chart['coordinate_radii']],13,18)

def main():
 require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed source');require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
 TARGET.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=[0];ticks=[0];counts={};seq=[0];stop=[False];resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
 signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGALRM,lambda *_:stop.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,240)
 def check():require(not stop[0]and time.monotonic()-start<240 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'240s/4GiB endpoint cap')
 def save(name,value):
  b=json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode();b=gzip.compress(b,mtime=0)if name.endswith('.gz')else b+b'\n';require(used[0]+len(b)<CAP,'128MiB endpoint cap')
  p=TARGET/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
  used[0]+=len(b)
 def tick():ticks[0]+=1;require(ticks[0]<=144,'36profile plus108 endpoint native ceiling');check()
 def observe(kind,row):counts[kind]=counts.get(kind,0)+1;save(f'events/{seq[0]:04d}-{kind}.json.gz',row);seq[0]+=1;check()
 source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();identity={n:entry(ROOT/n)for n in SOURCES}
 for n,v in identity.items():require(hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'exact committed closure '+n)
 save('runtime.json',{'commit':source,'sources':identity,'precision':384,'pid':os.getpid(),'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':dict.fromkeys(THREADS,'1'),'python':sys.version,'numpy':np.__version__,'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__,'tick_cap':144})
 for n in SOURCES:p=TARGET/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used[0]+=p.stat().st_size
 result={'passed':False,'status':'unresolved'}
 try:
  with ctx.workprec(384):
   cost=profile(tick,observe);save('profile.json',cost);require(cost['forecast_seconds']<240,'full704 and native384 measured admission');check()
   inp,full,old,binding=C.bind_root();known=full[:-1];t=full[-1];require(t==exact('1/32768'),'exact inherited endpoint parameter')
   pre=matrix(old['preconditioner']);jac=matrix(old['whole_jacobian']);whole=[decode(v)for v in old['whole_box']];weights=[exact(v)for v in old['weights']];del old;gc.collect()
   save('inputs.json',{'map_inputs':inp,'binding':binding,'known_endpoint':vector_json(full),'parameter':ball_json(t),'preconditioner_source':'exact source-bound original704 certificate','whole_jacobian_source':'same source-bound original704 whole domain'})
   center=[v.mid()for v in known];model=TwoRateCuspMap(inp)
   # Two bounded proposals reduce midpoint residual; existence remains inherited.
   for it in range(3):
    require(all(whole[i].contains(center[i])for i in range(704)),'entire point evaluation inside authenticated old domain')
    g,j,_,_=assemble(model,center,18,t,tick,observe);save(f'point-{it}.json.gz',{'center':vector_json(center),'value':matrix_json(g),'jacobian':matrix_json(j),'t':ball_json(t)});del j
    if it<2:
     correction=pre*g;save(f'newton-{it}.json',{'correction':matrix_json(correction)});center=[(center[i]-correction[i,0]).mid()for i in range(704)]
   require(ticks[0]==144 and counts=={'direction_return':144},'all144 complete native profile and point returns')
   proof=localize(pre,jac,g,center,known,whole,weights);save('localization.json',proof);small=[decode(v)for v in proof['box']]+[t];del pre,jac,whole;gc.collect();check()
   inventory(PARENT,PINS['outer']);inventory(PARENT/'run',PINS['worker']);C.archive_bound(PARENT,PINS['source']);la=read(PARENT/'launcher-result.json');out=read(PARENT/'run/outcome.json');ci=read(PARENT/'run/inputs.json')
   require(la['commit']==PINS['source']and la['exit_code']==2 and not la['owned_live_pids']and out['ticks']==253 and out['strict_inclusions']==705,'complete unchanged unjoined705 proof')
   require(ci['map_inputs']==inp and ci['bindings']==binding,'same map and exact independent native root source')
   chart=read(PARENT/'run/certificate.json.gz');member=join(chart,small);save('membership.json',member);check()
   result.update(passed=member['passed'],status='known_endpoint_joined705'if member['passed']else'completed_localization_membership_unresolved',native_calls=108,profile_native_calls=36,root_dimension=704,join_dimension=705,localized_known_root=True,anchor_coordinates=sum(member['coordinate_inclusions']),parameter_inside=member['parameter_inside'],weighted_bound=proof['weighted_bound'],parent=dict(root=str(PARENT),**PINS),scope='The previously independently proved704 endpoint is localized and tested inside an unchanged proved705 tube. Successful full membership connects that existing positive primitive18 unit-pair interval; complementary stability and eventual55 junction connection remain separate.')
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e))
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks[0],events=counts,bytes_before_final=used[0]);save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(TARGET)):entry(p)for p in sorted(TARGET.rglob('*'))if p.is_file()}})
 return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
