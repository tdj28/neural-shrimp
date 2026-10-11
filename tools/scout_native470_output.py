"""Fixed-output numerical section from the retained accepted native470 route."""
import gzip,json,resource,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import localize_native_fold_flip as F
C=F.C;need,read,entry=C.need,C.read,C.entry
from neural_shrimp.signed_flip_output_scout import evaluate,survey
OUT=F.OUT.parents[2]/'signed-flip-output-scout001/attempt001/run';CARD='docs/plans/native470-output-section-inputs.json'
SETTINGS=dict(C.SETTINGS,step=.000025,max_points=32,max_evaluations=240,native_calls=4536)
FILES=sorted(set(F.FILES+[CARD,'docs/plans/native470-output-section.md','docs/reviews/native-fold-flip-localization002.json','src/neural_shrimp/signed_flip_output_scout.py','tests/test_signed_flip_output_scout.py','tools/scout_native470_output.py','tools/launch_native470_output.py','tests/test_native470_output_driver.py']))

def terminal_association(flat,old):
 x=np.asarray(flat,dtype=float);y=np.asarray(old['native_flat'],dtype=float)
 need(x.shape==y.shape==(470,)and np.isfinite(x).all()and np.isfinite(y).all(),'all470 finite output-section association')
 errors=abs(x-y)/np.maximum(1.,abs(y));checks=errors<1e-10;same=bool(x[-1]==old['z'][13])
 return dict(passed=bool(checks.all()and same),coordinates=int(checks.sum()),coordinate_inclusions=checks.tolist(),maximum_relative_error=float(max(errors)),output_rate_exact_float=same,scope='Literal saved numerical state/mode/rates across the section change. No new interval branch authority.')

def binding(w,card,metadata_only=False):
 need(card['status']=='reviewed'and card['settings']==SETTINGS,'fixed reviewed output section')
 p=F.binding(w,card['inherited'],metadata_only=True)
 r=card['damped_failure'];rp=Path(r['root']);need(rp==F.OUT.parent,'literal damped failure');o=F.owned_source(w,rp,r,2)
 need(o['error']=='ValueError: fixed accepted Newton step cap'and o['native_calls']==2232,'complete damped failure retained')
 if metadata_only:return p
 def events(name):return [(f,json.loads(gzip.decompress(f.read_bytes())))for f in sorted((p/'run/events').glob('*-target-'+name+'.json.gz'))]
 points=events('accepted_point');attempts=events('continuation_attempt');ei=events('evaluation_input');er=events('evaluation_return');joins=[v for _,v in attempts if v['passed']]
 need(len(points)==17 and [v['index']for _,v in points]==list(range(17))and all(v['diagnostics']['passed']for _,v in points)and [v['index']for v in joins]==list(range(16))and all(v['reverse_coordinates']==470 and v['reverse_error']<1e-7 for v in joins),'entire accepted17 and literal16 reverse joins')
 need([v['index']for _,v in ei]==[v['index']for _,v in er]==list(range(1,161)),'all160 retained full evaluations')
 last=points[-1][1];matches=[i for i,(_,v)in enumerate(ei)if v['hidden']==last['hidden']and v['z']==last['z']and int(er[i][0].name.split('-')[0])<int(points[-1][0].name.split('-')[0])];need(matches,'same accepted return');i=matches[-1]
 old=np.asarray(last['z']);full=np.asarray(last['native_flat']);data=er[i][1]['native']
 need(old.shape==(27,)and full.shape==(470,)and np.array_equal(old[:13],full[:13])and np.array_equal(old[14:],full[13:26])and old[13]==full[-1]and last['hidden']==full[-2]and np.array_equal(np.asarray(data['orbit'])[0],old[:13])and np.array_equal(np.asarray(data['modes'])[0],old[14:]),'literal initial phase/mode/rates')
 z=np.r_[old[:13],last['hidden'],old[14:]];output=float(old[13]);inp=read(Path(card['inherited']['preceding']['proof']['root'])/'run/target-inputs.json')
 w.chart=last;w.save('input-binding.json',dict(inherited=card['inherited'],damped_failure=r,map_inputs=inp['map_inputs'],accepted_evaluation=ei[i][1]['index'],accepted_return=entry(er[i][0]),scope='Same numerical branch in an output section; no failed41 iterate used.'))
 w.save('seed-proposal.json',dict(z=z,output=output,full470=full,previous_index=16));return C.TanhMap(inp['map_inputs']),z,output

def profile(w):
 inp,x=C.C.L.K.fixture();n=13;z=np.r_[[float(v.mid())for v in x[:n]],float(x[n].mid()),[float(v.mid())for v in x[n+2:2*n+2]]];output=float(x[n+1].mid());model=C.TanhMap(inp);times=[];sizes=[];w.save('profile-inputs.json',dict(map_inputs=inp,z=z,output=output,period=18))
 for i in range(12):
  used=w.used;t=time.monotonic();w.observe('evaluation_input',dict(index=i,z=z,output=output));v,J,do,d=evaluate(model,z,output,18,w.tick,w.observe);w.observe('evaluation_return',dict(index=i,value=v,jacobian=J,parameter_derivative=do,native=d));times.append(time.monotonic()-t);sizes.append(w.used-used)
 old=read(ROOT/'docs/reviews/native470-first-diagnosis001.json')['outcome'];seconds=time.monotonic()-w.start+2*SETTINGS['max_evaluations']*max(times)+2*old['seconds']+20;raw=2*SETTINGS['max_evaluations']*max(sizes)+32*1024**2;rss=max(2*resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+256*1024,2*old['peak_rss_kib']+256*1024)
 a=dict(passed=seconds<240 and raw<256*1024**2 and rss<3*1024**2,seconds=times,bytes=sizes,forecast_seconds=seconds,forecast_bytes=raw,forecast_rss_kib=rss,target_arrays_unopened=True,native_calls=w.calls);w.save('profile.json',a);need(a['passed']and w.calls==216,'full unrelated output-section measured profile')

def install():
 C.OUT=OUT;C.CARD=CARD;C.SETTINGS=SETTINGS;C.FILES=FILES;C.binding=binding;C.association=terminal_association;C.profile=profile;C.survey=survey
if __name__=='__main__':install();raise SystemExit(C.main())
