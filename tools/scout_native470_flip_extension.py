"""Fixed numerical continuation from the exact retained terminal470 point."""
from pathlib import Path
import hashlib,json,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import scout_native470_flip as C
need,read,entry=C.need,C.read,C.entry
ORIGINAL_BINDING=C.binding
OUT=C.OUT.parents[1]/'attempt002/run';CARD='docs/plans/native470-flip-extension-inputs.json'
SETTINGS=dict(C.SETTINGS,step=-.125,max_points=24,max_evaluations=160,native_calls=3096)
FILES=sorted(set(C.FILES+[CARD,'docs/plans/native470-flip-extension.md','docs/reviews/native470-flip-scout001.json','tools/scout_native470_flip_extension.py','tools/launch_native470_flip_extension.py','tests/test_native470_flip_extension.py']))

def terminal_association(flat,old):
 x=np.asarray(flat,dtype=float);y=np.asarray(old['native_flat'],dtype=float)
 need(x.shape==y.shape==(470,)and np.isfinite(x).all()and np.isfinite(y).all(),'complete470 finite terminal association')
 errors=np.abs(x-y)/np.maximum(1.,np.abs(y));same_rate=bool(x[-2]==old['hidden']);checks=errors<1e-10
 return dict(passed=bool(checks.all()and same_rate),coordinates=int(checks.sum()),coordinate_inclusions=checks.tolist(),maximum_relative_error=float(max(errors)),hidden_rate_exact_float=same_rate,scope='Literal saved numerical native470 state/mode/rate association only; inherited previous reverse-linked scout and initial rigorous-tube association, not a new interval root.')

def binding(w,card,metadata_only=False):
 ORIGINAL_BINDING(w,card,metadata_only=True);r=card['prior_scout'];p=Path(r['root'])
 need(p==OUT.parents[1]/'attempt001','fixed previous scout')
 for b,k in((p,'outer'),(p/'run','worker')):C.C.L.K.inventory(b,r[k]['sha256'],w)
 rt=read(p/'run/runtime.json');la=read(p/'launcher-result.json');o=read(p/'run/outcome.json')
 need(rt['source_commit']==la['source_commit']==r['source']and o==r['outcome']and o['passed']and o['status']=='point_cap'and o['points']==48 and o['evaluations']==283 and o['native_calls']==5310,'complete bounded preceding numerical path')
 need(la['exit_code']==0 and la['failure']is None and not la['owned_live_pids']and all(not Path('/proc',str(la[k])).exists()for k in('launcher_pid','owned_group_pid')),'preceding owned group gone')
 for n,v in rt['sources'].items():need(entry(p/'run/source'/n)==v and hashlib.sha256(subprocess.check_output(['git','show',r['source']+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'literal original scout source');w.check()
 for n in ('src/neural_shrimp/signed_flip_scout.py','src/neural_shrimp/unit_pair_scout.py'):need(entry(ROOT/n)==rt['sources'][n],'unchanged actual numerical equations')
 need(entry(p/'run/survey.json')==r['survey']and entry(p/'run/initial-tube-association.json')==r['initial_association'],'whole prior point and initial tube association')
 if metadata_only:return p
 initial=read(p/'run/initial-tube-association.json');s=read(p/'run/survey.json');need(initial['passed']and initial['coordinates']==470 and len(s['points'])==48 and all(v['diagnostics']['passed']for v in s['points']),'previous fullnative qualification and initial association')
 joins=[]
 for path in sorted((p/'run/events').glob('*-target-continuation_attempt.json.gz')):
  import gzip
  joins.append(json.loads(gzip.decompress(path.read_bytes())))
 need(len(joins)==47 and all(v['passed']and v['reverse_coordinates']==470 and v['reverse_error']<1e-7 for v in joins),'all47 previous complete reverse joins')
 last=s['points'][-1];z=np.asarray(last['z'],dtype=float);need(z.shape==(27,)and np.array_equal(z[:13],np.asarray(last['native_flat'][:13]))and np.array_equal(z[14:],np.asarray(last['native_flat'][13:26]))and z[13]==last['native_flat'][-1]and last['hidden']==last['native_flat'][-2],'literal terminal phase/rates/mode')
 w.chart=last;inp=read(Path(card['proof']['root'])/'run/target-inputs.json');w.save('input-binding.json',dict(proof=card['proof'],independent=card['independent'],prior_scout=r,map_inputs=inp['map_inputs'],scope='Numerical terminal continuation only, no increased rigorous extent.'));w.save('seed-proposal.json',dict(z=z,hidden=last['hidden'],full470=last['native_flat'],previous_index=47));return C.TanhMap(inp['map_inputs']),z,last['hidden']

def install():
 C.OUT=OUT;C.CARD=CARD;C.SETTINGS=SETTINGS;C.FILES=FILES;C.binding=binding;C.association=terminal_association
if __name__=='__main__':install();raise SystemExit(C.main())
