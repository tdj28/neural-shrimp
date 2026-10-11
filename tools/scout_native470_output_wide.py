"""Bounded wider numerical route after a retained sampled near miss below+1."""
import functools,gzip,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import scout_native470_output_extension as E
D=E.D
C=D.C;need,read,entry=C.need,C.read,C.entry
OUT=D.OUT.parents[1]/'attempt003/run';CARD='docs/plans/native470-output-wide-inputs.json'
SETTINGS=dict(E.SETTINGS,step=.00025)
FILES=sorted(set(E.FILES+[CARD,'docs/plans/native470-output-wide.md','docs/reviews/native470-output-scout002.json','tools/scout_native470_output_wide.py','tools/launch_native470_output_wide.py','tests/test_native470_output_wide.py']))

def terminal_association(flat,old):
 adjusted=dict(old,z=[0.]*13+[old['output']]+[0.]*13)
 return D.terminal_association(flat,adjusted)

def binding(w,card,metadata_only=False):
 need(card['status']=='reviewed'and card['settings']==SETTINGS,'reviewed fixed output extension')
 E.binding(w,card['previous_card'],metadata_only=True)
 r=card['previous'];p=Path(r['root']);need(p==E.OUT.parent,'literal completed near-miss scout')
 o=D.F.owned_source(w,p,r,0);need(o['passed']and o['status']=='point_cap'and o['native_calls']==3582 and o['points']==32 and o['evaluations']==187,'entire completed output scout')
 need(entry(p/'run/survey.json')==r['survey']and entry(p/'run/initial-tube-association.json')==r['initial_association'],'exact full prior point records')
 need(read(p/'run/source'/E.CARD)==card['previous_card'],'literal prior reviewed card')
 if metadata_only:return p
 s=read(p/'run/survey.json');a=read(p/'run/initial-tube-association.json');need(a['passed']and a['coordinates']==470 and a['output_rate_exact_float']and len(s['points'])==32 and all(v['diagnostics']['passed']for v in s['points']),'all32 native points and initial section identity')
 joins=[json.loads(gzip.decompress(f.read_bytes()))for f in sorted((p/'run/events').glob('*-target-continuation_attempt.json.gz'))]
 need(len(joins)==31 and all(v['passed']and v['reverse_coordinates']==470 and v['reverse_error']<1e-7 for v in joins),'all31 unchanged native reverse associations')
 last=s['points'][-1];z=np.asarray(last['z']);flat=np.asarray(last['native_flat']);need(z.shape==(27,)and flat.shape==(470,)and np.array_equal(z[:13],flat[:13])and np.array_equal(z[14:],flat[13:26])and z[13]==flat[-2]==last['hidden']and last['output']==flat[-1],'literal terminal output-section packing')
 inp=read(p/'run/input-binding.json');w.chart=last;w.save('input-binding.json',dict(previous=r,previous_card=card['previous_card'],map_inputs=inp['map_inputs'],scope='Numerical continuation only, larger disclosed numerical step after sampled near miss; no new root authority.'));w.save('seed-proposal.json',dict(z=z,output=last['output'],full470=flat,previous_index=31));return C.TanhMap(inp['map_inputs']),z,last['output']

def install():
 C.OUT=OUT;C.CARD=CARD;C.SETTINGS=SETTINGS;C.FILES=FILES;C.binding=binding;C.association=terminal_association;C.profile=D.profile;C.survey=functools.partial(D.survey,near_plus_threshold=SETTINGS['near_plus_threshold'])
if __name__=='__main__':install();raise SystemExit(C.main())
