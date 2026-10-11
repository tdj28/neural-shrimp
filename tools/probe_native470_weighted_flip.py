"""One fresh same-width weighted signed470 tube after saved diagnosis."""
from fractions import Fraction as F
from pathlib import Path
import hashlib,subprocess,time
import probe_native470_first_flip as C
CARD='docs/plans/native470-weighted-flip-inputs.json'
C.CARD=CARD;C.OUT=C.OUT.parent.parent/'attempt002/run';C.WEIGHTS=C.read(C.ROOT/CARD)['settings']['weights']

def weights_from(d):
 values=d['scaling']['weights'];C.need(len(values)==470 and d['dimension']==470 and d['scaling']['collatz_max']<.95,'exact completed positive-scaling proposal')
 return [str(max(F(1,2**20),F.from_float(v)))for v in values]

def retained(w,card):
 C.need(card['schema']=='same-junction-weighted-signed470-v1'and C.SETTINGS==dict(period=18,sigma=-1,halfwidth_exponent=-8,radius_exponent=-14,newton_steps=2,precision=256),'same fixed width/rho/native gates')
 for key in ('failed','diagnosis'):
  spec=card[key];p=Path(spec['root'])
  for b,k in ((p,'outer'),(p/'run','worker')):C.L.K.inventory(b,spec[k]['sha256'],w)
  rt=C.read(p/'run/runtime.json');la=C.read(p/'launcher-result.json');o=C.read(p/'run/outcome.json');C.need(rt['source_commit']==la['source_commit']==spec['source']and o==spec['outcome']and la['exit_code']==(2 if key=='failed'else 0)and la['failure']is None and not la['owned_live_pids']and all(not Path('/proc',str(la[k])).exists()for k in ('launcher_pid','owned_group_pid')),'literal retained completed owned '+key)
  for n,v in rt['sources'].items():C.need(C.entry(p/'run/source'/n)==v and hashlib.sha256(subprocess.check_output(['git','show',spec['source']+':'+n],cwd=C.ROOT)).hexdigest()==v['sha256'],'complete historical source');w.check()
  if key=='failed':C.need(not o['passed']and o['status']=='complete_bound_failure'and o['native_calls']==276 and o['ticks']==278 and C.entry(p/'run/target-certificate.json.gz')==spec['certificate'],'retained complete failed acquisition')
  else:
   C.need(o['passed']and o['native_calls']==0 and C.entry(p/'run/diagnostic.json')==spec['diagnostic'],'exact saved-only proposal');d=C.read(p/'run/diagnostic.json');C.need(C.WEIGHTS==weights_from(d),'literal exact dyadic floor/weight proposal')
 old=card['failed']['outcome'];priorbytes=sum(v['bytes']for v in C.read(Path(card['failed']['root'])/'manifest.json')['files'].values());floor=dict(seconds=2*old['seconds']+time.monotonic()-w.start+15,rss_kib=2*old['peak_rss_kib']+256*1024,raw_bytes=2*priorbytes+16*1024**2);C.need(floor['seconds']<240 and floor['rss_kib']<4*1024**2 and floor['raw_bytes']<768*1024**2,'actual completed native cost floor');w.save('actual-cost-floor.json',floor)
 C.need(len(C.WEIGHTS)==470 and all(0<F(v)<=1 for v in C.WEIGHTS),'exact positive bounded weights');w.save('weighted-binding.json',dict(failed=card['failed'],diagnosis=card['diagnosis'],weights=C.WEIGHTS,scope='Fresh narrower correction box, same parameter width. Old approximate scaling is only a proposal; complete fresh470 proof and entire same-known-root anchor required.'))

C.EXTRA_BINDING=retained
C.FILES=sorted(set(C.FILES+[CARD,'docs/plans/native470-weighted-flip.md','docs/reviews/native470-first-diagnosis001.json','docs/reviews/native470-first-flip001.json','src/neural_shrimp/unit_pair_conditioning.py','tools/probe_native470_weighted_flip.py','tools/launch_native470_weighted_flip.py','tests/test_native470_weighted_flip.py']))
if __name__=='__main__':raise SystemExit(C.main())
