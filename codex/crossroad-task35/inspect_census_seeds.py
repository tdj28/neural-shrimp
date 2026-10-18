"""Read original census rows and pin attracting seed candidates; no model calls."""
import json,hashlib
from pathlib import Path
import numpy as np
base=Path('/data2/neural-shrimp-data/explore/llm-soft-loop/pod64/oc/census')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for line in (base/'census.jsonl').read_text().splitlines():
 d=json.loads(line);p=base/('orbits-'+d['name']+'.npy');x=np.load(p,allow_pickle=False)
 assert x.shape==(len(d['orbits']),896) and x.dtype==np.float64 and np.isfinite(x).all()
 accepted=[]
 for i,o in enumerate(d['orbits']):
  attempts=[a for a in d['attempts'] if a.get('orbit')==i]
  good=[a for a in attempts if a.get('converged') and a.get('primitive') and a['final_residual']<=1e-10]
  z=np.array([complex(*v) for v in o['multipliers']]);rho=float(max(abs(z)))
  assert abs(rho-o['leading_modulus'])<1e-14
  if o['attracting'] and rho<1 and good:
   accepted.append({'index':i,'saved_leading_ritz_modulus':rho,'saved_ritz_residuals':o['ritz_residuals'],'best_saved_newton_residual':min(a['final_residual'] for a in good),'original_phase_distance_list':o['distances_to_others'],'qualification':'original finite Ritz/primitive reports; full spectrum not saved'})
 rows.append({'name':d['name'],'alpha':d['alpha'],'alpha2':d['alpha2'],'state_path':str(p),'state_sha256':sha(p),'original_attempts':len(d['attempts']),'failed_attempts':sum(not a.get('converged',False) for a in d['attempts']),'candidates':accepted})
result={'census_path':str(base/'census.jsonl'),'census_sha256':sha(base/'census.jsonl'),'rows':rows,'preferred_C2_locator':'T-near-L indices0,2; closeness is parameter-only, continuation must establish branch connection','C3_seed_pair':'not identified by this census; all its cells are near the other cusp pocket','limitations':['phase0 state array alone cannot recompute phase-aligned distances without model calls','reported four Ritz values are not a complete eigensystem','original execution source/receipt crossbinding pending','no flip bracket has been manufactured or inferred from cusp+1 data']}
Path('codex/crossroad-task35/census-seed-candidates.json').write_text(json.dumps(result,indent=2)+'\n')
print([(r['name'],[c['index'] for c in r['candidates']]) for r in rows])
