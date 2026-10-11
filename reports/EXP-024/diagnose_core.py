"""Post hoc saved-array risk decomposition; no fitting or new integration."""
from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
import run_phase_memory_recovery as codec
root=Path(sys.argv[1]);manifest=json.loads((root/'manifest.json').read_text())
original=Path(manifest['reference_root']);seals=codec.load_packet(original/'observer-seal.json')
paths=json.loads((root/'finite-provenance.json').read_text());analysis=json.loads((root/'analysis.json').read_text())
records=[]
for dose,amplitude in enumerate([1e-4,1e-3,1e-2]):
 for window,start,stop in [('early',0,32),('later',32,64)]:
  cid='core--p21';reference=seals[cid][f'{dose}--state--{window}--second'];w=reference['w']
  Y=np.stack([codec.load_packet(paths[f'{cid}--finite--primary--dose-{dose}--sign1-{s1}--sign2-{s2}']['path'])['samples'][start:stop].ravel() for s1,s2 in [(1,1),(1,-1),(-1,1),(-1,-1)]])
  m=Y.mean(0);A=(Y[0]+Y[1]-Y[2]-Y[3])/4;B=(Y[0]-Y[1]+Y[2]-Y[3])/4;C=(Y[0]-Y[1]-Y[2]+Y[3])/4
  parts={'center_bias_squared':float((w@(m-reference['m']))**2),'first_write_leakage_squared':float((w@A)**2),'second_write_leakage_squared':float((w@B)**2),'parity_miscalibration_squared':float((1-w@C)**2),'measurement_noise':float(1e-8*(w@w))}
  reconstructed=1-sum(parts.values())
  actual=next(r['scores']['second'] for r in analysis if (r['context'],r['amplitude'],r['window'],r['observation'])==(cid,amplitude,window,'state'))
  assert abs(reconstructed-actual)/max(1,abs(actual))<1e-10
  records.append(dict(context=cid,amplitude=amplitude,window=window,risk_terms=parts,reconstructed_r_squared=reconstructed,original_r_squared=actual))
print(json.dumps(dict(status='post_hoc_saved_arithmetic_diagnostic',selection='core--p21 selected after its observed numerical/model failure; not a new prespecified endpoint',records=records),indent=2))
