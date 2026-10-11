from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(sys.argv[1]);dest=Path(sys.argv[2])
assert not dest.exists()
rows=json.loads((root/'analysis.json').read_text())
original=Path(json.loads((root/'manifest.json').read_text())['reference_root'])
phases={p['index']:p['fraction'] for p in json.loads((original/'manifest.json').read_text())['plan']['phases']}
fig,ax=plt.subplots(figsize=(10,4.3),dpi=150)
for a,c,m in [(1e-4,'#7393b3','o'),(1e-3,'#b88621','s'),(1e-2,'#d35732','^')]:
 r=[x for x in rows if (x['setting'],x['observation'],x['window'],x['amplitude'])==('failed-near-edge','state','early',a)]
 ax.scatter([phases[x['phase_index']] for x in r],[x['scores']['second'] for x in r],color=c,marker=m,s=48,alpha=.85,label=f'Input amplitude {a:g}')
ax.axhline(.3379013353634303,color='#d35732',ls='--',lw=1,label='Mean at 0.01: 0.338')
ax.set(xlim=(0,1),ylim=(-.04,1.05),xlabel='Initial phase (fraction of the neural cycle)',ylabel='Parity R² of sealed linear readout',title='The near-edge interaction depends strongly on when inputs arrive')
ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=9,loc='center right')
fig.text(.5,-.015,'32 held-out phases; early state window. Mean 95% phase-bootstrap interval at 0.01: [0.195, 0.491].',ha='center',fontsize=9)
fig.tight_layout();fig.savefig(dest,bbox_inches='tight');fig.savefig(dest.with_suffix('.pdf'),bbox_inches='tight')
