from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=False,parents=True)
rows=json.loads((root/'analysis.json').read_text());agg=json.loads((root/'aggregate.json').read_text())
assert agg['status']=='qualified_fixed_panel'
colors={'core':'#2463a6','failed-near-edge':'#d35732'};labels={'core':'Core','failed-near-edge':'Near-edge'}
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
fig,axes=plt.subplots(1,2,figsize=(11.5,4.3),sharey=True)
for ax,window in zip(axes,['early','later']):
 for setting,color in colors.items():
  r=sorted([x for x in agg['groups'] if (x['setting'],x['observation'],x['window'])==(setting,'state',window)],key=lambda x:x['amplitude'])
  x=np.array([v['amplitude'] for v in r]);y=np.array([v['mean_r_squared'] for v in r]);ci=np.array([v['phase_bootstrap_95'] for v in r])
  ax.plot(x,y,'o-',color=color,label=labels[setting]);ax.fill_between(x,ci[:,0],ci[:,1],color=color,alpha=.15)
 ax.axhline(0,color='#888',lw=.8);ax.set_yscale('symlog',linthresh=1.);ax.set_xscale('log');ax.set_xlabel('Write amplitude (state units)');ax.set_title(window.capitalize()+' state window')
axes[0].set_ylabel('Parity R² of sealed linear readout');axes[0].legend(frameon=False)
fig.suptitle('Cross-input feature recovery on 32 held-out phases',y=1.02)
fig.text(.5,-.02,'Bands: phase-bootstrap 95% intervals. Symmetric-log y axis retains large negative scores; early near-edge at 0.01 is primary.',ha='center',fontsize=9)
fig.tight_layout();fig.savefig(out/'parity-r2.png',bbox_inches='tight');fig.savefig(out/'parity-r2.pdf',bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11.5,4.3),sharey=True)
for ax,window in zip(axes,['early','later']):
 for setting,color in colors.items():
  direct=[];pred=[];lower=[];upper=[]
  for a in [1e-4,1e-3,1e-2]:
   r=[x for x in rows if (x['setting'],x['observation'],x['window'],x['amplitude'])==(setting,'state',window,a)]
   y=np.array([v['direct_contrast_norm'] for v in r]);direct.append(np.median(y));lower.append(np.quantile(y,.1));upper.append(np.quantile(y,.9));pred.append(np.median([v['predicted_contrast_norm'] for v in r]))
  ax.plot([1e-4,1e-3,1e-2],direct,'o-',color=color,label=labels[setting]+' direct')
  ax.plot([1e-4,1e-3,1e-2],pred,'--',color=color,label=labels[setting]+' second order')
  ax.fill_between([1e-4,1e-3,1e-2],lower,upper,color=color,alpha=.1)
 ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Write amplitude (state units)');ax.set_title(window.capitalize()+' state window')
axes[0].set_ylabel('Norm of factorial interaction C');axes[0].legend(frameon=False,fontsize=9)
fig.suptitle('Interaction scaling: direct trajectories and reference prediction',y=1.02)
fig.text(.5,-.02,'Lines: phase medians. Shading: 10th–90th phase percentiles, not confidence intervals.',ha='center',fontsize=9)
fig.tight_layout();fig.savefig(out/'contrast-scaling.png',bbox_inches='tight');fig.savefig(out/'contrast-scaling.pdf',bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11.5,4.5))
for ax,window in zip(axes,['early','later']):
 vals=[];unresolved=0
 for setting,color in colors.items():
  r=[v for v in rows if (v['setting'],v['observation'],v['window'])==(setting,'state',window)]
  resolved=[v for v in r if v['direct_contrast_norm']>10*v['contrast_numerical_error_norm'] and v['predicted_contrast_norm']>0]
  unresolved+=len(r)-len(resolved)
  x=[v['predicted_contrast_norm'] for v in resolved];y=[v['direct_contrast_norm'] for v in resolved]
  vals.extend(x+y);ax.scatter(x,y,s=20,color=color,alpha=.6,label=labels[setting])
 low,high=min(vals)*.5,max(vals)*2
 ax.plot([low,high],[low,high],color='#777',ls='--',lw=1)
 ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(low,high);ax.set_ylim(low,high)
 ax.set_xlabel('Second-order predicted contrast norm');ax.set_ylabel('Direct contrast norm');ax.set_title(window.capitalize()+f' window ({unresolved} unresolved omitted)')
axes[0].legend(frameon=False)
fig.suptitle('Interaction prediction across held-out phases and doses',y=1.02)
fig.text(.5,-.02,'Points require direct contrast >10× cross-method discrepancy; dashed line is identity. Norm agreement alone is not vector agreement.',ha='center',fontsize=9)
fig.tight_layout();fig.savefig(out/'predicted-vs-direct.png',bbox_inches='tight');fig.savefig(out/'predicted-vs-direct.pdf',bbox_inches='tight');plt.close(fig)
print(out)
