"""Display all frozen settings; no interpolation of a shrimp boundary."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(raw, output):
    summary = json.loads((raw/'analysis.json').read_text())
    output.mkdir(parents=True,exist_ok=False)
    fig, axes = plt.subplots(3,2,figsize=(12,10),sharex='col')
    colors = {'periodic':'#177b70','irregular-supported':'#bb5229','mixed-or-unresolved':'#777777'}
    markers = {'periodic':'o','irregular-supported':'^','mixed-or-unresolved':'x'}
    for col, alpha in enumerate(sorted({p['alpha'] for p in summary['points']})):
        points = [p for p in summary['points'] if p['alpha']==alpha]
        axes[0,col].set_title(f'α = {alpha:.7f}')
        for row, (target,window) in enumerate([(0,[55,60]),(2,[20,25])]):
            for quadratic in [False,True]:
                values=[]
                for p in points:
                    a=next(a for a in p['assays'] if a['amplitude']==.1 and a['window']==window and a['quadratic']==quadratic)
                    value=a['r2'][1][target]
                    values.append(value)
                    axes[row,col].scatter(p['gamma'],value,c=colors[p['status']],marker=markers[p['status']],
                                           s=40 if not quadratic else 20,alpha=1 if not quadratic else .55,zorder=3)
                axes[row,col].plot([p['gamma'] for p in points],values,color='#637080',
                                   ls='-' if not quadratic else '--',lw=1,alpha=.7,
                                   label='Affine readout' if not quadratic else 'Quadratic readout')
            axes[row,col].axhline(0,color='#999999',lw=.8)
            axes[row,col].set_ylabel(['Earlier-input memory R²\nt = 55–60','Input-product R²\nt = 20–25'][row])
        for p in points:
            axes[2,col].scatter(p['gamma'],max(p['repeatability'][1]['final_mean'],1e-15),
                                c=colors[p['status']],marker=markers[p['status']],s=40)
        axes[2,col].set_yscale('log')
        axes[2,col].set_ylabel('Mean activity separation at t = 60\ninitial state difference = 10⁻⁶')
        axes[2,col].set_xlabel('Connection weight γ')
        for ax in axes[:,col]:
            ax.grid(alpha=.18)
            ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(loc='best',fontsize=9)
    for status,color in colors.items():
        axes[2,1].scatter([],[],c=color,marker=markers[status],label=status.replace('-',' '))
    axes[2,1].legend(loc='best',fontsize=8)
    fig.suptitle('Neural behavior across the studied periodic window',fontsize=17,y=.995)
    subtitle = ('Numerically qualified' if summary['qualified'] else 'Numerical qualification FAILED')
    fig.text(.5,.962,f'{subtitle} • fixed inputs and clock • amplitude 0.1 • 128 held-out contexts per setting',ha='center',fontsize=10)
    fig.text(.5,.012,'Markers identify sampled regimes. Connecting lines guide the eye; they do not locate bifurcations.\n'
             'Negative R² means worse than predicting zero. Separation is unforced repeatability, not task accuracy.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.055,1,.95])
    for suffix in ['png','pdf']:
        fig.savefig(output/f'boundary-behavior.{suffix}',dpi=170,bbox_inches='tight')
    (output/'provenance.json').write_text(json.dumps(dict(source_analysis=str(raw/'analysis.json'),
        analysis_sha256=hashlib.sha256((raw/'analysis.json').read_bytes()).hexdigest(),
        receipt_sha256=hashlib.sha256((raw/'receipt.json').read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('raw',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    plot(args.raw,args.output)
