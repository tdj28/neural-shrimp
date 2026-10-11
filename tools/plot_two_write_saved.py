#!/usr/bin/env python3
"""Post-run EXP-016 diagnostic figures; sealed summary only, no dynamics or fits.

Every plotted phase remains visible, including blocked prerequisites. No setting
means, favorable complete-case aggregate, bootstrap or endpoint is constructed.
Gaussian-model probabilities are deterministic analytic scores, not noisy trials.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, SymLogNorm
from matplotlib.lines import Line2D

SETTINGS=('core','failed-near-edge');ARMS=('identity','timing','transverse');WINDOWS=('immediate','k2','k4','k8')


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def binding(path):return dict(path=str(path),bytes=Path(path).stat().st_size,sha256=sha(path))


def collect(summary):
    a=summary['analysis']
    if a['counts']['histories']!=1296 or a['counts']['trace_tasks']!=2700 or a['counts']['comparisons']!=1404:raise ValueError('Complete planned denominator differs')
    rows=a['balanced_accuracies'];balances={(r['setting'],r['phase_index'],r['arm'],r['window']):r for r in rows}
    expected={(s,p,arm,w) for s in SETTINGS for p in range(24) for arm in ARMS for w in WINDOWS}
    if set(balances)!=expected or len(rows)!=576:raise ValueError('Exact576 balance rows required')
    scores=a['scores'];by_id={r['task_id']:r for r in scores}
    if len(scores)!=2700 or len(by_id)!=2700:raise ValueError('Exact2700 retained scores required')
    fine={}
    for row in scores:
        task=row['task']
        if task['method']=='rk4-fine':
            key=(task['setting'],task['phase_index'],task['arm'],task['first_sign'],task['second_sign'])
            if key in fine:raise ValueError('Duplicate fine history')
            fine[key]=row
    expected_fine={(s,p,arm,f,b) for s in SETTINGS for p in range(24) for arm in ARMS for f in (-1,0,1) for b in (-1,0,1)}
    if set(fine)!=expected_fine:raise ValueError('Exact1296 fine histories required')
    contexts=a['contexts'];cids={f'{s}--p{p:02d}' for s in SETTINGS for p in range(24)}
    if set(contexts)!=cids:raise ValueError('Exact48 context ledger required')
    for key,row in balances.items():
        s,p,arm,w=key;cid=f'{s}--p{p:02d}'
        if row['qualified']!=contexts[cid]['qualified']:raise ValueError('Cell/context qualification differs')
        if row['complete']:
            if any(not isinstance(row[k],(int,float)) or isinstance(row[k],bool) or not math.isfinite(row[k]) or not 0<=row[k]<=1 for k in ('with_second','without_second')):raise ValueError('Finite probabilities required')
        elif row.get('with_second') is not None or row.get('without_second') is not None:raise ValueError('Unavailable cell has invented mean')
    return balances,fine,contexts


def matrix(balances,setting,rows):
    values=np.full((len(rows),24),np.nan)
    for i,(arm,window,key) in enumerate(rows):
        for p in range(24):
            row=balances[setting,p,arm,window]
            if row['complete'] and row['qualified']:values[i,p]=row[key]
    return values


def margin_matrix(fine,contexts,setting):
    values=np.full((6,24),np.nan)
    for i,arm in enumerate(ARMS):
        for p in range(24):
            if not contexts[f'{setting}--p{p:02d}']['qualified']:continue
            rows=[fine[setting,p,arm,f,s] for f in (-1,1) for s in (-1,1)]
            if not all(r['complete'] for r in rows):continue
            z=[r['windows']['k4']['primary']['signed_margin'] for r in rows]
            if not np.isfinite(z).all():raise ValueError('Finite margins required')
            values[2*i,p]=min(z);values[2*i+1,p]=max(z)
    return values


def show(ax,values,labels,norm,cmap):
    color=plt.get_cmap(cmap).copy();color.set_bad('#d7d9dd')
    im=ax.imshow(np.ma.masked_invalid(values),cmap=color,norm=norm,aspect='auto',interpolation='nearest')
    yy,xx=np.where(np.isnan(values));ax.scatter(xx,yy,marker='x',color='#535860',s=24,linewidths=.8)
    ax.axvline(7.5,color='#141923',lw=1.5,ls='--')
    ax.set_xticks(range(24));ax.set_xticklabels([f'{p:02d}' for p in range(24)],fontsize=8)
    ax.set_yticks(range(len(labels)));ax.set_yticklabels(labels,fontsize=9)
    ax.set_xlim(-.5,23.5);ax.tick_params(length=0,pad=5)
    for spine in ax.spines.values():spine.set_color('#b9bec6')
    ax.set_xlabel('Shared phase index: development 00–07  |  held-out 08–23',fontsize=9)
    return im


def render(summary,output):
    balances,fine,contexts=collect(summary);output=Path(output)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'figure.facecolor':'white'})
    names={'core':'Core cycle','failed-near-edge':'Near-edge cycle'}
    norm=TwoSlopeNorm(vmin=0.,vcenter=.5,vmax=1.)
    fig,axes=plt.subplots(2,2,figsize=(17,7.9),layout='constrained')
    for i,setting in enumerate(SETTINGS):
        qualified=sum(contexts[f'{setting}--p{p:02d}']['qualified'] for p in range(24))
        for j,(key,label) in enumerate([('without_second','second write absent'),('with_second','second write present')]):
            rows=[('identity',w,key) for w in WINDOWS]
            im=show(axes[i,j],matrix(balances,setting,rows),['Immediate','2T','4T','8T'],norm,'RdBu_r')
            axes[i,j].set_title(f'{names[setting]} · {label}\n{qualified}/24 qualified contexts; identity-arm observer',loc='left')
    bar=fig.colorbar(im,ax=axes,location='right',shrink=.8,pad=.02,ticks=[0,.25,.5,.75,1]);bar.set_label('Analytic Gaussian balanced accuracy')
    fig.suptitle('EXP-016: phase-resolved two-write readout\nAll planned phases shown; gray × = blocked/unavailable. Setting means and primary contrast are unqualified.',fontsize=15)
    fig.supxlabel('Deterministic trajectories with assumed i.i.d. Gaussian measurement noise (σ = 10⁻⁴); no sampled noisy trials.\nFine RK4 is primary; exact clocks and the same sealed identity observer are used with and without the second write.',fontsize=10)
    path1=output/'identity-readout-by-phase.png';fig.savefig(path1,dpi=180,bbox_inches='tight');plt.close(fig)
    margins={s:margin_matrix(fine,contexts,s) for s in SETTINGS}
    finite=np.concatenate([v[np.isfinite(v)] for v in margins.values()]);bound=max(1.,float(np.ceil(np.max(np.abs(finite))))) if finite.size else 1.
    mnorm=SymLogNorm(linthresh=.01,linscale=1.,vmin=-bound,vmax=bound,base=10)
    fig,axes=plt.subplots(2,2,figsize=(17,9.1),layout='constrained')
    labels=[];ba_rows=[]
    for arm,name in [('identity','Identity'),('timing','Retain timing'),('transverse','Retain transverse')]:
        for key,condition in [('without_second','absent'),('with_second','present')]:
            labels.append(f'{name} · 2nd {condition}');ba_rows.append((arm,'k4',key))
    margin_labels=[f'{name} · {limit}' for name in ('Identity','Retain timing','Retain transverse') for limit in ('min z','max z')]
    for i,setting in enumerate(SETTINGS):
        left=show(axes[i,0],matrix(balances,setting,ba_rows),labels,norm,'RdBu_r')
        right=show(axes[i,1],margins[setting],margin_labels,mnorm,'RdBu_r')
        axes[i,0].set_title(f'{names[setting]} · 4T analytic accuracy',loc='left')
        axes[i,1].set_title(f'{names[setting]} · 4T unsaturated signed margins\nMin/max over the four prespecified ± first / ± second sign pairs',loc='left')
    b1=fig.colorbar(left,ax=axes[:,0],shrink=.79,pad=.02,ticks=[0,.25,.5,.75,1]);b1.set_label('Analytic Gaussian balanced accuracy')
    ticks=sorted(set([-bound,-1.,-.1,-.01,0.,.01,.1,1.,bound]))
    b2=fig.colorbar(right,ax=axes[:,1],shrink=.79,pad=.02,ticks=ticks);b2.set_label('Signed margin / noise SD (symmetric log color scale)')
    fig.suptitle('EXP-016: interference and local component retention\nAll arms use the same identity-derived observer; unavailable phase cells remain visible.',fontsize=15)
    fig.supxlabel('Analytic Gaussian probabilities from deterministic trajectories; no noisy trials. σ = 10⁻⁴, write magnitude ε = 10⁻⁴.\nIndividual-context diagnostics only: neither setting qualifies for a mean. Local resets do not establish information erasure or memory capacity.',fontsize=10)
    path2=output/'interference-and-margins-at-4T.png';fig.savefig(path2,dpi=180,bbox_inches='tight');plt.close(fig)
    with (output/'phase-readouts.csv').open('x',newline='') as stream:
        fields=['setting','phase_index','split','arm','window','complete','qualified','with_second','without_second','interference_loss'];writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in balances.values():writer.writerow({k:row[k] for k in fields})
    with (output/'signed-margins-at-4T.csv').open('x',newline='') as stream:
        fields=['setting','phase_index','split','arm','first_sign','second_sign','complete','qualified','projected_mean','signed_margin','probability_correct'];writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for (s,p,arm,first,second),row in fine.items():
            if first==0 or second==0:continue
            data={k:row['task'][k] for k in fields[:6]};data.update(complete=row['complete'],qualified=contexts[f'{s}--p{p:02d}']['qualified'])
            if row['complete']:
                score=row['windows']['k4']['primary'];data.update({k:score[k] for k in fields[-3:]})
            writer.writerow(data)
    return dict(images=[str(path1),str(path2)],phase_rows=576,signed_margin_rows=576,margin_color_max=bound,
        qualified_contexts={s:sum(contexts[f'{s}--p{p:02d}']['qualified'] for p in range(24)) for s in SETTINGS})


def run(raw,audit_report,out,expected_receipt,expected_audit):
    raw=Path(raw).resolve();out=Path(out).resolve();audit_report=Path(audit_report).resolve()
    if out==raw or out.is_relative_to(raw) or out.is_relative_to(Path('/data2/neural-shrimp')):raise ValueError('External exclusive output required')
    if sha(raw/'receipt.json')!=expected_receipt or sha(audit_report)!=expected_audit:raise ValueError('Receipt/audit binding differs')
    receipt=read(raw/'receipt.json');audit=read(audit_report)
    if audit['passed'] is not True or audit['failure_count']!=0 or audit['receipt']['sha256']!=expected_receipt:raise ValueError('Passing bound audit required')
    summary_path=raw/'summary.json';item=next(r for r in receipt['artifacts'] if r['path']=='summary.json')
    if summary_path.stat().st_size!=item['bytes'] or sha(summary_path)!=item['sha256']:raise ValueError('Sealed summary changed')
    summary=read(summary_path);collect(summary);out.mkdir(parents=True,exist_ok=False)
    details=render(summary,out)
    report=dict(experiment='EXP-016',scope=__doc__,source_commit=receipt['source_commit'],plan_sha256=receipt['plan_sha256'],
        raw_receipt=binding(raw/'receipt.json'),summary=binding(summary_path),audit_report=binding(audit_report),
        helper=binding(Path(__file__).resolve()),details=details,
        suppression=dict(setting_means={s:summary['analysis']['settings'][s]['means'] is None for s in SETTINGS},primary=summary['analysis']['primary'] is None),
        outputs=[binding(p) for p in sorted(out.iterdir()) if p.is_file()])
    (out/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('raw','audit-report','out','expected-receipt','expected-audit'):p.add_argument('--'+key,required=True)
    a=p.parse_args();result=run(a.raw,a.audit_report,a.out,a.expected_receipt,a.expected_audit)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
