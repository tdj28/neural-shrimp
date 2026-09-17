#!/usr/bin/env python3
"""Post-run EXP-019 regression figures; sealed summary only, no dynamics or fits.

Every plotted phase remains visible, including blocked prerequisites. No setting
means, favorable complete-case aggregate, bootstrap or endpoint is constructed.
Gaussian-model probabilities are deterministic analytic scores, not noisy trials.
Source/summary hashing is evidence binding, not independent numerical replay.
The historical development/heldout labels do not make these fresh test phases.
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
    if summary.get('experiment_id')!='EXP-019' or summary.get('reused_regression_phases') is not True or summary.get('fresh_holdout') is not False:
        raise ValueError('Explicit EXP-019 reused-regression scope required')
    for key in ('complete','source_valid'):
        if type(summary.get(key)) is not bool:raise ValueError('Explicit producer completeness/source validity required')
    if a is None:
        if summary['complete']:raise ValueError('Complete producer lacks analysis')
        return unavailable_ledger()
    if a['counts']['histories']!=1296 or a['counts']['trace_tasks']!=2700 or a['counts']['comparisons']!=1404:raise ValueError('Complete planned denominator differs')
    rows=a['balanced_accuracies'];balances={(r['setting'],r['phase_index'],r['arm'],r['window']):r for r in rows}
    expected={(s,p,arm,w) for s in SETTINGS for p in range(24) for arm in ARMS for w in WINDOWS}
    if set(balances)!=expected or len(rows)!=576:raise ValueError('Exact576 balance rows required')
    scores=a['scores'];by_id={r['task_id']:r for r in scores}
    if len(scores)!=2700 or len(by_id)!=2700:raise ValueError('Exact2700 retained scores required')
    expected_tasks={(s,p,arm,f,b,m) for s in SETTINGS for p in range(24) for arm in ARMS
        for f in (-1,0,1) for b in (-1,0,1) for m in (('rk4-coarse','rk4-fine','dop853-independent') if p in (0,8) else ('rk4-coarse','rk4-fine'))}
    actual_tasks=[]
    for row in scores:
        t=row['task'];key=tuple(t[k] for k in ('setting','phase_index','arm','first_sign','second_sign','method'))
        if any(type(t[k]) is not int for k in ('phase_index','first_sign','second_sign')):raise ValueError('Integer task identities required')
        wanted=f"{t['setting']}--p{t['phase_index']:02d}--{t['arm']}--s1{t['first_sign']:+d}--s2{t['second_sign']:+d}--{t['method']}"
        if row['task_id']!=wanted or t.get('id',wanted)!=wanted:raise ValueError('Saved task identity differs')
        if t['split']!=('development' if t['phase_index']<8 else 'heldout'):raise ValueError('Historical split differs')
        if any(type(row.get(flag)) is not bool for flag in ('complete','qualified')) or row['qualified'] and not row['complete']:raise ValueError('Invalid score completeness')
        actual_tasks.append(key)
    if len(set(actual_tasks))!=2700 or set(actual_tasks)!=expected_tasks:raise ValueError('Exact2700 solver/sign identities required')
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
        if type(p) is not int or row.get('context_id')!=cid or row.get('split')!=('development' if p<8 else 'heldout'):
            raise ValueError('Exact phase/window/arm identity required')
        for flag in ('complete','qualified'):
            if type(row.get(flag)) is not bool or type(contexts[cid].get(flag)) is not bool:raise ValueError('Boolean cell/context flags required')
        if row['qualified'] and (not row['complete'] or not contexts[cid]['complete']):raise ValueError('Incomplete context cannot qualify')
        if row['qualified']!=contexts[cid]['qualified']:raise ValueError('Cell/context qualification differs')
        if row['complete']:
            if any(not isinstance(row[k],(int,float)) or isinstance(row[k],bool) or not math.isfinite(row[k]) or not 0<=row[k]<=1 for k in ('with_second','without_second')):raise ValueError('Finite probabilities required')
        elif row.get('with_second') is not None or row.get('without_second') is not None:raise ValueError('Unavailable cell has invented mean')
    for s in SETTINGS:
        qualified=all(contexts[f'{s}--p{p:02d}']['qualified'] for p in range(24))
        declared=a['settings'][s]
        if declared['qualified'] is not qualified:raise ValueError('Setting qualification differs')
        if not qualified and declared['means'] is not None:raise ValueError('Failed setting has invented aggregate')
    if not all(a['settings'][s]['qualified'] for s in SETTINGS) and a['primary'] is not None:raise ValueError('Unqualified cohort has primary contrast')
    if not summary['source_valid']:
        # Keep raw probabilities in CSV but suppress every numerical color.
        balances={k:dict(v,qualified=False,saved_qualified=v['qualified'],source_qualification_suppressed=True) for k,v in balances.items()}
        contexts={k:dict(v,qualified=False,source_qualification_suppressed=True) for k,v in contexts.items()}
    return balances,fine,contexts


def unavailable_ledger():
    """Fixed planned identities only; never invent outcomes after analysis loss."""
    balances={};fine={};contexts={}
    for s in SETTINGS:
        for p in range(24):
            cid=f'{s}--p{p:02d}';split='development' if p<8 else 'heldout'
            contexts[cid]=dict(context_id=cid,complete=False,qualified=False)
            for arm in ARMS:
                for w in WINDOWS:
                    balances[s,p,arm,w]=dict(context_id=cid,setting=s,phase_index=p,split=split,arm=arm,window=w,
                        complete=False,qualified=False,with_second=None,without_second=None,interference_loss=None)
                for first in (-1,0,1):
                    for second in (-1,0,1):
                        fine[s,p,arm,first,second]=dict(complete=False,qualified=False,windows={},task=dict(
                            setting=s,phase_index=p,split=split,arm=arm,first_sign=first,second_sign=second))
    return balances,fine,contexts


def periods_from_saved_clocks(fine):
    """Infer T only from complete stored readout clocks, without reference access."""
    periods={s:[] for s in SETTINGS}
    for (s,p,arm,first,second),row in fine.items():
        if not row['complete']:continue
        window=row['windows']['k4'];times=np.asarray(window['times'],dtype=float)
        if times.shape!=(32,) or not np.isfinite(times).all() or not np.all(np.diff(times)>0):raise ValueError('Complete fixed clock required')
        period=float(times[0]/(3+.5/32))
        if period<=0 or not np.allclose(times,period*(3+(np.arange(32)+.5)/32),rtol=1e-12,atol=1e-14):raise ValueError('Saved 4T midpoint clock differs')
        periods[s].append(period)
    result={}
    for s,values in periods.items():
        if values and not np.allclose(values,values[0],rtol=1e-12,atol=1e-14):raise ValueError('Setting periods differ')
        result[s]=values[0] if values else None
    return result


def qualification_label(summary,contexts):
    if not summary['source_valid']:return 'Source validity failed; all numerical cells suppressed.'
    if summary['analysis'] is None:return 'Analysis unavailable; all planned phase cells retained.'
    if all(contexts[c]['qualified'] for c in contexts):return 'All 48 contexts qualified. Figures show individual phases; no setting means.'
    return 'Incomplete qualification. Individual qualified phases only; no complete-case averages.'


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
    color=plt.get_cmap(cmap).with_extremes(bad='#d7d9dd')
    im=ax.imshow(np.ma.masked_invalid(values),cmap=color,norm=norm,aspect='auto',interpolation='nearest')
    yy,xx=np.where(np.isnan(values));ax.scatter(xx,yy,marker='x',color='#535860',s=24,linewidths=.8)
    ax.axvline(7.5,color='#141923',lw=1.5,ls='--')
    ax.set_xticks(range(24));ax.set_xticklabels([f'{p:02d}' for p in range(24)],fontsize=8)
    ax.set_yticks(range(len(labels)));ax.set_yticklabels(labels,fontsize=9)
    ax.set_xlim(-.5,23.5);ax.tick_params(length=0,pad=5)
    for spine in ax.spines.values():spine.set_color('#b9bec6')
    ax.set_xlabel('Reused phase index · historical development 00–07 | historical heldout 08–23',fontsize=9)
    return im


def render(summary,output):
    balances,fine,contexts=collect(summary);output=Path(output)
    if any(output.iterdir()):raise FileExistsError('Figure output must be empty; existing artifacts are immutable')
    periods=periods_from_saved_clocks(fine)
    status=qualification_label(summary,contexts)
    prefix='SYNTHETIC FIXTURE — ' if summary.get('synthetic_fixture') is True else ''
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'figure.facecolor':'white'})
    names={'core':'Core cycle','failed-near-edge':'Near-edge cycle'}
    norm=TwoSlopeNorm(vmin=0.,vcenter=.5,vmax=1.)
    fig,axes=plt.subplots(2,2,figsize=(17,7.9),layout='constrained')
    for i,setting in enumerate(SETTINGS):
        qualified=sum(contexts[f'{setting}--p{p:02d}']['qualified'] for p in range(24))
        for j,(key,label) in enumerate([('without_second','second write absent'),('with_second','second write present')]):
            rows=[('identity',w,key) for w in WINDOWS]
            im=show(axes[i,j],matrix(balances,setting,rows),['Immediate','2T','4T','8T'],norm,'RdBu_r')
            period_label='T unavailable' if periods[setting] is None else f'T = {periods[setting]:.6g} model-time units'
            axes[i,j].set_title(f'{names[setting]} · {label}\n{qualified}/24 qualified contexts; {period_label}',loc='left')
    bar=fig.colorbar(im,ax=axes,location='right',shrink=.8,pad=.02,ticks=[0,.25,.5,.75,1]);bar.set_label('Analytic Gaussian balanced accuracy')
    fig.suptitle(prefix+'EXP-019: phase-resolved two-write memory regression\n'+status+' Gray × = unavailable or unqualified.',fontsize=14)
    fig.supxlabel('Deterministic trajectories with assumed i.i.d. Gaussian measurement noise (σ = 10⁻⁴); no sampled noisy trials.\nFine RK4, identity arm, same sealed observer. All 48 contexts are reused regression phases; equal cycles need not mean equal elapsed model time.',fontsize=10)
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
    fig.suptitle(prefix+'EXP-019: interference and local component retention at 4T\n'+status,fontsize=14)
    fig.supxlabel('Analytic Gaussian probabilities from deterministic trajectories; no noisy trials. σ = 10⁻⁴, ε = 10⁻⁴.\nAll arms use the same identity-derived observer. Gray × = unavailable/unqualified. Reused phases; no information-erasure or memory-capacity claim.',fontsize=10)
    path2=output/'interference-and-margins-at-4T.png';fig.savefig(path2,dpi=180,bbox_inches='tight');plt.close(fig)
    with (output/'phase-readouts.csv').open('x',newline='') as stream:
        fields=['setting','phase_index','split','arm','window','complete','qualified','with_second','without_second','interference_loss','period','window_start','window_end','plotted','reused_regression_phase','saved_qualified'];writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in balances.values():
            data={k:row[k] for k in fields[:10]};T=periods[row['setting']];w=row['window']
            left,right=(0.,.125) if w=='immediate' else (int(w[1:])-1,int(w[1:]))
            data.update(period=T,window_start=None if T is None else left*T,window_end=None if T is None else right*T,
                plotted=row['complete'] and row['qualified'],reused_regression_phase=True,
                saved_qualified=row.get('saved_qualified',row['qualified']))
            writer.writerow(data)
    with (output/'signed-margins-at-4T.csv').open('x',newline='') as stream:
        fields=['setting','phase_index','split','arm','first_sign','second_sign','complete','qualified','projected_mean','signed_margin','probability_correct'];writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for (s,p,arm,first,second),row in fine.items():
            if first==0 or second==0:continue
            data={k:row['task'][k] for k in fields[:6]};data.update(complete=row['complete'],qualified=contexts[f'{s}--p{p:02d}']['qualified'])
            if row['complete']:
                score=row['windows']['k4']['primary'];data.update({k:score[k] for k in fields[-3:]})
            writer.writerow(data)
    return dict(images=[str(path1),str(path2)],phase_rows=576,signed_margin_rows=576,margin_color_max=bound,
        periods=periods,status=status,setting_means_plotted=False,reused_regression_phases=True,fresh_holdout=False,
        qualified_contexts={s:sum(contexts[f'{s}--p{p:02d}']['qualified'] for p in range(24)) for s in SETTINGS})


def run(raw,out,expected_receipt,*,audit_report=None,expected_audit=None):
    raw=Path(raw).resolve();out=Path(out).resolve()
    if out==raw or out.is_relative_to(raw) or out.is_relative_to(Path('/data2/neural-shrimp')):raise ValueError('External exclusive output required')
    if any((parent/'receipt.json').exists() for parent in out.parents):raise ValueError('Output must not nest inside any sealed evidence directory')
    if sha(raw/'receipt.json')!=expected_receipt:raise ValueError('Receipt binding differs')
    receipt=read(raw/'receipt.json')
    if receipt['experiment_id']!='EXP-019':raise ValueError('Wrong experiment receipt')
    audit_binding=None
    if audit_report is not None or expected_audit is not None:
        if audit_report is None or expected_audit is None:raise ValueError('Both audit path and expected hash required')
        audit_report=Path(audit_report).resolve()
        if sha(audit_report)!=expected_audit:raise ValueError('Audit hash binding differs')
        audit_binding=binding(audit_report)
    summary_path=raw/'summary.json';items=[r for r in receipt['artifacts'] if r['path']=='summary.json']
    if len(items)!=1:raise ValueError('Exactly one sealed summary binding required')
    item=items[0]
    if summary_path.stat().st_size!=item['bytes'] or sha(summary_path)!=item['sha256']:raise ValueError('Sealed summary changed')
    summary=read(summary_path);collect(summary);out.mkdir(parents=True,exist_ok=False)
    details=render(summary,out)
    analysis=summary['analysis']
    report=dict(experiment='EXP-019',scope=__doc__,source_commit=receipt['source_commit'],plan_sha256=receipt['plan_sha256'],
        producer=dict(complete=summary['complete'],source_valid=summary['source_valid'],analysis_available=analysis is not None),
        raw_receipt=binding(raw/'receipt.json'),summary=binding(summary_path),associated_audit_report=audit_binding,
        audit_scope='Optional audit file hash bound only; this plotter neither asserts its verdict nor performs numerical replay.',
        helper=binding(Path(__file__).resolve()),details=details,
        suppression=dict(setting_means_never_plotted=True,setting_means_unavailable={s:analysis is None or analysis['settings'][s]['means'] is None for s in SETTINGS},primary_unavailable=analysis is None or analysis['primary'] is None),
        outputs=[binding(p) for p in sorted(out.iterdir()) if p.is_file()])
    (out/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('raw','out','expected-receipt'):p.add_argument('--'+key,required=True)
    for key in ('audit-report','expected-audit'):p.add_argument('--'+key)
    a=p.parse_args();result=run(a.raw,a.out,a.expected_receipt,audit_report=a.audit_report,expected_audit=a.expected_audit)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
