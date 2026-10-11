#!/usr/bin/env python3
"""Saved EXP-021 summary figures; no ODE, fitting or decoder evaluation.

The only research inputs are receipt.json, manifest.json and summary.json.
Their hashes are checked; this is not a replacement for numerical replay.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

SETTINGS = ('core', 'failed-near-edge')
WINDOWS = ('k2', 'k4', 'k8')
RECIPES = ('original_linear', 'primary', 'one_mode')
LABELS = {'original_linear': 'Original covariance-linear',
          'primary': 'Full-reference mixture', 'one_mode': 'One-mode mixture (secondary)'}
COLORS = {'original_linear': '#68737D', 'primary': '#0072B2', 'one_mode': '#D55E00'}
MARKERS = {'original_linear': 's', 'primary': 'o', 'one_mode': '^'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def _number(x, name):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(name+' requires finite number')
    return float(x)


def _interval(value, name):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(name+' requires two endpoints')
    lo, hi = [_number(x, name) for x in value]
    if not 0 <= lo <= hi <= 1:
        raise ValueError(name+' outside probability range')
    return lo, hi


def collect(saved, planned_cells=None):
    """Validate full phase census and return432 phase +18 aggregate rows.

    Saved values are never converted into a favorable complete-case mean.
    Invalid source suppresses every plotted value while retaining the CSV.
    """
    if not isinstance(saved, dict) or saved.get('experiment_id') != 'EXP-021':
        raise ValueError('EXP-021 summary required')
    for key in ('source_valid','complete','qualified'):
        if type(saved.get(key)) is not bool:
            raise ValueError('Explicit Boolean producer status required')
    if saved['qualified'] and not (saved['complete'] and saved['source_valid']):
        raise ValueError('Invalid/incomplete source marked qualified')
    source_valid = saved.get('source_valid') is True
    analysis = saved.get('analysis')
    if not isinstance(analysis, dict) or not isinstance(analysis.get('cells'), list):
        # Retain the exact source-bound planned positions if analysis itself
        # failed. This creates no values, qualification or successful subset.
        if not isinstance(planned_cells,list) or len(planned_cells)!=144:
            raise ValueError('Complete planned phase ledger unavailable for plotting')
        analysis={'primary':None,'settings':{s:dict(cells=72,qualified=False,secondary_qualified=False,
                    means=None,secondary_means=None) for s in SETTINGS},'cells':[]}
        for cell in planned_cells:
            absent=dict(complete=False,qualified=False,nominal=None,interval=None)
            analysis['cells'].append(dict(id=cell['id'],cell=cell,complete=False,qualified=False,
                secondary_complete=False,secondary_qualified=False,
                groups=[dict(cell_id=cell['id'],method='rk4-fine',original_linear=None,
                    original_linear_interval=None,primary=absent.copy(),one_mode=absent.copy())]))
    packets = analysis['cells']
    if len(packets) != 144:
        raise ValueError('Exactly144 retained cells required')
    rows, seen, phases = [], set(), {}
    for packet in packets:
        cell = packet['cell']
        setting, index, window = cell['setting'], cell['phase_index'], cell['window']
        if setting not in SETTINGS or type(index) is not int or index not in range(24) or window not in WINDOWS:
            raise ValueError('Unknown phase cell identity')
        ident = f'{setting}--p{index:02d}--{window}'
        if packet['id'] != ident or cell['id'] != ident or cell['context_id'] != f'{setting}--p{index:02d}' or ident in seen:
            raise ValueError('Duplicate/misbound phase cell')
        seen.add(ident)
        fraction = _number(cell['phase_fraction'], 'phase fraction')
        if not 0 <= fraction < 1 or cell['split'] != ('development' if index < 8 else 'heldout'):
            raise ValueError('Invalid inherited phase metadata')
        if index in phases and phases[index] != fraction:
            raise ValueError('Shared phase fractions differ')
        phases[index] = fraction
        for key in ('complete', 'qualified', 'secondary_complete', 'secondary_qualified'):
            if type(packet.get(key)) is not bool:
                raise ValueError('Explicit Boolean cell qualification required')
        if packet['qualified'] and not packet['complete']:
            raise ValueError('Incomplete cell marked qualified')
        if packet['secondary_qualified'] and not (packet['secondary_complete'] and packet['qualified']):
            raise ValueError('Secondary qualification lacks required primary/completeness')
        groups = [g for g in packet['groups'] if g['method'] == 'rk4-fine']
        if len(groups) != 1 or groups[0]['cell_id'] != ident:
            raise ValueError('Exactly one bound primary fine group required')
        group = groups[0]
        for recipe in RECIPES:
            if recipe == 'original_linear':
                nominal, interval = group.get('original_linear'), group.get('original_linear_interval')
                complete = qualified = nominal is not None and interval is not None
            else:
                p = group[recipe]
                if type(p.get('complete')) is not bool or type(p.get('qualified')) is not bool:
                    raise ValueError('Explicit probability group flags required')
                complete = p['complete']
                qualified = p['qualified'] and packet['qualified' if recipe == 'primary' else 'secondary_qualified']
                nominal, interval = p.get('nominal'), p.get('interval')
                if p['qualified'] and not complete:
                    raise ValueError('Incomplete probability marked qualified')
                if p['qualified'] and not packet['qualified' if recipe=='primary' else 'secondary_qualified']:
                    raise ValueError('Probability qualification contradicts cell gate')
            lo = hi = None
            if complete:
                nominal = _number(nominal, 'nominal BA')
                lo, hi = _interval(interval, 'BA interval')
                if not 0 <= nominal <= 1:
                    raise ValueError('Nominal BA outside probability range')
            display = bool(source_valid and complete and qualified)
            rows.append(dict(cell_id=ident, setting=setting, phase_index=index,
                phase_fraction=fraction, inherited_split=cell['split'], window=window,
                method='rk4-fine', recipe=recipe, complete=bool(complete), qualified=bool(qualified),
                source_valid=source_valid, displayed=display, nominal=nominal,
                lower=lo, upper=hi, reason='' if display else 'source invalid' if not source_valid
                    else 'unavailable' if not complete else 'qualification failed'))
    expected = {f'{s}--p{p:02d}--{w}' for s in SETTINGS for p in range(24) for w in WINDOWS}
    if seen != expected:
        raise ValueError('Incomplete or extra cell census')
    aggregates = []
    for setting in SETTINGS:
        status = analysis['settings'][setting]
        if status['cells'] != 72 or type(status.get('qualified')) is not bool or type(status.get('secondary_qualified')) is not bool:
            raise ValueError('Setting qualification census differs')
        own = [p for p in packets if p['cell']['setting'] == setting]
        if status['qualified'] != all(p['qualified'] for p in own) or status['secondary_qualified'] != all(p['secondary_qualified'] for p in own):
            raise ValueError('Setting qualification differs from complete cell denominator')
        for window in WINDOWS:
            for recipe in RECIPES:
                allowed = status['secondary_qualified'] if recipe == 'one_mode' else status['qualified']
                means = status.get('secondary_means' if recipe == 'one_mode' else 'means')
                value = None
                if allowed:
                    if not isinstance(means, list) or len(means) != 3 or {m['window'] for m in means} != set(WINDOWS):
                        raise ValueError('Qualified setting needs all three full-cohort means')
                    value = next(m for m in means if m['window'] == window)
                    if value['phases'] != 24:
                        raise ValueError('Full24-phase aggregate required')
                elif means is not None:
                    raise ValueError('Unqualified setting must suppress its aggregate means')
                nominal = lo = hi = None
                if value is not None:
                    if recipe == 'original_linear':
                        nominal = _number(value['original_linear'], 'original setting mean')
                        lo = hi = nominal
                    else:
                        nominal = _number(value['nominal_accuracy'], 'setting mean')
                        lo, hi = _interval(value['accuracy_interval'], 'setting mean interval')
                    selected = [r for r in rows if r['setting'] == setting and r['window'] == window and r['recipe'] == recipe]
                    if len(selected) != 24 or not all(r['complete'] and r['qualified'] for r in selected):
                        raise ValueError('Aggregate lacks complete qualified24-phase inputs')
                    expected_mean = math.fsum(r['nominal'] for r in selected)/24
                    if abs(nominal-expected_mean) > 1e-14:
                        raise ValueError('Saved setting mean differs from all24 phases')
                aggregates.append(dict(setting=setting, window=window, recipe=recipe, phases=24,
                    qualified=bool(allowed), source_valid=source_valid, displayed=bool(allowed and source_valid),
                    nominal=nominal, lower=lo, upper=hi,
                    reason='' if allowed and source_valid else 'source invalid' if not source_valid else 'setting mean withheld'))
    return dict(phase_rows=rows, aggregate_rows=aggregates, source_valid=source_valid,
                producer_complete=saved.get('complete') is True,
                producer_qualified=saved.get('qualified') is True,
                primary_reported=analysis.get('primary') is not None)


def _footer(fig, provenance):
    fig.text(.5, .045, 'Analytic Gaussian readout accuracy; known phase; σ=10⁻⁴; no noisy trials. '
             'Fixed three-neuron dynamics and reference-defined readouts.', ha='center', fontsize=9)
    fig.text(.5, .025, 'All24 phases per setting are reused regression phases. Bars are numerical enclosures, '
             'not confidence intervals. No memory-capacity or erasure claim.', ha='center', fontsize=9)
    fig.text(.5, .008, provenance, ha='center', fontsize=8, color='#555555')


def figures(data, output, provenance='Synthetic fixture; no research outcomes'):
    output = Path(output)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.spines.top':False, 'axes.spines.right':False})
    handles = [Line2D([0],[0],marker=MARKERS[r],color=COLORS[r],linestyle='none',label=LABELS[r]) for r in RECIPES]
    handles.append(Line2D([0],[0],marker='x',color='#B22222',linestyle='none',label='Unavailable / unqualified'))
    fig, axes = plt.subplots(2,3,figsize=(15,8.6),sharex=True,sharey=True)
    for i, setting in enumerate(SETTINGS):
        for j, window in enumerate(WINDOWS):
            ax = axes[i,j]
            for ri, recipe in enumerate(RECIPES):
                selected = sorted([r for r in data['phase_rows'] if r['setting']==setting and r['window']==window and r['recipe']==recipe],key=lambda r:r['phase_fraction'])
                for row in selected:
                    x = row['phase_fraction']
                    if row['displayed']:
                        y = row['nominal']
                        ax.errorbar(x,y,yerr=[[max(0.,y-row['lower'])],[max(0.,row['upper']-y)]],
                            fmt=MARKERS[recipe],color=COLORS[recipe],markersize=4.5,capsize=2,
                            alpha=.85,markerfacecolor='none' if recipe=='one_mode' else COLORS[recipe])
                    else:
                        ax.plot(x,-.035-.02*ri,'x',color='#B22222',markersize=4)
            displayed = sum(r['displayed'] for r in data['phase_rows'] if r['setting']==setting and r['window']==window and r['recipe']=='primary')
            ax.set_title(f'{setting} · window ending {window[1:]}T\nFull mixture: {displayed}/24 qualified phase values',fontsize=10)
            ax.axhline(.5,color='#BBBBBB',linestyle=':',linewidth=1)
            ax.set_xlim(-.025,1.025);ax.set_ylim(-.10,1.04)
            ax.set_xticks([0,.25,.5,.75,1]);ax.set_yticks([0,.25,.5,.75,1])
            ax.grid(alpha=.15)
            if i==1:ax.set_xlabel('Known initial phase fraction (24 fixed reused phases)')
            if j==0:ax.set_ylabel('First-bit balanced accuracy\nunknown balanced second bit')
    fig.suptitle('EXP-021 · Phase-resolved reference-defined decoding',fontsize=16,y=.985)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.945),ncol=4,frameon=False)
    fig.subplots_adjust(top=.84,bottom=.14,hspace=.38,wspace=.13)
    _footer(fig,provenance)
    fig.savefig(output/'phase-resolved-readouts.png',dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(12,5.8),sharey=True)
    for ax,setting in zip(axes,SETTINGS):
        plotted = False
        for recipe in RECIPES:
            rows = [next(r for r in data['aggregate_rows'] if r['setting']==setting and r['window']==window and r['recipe']==recipe) for window in WINDOWS]
            if all(r['displayed'] for r in rows):
                x=[2,4,8];y=[r['nominal'] for r in rows]
                errors=[[max(0.,r['nominal']-r['lower']) for r in rows],[max(0.,r['upper']-r['nominal']) for r in rows]]
                ax.errorbar(x,y,yerr=errors,fmt=MARKERS[recipe]+'-',color=COLORS[recipe],capsize=3,label=LABELS[recipe]);plotted=True
        if not plotted:
            ax.text(.5,.5,'Setting averages withheld\nFull required qualification unavailable',ha='center',va='center',transform=ax.transAxes,color='#A32222')
        elif not all(r['displayed'] for r in data['aggregate_rows'] if r['setting']==setting and r['recipe']=='one_mode'):
            ax.text(.02,.03,'Secondary averages withheld',transform=ax.transAxes,color='#A32222',fontsize=9)
        ax.set_title(setting);ax.set_xticks([2,4,8]);ax.set_xlim(1.5,8.5);ax.set_ylim(0,1.025)
        ax.axhline(.5,color='#BBBBBB',linestyle=':');ax.grid(alpha=.15)
        ax.set_xlabel('Window ending at kT (cycles; model time)')
    axes[0].set_ylabel('Balanced accuracy · all24 phases equally weighted')
    fig.suptitle('EXP-021 · Full-cohort descriptive averages',fontsize=15,y=.98)
    fig.legend(handles=handles[:3],loc='upper center',bbox_to_anchor=(.5,.925),ncol=3,frameon=False)
    fig.text(.5,.14,'Same cycle counts do not imply equal elapsed model time across settings. '
             +('Complete-study primary reported.' if data['primary_reported'] and data['source_valid'] else 'Complete-study primary withheld.'),ha='center',fontsize=9)
    fig.subplots_adjust(top=.80,bottom=.25,wspace=.15)
    _footer(fig,provenance)
    fig.savefig(output/'full-cohort-readout-averages.png',dpi=180)
    plt.close(fig)


def _csv(path, rows):
    with Path(path).open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def render(raw, output, *, expected_receipt, expected_source, expected_plan):
    raw, output = Path(raw).resolve(),Path(output).resolve()
    if output.exists():raise ValueError('Output must be a new exclusive directory')
    repository = Path('/data2/neural-shrimp').resolve()
    if output==repository or repository in output.parents or raw==output or raw in output.parents:
        raise ValueError('Output must be external to repository and raw input')
    if any((p/'receipt.json').exists() for p in (output,*output.parents)):
        raise ValueError('Output may not nest inside sealed evidence')
    receipt_path=raw/'receipt.json'
    if digest(receipt_path)!=expected_receipt:raise ValueError('Expected receipt hash differs')
    receipt=read(receipt_path)
    if receipt.get('experiment_id')!='EXP-021' or receipt.get('source_commit')!=expected_source or receipt.get('plan_sha256')!=expected_plan:
        raise ValueError('Exact source/plan/experiment binding differs')
    inventory={x['path']:x for x in receipt['artifacts']}
    if len(inventory)!=len(receipt['artifacts']):raise ValueError('Duplicate receipt path')
    inputs=[]
    for name in ['manifest.json','summary.json']:
        path=raw/name;entry=inventory.get(name)
        if entry is None or path.is_symlink() or path.stat().st_size!=entry['bytes'] or digest(path)!=entry['sha256']:
            raise ValueError('Input hash/size/membership differs: '+name)
        inputs.append(binding(path))
    manifest=read(raw/'manifest.json');saved=read(raw/'summary.json')
    if manifest.get('source_commit')!=expected_source or manifest.get('plan_sha256')!=expected_plan:
        raise ValueError('Manifest source/plan differs')
    if receipt.get('qualified') is not saved.get('qualified'):
        raise ValueError('Receipt/summary qualification differs')
    data=collect(saved,manifest.get('plan',{}).get('cells'))
    output.mkdir(parents=True,exist_ok=False)
    figures(data,output,f'Source {expected_source[:12]} · receipt {expected_receipt[:12]} · saved producer qualification; hash check is not numerical replay')
    _csv(output/'phase-readouts.csv',data['phase_rows'])
    _csv(output/'aggregate-readouts.csv',data['aggregate_rows'])
    result=dict(experiment='EXP-021',scope='Read-only saved-summary figures; no new decoder/ODE calculation or numerical replay',
        source_commit=expected_source,plan_sha256=expected_plan,raw_receipt=binding(receipt_path),inputs=inputs,
        generator=binding(Path(__file__).resolve()),phase_rows=len(data['phase_rows']),aggregate_rows=len(data['aggregate_rows']),
        displayed_phase_rows=sum(r['displayed'] for r in data['phase_rows']),producer_complete=data['producer_complete'],
        producer_qualified=data['producer_qualified'],source_valid=data['source_valid'],
        outputs=[binding(output/name) for name in ['phase-resolved-readouts.png','full-cohort-readout-averages.png','phase-readouts.csv','aggregate-readouts.csv']])
    (output/'receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['raw','out','expected-receipt','expected-source','expected-plan']:
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    result=render(args.raw,args.out,expected_receipt=args.expected_receipt,expected_source=args.expected_source,expected_plan=args.expected_plan)
    print(json.dumps({k:v for k,v in result.items() if k not in ('outputs','inputs')},indent=2))


if __name__=='__main__':main()
