#!/usr/bin/env python3
"""Read-only EXP014 saved-summary figures. Never integrate, fit or relabel.

Plotting is not an independent replay of raw numerical evidence. The CLI checks
receipt bindings of summary/manifest/postflight and marks all unqualified output.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys
import numpy as np

SETTINGS=('core','failed-near-edge')
DIRECTIONS=('phase_write','phase_null','neuron_one')
METHODS=('rk4-0.0025','rk4-0.00125','dop853')
PRIMARY='rk4-0.00125'
DELAYS=(1,2,4,8)
NOISES=(1e-5,1e-4,1e-3)
DOSES=(1e-4,1e-3,1e-2)
COLORS={'phase_write':'#146d96','phase_null':'#ba4439','neuron_one':'#7753a2'}
LABELS={'phase_write':'Phase writing','phase_null':'Phase null','neuron_one':'Neuron 1'}
ROOT=Path('/data2/neural-shrimp')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):return json.loads(Path(path).read_text())

def expected_cells():
    return list(itertools.product(SETTINGS,range(8),range(3),DIRECTIONS,METHODS,DELAYS,NOISES))

def cell_key(row):
    names=('setting','phase_index','dose_index','direction','method','delay_cycles','sigma')
    if any(type(row.get(k)) is not int for k in ('phase_index','dose_index','delay_cycles')):
        raise ValueError('Strict integer cell identities required')
    if type(row.get('sigma')) is not float:raise ValueError('Strict declared noise identity required')
    return tuple(row[k] for k in names)

def finite_ba(value):return type(value) in (int,float) and math.isfinite(value) and 0<=value<=1


def setting_evidence_complete(science,setting):
    """Recheck exact visible denominators/flags; no numerical reintegration."""
    try:
        cases=[]
        for phase in range(8):
            cases.extend(f'{setting}--p{phase}--d{dose}--{direction}--s{sign:+d}'
                for dose,direction,sign in itertools.product(range(3),DIRECTIONS,(-1,1)))
            cases.append(f'{setting}--p{phase}--sham')
        expected_ids={case+'--'+method for case,method in itertools.product(cases,METHODS)}
        terminal=[r for r in science['terminal_rows'] if r['setting']==setting]
        if (len(terminal)!=456 or {r['id'] for r in terminal}!=expected_ids
                or any(r['status']!='complete' or r.get('sham_pass') is not True for r in terminal)):return False
        pairs=((METHODS[0],METHODS[1]),(METHODS[1],METHODS[2]))
        expected_pairs={(case+'--'+a,case+'--'+b) for case in cases for a,b in pairs}
        comparisons=[r for r in science['trace_comparisons'] if r['setting']==setting]
        if (len(comparisons)!=304 or {tuple(r['ids']) for r in comparisons}!=expected_pairs
                or any(r['qualified'] is not True for r in comparisons)):return False
        ba=[r for r in science['balanced_accuracy_comparisons'] if r['setting']==setting]
        expected_ba=set(itertools.product(range(8),range(3),DIRECTIONS,pairs,DELAYS,NOISES))
        observed_ba={(r['phase_index'],r['dose_index'],r['direction'],tuple(r['methods']),r['delay_cycles'],r['sigma']) for r in ba}
        if len(ba)!=1728 or observed_ba!=expected_ba or any(r['qualified'] is not True for r in ba):return False
        prereq=science['prerequisites'][setting]
        return (prereq['reference'] is True and len(prereq['propagators'])==8
                and all(x is True for x in prereq['propagators']))
    except (KeyError,TypeError,ValueError):return False


def collect(summary):
    """Complete5184-cell display ledger, no means for unqualified settings."""
    if summary.get('experiment_id')!='EXP-014':raise ValueError('EXP014 summary required')
    science=summary['scientific'];expected=expected_cells();known=set(expected);indexed={};errors=[]
    for position,row in enumerate(science.get('paired_readouts',[])):
        try:
            key=cell_key(row)
            if key not in known or key in indexed:raise ValueError('Unknown or duplicate paired-cell identity')
            indexed[key]=row
        except (KeyError,TypeError,ValueError) as exc:errors.append(dict(position=position,error=str(exc)))
    numerical_failures=set();failed_sham_contexts=set()
    pulse_pattern=re.compile(r'^(core|failed-near-edge)--p([0-7])--d([0-2])--(phase_write|phase_null|neuron_one)--s[+-]1--')
    sham_pattern=re.compile(r'^(core|failed-near-edge)--p([0-7])--sham--')
    for comparison in science.get('trace_comparisons',[]):
        if comparison.get('qualified') is True:continue
        for ident in comparison.get('ids',[]):
            match=pulse_pattern.match(ident) if isinstance(ident,str) else None
            if match:numerical_failures.add((match[1],int(match[2]),int(match[3]),match[4]))
            match=sham_pattern.match(ident) if isinstance(ident,str) else None
            if match:failed_sham_contexts.add((match[1],int(match[2])))
    for comparison in science.get('balanced_accuracy_comparisons',[]):
        if comparison.get('qualified') is not True:
            numerical_failures.add(tuple(comparison.get(k) for k in ('setting','phase_index','dose_index','direction')))
    for terminal in science.get('terminal_rows',[]):
        match=sham_pattern.match(terminal.get('id',''))
        if match and terminal.get('sham_pass') is not True:failed_sham_contexts.add((match[1],int(match[2])))
    columns=('setting','phase_index','dose_index','direction','method','delay_cycles','sigma')
    rows=[]
    for key in expected:
        original=indexed.get(key);row=dict(zip(columns,key));row.update(status='unavailable',available=False,
            balanced_accuracy=None,phase_only_balanced_accuracy=None,full_linear_balanced_accuracy=None,
            actual_minus_phase_only=None,error=None,numerical_failure=(key[:4] in numerical_failures or key[:2] in failed_sham_contexts))
        if original is None:row['error']='missing_planned_cell'
        elif original.get('status')!='complete':row['error']=str(original.get('error','unavailable_saved_cell'))
        elif not all(finite_ba(original.get(name)) for name in ('balanced_accuracy','phase_only_balanced_accuracy','full_linear_balanced_accuracy')):
            row['error']='nonfinite_or_out_of_range_accuracy'
        elif original.get('zero_balanced_accuracy')!=.5:row['error']='zero_null_not_chance'
        else:
            row.update(status='complete',available=True,**{name:float(original[name]) for name in
                ('balanced_accuracy','phase_only_balanced_accuracy','full_linear_balanced_accuracy')})
            row['actual_minus_phase_only']=row['balanced_accuracy']-row['phase_only_balanced_accuracy']
        rows.append(row)
    settings={};means=[]
    for setting in SETTINGS:
        packet=science.get('settings',{}).get(setting,{})
        setting_rows=[r for r in rows if r['setting']==setting]
        # A source/postflight failure suppresses all means. A different setting's
        # missing prerequisite does not suppress this setting's qualified result.
        qualified=(packet.get('qualified') is True and packet.get('complete') is True
            and summary.get('postflight_error') is None and summary.get('error') is None and not errors
            and setting_evidence_complete(science,setting) and all(r['available'] for r in setting_rows))
        setting_means=[]
        if qualified:
            for method,delay,sigma,dose,direction in itertools.product(METHODS,DELAYS,NOISES,range(3),DIRECTIONS):
                cells=[r for r in setting_rows if (r['method'],r['delay_cycles'],r['sigma'],r['dose_index'],r['direction'])==(method,delay,sigma,dose,direction)]
                if len(cells)!=8:raise ValueError('Exact eight-phase mean denominator required')
                record=dict(setting=setting,method=method,delay_cycles=delay,sigma=sigma,dose_index=dose,
                    epsilon=DOSES[dose],direction=direction,phase_count=8,descriptive_only=True)
                for name in ('balanced_accuracy','phase_only_balanced_accuracy','full_linear_balanced_accuracy','actual_minus_phase_only'):
                    record[name]=float(np.mean([c[name] for c in cells]))
                setting_means.append(record)
        means.extend(setting_means)
        terminal=[r for r in science.get('terminal_rows',[]) if r.get('setting')==setting]
        comparisons=[r for r in science.get('trace_comparisons',[]) if r.get('setting')==setting]
        ba_comparisons=[r for r in science.get('balanced_accuracy_comparisons',[]) if r.get('setting')==setting]
        settings[setting]=dict(saved_qualified=packet.get('qualified') is True,display_qualified=bool(qualified),
            required_paired_cells=2592,available_paired_cells=sum(r['available'] for r in setting_rows),
            primary_method_required_cells=864,primary_method_available_cells=sum(r['available'] and r['method']==PRIMARY for r in setting_rows),
            terminal_status_counts=dict(Counter(r.get('status','missing') for r in terminal)),
            reported_trace_comparisons=len(comparisons),failed_trace_comparisons=sum(r.get('qualified') is not True for r in comparisons),
            reported_ba_comparisons=len(ba_comparisons),failed_ba_comparisons=sum(r.get('qualified') is not True for r in ba_comparisons),
            means_emitted=len(setting_means),saved_prerequisites=science.get('prerequisites',{}).get(setting),
            view='Qualified setting: eight-phase descriptive means by dose.' if qualified else 'DIAGNOSTIC ONLY: unqualified setting; no aggregate means.')
    return dict(schema_version='EXP014-display-v1',primary_method=PRIMARY,required_cells=5184,rows=rows,means=means,
        settings=settings,input_errors=errors,source_reported_complete_study_qualified=summary.get('qualified') is True,
        complete_study_qualified=summary.get('qualified') is True and all(r['display_qualified'] for r in settings.values()),
        postflight_error=summary.get('postflight_error'),
        scope='Read-only presentation of saved readouts. No raw metric replay, inferential intervals, noise realizations, solver pooling, geometry utility, or neural-network generalization.')


def load_saved(directory):
    directory=Path(directory).resolve();receipt=read(directory/'receipt.json');manifest=read(directory/'manifest.json')
    index={r['path']:r for r in receipt['artifacts']}
    if len(index)!=len(receipt['artifacts']):raise ValueError('Duplicate receipt artifact identity')
    bindings=[]
    for name in ('summary.json','manifest.json','postflight.json'):
        path=directory/name
        if path.is_symlink():raise ValueError('Raw input aliases are forbidden')
        actual=dict(path=name,bytes=path.stat().st_size,sha256=sha(path))
        if index.get(name)!=actual:raise ValueError('Saved input hash/bytes mismatch: '+name)
        bindings.append(actual)
    if receipt['source_commit']!=manifest['source_commit'] or receipt['plan_sha256']!=manifest['plan_sha256']:
        raise ValueError('Manifest/receipt source binding differs')
    summary=read(directory/'summary.json');postflight=read(directory/'postflight.json')
    if summary.get('postflight_error')!=postflight.get('error'):raise ValueError('Postflight declaration differs')
    if receipt.get('experiment_id')!='EXP-014' or summary.get('experiment_id')!='EXP-014':raise ValueError('Wrong experiment')
    return summary,dict(raw_directory=str(directory),receipt_sha256=sha(directory/'receipt.json'),
        receipt_bytes=(directory/'receipt.json').stat().st_size,source_commit=receipt['source_commit'],
        plan_sha256=receipt['plan_sha256'],inputs=bindings,
        binding_scope='Summary, manifest and postflight hashes checked against receipt; no raw trajectory or full receipt inventory replay.')


def _matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _save(figure,output,stem,plt):
    for extension in ('png','pdf'):figure.savefig(output/f'{stem}.{extension}',dpi=180,bbox_inches='tight')
    plt.close(figure)


def plot_qualified(report,setting,sigma,output):
    plt=_matplotlib();fig,axes=plt.subplots(3,2,figsize=(11,10),sharex=True)
    for dose in range(3):
        left,right=axes[dose]
        for direction in DIRECTIONS:
            cells=[next(r for r in report['means'] if (r['setting'],r['method'],r['sigma'],r['dose_index'],r['direction'],r['delay_cycles'])==(setting,PRIMARY,sigma,dose,direction,k)) for k in DELAYS]
            left.plot(DELAYS,[r['balanced_accuracy'] for r in cells],marker='o',color=COLORS[direction],label=LABELS[direction])
            left.plot(DELAYS,[r['phase_only_balanced_accuracy'] for r in cells],linestyle='--',color=COLORS[direction],alpha=.8)
            right.plot(DELAYS,[r['actual_minus_phase_only'] for r in cells],marker='o',color=COLORS[direction],label=LABELS[direction])
        left.axhline(.5,color='.5',lw=.8);left.set_ylim(0,1);left.set_ylabel(f'Analytic BA | dose {DOSES[dose]:g}')
        right.axhline(0,color='.5',lw=.8);right.set_ylim(-1,1);right.set_ylabel('Δ analytic BA')
        for ax in (left,right):ax.set_xscale('log',base=2);ax.set_xticks(DELAYS,labels=[str(k) for k in DELAYS]);ax.grid(alpha=.18)
    axes[0,0].legend(fontsize=8);axes[-1,0].set_xlabel('Delay (cycles)');axes[-1,1].set_xlabel('Delay (cycles)')
    axes[0,0].set_title('Full dynamics solid · phase-only dashed');axes[0,1].set_title('Full dynamics − phase-only; same paired cells')
    fig.suptitle(f'{setting} | qualified setting | σ={sigma:g}\nFine RK4; eight-phase descriptive means per dose; no confidence intervals\nAnalytic Gaussian readout BA: deterministic trajectories, assumed i.i.d. measurement noise; no noisy trials',fontsize=10)
    fig.tight_layout(rect=[0,0,1,.91]);_save(fig,output,f'{setting}-sigma-{sigma:g}-qualified',plt)


def plot_diagnostic(report,setting,sigma,output,value):
    plt=_matplotlib();fig,axes=plt.subplots(3,3,figsize=(11,10),sharex=True,sharey=True)
    difference=value=='actual_minus_phase_only';minimum,maximum=(-1,1) if difference else (0,1)
    cmap=plt.get_cmap('RdBu_r' if difference else 'viridis').with_extremes(bad='#dddddd')
    selected=[r for r in report['rows'] if (r['setting'],r['method'],r['sigma'])==(setting,PRIMARY,sigma)]
    for dose,direction in itertools.product(range(3),DIRECTIONS):
        ax=axes[dose,DIRECTIONS.index(direction)];matrix=np.full((8,4),np.nan)
        for r in selected:
            if r['dose_index']==dose and r['direction']==direction and r['available']:
                matrix[r['phase_index'],DELAYS.index(r['delay_cycles'])]=r[value]
        picture=ax.imshow(matrix,origin='lower',aspect='auto',vmin=minimum,vmax=maximum,cmap=cmap)
        for row,col in np.argwhere(~np.isfinite(matrix)):ax.text(col,row,'×',ha='center',va='center',color='#a42121',fontsize=11)
        from matplotlib.patches import Rectangle
        for cell in selected:
            if cell['dose_index']==dose and cell['direction']==direction and cell['numerical_failure']:
                col=DELAYS.index(cell['delay_cycles']);row=cell['phase_index']
                ax.add_patch(Rectangle((col-.5,row-.5),1,1,fill=False,edgecolor='#de3a32',linewidth=1.2))
        ax.set_xticks(range(4),labels=[str(k) for k in DELAYS]);ax.set_yticks(range(8))
        ax.set_title(f'{LABELS[direction]} | dose {DOSES[dose]:g}',fontsize=10)
        if dose==2:ax.set_xlabel('Delay (cycles)')
        if direction==DIRECTIONS[0]:ax.set_ylabel('Reference phase index')
    colorlabel='Full dynamics − phase-only analytic Gaussian BA' if difference else 'Analytic Gaussian readout balanced accuracy'
    fig.subplots_adjust(top=.87,right=.84,hspace=.35,wspace=.2)
    color_axis=fig.add_axes([.88,.16,.018,.65])
    fig.colorbar(picture,cax=color_axis,label=colorlabel)
    count=sum(r['available'] for r in selected)
    fig.suptitle(f'{setting} | DIAGNOSTIC ONLY — setting unqualified | σ={sigma:g}\nFine RK4: {count}/288 displayed cells available; × unavailable; red border numerical failure; no means\nAnalytic Gaussian readout BA: deterministic trajectories, assumed i.i.d. measurement noise; no noisy trials',fontsize=10)
    _save(fig,output,f'{setting}-sigma-{sigma:g}-diagnostic-{value}',plt)


def write_report(summary,provenance,output,*,source_directory=None):
    output=Path(output).resolve();source=Path(source_directory).resolve() if source_directory else None
    if output==ROOT or ROOT in output.parents or (source is not None and (output==source or source in output.parents)):
        raise ValueError('Report must be external to repository and immutable source attempt')
    output.parent.mkdir(parents=True,exist_ok=True);output.mkdir()
    report=collect(summary);report['provenance']=provenance
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (output/'paired-readouts.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(report['rows'][0]));writer.writeheader();writer.writerows(report['rows'])
    for setting,sigma in itertools.product(SETTINGS,NOISES):
        if report['settings'][setting]['display_qualified']:plot_qualified(report,setting,sigma,output)
        else:
            for value in ('balanced_accuracy','actual_minus_phase_only'):plot_diagnostic(report,setting,sigma,output,value)
    notes=['# EXP-014 saved-readout report','',report['scope'],'',
        'Only fine RK4 appears in figures; coarse RK4 and DOP853 remain numerical checks, never independent replicates.',
        'Three Gaussian observation-noise levels are analytic readout uncertainties, not simulated noise paths.',
        'Phase-only predictions use the requested pulse; full-linear predictions use the realized displacement.',
        'Unqualified settings have individual diagnostic heatmaps and no aggregate means. A different setting may qualify independently; producer or source errors suppress all means.',
        'Grey cells marked × are unavailable. Red borders mean at least one required numerical comparison for the signed pair, or its sham context, failed; a colored diagnostic cell is not a claim that its setting qualified.',
        'No observed sham subtraction, peak matching, response-based clock shift or classifier refitting is performed.','']
    for setting,record in report['settings'].items():
        notes.extend([f'## {setting}', '',record['view'],f"Available paired cells: {record['available_paired_cells']}/2592. Failed trace comparisons: {record['failed_trace_comparisons']}/{record['reported_trace_comparisons']}; failed BA comparisons: {record['failed_ba_comparisons']}/{record['reported_ba_comparisons']}.",''])
    (output/'README.md').write_text('\n'.join(notes))
    items=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(output.iterdir()) if p.is_file()]
    receipt=dict(schema='EXP014-report-receipt-v1',source=provenance,artifacts=items,plotting_source_sha256=sha(__file__),new_dynamics=False,model_fitting=False)
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    return dict(directory=str(output),receipt_sha256=sha(output/'receipt.json'),artifacts=len(items),
        display_qualified={key:value['display_qualified'] for key,value in report['settings'].items()})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();summary,provenance=load_saved(args.directory)
    print(json.dumps(write_report(summary,provenance,args.out,source_directory=args.directory),indent=2))

if __name__=='__main__':main()
