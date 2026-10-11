#!/usr/bin/env python3
"""Read-only fixed-census EXP014 refinement report; never solve, fit or select.

Only run on a completed, sealed attempt. Hashes all receipt artifacts, checks
visible summary identities/gates, but does not replay raw numerical trajectories.
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
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name=='tools' else Path('/data2/neural-shrimp')
OLD_METHODS=('rk4-0.0025','rk4-0.00125','dop853')
NEW_METHODS=('rk4-0.000625','rk4-0.0003125','rk4-0.00015625','dop853-1e-13-h0.0125','dop853-1e-13-h0.00625')
METHODS=OLD_METHODS+NEW_METHODS
GROUPS=('old-old','old-new','new-new')
DELAYS=(1,2,4,8);NOISES=(1e-5,1e-4,1e-3);PRIMARY=1e-4
EXPECTED_COUNTS=dict(physical_cases=85,methods=8,trace_rows=680,pair_rows=2380,same_method_ba_cells=3744,required_new_traces=425,required_old_traces=255)
SCOPE=('Outcome-selected 85-case diagnosis. No setting means, incidence comparison, solver selection, '
       'continuous crossing estimate, convergence-order claim or original-study requalification. '
       'BA is analytic Gaussian readout accuracy for deterministic trajectories with assumed i.i.d. measurement noise; no noisy trials. '
       'Raw hash verification and visible-summary checks are not independent trajectory replay.')


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def binding(path,root):return dict(path=str(Path(path).relative_to(root)),bytes=Path(path).stat().st_size,sha256=sha(path))
def number(value,*,minimum=None,maximum=None):
    if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('Finite nonboolean scalar required')
    if minimum is not None and value<minimum or maximum is not None and value>maximum:raise ValueError('Scalar outside declared range')
    return float(value)
def boolean(value):
    if type(value) is not bool:raise ValueError('Strict Boolean required')
    return value

def method_group(a,b):return 'old-old' if a in OLD_METHODS and b in OLD_METHODS else 'new-new' if a in NEW_METHODS and b in NEW_METHODS else 'old-new'

def cases_from_manifest(manifest):
    plan=manifest['plan'];cases=plan['selection']['cases'];seen=set()
    for case in cases:
        if set(case)!={'id','setting','phase_index','kind','dose_index','epsilon','direction','sign'}:raise ValueError('Physical case fields differ')
        if type(case['phase_index']) is not int or case['phase_index'] not in range(8) or case['setting'] not in ('core','failed-near-edge'):raise ValueError('Case phase/setting differs')
        stem=f'{case["setting"]}--p{case["phase_index"]}'
        if case['kind']=='sham':
            if any(case[k] is not None for k in ('dose_index','epsilon','direction','sign')):raise ValueError('Sham metadata differs')
            ident=stem+'--sham'
        else:
            if case['kind']!='pulse' or type(case['dose_index']) is not int or case['dose_index'] not in range(3) or type(case['sign']) is not int or case['sign'] not in (-1,1):raise ValueError('Pulse identity differs')
            if case['direction'] not in ('phase_write','phase_null','neuron_one') or type(case['epsilon']) is not float or case['epsilon']!=(1e-4,1e-3,1e-2)[case['dose_index']]:raise ValueError('Pulse dose/direction differs')
            ident=stem+f'--d{case["dose_index"]}--{case["direction"]}--s{case["sign"]:+d}'
        if case['id']!=ident or ident in seen:raise ValueError('Duplicate/misbound case ID')
        seen.add(ident)
    contexts={(c['setting'],c['phase_index'],c['direction']) for c in cases if c['kind']=='pulse'}
    expected={(s,p,d,e,sign) for s,p,d in contexts for e in range(3) for sign in (-1,1)}
    observed={(c['setting'],c['phase_index'],c['direction'],c['dose_index'],c['sign']) for c in cases if c['kind']=='pulse'}
    phases={(s,p) for s,p,d in contexts};shams={(c['setting'],c['phase_index']) for c in cases if c['kind']=='sham'}
    if len(cases)!=85 or len(contexts)!=13 or len(phases)!=7 or expected!=observed or shams!=phases or Counter(c['setting'] for c in cases)!={'core':26,'failed-near-edge':59}:raise ValueError('Exact85 physical census differs')
    failed=plan['selection']['failed_signed_ids']
    if len(failed)!=18 or len(set(failed))!=18 or not set(failed)<=seen:raise ValueError('Exact18 original failed sign identities required')
    if any(c['dose_index']!=2 for c in cases if c['id'] in failed):raise ValueError('Original failed signs must be strong-dose pulses')
    categories=plan['selection']['selection_categories']
    expected_categories={c['id']:('matched-sham' if c['kind']=='sham' else 'original-failed-strong-sign' if c['id'] in failed else 'matched-strong-opposite-sign' if c['dose_index']==2 else 'matched-lower-dose-both-signs') for c in cases}
    if categories!=expected_categories:raise ValueError('Selection category differs')
    return cases,set(failed),contexts


def pair_specs(cases):
    return [dict(id=c['id']+'--'+a+'--vs--'+b,case_id=c['id'],setting=c['setting'],pair=[a,b],group=method_group(a,b)) for c in cases for a,b in itertools.combinations(METHODS,2)]


def validate_difference(row,threshold,start=0.):
    peak=number(row['maximum_absolute'],minimum=0);number(row['rms'],minimum=0,maximum=peak+1e-14)
    t=number(row['maximum_time'],minimum=start,maximum=320.)
    if abs(t/.025-round(t/.025))>1e-9:raise ValueError('Maximum time off saved grid')
    if type(row['maximum_component']) is not int or row['maximum_component'] not in range(3):raise ValueError('Maximum component differs')
    if abs(number(row['value_at_maximum']))!=peak:raise ValueError('Signed maximum differs')
    if len(row['component_rms'])!=3:raise ValueError('Component RMS census differs')
    for v in row['component_rms']:number(v,minimum=0,maximum=peak+1e-14)
    crossing=row['first_threshold_crossing_time']
    if crossing is None:
        if peak>threshold:raise ValueError('Missing sampled crossing')
    else:
        crossing=number(crossing,minimum=start,maximum=320.)
        if peak<=threshold or crossing>t+1e-12 or abs(crossing/.025-round(crossing/.025))>1e-9:raise ValueError('Sampled crossing differs')
    return peak,crossing


def collect(summary,manifest):
    cases,original_failed,contexts=cases_from_manifest(manifest)
    if summary.get('experiment_id')!='EXP-014-refinement' or summary.get('original_unchanged') is not True:raise ValueError('Refinement summary scope differs')
    science=summary['analysis']
    if science.get('counts')!=EXPECTED_COUNTS or science.get('setting_means') is not None or science.get('original_study_requalified') is not False:raise ValueError('Fixed summary counts/scope differ')
    terminal=science['terminal_rows'];expected_tasks={c['id']+'--'+m:dict(c,id=c['id']+'--'+m,method=m) for c in cases for m in METHODS}
    if len(terminal)!=680 or {r['task']['id'] for r in terminal}!=set(expected_tasks):raise ValueError('Exact680 terminal score census differs')
    readouts=[]
    for row in terminal:
        task=row['task'];ident=task['id']
        if task!=expected_tasks[ident]:raise ValueError('Terminal score identity differs')
        score=row.get('score');complete=row['status']=='complete'
        if complete and (not isinstance(score,dict) or score.get('status')!='complete' or score.get('task')!=task or set(score['readouts'])!={str(d) for d in DELAYS}):raise ValueError('Complete score packet missing or misbound')
        for d in DELAYS:
            out=dict(task_id=ident,case_id=ident.rsplit('--',1)[0],method=task['method'],delay_cycles=d,status='complete' if complete else 'unavailable',q=None,primary_standardized_q=None,noise_gain=None,orientation=None)
            if complete:
                value=score['readouts'][str(d)];q=number(value['q']);gain=number(value['noise_gain'],minimum=0)
                if gain==0 or type(score['orientation']) is not int or score['orientation'] not in (-1,1):raise ValueError('Readout gain/orientation differs')
                out.update(q=q,primary_standardized_q=q/(PRIMARY*gain),noise_gain=gain,orientation=score['orientation'])
            readouts.append(out)
    specs=pair_specs(cases);rows=science['comparisons']
    if len(rows)!=2380 or {r['spec']['id'] for r in rows}!={s['id'] for s in specs}:raise ValueError('Exact2380 comparison census differs')
    indexed={r['spec']['id']:r for r in rows};pair_table=[]
    for spec in specs:
        row=indexed[spec['id']]
        if row['spec']!=spec:raise ValueError('Pair identity/group differs')
        complete=boolean(row['complete']);qualified=boolean(row['qualified'])
        out=dict(**spec,complete=complete,qualified=qualified,state_max=None,activation_max=None,state_first_sample=None,activation_first_sample=None,readout_max_standardized_difference=None,readout_first_declared_delay=None,error=row.get('error'))
        if not complete:
            if qualified:raise ValueError('Incomplete pair cannot qualify')
        else:
            state,st=validate_difference(row['state_difference'],1e-6);act,at=validate_difference(row['activation_difference'],1e-7)
            if row['late_window']['start']!=240. or row['late_window']['end']!=320. or row['late_window']['samples']!=3201:raise ValueError('Late window changed')
            validate_difference(row['late_window']['state_difference'],1e-6,240.);validate_difference(row['late_window']['activation_difference'],1e-7,240.)
            if len(row['last_state_difference'])!=3 or len(row['last_activation_difference'])!=3:raise ValueError('Final difference census differs')
            for v in row['last_state_difference']+row['last_activation_difference']:number(v)
            if set(row['readouts'])!={str(d) for d in DELAYS}:raise ValueError('Pair readout census differs')
            differences=[];first=None
            for d in DELAYS:
                value=row['readouts'][str(d)];diff=number(value['standardized_mean_difference'],minimum=0);differences.append(diff)
                if boolean(value['passed'])!=(diff<=.01):raise ValueError('Readout gate differs')
                end=number(value['completed_window_end'],minimum=0,maximum=320.);last=number(value['last_sample_time'],minimum=0,maximum=end)
                if first is None and diff>.01:first=dict(delay_cycles=d,completed_window_end=end,last_sample_time=last)
            if row['first_completed_readout_threshold_crossing']!=first:raise ValueError('Earliest declared readout differs')
            if boolean(row['state_pass'])!=(state<=1e-6) or boolean(row['activation_pass'])!=(act<=1e-7) or qualified!=(state<=1e-6 and act<=1e-7 and max(differences)<=.01):raise ValueError('Pair qualification differs')
            out.update(state_max=state,activation_max=act,state_first_sample=st,activation_first_sample=at,readout_max_standardized_difference=max(differences),readout_first_declared_delay=first['delay_cycles'] if first else None)
        pair_table.append(out)
    groups={g:dict(required=sum(x['group']==g for x in pair_table),complete=sum(x['group']==g and x['complete'] for x in pair_table),passed=sum(x['group']==g and x['qualified'] for x in pair_table)) for g in GROUPS}
    if science['comparison_groups']!=groups:raise ValueError('Saved comparison group counts differ')
    ba=science['paired_readouts'];names=('setting','phase_index','direction','dose_index','method','delay_cycles','sigma')
    expected_ba={(*c,e,m,d,n) for c in contexts for e in range(3) for m in METHODS for d in DELAYS for n in NOISES}
    for row in ba:
        if any(type(row[k]) is not int for k in ('phase_index','dose_index','delay_cycles')) or type(row['sigma']) is not float:raise ValueError('BA cell identity type differs')
    if len(ba)!=3744 or {tuple(r[k] for k in names) for r in ba}!=expected_ba:raise ValueError('Exact3744 BA census differs')
    ba_index={tuple(r[k] for k in names):r for r in ba}
    for row in ba:
        if row['status']=='complete':
            for k in ('balanced_accuracy','phase_only_balanced_accuracy','full_linear_balanced_accuracy','zero_balanced_accuracy'):number(row[k],minimum=0,maximum=1)
            if row['zero_balanced_accuracy']!=.5 or len(row['standardized_correct_means'])!=2:raise ValueError('BA baseline/mean census differs')
            for v in row['standardized_correct_means']:number(v)
    case_table=[]
    for case in cases:
        members=[x for x in pair_table if x['case_id']==case['id']];group_rows={}
        for group in GROUPS:
            subset=[x for x in members if x['group']==group];done=[x for x in subset if x['complete']]
            def maximum(name):return max((x[name] for x in done),default=None)
            def earliest(name):return min((x[name] for x in done if x[name] is not None),default=None)
            group_rows[group]=dict(required=len(subset),complete=len(done),failed=sum(not x['qualified'] for x in done),unavailable=len(subset)-len(done),
                state_max=maximum('state_max'),activation_max=maximum('activation_max'),readout_max_standardized_difference=maximum('readout_max_standardized_difference'),
                state_first_sample=earliest('state_first_sample'),activation_first_sample=earliest('activation_first_sample'),readout_first_declared_delay=earliest('readout_first_declared_delay'))
        pulse=case['kind']=='pulse';ba_changes=[];ba_complete=0
        if pulse:
            for d in DELAYS:
                values=[ba_index[(case['setting'],case['phase_index'],case['direction'],case['dose_index'],m,d,PRIMARY)] for m in METHODS]
                ba_complete+=sum(x['status']=='complete' for x in values)
                if all(x['status']=='complete' for x in values):ba_changes.append(max(x['balanced_accuracy'] for x in values)-min(x['balanced_accuracy'] for x in values))
        case_table.append(dict(case=case,original_failed_sign=case['id'] in original_failed,selection_category=manifest['plan']['selection']['selection_categories'][case['id']],groups=group_rows,
            maximum_readout_difference=max((x['readout_max_standardized_difference'] for x in members if x['complete']),default=None),
            maximum_primary_ba_method_change=max(ba_changes,default=None),primary_ba_cells_complete=ba_complete,primary_ba_cells_required=32 if pulse else 0))
    computed_complete=all(x['status']=='complete' for x in terminal+ba) and all(x['complete'] for x in rows)
    if boolean(science['complete'])!=computed_complete:raise ValueError('Scientific analysis completion differs')
    return dict(schema_version='EXP014-refinement-report-v1',scope=SCOPE,counts=EXPECTED_COUNTS,analysis_complete=computed_complete,
        source_valid=summary.get('source_valid') is True,producer_complete=summary.get('complete') is True,producer_error=summary.get('error'),
        original_failed_signed_cases=18,comparison_groups=groups,cases=case_table,pairs=pair_table,signed_readouts=readouts,paired_readouts=ba,
        display=dict(primary_sigma=PRIMARY,first_crossing='first saved .025-grid sample; readout crossing only among completed delays1,2,4,8',ba_changes='Maximum method range at each fixed delay, then maximum over4delays; same-method signs only. Shared binary BA diagnostic repeated on both signed-case columns; shams N/A.',means=None))


def load_saved(directory):
    directory=Path(directory).resolve();receipt=read(directory/'receipt.json');manifest=read(directory/'manifest.json');summary=read(directory/'summary.json');post=read(directory/'postflight.json')
    artifacts=receipt['artifacts'];paths=[r['path'] for r in artifacts]
    actual=[]
    for p in sorted(directory.rglob('*')):
        if p.is_symlink():raise ValueError('Raw evidence alias forbidden')
        if p.is_file() and p!=directory/'receipt.json':actual.append(str(p.relative_to(directory)))
    if len(paths)!=len(set(paths)) or sorted(paths)!=actual:raise ValueError('Exact artifact inventory differs')
    for item in artifacts:
        p=directory/item['path']
        if not p.resolve().is_relative_to(directory) or binding(p,directory)!=item:raise ValueError('Artifact bytes/hash differ: '+item['path'])
    if receipt['artifact_bytes']!=sum(x['bytes'] for x in artifacts):raise ValueError('Artifact byte total differs')
    if receipt['source_commit']!=manifest['source_commit'] or receipt['plan_sha256']!=manifest['plan_sha256'] or receipt['experiment_id']!='EXP-014-refinement':raise ValueError('Source/plan receipt identity differs')
    if summary['postflight_error']!=post['error'] or summary['source_valid'] is not (post['error'] is None) or post['source_bindings_unchanged'] is not (post['error'] is None):raise ValueError('Source postflight flags differ')
    expected_status='complete' if summary['complete'] and summary['source_valid'] else 'incomplete_or_invalid'
    if receipt['status']!=expected_status:raise ValueError('Receipt completion differs')
    return summary,manifest,dict(source_directory=str(directory),source_commit=manifest['source_commit'],plan_sha256=manifest['plan_sha256'],receipt=binding(directory/'receipt.json',directory),artifacts_verified=len(artifacts),artifact_bytes_verified=receipt['artifact_bytes'],scope='Full immutable inventory bytes and hashes, visible summary census/gates; no independent raw trajectory replay.')


def label(case,failed):
    stem=('C' if case['setting']=='core' else 'E')+str(case['phase_index'])
    if case['kind']=='sham':return stem+'/sham'
    direction={'phase_write':'W','phase_null':'N','neuron_one':'1'}[case['direction']]
    return stem+'/'+direction+('+' if case['sign']==1 else '−')+'/d'+str(case['dose_index'])+('*' if failed else '')


def plot_report(report,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    output=Path(output);cases=report['cases'];n=len(cases);x=np.arange(n)
    figure,axes=plt.subplots(3,1,figsize=(23,12),gridspec_kw=dict(height_ratios=[2.0,3.0,3.8]),sharex=True)
    colorbar_axes=[]
    failures=np.array([[c['groups'][g]['failed']/c['groups'][g]['required'] for c in cases] for g in GROUPS])
    image=axes[0].imshow(failures,aspect='auto',cmap='Reds',vmin=0,vmax=1,interpolation='nearest')
    axes[0].set_yticks(range(3),['Old–old /3','Old–new /15','New–new /10']);axes[0].set_title('A. Failed numerical pairs per selected case (fraction of fixed pair denominator)',loc='left',fontsize=12)
    for gi,g in enumerate(GROUPS):
        for ci,c in enumerate(cases):
            row=c['groups'][g]
            if row['unavailable']:axes[0].text(ci,gi,'×',ha='center',va='center',fontsize=10,color='#24588e')
    colorbar=figure.colorbar(image,ax=axes[0],pad=.007,fraction=.014);colorbar.set_label('Failure fraction',fontsize=9);colorbar_axes.append(colorbar.ax)
    crossings=np.full((6,n),np.nan)
    for gi,g in enumerate(GROUPS):
        for ci,c in enumerate(cases):
            for vi,k in enumerate(('state_first_sample','activation_first_sample')):
                if c['groups'][g][k] is not None:crossings[2*gi+vi,ci]=c['groups'][g][k]
    cmap=plt.get_cmap('viridis').with_extremes(bad='#eeeeee')
    image=axes[1].imshow(crossings,aspect='auto',cmap=cmap,vmin=0,vmax=320,interpolation='nearest')
    axes[1].set_yticks(range(6),[f'{g.replace("-","–")} {k}' for g in GROUPS for k in ('state','activation')]);axes[1].set_title('B. Earliest sampled threshold exceedance among each group’s pairs (.025 grid; gray = none observed)',loc='left',fontsize=12)
    for gi,g in enumerate(GROUPS):
        for ci,c in enumerate(cases):
            if c['groups'][g]['unavailable']:
                for vi in (0,1):axes[1].text(ci,2*gi+vi,'×',ha='center',va='center',fontsize=10,color='#d53d30')
    colorbar=figure.colorbar(image,ax=axes[1],pad=.007,fraction=.014);colorbar.set_label('Saved time',fontsize=9);colorbar_axes.append(colorbar.ax)
    # Matching axes widths leave all85 signed cases aligned between panels.
    ax=axes[2];right=ax.twinx();q=np.array([np.nan if c['maximum_readout_difference'] is None else c['maximum_readout_difference'] for c in cases]);ba=np.array([np.nan if c['maximum_primary_ba_method_change'] is None else c['maximum_primary_ba_method_change'] for c in cases])
    ax.scatter(x-.12,q,s=17,c='#166b8e',marker='o',label='Max |Δq| / (σ ||w||)');right.scatter(x+.12,ba,s=20,c='#bd5b29',marker='^',label='Max analytic BA change')
    ax.axhline(.01,ls='--',lw=.8,color='#166b8e',alpha=.6);ax.set_yscale('symlog',linthresh=.01);ax.set_ylim(-.001,max(.02,float(np.nanmax(q))*1.5) if np.isfinite(q).any() else .02)
    right.set_ylim(-.03,1.03);ax.set_ylabel('Standardized q difference (symlog)',color='#166b8e');right.set_ylabel('Analytic Gaussian readout BA change',color='#bd5b29')
    ax.set_title('C. Readout disagreement versus analytic BA change (σ = 10⁻⁴; max across methods and four fixed delays)',loc='left',fontsize=12)
    ax.grid(axis='y',alpha=.2);ax.legend(loc='upper left',fontsize=9);right.legend(loc='upper right',fontsize=9)
    for i,c in enumerate(cases):
        case=c['case']
        if any(row['unavailable'] for row in c['groups'].values()):
            ax.scatter(i-.12,q[i] if np.isfinite(q[i]) else 0.,marker='x',s=35,color='#166b8e')
        if case['kind']=='sham':ax.axvspan(i-.5,i+.5,color='#888888',alpha=.15);ax.text(i,-.001,'S',ha='center',va='bottom',fontsize=7,color='#555555')
        elif c['primary_ba_cells_complete']!=32:right.scatter(i+.12,.5,c='#bd5b29',marker='x',s=24)
        if c['original_failed_sign']:
            for a in axes:a.axvline(i,alpha=.08,color='#9f1717',lw=7)
    labels=[label(c['case'],c['original_failed_sign']) for c in cases];axes[-1].set_xticks(x,labels,rotation=90,fontsize=6.5)
    for a in axes:
        a.set_xlim(-.5,n-.5);a.tick_params(axis='y',labelsize=8)
        for i in range(1,n):
            if (cases[i]['case']['setting'],cases[i]['case']['phase_index'])!=(cases[i-1]['case']['setting'],cases[i-1]['case']['phase_index']):a.axvline(i-.5,color='#333333',lw=.6,alpha=.6)
    state='Complete saved analysis' if report['analysis_complete'] and report['source_valid'] and report['producer_complete'] else 'INCOMPLETE / INVALID: descriptive partial evidence only'
    figure.suptitle('EXP-014 refinement: all 85 outcome-selected physical cases — '+state+'\nDeterministic trajectories; analytic Gaussian measurement-noise readout; no sampled noisy trials',fontsize=14,y=.985)
    figure.text(.065,.025,'C/E = core/failed-near-edge; numeral = phase index; W = phase write, N = phase null, 1 = neuron1; d0/d1/d2 = 10⁻⁴/10⁻³/10⁻²; * = one of 18 original failed signed cases.\n× = unavailable comparison/cell (not a pass); S = sham, BA not defined. BA diagnostic is shared by the two signed columns; no setting means or incidence comparison.\nState bound 10⁻⁶, activation bound 10⁻⁷; readout bound .01. Sampled crossings are not continuous-time onsets. Gray cells with × have unavailable evidence.',fontsize=9)
    figure.subplots_adjust(left=.065,right=.92,top=.91,bottom=.19,hspace=.42)
    # Colorbars narrow top panels; align their data areas to the lower axis.
    position=axes[2].get_position()
    for a,bar in zip(axes[:2],colorbar_axes,strict=True):
        old=a.get_position();a.set_position([position.x0,old.y0,position.width,old.height]);bar.set_position([position.x1+.01,old.y0,.008,old.height])
    for extension in ('png','pdf'):figure.savefig(output/f'refinement-diagnostics.{extension}',dpi=170,facecolor='white')
    plt.close(figure)


def csv_write(path,rows):
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        writer.writerows({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows)


def write_report(summary,manifest,provenance,out,*,source_directory=None):
    out=Path(out).resolve()
    if out==ROOT or ROOT in out.parents:raise ValueError('Report must be outside repository')
    if source_directory is not None:
        source=Path(source_directory).resolve()
        if out==source or source in out.parents:raise ValueError('Report must be outside immutable source attempt')
    report=collect(summary,manifest);out.parent.mkdir(parents=True,exist_ok=True);out.mkdir()
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    case_rows=[dict(**c['case'],original_failed_sign=c['original_failed_sign'],selection_category=c['selection_category'],group=g,**c['groups'][g],maximum_primary_ba_method_change=c['maximum_primary_ba_method_change'],primary_ba_cells_complete=c['primary_ba_cells_complete'],primary_ba_cells_required=c['primary_ba_cells_required']) for c in report['cases'] for g in GROUPS]
    csv_write(out/'case-groups.csv',case_rows);csv_write(out/'comparisons.csv',report['pairs']);csv_write(out/'signed-readouts.csv',report['signed_readouts']);csv_write(out/'paired-readouts.csv',report['paired_readouts'])
    plot_report(report,out)
    (out/'README.md').write_text('# EXP-014 refinement selected-case report\n\n'+SCOPE+'\n\nAll 85 physical cases, 680 old/new score rows, 2,380 pair comparisons and 3,744 same-method analytic BA cells are retained. The signed readout export has 2,720 delay rows. No complete-case averaging is performed. Numerical gate failures remain visible in a complete diagnostic run.\n\nThe three panels show fixed-denominator pair failures, earliest saved-grid state/activation threshold exceedances, and primary-noise standardized q disagreement beside the range of analytic BA across methods at fixed delays. The BA range is shared by each signed pair; shams have no binary BA. Detailed crossings and all noise cells are in CSV/JSON. Gray without a cross means no observed exceedance; a cross marks unavailable evidence.\n\n'+json.dumps(provenance,indent=2)+'\n')
    files=[binding(p,out) for p in sorted(out.iterdir()) if p.is_file()]
    receipt=dict(schema_version='EXP014-refinement-report-receipt-v1',source=provenance,artifacts=files,artifact_bytes=sum(x['bytes'] for x in files),script=dict(path=str(Path(__file__).resolve()),bytes=Path(__file__).stat().st_size,sha256=sha(__file__)),scope=SCOPE)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    return dict(directory=str(out),receipt_sha256=sha(out/'receipt.json'),counts=report['counts'],analysis_complete=report['analysis_complete'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True);parser.add_argument('--out',required=True);args=parser.parse_args()
    summary,manifest,provenance=load_saved(args.directory)
    print(json.dumps(write_report(summary,manifest,provenance,args.out,source_directory=args.directory),indent=2),flush=True)

if __name__=='__main__':main()
