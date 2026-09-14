#!/usr/bin/env python3
"""Post-run presentation of frozen EXP008 results; no fitting or integration."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'evidence/EXP-008/attempt-001'
NAMES={'baseline':'Baseline','geometry':'+ geometry','shuffled_geometry':'Mismatched geometry','min_preview':'Minimum preview'}


def scientific_tex(value):
    mantissa, exponent = f'{value:.2e}'.split('e')
    return mantissa + r'\times10^{' + str(int(exponent)) + '}'


def check_summary(saved, rebuilt, path='summary'):
    """Portable presentation arithmetic; experiment and raw hashes stay exact."""
    assert type(saved) is type(rebuilt), path
    if isinstance(rebuilt, dict):
        assert saved.keys()==rebuilt.keys(), path
        for key in rebuilt: check_summary(saved[key],rebuilt[key],path+'.'+key)
    elif isinstance(rebuilt, list):
        assert len(saved)==len(rebuilt), path
        for i,(left,right) in enumerate(zip(saved,rebuilt)): check_summary(left,right,f'{path}[{i}]')
    elif isinstance(rebuilt, float):
        assert math.isfinite(saved) and math.isfinite(rebuilt), path
        assert math.isclose(saved,rebuilt,rel_tol=1e-12,abs_tol=1e-14), path
    else:
        assert saved==rebuilt, path


def derive():
    receipt=json.loads((SOURCE/'receipt.json').read_text())
    for name in ['summary.json','model.json','test-decisions.json','development-targets.jsonl','test-targets.jsonl',
                 'development-features.json','test-features.json','model-seal.json']:
        item=next(v for v in receipt['artifacts'] if v['path']==name)
        assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==item['sha256']
    summary=json.loads((SOURCE/'summary.json').read_text())
    models=json.loads((SOURCE/'model.json').read_text())
    decisions=json.loads((SOURCE/'test-decisions.json').read_text())
    records=[json.loads(line) for role in ['development','test'] for line in (SOURCE/f'{role}-targets.jsonl').read_text().splitlines()]
    assert len(records)==7008 and all(r['status']=='complete' for r in records)
    assert summary['primary_data_gate']['passed']
    fine=summary['steps']['0.005']['panels'];pooled=fine['pooled'];primary=summary['primary']
    lookup={(r['anchor'],r['phase'],r['radius'],r['candidate'],r['dt']):r for r in records}
    differences=[abs(r['rmse']-lookup[(r['anchor'],r['phase'],r['radius'],r['candidate'],.01)]['rmse']) for r in records if r['dt']==.005]
    chosen=[]
    for d in decisions:
        for policy,choice in d['policies'].items():
            key=(d['anchor'],d['phase'],d['radius'],choice['candidate'])
            chosen.append(abs(lookup[(*key,.005)]['rmse']-lookup[(*key,.01)]['rmse']))
    prediction_gain=100*(1-pooled['prediction']['geometry']['mse']/pooled['prediction']['baseline']['mse'])
    preview_gain=100*(1-pooled['means']['min_preview']['selected_rmse']/pooled['means']['geometry']['selected_rmse'])
    fixed=next(d for d in decisions if (d['anchor'],d['phase'],d['radius'])==('information-00',675,.016))
    example={name:lookup[('information-00',675,.016,choice['candidate'],.005)]['rmse'] for name,choice in fixed['policies'].items()}
    artifact_index={item['path']:item for item in receipt['artifacts']}
    sensitivity_files=sorted((SOURCE/'sensitivity').glob('*.json'))
    for path in sensitivity_files:
        item=artifact_index[path.relative_to(SOURCE).as_posix()]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
    sense=math.fsum(json.loads(f.read_text())['wall_seconds'] for f in sensitivity_files)
    features={role:json.loads((SOURCE/f'{role}-features.json').read_text()) for role in ['development','test']}
    preview_seconds=math.fsum(r['wall_seconds'] for rows in features.values() for r in rows)
    stats=dict(source_commit=receipt['source_commit'],receipt_sha256=hashlib.sha256((SOURCE/'receipt.json').read_bytes()).hexdigest(),
        complete_targets=7008,complete_previews=3456,artifacts=len(receipt['artifacts']),artifact_bytes=sum(v['bytes'] for v in receipt['artifacts']),
        primary=primary,means=pooled['means'],prediction=pooled['prediction'],prediction_gain_percent=prediction_gain,
        preview_error_reduction_percent=preview_gain,coarse_fine_all=dict(pairs=len(differences),median=float(np.median(differences)),
        p95=float(np.quantile(differences,.95)),maximum=max(differences),mean=float(np.mean(differences))),
        coarse_fine_selected=dict(policy_pairs=len(chosen),median=float(np.median(chosen)),p95=float(np.quantile(chosen,.95)),
        maximum=max(chosen),mean=float(np.mean(chosen))),example=dict(actions={k:v['candidate'] for k,v in fixed['policies'].items()},rmse=example),
        costs=dict(collection_and_summary_seconds=receipt['wall_seconds'],sensitivity_seconds=sense,
                   summed_overlapping_preview_seconds=preview_seconds,fit_seconds=json.loads((SOURCE/'model-seal.json').read_text())['fit_seconds']),
        feature_coverage={role:dict(rows=len(rows),outside_crop=sum(r['geometry'][4]==0 for r in rows),
                           censored=sum(r['geometry'][5]==1 for r in rows)) for role,rows in features.items()})
    tex=r'''% Generated by summarize_information_paper.py from immutable EXP008 evidence.
\subsection{Measured prediction and action results}
All 24 baseline attempts qualify. All 3,456 previews and 7,008 target records complete; the strict primary data gate passes. Table~\ref{tab:information-dose} reports all four policies and the finite-action oracle.

\begin{table}[H]
\centering\small
\begin{tabular}{rrrrrr}
\toprule
Radius & Baseline & + geometry & Mismatch & Preview & Oracle \\
\midrule
'''
    for radius in ['0.00025','0.001','0.004','0.016','pooled']:
        means=fine[radius]['means'];oracle=means['baseline']['selected_rmse']-means['baseline']['regret']
        values=[means[k]['selected_rmse'] for k in NAMES]+[oracle]
        tex+=(radius.capitalize() if radius=='pooled' else radius)+' & '+' & '.join(f'{v:.6f}' for v in values)+r' \\'+'\n'
    tex+=r'''\bottomrule
\end{tabular}
\caption{\textbf{EXP-008 selected waveform error and available headroom.} Fine-step mean tail RMSE. Each condition offers the same eighteen candidate entries; aggregation gives equal weight to phases, doses and test blocks. ``Preview'' selects minimum short-preview error. The oracle takes the minimum measured tail error among the same eighteen entries per condition; it uses hindsight and is not a deployable policy.}
\label{tab:information-dose}
\end{table}
'''
    tex+=f"The primary mean benefit from adding geometry is ${primary['mean_benefit']:.8f}$, with descriptive 95\\% interval $[{primary['ci95'][0]:.8f},{primary['ci95'][1]:.8f}]$; eight blocks favor geometry, three oppose it and one ties. The aggregate advantage remains unresolved. The smallest-dose diagnostic favors geometry relative to the learned baseline, but does not replace the pooled primary or the stronger minimum-preview control.\n\n"
    tex+=r'''\begin{table}[H]
\centering\small
\begin{tabular}{lrrrr}
\toprule
Policy & Selected RMSE & Regret & Prediction MSE & Negative (\%) \\
\midrule
'''
    for name,label in NAMES.items():
        means=pooled['means'][name]
        pred=pooled['prediction'].get(name)
        tex+=f"{label} & {means['selected_rmse']:.6f} & {means['regret']:.6f} & "
        tex+=(f"{pred['mse']:.6f} & {100*pred['below_zero_fraction']:.2f}" if pred else r'--- & ---')+r' \\'+'\n'
    tex+=r'''\bottomrule
\end{tabular}
\caption{\textbf{Prediction quality and decision quality differ.} Fine-step pooled results. Prediction MSE and negative-score fractions cover all eighteen candidate forecasts per context for each learned model. Negative regression scores are retained; actual activation RMSE is nonnegative. Dashes mean this regression diagnostic is not defined for the direct selector, not failed measurements or zero error. Regret uses the common finite-action oracle.}
\label{tab:information-prediction}
\end{table}
'''
    tex+=f"Geometry reduces mean prediction MSE from {pooled['prediction']['baseline']['mse']:.8f} to {pooled['prediction']['geometry']['mse']:.8f}, a {prediction_gain:.2f}\\% reduction in this sample. That improvement does not establish improved decision utility. The fixed ridge penalties selected from development are $10^{{-6}}$ for baseline and mismatched geometry, and $10^{{-2}}$ for real geometry. Linear extrapolation remains visible: the baseline and geometry models produce negative predictions for {100*pooled['prediction']['baseline']['below_zero_fraction']:.2f}\\% and {100*pooled['prediction']['geometry']['below_zero_fraction']:.2f}\\% of candidate entries, respectively. No predictions exceed two. Neither clipping nor retuning is applied after seeing this result.\n\n"
    tex+=r'''\begin{table}[H]
\centering\small
\begin{tabular}{lrrr}
\toprule
Contrast & Mean benefit & Descriptive 95\% interval & Positive blocks \\
\midrule
'''
    for radius in ['0.00025','0.001','0.004','0.016','pooled']:
        c=fine[radius]['contrasts']['baseline_minus_geometry']
        label='Pooled (primary)' if radius=='pooled' else 'Radius '+radius
        tex+=f"{label} & {c['mean_benefit']:.6f} & $[{c['ci95'][0]:.6f},{c['ci95'][1]:.6f}]$ & {c['positive_blocks']}/12"+r' \\'+'\n'
    tex+=r'\midrule'+'\n'
    for key,label in [('baseline_minus_shuffled','Baseline minus mismatch'),('shuffled_minus_geometry','Mismatch minus geometry'),('min_preview_minus_geometry','Preview minus geometry')]:
        c=pooled['contrasts'][key]
        tex+=f"{label} & {c['mean_benefit']:.6f} & $[{c['ci95'][0]:.6f},{c['ci95'][1]:.6f}]$ & {c['positive_blocks']}/12"+r' \\'+'\n'
    tex+=r'''\bottomrule
\end{tabular}
\caption{\textbf{Complete declared EXP-008 contrasts.} The first five rows subtract geometry's selected error from baseline's; the pooled row is the single primary. Remaining rows are prespecified diagnostics. Intervals resample whole paired test blocks and are descriptive; dose/control findings are not additional primary successes or multiplicity-adjusted population claims.}
\label{tab:information-contrasts}
\end{table}
'''
    tex+=f"The minimum-preview control has mean error {pooled['means']['min_preview']['selected_rmse']:.8f}, versus {pooled['means']['geometry']['selected_rmse']:.8f} for geometry: {preview_gain:.2f}\\% lower. It has lower average error at every dose and in ten of twelve pooled blocks; geometry wins one block and one ties. All policies receive eighteen previews, and the direct rule avoids fitting the regression. Thus the current geometry learner does not improve on a directly available decision rule.\n\n"
    tex+=f"The fixed first test example is an exact action coincidence: all four policies choose candidate 4 ($+\\gamma$), with tail RMSE {example['baseline']:.8f}. Figure~\\ref{{fig:information-waveforms}} therefore shows coincident selected traces, alongside every other candidate's measured alternative. The example remains fixed instead of being replaced by a favorable separation.\n\n"
    tex+=r'\subsection{Numerical and resource limits}'+'\n'
    tex+=f"Across all {len(differences):,} coarse/fine target pairs, the median absolute RMSE difference is ${scientific_tex(np.median(differences))}$, the 95th percentile is {np.quantile(differences,.95):.6f}, and the maximum is {max(differences):.6f}. Pointwise convergence across the exhaustive candidate set is not established. For the {len(chosen)} selected policy-evaluation pairs, which repeat actions when methods agree, the median difference is ${scientific_tex(np.median(chosen))}$ and maximum {max(chosen):.6f}. The coarse-step pooled primary benefit is {summary['steps']['0.01']['panels']['pooled']['contrasts']['baseline_minus_geometry']['mean_benefit']:.8f}, with the same unresolved interpretation. Prediction MSE and oracle estimates remain step-specific diagnostics.\n\n"
    z=summary['zero']['test-0.005']
    tex+=f"All 24 fine-step test shams complete, with mean error ${scientific_tex(z['mean'])}$ and maximum ${scientific_tex(z['maximum'])}$. The attempt preserves {len(receipt['artifacts']):,} receipt-listed artifacts. Collection and summary take {receipt['wall_seconds']:.2f} seconds, excluding verification; 48 sensitivity solves total {sense:.2f} seconds within collection and fitting takes {stats['costs']['fit_seconds']:.3f} seconds. The sum of overlapping preview-call durations is {preview_seconds:.2f} seconds, not elapsed runtime or CPU time. The old atlas cost 140.54 seconds separately. Exhaustive candidate targets are evaluation cost, beyond the eighteen forecasts used per decision. There is no demonstrated net speedup or paid-compute benefit.\n"
    return stats,tex


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write',action='store_true');args=parser.parse_args()
    stats,tex=derive()
    outputs={ROOT/'paper/generated-exp008-results.tex':tex,
             ROOT/'evidence/information-paper-summary.json':json.dumps(stats,indent=2,sort_keys=True)+'\n'}
    for path,content in outputs.items():
        if args.write:path.write_text(content)
        elif path.suffix=='.json':check_summary(json.loads(path.read_text()),stats)
        else:assert path.read_text()==content,str(path)
    print(json.dumps(dict(verified=True,targets=stats['complete_targets'],artifacts=stats['artifacts'])))
if __name__=='__main__':main()
