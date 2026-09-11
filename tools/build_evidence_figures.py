#!/usr/bin/env python3
"""Deterministic, post-run manuscript figures from immutable pilot evidence.

This script does not run training, orbit searches, or new target experiments.
Use PYTHONPATH=src python tools/build_evidence_figures.py [--check].
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

from neural_shrimp.cli import sha256, verify
from neural_shrimp.models import TanhRegressor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/figures'
INDEX = OUT / 'empirical-provenance.json'
CHECKING = False
DERIVED_ATOL = 1e-14
DERIVED_RTOL = 1e-12
SCRIPT = Path(__file__).resolve()
EXPECTED = {'EXP-001': 65, 'EXP-002': 162, 'EXP-003': 96}
POLICIES = ['fixed', 'scheduled_decay', 'stagnation_decay', 'recurrence_decay']
SHORT = ['Fixed', 'Schedule', 'Stagnation', 'Recurrence']
COLORS = ['#27384b', '#d55e00', '#0072b2', '#009e73']
STYLES = ['-', '--', '-.', ':']
PERIOD_COLORS = {2: '#0072b2', 4: '#e69f00', 7: '#009e73', 8: '#cc79a7', 16: '#7a5195'}
SOURCE_CODE = ['src/neural_shrimp/cli.py', 'src/neural_shrimp/models.py']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.2,
 'axes.labelsize': 8.2, 'axes.titlesize': 9, 'xtick.labelsize': 7.5,
 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'figure.titlesize': 11,
 'axes.spines.top': False, 'axes.spines.right': False,
 'savefig.dpi': 300, 'pdf.fonttype': 42, 'ps.fonttype': 42,
 'svg.hashsalt': 'neural-shrimp-empirical-v1', 'svg.fonttype': 'none'})


def serial(value):
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError('Nonfinite derived value must be explicitly represented as missing')
    return value


def compare_derived(expected, actual, path='plotted_data'):
    """Re-derive all arrays; permit only declared floating roundoff across hosts."""
    if isinstance(expected, dict):
        assert isinstance(actual,dict) and expected.keys()==actual.keys(), path
        for key in expected: compare_derived(expected[key],actual[key],path+'.'+key)
    elif isinstance(expected,list):
        assert isinstance(actual,list) and len(expected)==len(actual), path
        for i,(a,b) in enumerate(zip(expected,actual)): compare_derived(a,b,f'{path}[{i}]')
    elif isinstance(expected,float):
        assert np.isclose(expected,actual,atol=DERIVED_ATOL,rtol=DERIVED_RTOL),path
    else:
        assert expected==actual,path


def write_json(path, value):
    path.write_text(json.dumps(serial(value), indent=2, sort_keys=True, allow_nan=False) + '\n')


def relative(path):
    return str(Path(path).relative_to(ROOT))


def binding(path):
    return {'path': relative(path), 'sha256': sha256(path)}


def raw(ex, row):
    with np.load(ROOT / 'evidence' / ex / 'attempt-001/trials' / (row['trial_id'] + '.npz'), allow_pickle=False) as arrays:
        return {k: arrays[k] for k in arrays.files}


def inputs(ex, seed):
    with np.load(ROOT / 'evidence' / ex / 'attempt-001/inputs' / f'seed-{seed}.npz', allow_pickle=False) as a:
        return {k: a[k] for k in a.files}


def load_evidence():
    experiments = {}
    readout_gates = {}
    for ex, count in EXPECTED.items():
        directory = ROOT / 'evidence' / ex / 'attempt-001'
        rows = verify(directory)
        assert len(rows) == count and all(r['experiment_id'] == ex for r in rows)
        assert dict(Counter(r['status'] for r in rows)) == {
            'EXP-001': {'periodic_candidate':39,'unresolved':26},
            'EXP-002': {'unresolved':107,'divergent':55},
            'EXP-003': {'completed':60,'divergent':36}}[ex]
        if ex=='EXP-001':
            assert all(r.get('cycle_certificate',{}).get('verified') and r['cycle_certificate']['attracting'] for r in rows if r['status']=='periodic_candidate')
        config = json.loads((directory / 'manifest.json').read_text())
        receipt = json.loads((directory / 'receipt.json').read_text())
        experiments[ex] = {'rows': rows, 'config': config, 'receipt': receipt}
        max_abs_error = 0.0
        checked = 0
        if ex != 'EXP-001':
            model = TanhRegressor(config['d'], config['width'], config['ridge'])
            datasets = {seed: inputs(ex, seed) for seed in config['seeds']}
            for r in rows:
                if r['status'] == 'divergent':
                    assert r.get('terminal_test_mse') is None
                    continue
                states = raw(ex, r)['states']
                data = datasets[r['seed']]
                value = float(np.mean((model.predict(states[-1], data['test_x']) - data['test_y']) ** 2))
                error = abs(value - r['terminal_test_mse'])
                max_abs_error = max(max_abs_error, error)
                assert np.isclose(value, r['terminal_test_mse'], atol=1e-14, rtol=1e-12)
                checked += 1
            readout_gates[ex] = {'checked_terminal_test_mse_rows': checked,
                'max_absolute_error': max_abs_error, 'atol': 1e-14, 'rtol': 1e-12,
                'status': 'passed', 'description': 'Recomputed released terminal test MSE from each finite saved full state and frozen test split before post-run analysis.'}
    return experiments, readout_gates


def save(fig, stem, title, description, alt_text, experiments, selections, fields,
         transformations, plotted, exclusions, accessibility, gates, intervals='None; deterministic census or descriptive trajectories, no inferential intervals.'):
    if CHECKING:
        plt.close(fig)
        expected=json.loads((OUT/(stem+'.receipt.json')).read_text())
        compare_derived(expected['plotted_data'],serial(plotted))
        assert expected['data_source']['aggregation_and_transformation']==transformations
        assert expected['data_source']['exclusions']==exclusions
        for ex,rows in selections.items():
            assert expected['data_source']['studies'][ex]['trial_selection']==[r['trial_id'] for r in rows]
        return binding(OUT/(stem+'.receipt.json'))
    fig.suptitle(title, x=.055, ha='left', fontweight='bold')
    paths = []
    for suffix in ['pdf', 'svg', 'png']:
        path = OUT / f'{stem}.{suffix}'
        metadata = {'Creator': 'tools/build_evidence_figures.py'} if suffix != 'svg' else {'Creator': 'tools/build_evidence_figures.py', 'Date': None}
        if suffix == 'pdf':
            metadata.update({'CreationDate': None, 'ModDate': None})
        fig.savefig(path, dpi=300, metadata=metadata, facecolor='white')
        paths.append(binding(path))
    plt.close(fig)
    sources = {}
    for ex, selected in selections.items():
        p = ROOT / 'evidence' / ex / 'attempt-001'
        source_paths = [p / 'receipt.json', p / 'manifest.json']
        source_paths += [p / 'trials' / (r['trial_id'] + suffix) for r in selected for suffix in ['.json', '.npz']]
        if ex != 'EXP-001':
            source_paths += [p / 'inputs' / f'seed-{seed}.npz' for seed in sorted({r['seed'] for r in selected})]
        sources[ex] = {'artifacts': [binding(s) for s in source_paths],
            'trial_selection': [r['trial_id'] for r in selected],
            'source_commit': experiments[ex]['receipt']['source_commit'],
            'root_receipt': binding(p / 'receipt.json')}
    receipt = {'figure_id': stem, 'title': title, 'description': description,
        'alt_text': alt_text, 'data_source': {'studies': sources, 'schema_fields': fields,
        'exclusions': exclusions, 'aggregation_and_transformation': transformations},
        'provenance': {'generator': binding(SCRIPT),
            'supporting_code': [binding(ROOT / p) for p in SOURCE_CODE],
            'matplotlib_version': matplotlib.__version__, 'numpy_version': np.__version__,
            'outputs': paths}, 'plotted_data': plotted, 'interval_semantics': intervals,
        'claim_scope_exclusions': ['No new target execution.', 'No verified neural cycle or shrimp geometry claim.', 'Unresolved does not mean chaotic.', 'No statistical generalization beyond these finite pilot tasks.'],
        'accessibility_checks': accessibility,
        'derived_value_check_tolerances': {'atol':DERIVED_ATOL,'rtol':DERIVED_RTOL},
        'hard_guards': {'source_hash_verification': 'neural_shrimp.cli.verify passed for all 323 trials',
            'expected_trial_counts': EXPECTED, 'study_identities_checked': True,
            'test_readout_reconstruction': gates, 'new_target_runs': 0}}
    path = OUT / (stem + '.receipt.json')
    write_json(path, receipt)
    return binding(path)


def map_axis(ax, logarithmic=False):
    if logarithmic:
        ticks = [.03, .3, 1.8, 7]
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xticks(ticks, ['0.03', '0.3', '1.8', '7'])
        ax.set_yticks(ticks, ['0.03', '0.3', '1.8', '7'])
        ax.minorticks_off()
        ax.set(xlim=(.018, 11), ylim=(.018, 11), xlabel='Hidden-layer learning rate', ylabel='Output-layer learning rate')
    else:
        ax.set(xlim=(.77, 1.43), ylim=(.075, .325), xlabel='$a$', ylabel='$b$')
        ax.set_xticks([.8, 1, 1.2, 1.4]); ax.set_yticks([.1, .2, .3])
    ax.grid(alpha=.14, zorder=0)


def henon_map(experiments, gates):
    rows = sorted(experiments['EXP-001']['rows'], key=lambda r: (r['a'], r['b']))
    values = []
    for r in rows:
        c = r.get('cycle_certificate', {})
        values.append({'trial_id': r['trial_id'], 'a': r['a'], 'b': r['b'],
            'primitive_period': c.get('primitive_period') if c.get('verified') and c.get('attracting') else None,
            'spectral_radius': c.get('spectral_radius') if c.get('verified') and c.get('attracting') else None,
            'status': 'verified attracting' if c.get('verified') and c.get('attracting') else 'unresolved'})
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.3))
    fig.subplots_adjust(left=.10, right=.86, bottom=.18, top=.87, hspace=.57)
    for ax in axes: map_axis(ax)
    axes[0].set_title('A  Primitive period', loc='left'); axes[1].set_title('B  Local contraction', loc='left')
    for v in values:
        if v['primitive_period'] is None:
            for ax in axes: ax.scatter(v['a'], v['b'], marker='x', c='#828282', s=25, linewidths=1)
        else:
            p = v['primitive_period']
            axes[0].scatter(v['a'], v['b'], s=97, c=PERIOD_COLORS[p], edgecolors='white', linewidths=.4)
            axes[0].text(v['a'], v['b'], str(p), fontsize=6, color='white' if p != 4 else '#222222', ha='center', va='center')
            axes[1].scatter(v['a'], v['b'], c=[v['spectral_radius']], norm=Normalize(0, 1), cmap='viridis', s=55, edgecolors='#444444', linewidths=.2)
    cax = fig.add_axes([.90, .19, .018, .23])
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap='viridis'), cax=cax)
    cax.set_title(r'$\rho$', pad=7)
    fig.text(.10, .025, 'Numbers: verified primitive periods.  ×: unresolved.\n65 sampled controls; 39 attracting cycles; no boundary continuation.', fontsize=8)
    return save(fig, 'empirical-henon-map', 'The control resolves periodic regimes',
        'A complete sampled Hénon parameter census separates numerically verified primitive periods from unresolved tails and displays local contraction only for certified roots.',
        'Two scatter maps of 65 sampled Hénon controls. The top prints periods 2, 4, 7, 8 and 16 on colored circles; 26 unresolved controls are gray crosses. The bottom shows spectral radius between zero and one for the 39 verified attracting cycles, with the same unresolved crosses. No interpolation or chaos label appears.',
        experiments, {'EXP-001': rows}, ['a', 'b', 'status', 'cycle_certificate.verified', 'cycle_certificate.attracting', 'cycle_certificate.primitive_period', 'cycle_certificate.spectral_radius'],
        ['All 65 trials; identity control coordinates, no interpolation.', 'Plot period and spectral radius only when verified and attracting are both true.'],
        {'rows': values, 'period_counts': dict(Counter(str(v['primitive_period']) for v in values))},
        'No trials excluded. Spectral radius unavailable for unresolved points is marked with a cross, never imputed.',
        ['Numeric period labels supplement color.', 'Crosses identify unresolved controls.', 'Spectral radius has labeled numeric colorbar.'], gates)


def henon_orbits(experiments, gates):
    rows = experiments['EXP-001']['rows']
    selected = []
    for period in [2, 7]:
        eligible = [r for r in rows if r.get('cycle_certificate', {}).get('verified') and r['cycle_certificate'].get('attracting') and r['cycle_certificate']['primitive_period'] == period]
        selected.append(min(eligible, key=lambda r:(r['a'], r['b'], r['trial_id'])))
    selected.append(max([r for r in rows if r['status'] == 'unresolved'], key=lambda r:(r['a'], r['b'], r['trial_id'])))
    fig, axes = plt.subplots(3, 2, figsize=(6.4, 6.5))
    fig.subplots_adjust(left=.10, right=.97, bottom=.10, top=.89, wspace=.33, hspace=.60)
    plotted = []
    for i, r in enumerate(selected):
        states = raw('EXP-001', r)['states']; t = np.arange(r['gradient_updates'] - len(states) + 1, r['gradient_updates'] + 1)
        p = r.get('cycle_certificate', {}).get('primitive_period')
        color = PERIOD_COLORS.get(p, '#666666')
        title = f'Period {p}' if p else 'Unresolved'
        axes[i,0].plot(states[:,0], states[:,1], lw=.45, c=color, alpha=.35)
        axes[i,0].scatter(states[:,0], states[:,1], s=5 if p is None else 27, c=color, edgecolors='none')
        axes[i,0].set(xlabel='$x_t$', ylabel='$y_t$', title=f'{chr(65+2*i)}  {title}: $a={r["a"]:g}, b={r["b"]:g}$')
        for j, (label, style, c) in enumerate([('$x_t$', '-', '#0072b2'), ('$y_t$', '--', '#d55e00')]):
            axes[i,1].plot(t[-48:], states[-48:,j], linestyle=style, color=c, lw=1, label=label)
        axes[i,1].set(xlabel='Saved iteration', ylabel='State coordinate', title=f'{chr(66+2*i)}  Last 48 updates')
        axes[i,1].legend(loc='upper right', ncol=2, frameon=False, handlelength=1.5)
        for ax in axes[i]: ax.grid(alpha=.15)
        plotted.append({'trial_id': r['trial_id'], 'a': r['a'], 'b': r['b'], 'label': title,
                        'saved_iteration': t, 'saved_states': states, 'time_trace_slice': [-48, None]})
    fig.text(.10, .025, 'Post-run representatives from saved full states; unresolved behavior is not a chaos certificate.', fontsize=7.8)
    return save(fig, 'empirical-henon-orbits', 'What a verified cycle looks like in state space',
        'Three saved Hénon trajectories connect the parameter-map labels to actual full-state phase portraits and iteration traces.',
        'Three rows show period 2 at a 0.8 b 0.1, period 7 at a 1.25 b 0.3, and an unresolved trajectory at a 1.4 b 0.3. Left panels plot all 512 saved pairs of x and y; right panels plot the final 48 x and y values with solid and dashed lines. The unresolved trajectory has many distinct states, without an asymptotic chaos claim.',
        experiments, {'EXP-001': selected}, ['a', 'b', 'gradient_updates', 'states', 'status', 'cycle_certificate.primitive_period'],
        ['Post-run illustrative selection: lexicographically smallest (a,b,trial_id) verified attracting period-2 and period-7 trials; largest (a,b,trial_id) unresolved trial.', 'Use all 512 saved states for phase portraits and final 48 states for time traces; no new map iterations.', 'Connecting line segments indicate discrete update ordering, not continuous flow between iterates.'],
        {'selection': plotted}, '62 other controls are omitted from this illustrative view and retained in the complete map.',
        ['Each regime named in panel title.', 'x/y series use solid/dashed lines.', 'All axes labeled in state or iteration units.'], gates)


def neural_map(experiments, gates):
    rows = sorted(experiments['EXP-002']['rows'], key=lambda r:(r['seed'], r['lr_hidden'], r['lr_output']))
    seeds = experiments['EXP-002']['config']['seeds']; finite = [r['terminal_test_mse'] for r in rows if r['status'] != 'divergent']
    norm = LogNorm(min(finite), max(finite))
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.5))
    fig.subplots_adjust(left=.105, right=.87, top=.83, bottom=.18, wspace=.35, hspace=.50)
    counts = {}
    for j, seed in enumerate(seeds):
        group = [r for r in rows if r['seed'] == seed]; counts[str(seed)] = dict(Counter(r['status'] for r in group))
        for ax in axes[:,j]: map_axis(ax, logarithmic=True)
        axes[0,j].set_title(f'{chr(65+j)}  Seed {seed}: classification\n{counts[str(seed)].get("unresolved",0)} unresolved; {counts[str(seed)].get("divergent",0)} divergent', loc='left', fontsize=8.4)
        axes[1,j].set_title(f'{chr(67+j)}  Seed {seed}: test MSE', loc='left')
        for r in group:
            x,y = r['lr_hidden'],r['lr_output']
            if r['status']=='divergent':
                for ax in axes[:,j]: ax.scatter(x,y,marker='x',s=28,c='#777777',linewidths=1)
            else:
                axes[0,j].scatter(x,y,s=28,c='#0072b2',marker='o',edgecolors='none')
                axes[1,j].scatter(x,y,s=30,c=[r['terminal_test_mse']],cmap='viridis',norm=norm,edgecolors='#444444',linewidths=.15)
    cax = fig.add_axes([.90,.22,.018,.22])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='viridis'),cax=cax,label='Terminal test MSE')
    fig.text(.105,.025,'Circles: finite, unresolved.  ×: threshold divergence.\n162 sampled controls; 2,000 updates; no verified neural cycles.',fontsize=8)
    return save(fig,'empirical-neural-map','Low test error can coexist with unresolved dynamics',
        'The complete two-seed learning-rate scan displays dynamical classification and task readout on the actual sampled logarithmic learning-rate coordinates.',
        'Four scatter maps share logarithmic hidden-layer and output-layer learning-rate axes. Two top panels show unresolved finite trials as circles and operational threshold divergence as crosses. Two bottom panels color finite trials by held-out test MSE on a common logarithmic color scale. All 162 controls appear. None of the 107 finite trials produced a verified cycle by 2000 updates; the 55 divergent trials retain their crosses.',
        experiments, {'EXP-002':rows}, ['seed','lr_hidden','lr_output','status','terminal_test_mse'],
        ['All 162 trials; scatter at exact learning-rate coordinates on logarithmic axes; no interpolation or filled cells.', 'Shared logarithmic test-MSE color normalization using all finite trial readouts.'],
        {'rows':[{k:r.get(k) for k in ['trial_id','seed','lr_hidden','lr_output','status','terminal_test_mse']} for r in rows], 'counts_by_seed':counts,'color_norm':{'type':'log','min':min(finite),'max':max(finite)}},
        'No trials excluded. Divergent runs have undefined test MSE and appear as crosses, not as a finite quality value.',
        ['Circle/cross shapes distinguish statuses.', 'Counts and period-absence statement are textual.', 'Common numeric MSE colorbar, exact rate tick labels.'], gates)


def tail_diagnostics(experiments, gates):
    ex='EXP-002'; config=experiments[ex]['config']; model=TanhRegressor(config['d'],config['width'],config['ridge'])
    fig,axes=plt.subplots(2,2,figsize=(6.4,5.4)); fig.subplots_adjust(left=.115,right=.97,top=.86,bottom=.17,wspace=.35,hspace=.42)
    selected=[]; plotted=[]
    for i,seed in enumerate([101,102]):
        d=inputs(ex,seed)
        for j,rate in enumerate([.3,1.8]):
            r=next(r for r in experiments[ex]['rows'] if r['seed']==seed and r['lr_hidden']==rate and r['lr_output']==rate)
            assert r['status']=='unresolved'; selected.append(r); states=raw(ex,r)['states']
            predictions=np.asarray([model.predict(s,d['test_x']) for s in states]); losses=np.asarray([model.loss(s,d['train_x'],d['train_y']) for s in states])
            weight_abs=np.linalg.norm(np.diff(states,axis=0),axis=1)
            prediction_abs=np.sqrt(np.mean(np.diff(predictions,axis=0)**2,axis=1)); loss_abs=np.abs(np.diff(losses))
            weight_rel=weight_abs/np.maximum(np.linalg.norm(states[:-1],axis=1),1e-15)
            prediction_rel=prediction_abs/np.maximum(np.sqrt(np.mean(predictions[:-1]**2,axis=1)),1e-15)
            loss_rel=loss_abs/np.maximum(np.abs(losses[:-1]),1e-15)
            t=np.arange(r['gradient_updates']-len(states)+2,r['gradient_updates']+1)
            ax=axes[i,j]
            for values,label,color,style in [(weight_rel,'Full weights','#27384b','-'),(prediction_rel,'Test predictions','#0072b2','--'),(loss_rel,'Training loss','#d55e00',':')]:
                # Mask exact zero for a log display; counts remain explicit in plotted data.
                ax.plot(t,np.ma.masked_equal(values,0),label=label,c=color,ls=style,lw=1.15)
            ax.set(yscale='log',xlabel='Saved iteration',ylabel='Relative one-step change',title=f'{chr(65+2*i+j)}  Seed {seed}, rates ({rate:g}, {rate:g})')
            ax.grid(alpha=.18)
            plotted.append({'trial_id':r['trial_id'],'seed':seed,'rates':[rate,rate],'iteration':t,
                'weight_step_l2':weight_abs,'prediction_step_rms':prediction_abs,'loss_step_absolute':loss_abs,
                'weight_relative_step':weight_rel,'prediction_relative_step':prediction_rel,'loss_relative_step':loss_rel,
                'training_loss':losses,'zero_change_counts':{'weights':int(sum(weight_rel==0)),'predictions':int(sum(prediction_rel==0)),'loss':int(sum(loss_rel==0))}})
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.055),ncol=3,frameon=False)
    fig.text(.115,.015,'Post-run diagnostics of saved states; different readouts can move on different scales.',fontsize=7.8)
    return save(fig,'empirical-neural-tail-diagnostics','Weights, predictions, and loss need separate measurements',
        'Four explicitly chosen neural tails compare full-weight motion, prediction motion, and loss motion without mistaking a smooth scalar readout for a full-state periodic orbit.',
        'A two-by-two grid shows seeds 101 and 102 at equal learning rates 0.3 and 1.8. Each panel uses the final 512 saved weight states to plot 511 successive relative changes in the full weights, predictions on the frozen test inputs, and regularized training loss. Solid, dashed, and dotted lines identify the three readouts. These are post-run descriptive diagnostics, and all four tails remain unresolved under the frozen recurrence screen.',
        experiments, {ex:selected}, ['states','gradient_updates','seed','lr_hidden','lr_output','frozen train_x/train_y/test_x','model.loss','model.predict'],
        ['Preset post-run display conditions: seeds 101/102 crossed with equal learning rates 0.3/1.8. All four existed and were unresolved; no replacement selection.', 'Weight relative step = L2(theta[t]-theta[t-1])/max(L2(theta[t-1]),1e-15).', 'Prediction relative step = RMS(f_t(test_x)-f_(t-1)(test_x))/max(RMS(f_(t-1)(test_x)),1e-15).', 'Loss relative step = abs(L_t-L_(t-1))/max(abs(L_(t-1)),1e-15). Training loss includes ridge.', 'Recompute predictions and loss only at saved states; no gradient updates or map advancement.', 'Dimensionless relative changes use separate denominators; equality of their values does not imply equality of task impact.', 'Exact zeros are masked on logarithmic axes and counted in the receipt.'],
        {'conditions':plotted}, '158 other scan conditions are absent from this illustrative tail diagnostic but retained in the census map.',
        ['Solid/dashed/dotted series distinguish readouts.', 'Each panel identifies its seed and rates.', 'Log y axes labeled relative one-step change.'],gates)


def utility_census(experiments,gates):
    ex='EXP-003'; rows=experiments[ex]['rows']; config=experiments[ex]['config']
    cases=[(seed,*rates) for seed in config['seeds'] for rates in config['learning_rate_pairs']]
    assert len(cases)==24
    lookup={(r['seed'],r['lr_hidden'],r['lr_output'],r['policy']):r for r in rows}
    assert len(lookup)==96
    matrix=np.full((24,4),np.nan); values=[]
    for i,case in enumerate(cases):
        v={'case':i+1,'seed':case[0],'rates':list(case[1:]),'policies':{}}
        for j,p in enumerate(POLICIES):
            r=lookup[(*case,p)];v['policies'][p]={k:r.get(k) for k in ['trial_id','status','terminal_test_mse','events']}
            if r['status']!='divergent':matrix[i,j]=r['terminal_test_mse']
        values.append(v)
    finite=matrix[np.isfinite(matrix)];norm=LogNorm(finite.min(),finite.max())
    fig=plt.figure(figsize=(6.4,7.25));gs=fig.add_gridspec(2,2,height_ratios=[4.5,1.3],left=.27,right=.96,top=.855,bottom=.1,hspace=.22,wspace=.65)
    ax=fig.add_subplot(gs[0,:]); ax.imshow(np.ma.masked_invalid(matrix),cmap='viridis',norm=norm,aspect='auto',interpolation='none')
    ax.set_xticks(range(4),SHORT);ax.xaxis.tick_top();ax.tick_params(top=False,bottom=False,left=False)
    labels=[f'{seed}  ({h:g}, {o:g})' for seed,h,o in cases];ax.set_yticks(range(24),labels,fontsize=7.1)
    ax.set_ylabel('Seed  (hidden rate, output rate)',labelpad=8); ax.spines[['left','bottom']].set_visible(False)
    for i in range(24):
        for j in range(4):
            if np.isnan(matrix[i,j]):
                ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor='#ededed',edgecolor='white',linewidth=.4))
                ax.text(j,i,'FAIL',ha='center',va='center',fontsize=7,c='#555555')
            else:
                color='white' if norm(matrix[i,j])<.56 else '#17212b'
                ax.text(j,i,f'{matrix[i,j]:.1e}',ha='center',va='center',fontsize=6.9,c=color)
    for y in [5.5,11.5,17.5]:ax.axhline(y,color='white',lw=2)
    ax.set_title('A  Every paired case: terminal test MSE',loc='left',pad=29)
    summaries=[]
    for p in POLICIES:
        group=[r for r in rows if r['policy']==p]
        summaries.append({'policy':p,'n':len(group),'failed':sum(r['status']=='divergent' for r in group),'triggered_runs':sum(bool(r['events']) for r in group),'events':sum(len(r['events']) for r in group)})
    b=fig.add_subplot(gs[1,0]);c=fig.add_subplot(gs[1,1]);y=np.arange(4)
    for axis,key,title in [(b,'failed','B  Failed / 24'),(c,'triggered_runs','C  Triggered / 24')]:
        nums=[s[key] for s in summaries];axis.barh(y,nums,color='#57718c',height=.6)
        for i,n in enumerate(nums):axis.text(n+.5,i,str(n),va='center',fontsize=8)
        axis.set(yticks=y,yticklabels=SHORT,xlim=(0,24),xticks=[0,12,24],title=title)
        axis.invert_yaxis();axis.grid(axis='x',alpha=.15);axis.tick_params(axis='y',length=0)
    fig.text(.07,.018,'All 24 cases per policy retained. FAIL denotes operational threshold divergence.\nEqual update budget, not equal total compute; recurrence triggered in 0 of 24 cases.',fontsize=7.8)
    # Recurrence never fired: establish complete state-array equality, not merely rounded terminal MSE equality.
    identical=[]
    for case in cases:
        a=lookup[(*case,'fixed')]; b=lookup[(*case,'recurrence_decay')]
        assert not b['events']
        assert np.array_equal(raw(ex,a)['states'],raw(ex,b)['states'],equal_nan=True)
        identical.append({'case':list(case),'fixed':a['trial_id'],'recurrence':b['trial_id'],'full_saved_states_equal':True})
    return save(fig,'empirical-utility-census','Policy outcomes depend on the task and learning rates',
        'A complete paired census keeps all seed and rate combinations aligned, displays the task metric per policy, retains each divergent run, and shows whether a policy actually intervened.',
        'The main table heatmap contains 24 rows from four seeds and six rate pairs, with four policy columns. Every finite terminal test MSE is printed in scientific notation; all 36 divergent runs are marked FAIL. Two horizontal bar charts show nine failures under each policy and triggered-run counts of zero for fixed, fifteen for schedule, twelve for stagnation, and zero for recurrence. Fixed and recurrence saved full states are exactly equal in every paired case.',
        experiments,{ex:sorted(rows,key=lambda r:(cases.index((r['seed'],r['lr_hidden'],r['lr_output'])),POLICIES.index(r['policy'])))},
        ['seed','lr_hidden','lr_output','policy','status','terminal_test_mse','events','states'],
        ['All 96 trials aligned by seed and learning-rate pair; row order is manifest seed order crossed with manifest rate-pair order.', 'Finite MSE colored with common logarithmic normalization and printed to two significant digits; exact values retained here.', 'No success-only mean, pooled quality summary, or inferential interval.', 'Failures count divergent runs; triggered runs count nonempty events; event counts retained in receipt.', 'Compare full saved state arrays for fixed/recurrence pairs with np.array_equal(equal_nan=True).'],
        {'cases':values,'policy_census':summaries,'fixed_recurrence_identity':identical,'color_norm':{'type':'log','min':float(finite.min()),'max':float(finite.max())}},
        'No cases or policies excluded. Failed-case MSE is undefined and visibly marked FAIL.',
        ['Every heatmap cell contains a numeric value or FAIL.', 'Policy counts printed on bars.', 'Seed and both learning rates printed per row.'],gates)


def utility_trajectories(experiments,gates):
    ex='EXP-003';rows=experiments[ex]['rows'];config=experiments[ex]['config'];seed=202
    # Post-run display rule disclosed in advance of this figure, without a success claim.
    pairs=[pair for pair in config['learning_rate_pairs'] if all(next(r for r in rows if r['seed']==seed and [r['lr_hidden'],r['lr_output']]==pair and r['policy']==p)['status']!='divergent' for p in POLICIES)][:3]
    assert pairs==[[.3,.3],[1.2,1.2],[1.8,1.8]]
    fig,axes=plt.subplots(3,3,figsize=(6.4,6.4));fig.subplots_adjust(left=.13,right=.975,top=.86,bottom=.19,wspace=.22,hspace=.25)
    selected=[];plotted=[]
    for j,pair in enumerate(pairs):
        axes[0,j].set_title(f'Rates ({pair[0]:g}, {pair[1]:g})')
        for k,p in enumerate(POLICIES):
            r=next(r for r in rows if r['seed']==seed and [r['lr_hidden'],r['lr_output']]==pair and r['policy']==p)
            selected.append(r);a=raw(ex,r);losses=a['losses'];updates=np.linalg.norm(np.diff(a['states'],axis=0),axis=1);rates=a['learning_rates']
            assert len(losses)==len(updates)==len(rates)==1200
            assert np.array_equal(rates[:,0],rates[:,1])
            for i,y in enumerate([losses,updates,rates[:,0]]):
                axes[i,j].plot(np.arange(len(y)),np.ma.masked_equal(y,0),c=COLORS[k],ls=STYLES[k],lw=1 if k!=3 else 1.25,label=SHORT[k],alpha=.95)
            for event in r['events']:
                for ax in axes[:,j]:ax.axvline(event['step'],c=COLORS[k],lw=.6,ls=':',alpha=.38)
            plotted.append({'trial_id':r['trial_id'],'seed':seed,'initial_rates':pair,'policy':p,'iteration':np.arange(len(losses)),
                'training_loss':losses,'weight_update_l2':updates,'learning_rates':rates,'events':r['events'],'terminal_test_mse':r['terminal_test_mse']})
        for i in range(3):
            axes[i,j].set_yscale('log');axes[i,j].grid(alpha=.15);axes[i,j].set_xlim(0,1200)
            axes[i,j].set_xticks([0,600,1200]);axes[i,j].text(.02,.96,chr(65+3*i+j),transform=axes[i,j].transAxes,ha='left',va='top',fontweight='bold',fontsize=8)
            if i<2:axes[i,j].tick_params(labelbottom=False)
            else:axes[i,j].set_xlabel('Gradient-update index')
    # Common limits per row expose actual cross-condition scale differences.
    for i,key in enumerate(['training_loss','weight_update_l2','learning_rates']):
        vals=np.concatenate([np.asarray(item[key]).ravel() for item in plotted]); vals=vals[vals>0]
        for j in range(3):
            axes[i,j].set_ylim(float(vals.min())*.75,float(vals.max())*1.3)
            axes[i,j].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            if j>0: axes[i,j].tick_params(labelleft=False)
    axes[0,0].set_ylabel('Regularized train loss');axes[1,0].set_ylabel(r'$\|\theta_{t+1}-\theta_t\|_2$');axes[2,0].set_ylabel('Realized learning rate')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.075),ncol=4,frameon=False)
    fig.text(.105,.012,'Seed 202; post-run finite-case illustrations. Vertical lines mark interventions.\nFixed and recurrence traces coincide; all failed cases remain in the complete census.',fontsize=7.5)
    return save(fig,'empirical-utility-trajectories','An intervention changes the path, but needs a task readout',
        'Recorded training paths show how scheduled or loss-stagnation learning-rate changes alter weight motion and objective loss in three illustrative finite cases.',
        'Nine panels show seed 202 at equal starting learning rates 0.3, 1.2 and 1.8. The rows display saved regularized training loss, full-weight update norm, and realized learning rate for fixed, scheduled decay, stagnation decay, and recurrence decay. Policy lines differ by both color and dash pattern. Vertical lines mark actual recorded interventions. At rates 1.8, scheduled and stagnation decay sharply reduce loss and motion; recurrence exactly overlaps fixed because it never triggers. These examples were selected after the run from finite cases and do not summarize all outcomes.',
        experiments,{ex:selected},['states','losses','learning_rates','events','policy','seed','lr_hidden','lr_output','terminal_test_mse'],
        ['Post-run illustrative rule: seed 202, first three manifest-order rate pairs for which all four policies remained finite. Selected pairs are (0.3,0.3), (1.2,1.2), (1.8,1.8).', 'Plot all 1200 saved pre-update regularized losses and learning rates; derive full-weight L2 update norms from successive saved states.', 'Selected pairs retain equal hidden/output realized rates; plot this common rate after asserting exact equality.', 'Vertical intervention markers use exact recorded event steps.', 'Common y-axis limits within each quantity row, derived from all selected trajectories.', 'No smoothing, downsampling, averaging, benchmark claim, or new training.'],
        {'conditions':plotted}, 'Illustrative figure omits the other 84 trials; the census figure retains all finite and divergent cases. Selection uses post-run finiteness and cannot support an aggregate success claim.',
        ['Policy colors also use distinct line styles and named legend.', 'Each row has a distinct quantity label.', 'Overlapping fixed/recurrence paths explicitly disclosed.'],gates)


def check():
    global CHECKING
    index=json.loads(INDEX.read_text())
    for artifact in index['source_receipts']+[index['generator']]+index['supporting_code']+index['figures']:
        assert sha256(ROOT/artifact['path'])==artifact['sha256'],f'Hash drift: {artifact["path"]}'
    experiments,gates=load_evidence()
    for ex,gate in gates.items():
        for key in ['checked_terminal_test_mse_rows','atol','rtol','status']:
            assert gate[key]==index['readout_gates'][ex][key]
        # load_evidence already asserts each replayed readout within its frozen tolerance.
        # max_absolute_error is host-dependent descriptive metadata, not a bitwise gate.
    for ref in index['figures']:
        receipt=json.loads((ROOT/ref['path']).read_text())
        assert receipt['provenance']['generator']==index['generator']
        for artifact in receipt['provenance']['outputs']+receipt['provenance']['supporting_code']:
            assert sha256(ROOT/artifact['path'])==artifact['sha256'],f'Hash drift: {artifact["path"]}'
        for study in receipt['data_source']['studies'].values():
            for artifact in study['artifacts']:
                assert sha256(ROOT/artifact['path'])==artifact['sha256'],f'Hash drift: {artifact["path"]}'
    CHECKING=True
    try:
        for fn in [henon_map,henon_orbits,neural_map,tail_diagnostics,utility_census,utility_trajectories]:
            fn(experiments,gates)
    finally:
        CHECKING=False
    print('PASS: 323 source trials, terminal readout gates, six figure receipts, all plotted arrays and selections re-derived, 18 output hashes, generator and supporting code.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if args.check:return check()
    OUT.mkdir(parents=True,exist_ok=True);experiments,gates=load_evidence()
    receipts=[fn(experiments,gates) for fn in [henon_map,henon_orbits,neural_map,tail_diagnostics,utility_census,utility_trajectories]]
    write_json(INDEX,{'schema_version':1,'analysis_status':'post-run descriptive visualization of immutable exploratory pilots',
        'generator':binding(SCRIPT),'supporting_code':[binding(ROOT/p) for p in SOURCE_CODE],
        'source_receipts':[binding(ROOT/'evidence'/ex/'attempt-001/receipt.json') for ex in EXPECTED],
        'figures':receipts,'readout_gates':gates,'new_target_runs':0,'empirical_figures':6,
        'notes':'Each receipt includes exact plotted arrays, trial IDs, transformations, source and output SHA-256. Regeneration uses original recorded states, never new target execution.'})
    check()


if __name__=='__main__':main()
