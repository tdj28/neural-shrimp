"""Plot audited FGR3 evidence only; no model calls, fitting, or new trajectories."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from fold_global_return import sha, write_json, backup_file

DATA = Path('/data2/neural-shrimp-data')
AUDITS = DATA/'audits/fold-global-return'
ARCHIVE = Path('/archive/neural-shrimp-backups')
RAW = DATA/'explore/llm-soft-loop/global-return'


def read(path):
    return json.loads(path.read_text())


def checked_audit(stage, sources):
    name = f'20261113-fgr3-{stage}-01'
    root = AUDITS/name
    backup = ARCHIVE/name
    if stage in ('training', 'validation', 'growth'):
        backup /= 'audit'
    if stage == 'input-bindings':
        # This one-file auditor copies audit.json directly, without a manifest.
        assert sha(root/'audit.json') == sha(backup/'audit.json')
        sources[str(root/'audit.json')] = sha(root/'audit.json')
        return root, read(root/'audit.json')
    assert sha(root/'manifest.json') == sha(backup/'manifest.json')
    for f in read(root/'manifest.json'):
        for base in (root, backup):
            p = base/f['path']
            assert p.stat().st_size == f['bytes'] and sha(p) == f['sha256'], p
        sources[str(root/f['path'])] = f['sha256']
    sources[str(root/'manifest.json')] = sha(root/'manifest.json')
    return root, read(root/'audit.json')


def save(fig, out, name):
    for ext in ('png', 'pdf'):
        meta = {'CreationDate': None, 'ModDate': None} if ext == 'pdf' else {'Software': 'Matplotlib'}
        fig.savefig(out/f'{name}.{ext}', dpi=240, metadata=meta)
    plt.close(fig)


def returns(out, sources):
    _, training = checked_audit('training', sources)
    root, validation = checked_audit('validation', sources)
    _, inputs = checked_audit('input-bindings', sources)
    _, diagnosis = checked_audit('error-diagnostic', sources)
    assert inputs['validation']['frozen_table_binding_passed']
    assert inputs['training_manifest_sha256'] == training['bindings']['manifest_sha256']
    assert inputs['validation_manifest_sha256'] == validation['bindings']['manifest_sha256']
    model_path = RAW/'20261113-fgr3-validation-01/run/frozen-model.json'
    assert sha(model_path) == diagnosis['model_sha256']
    sources[str(model_path)] = sha(model_path)
    model = read(model_path)
    samples = model['rows']
    assert len(samples) == training['rows']
    returning = [r for r in samples if r['kind'] == 'return']
    captured = [r for r in samples if r['kind'] == 'reference_p3']
    other = [r for r in samples if r['kind'] == 'other_recurrence_candidate']
    assert len(returning) == training['outcomes']['return']
    horizons = validation['autonomous']['piecewise']['horizons']
    baseline = validation['autonomous']['nearest']['horizons']
    counts = validation['one_cycle_route']
    for mobile in (False, True):
        fig, axes = plt.subplots(2 if mobile else 1, 1 if mobile else 2,
                                figsize=(5.5, 9) if mobile else (11, 4.8), layout='constrained')
        a, b = axes.ravel()
        a.scatter([r['q'] for r in returning], [r['next_q'] for r in returning],
                  s=7, color='#1f6785', rasterized=True)
        a.axhline(4.17, color='#ccc', lw=.8)
        a.scatter([r['q'] for r in captured], np.full(len(captured), 4.25),
                  s=30, marker='|', color='#a43b43', label=f'{len(captured)} P3 approaches')
        a.scatter([r['q'] for r in other], np.full(len(other), 4.25),
                  s=27, marker='x', color='#333', label='1 other recurrence candidate')
        a.set(xlim=(2.99, 4.13), ylim=(2.97, 4.30), xlabel=r'Current departure $q=v/U$',
              ylabel='Next departure q',
              title='(a) The measured return is finely structured')
        a.set_yticks([3, 3.25, 3.5, 3.75, 4, 4.25], ['3.00', '3.25', '3.50', '3.75', '4.00', 'Stopped'])
        a.legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.18),
                 title=f'{len(samples):,} development starts; {len(returning)} returns', title_fontsize=8)
        x = np.arange(len(horizons))
        b.plot(x, [h['coordinate_RMSE_U'] for h in horizons], 'o-', color='#a43b43', label='Frozen piecewise table')
        b.plot(x, [h['coordinate_RMSE_U'] for h in baseline], 's--', mfc='white', color='#1f6785', label='Nearest-sample baseline')
        b.set_xticks(x, [str(h['cycles']) for h in horizons])
        b.set(xlabel='Cycles predicted without resetting to the network', ylabel=r'Next-departure RMSE (units of $U$)',
              ylim=(0, .58), title='(b) Independent rollouts lose accuracy')
        b.grid(alpha=.15)
        b.legend(fontsize=8, loc='upper left')
        for k, h in enumerate(horizons):
            b.annotate(f"{h['predicted_returning_rows']}/{h['observed_returning_rows']}",
                       (k, h['coordinate_RMSE_U']), xytext=(0, 11), textcoords='offset points',
                       ha='center', fontsize=8, color='#a43b43')
        b.text(.5, -.18, 'Fractions: table / network continuing returns.\nBaseline counts: 16, 16, 14, 14.\nErrors exclude premature terminal predictions.',
               ha='center', va='top', transform=b.transAxes, fontsize=8)
        save(fig, out, 'fig-fold-autonomous-test' + ('-mobile' if mobile else ''))
    return {'training': training, 'validation': validation, 'input_checks': inputs,
            'post_test_diagnosis': diagnosis, 'model_sha256': sha(model_path),
            'selection': 'All training states; all fresh trajectories. Rollout errors conditional on both model and network returning.',
            'capture_warning': f"Reset-at-observed-exit tests: {counts['capture_true_positive']} captures found, "
                               f"{counts['capture_false_negative']} missed, {counts['capture_false_positive']} false captures. "
                               'This is separate from autonomous rollout accuracy.'}


def growth(out, sources):
    root, validation = checked_audit('validation', sources)
    _, audited = checked_audit('growth', sources)
    rows = read(root/'rows.json')
    captures = sorted(r['capture']['first_128_match_start'] for r in rows
                      if r['capture']['first_128_match_start'] is not None)
    windows = [w for w in audited['windows'] if w['start'] is not None]
    control = next(w for w in audited['windows'] if w['start'] is None)
    assert control['control_passed']
    for mobile in (False, True):
        fig, axes = plt.subplots(2 if mobile else 1, 1 if mobile else 2,
                                figsize=(5.5, 9.8) if mobile else (11, 4.8), layout='constrained')
        a, b = axes.ravel()
        times = [0] + captures + [16384]
        survivors = [len(rows)] + [len(rows)-i-1 for i in range(len(captures))] + [len(rows)-len(captures)]
        a.step(times, survivors, where='post', color='#1f6785', lw=2)
        a.plot(16384, survivors[-1], marker='+', ms=11, mew=2, color='#1f6785')
        a.set(xlim=(0, 16900), ylim=(0, 17), xlabel='Individual network updates', ylabel='Starts not yet matched to period 3',
              title='(a) Repeated returns usually ended in capture')
        a.set_xticks([0,4096,8192,12288,16384], ['0','4,096','8,192','12,288','16,384'])
        a.set_yticks([0,2,4,8,12,16])
        a.grid(alpha=.15)
        a.text(.98,.95,f"{len(captures)} of {len(rows)} matched period 3\n{len(rows)-len(captures)} unresolved at the horizon",
               transform=a.transAxes,ha='right',va='top',fontsize=10)
        a.text(.5,-.18,'Fresh stratified starts on a constructed line.\nA survival count, not a basin-volume estimate.',
               transform=a.transAxes,ha='center',va='top',fontsize=8)
        b.axhline(0,color='#777',ls='--',lw=1)
        ticks=[]
        for k,w in enumerate(windows+[control]):
            complete = [r for r in w['rows'] if r['full_planned_window']]
            partial = [r for r in w['rows'] if not r['full_planned_window'] and r['mean'] is not None]
            color = '#1f6785' if w['start'] is None else '#a43b43'
            for r in complete+partial:
                x = k + (r['row']-7.5)*.027
                b.plot([x,x],[r['first_half'],r['second_half']],color=color,alpha=.4,lw=.9)
                b.scatter(x,r['mean'],s=25,facecolors=color if r in complete else 'white',edgecolors=color,zorder=3)
            ticks.append((f"{w['start']:,}" if w['start'] is not None else 'P3 control')+f"\n{len(complete)} full")
        b.set_xticks(range(len(ticks)),ticks)
        b.set(xlabel='Window start (individual network updates)', ylabel='Mean log tangent growth per update',
              title='(b) Growth measured before capture')
        b.text(.5,-.28 if mobile else -.18,'Filled: full window; open: truncated at capture.\nVertical spans join the two measurement-half means.',
               transform=b.transAxes,ha='center',va='top',fontsize=8)
        b.grid(axis='y',alpha=.15)
        save(fig,out,'fig-fold-growth-capture'+('-mobile' if mobile else ''))
    return {'growth':audited,'survival':validation['survival'],
            'selection':'All planned windows; no post-capture padding. Partial windows identified. Survival is conditional on the constructed sample.',
            'limits':'Finite-time tangent growth and finite survival are not an invariant-set certificate or proof of persistent chaos.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=('returns','growth'),required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--backup',type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);args.backup.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    sources={}
    claims=(returns if args.kind=='returns' else growth)(args.out,sources)
    receipt={'kind':args.kind,'sources':sources,'claims':claims,
             'generator':{'path':'tools/publish_fold_autonomous_figures.py','sha256':sha(__file__)},
             'matplotlib':matplotlib.__version__,
             'outputs':{f.name:{'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(args.out.iterdir())}}
    write_json(args.out/'receipt.json',receipt)
    manifest=[backup_file(f,args.out,args.backup) for f in sorted(args.out.iterdir())]
    write_json(args.out/'manifest.json',manifest);backup_file(args.out/'manifest.json',args.out,args.backup)
    print(json.dumps({'kind':args.kind,'out':str(args.out),'outputs':receipt['outputs']},indent=2))


if __name__=='__main__':
    main()
