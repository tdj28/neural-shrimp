"""Collect completed FGR3 audits and figure bindings into a public receipt."""
import argparse
import json
from collections import Counter
from pathlib import Path

from fold_global_return import sha, write_json
from publish_fold_autonomous_figures import checked_audit, RAW, ARCHIVE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--return-figures', type=Path, required=True)
    parser.add_argument('--growth-figures', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    controller = RAW/'20261113-fgr3-controller-01'
    status = json.loads((controller/'status.json').read_text())
    assert status['stage'] == 'ready_for_publication', status
    assert (controller/'complete.json').exists()
    sources = {}
    audits = {}
    bindings = {}
    for stage in ('training', 'validation', 'growth', 'input-bindings', 'error-diagnostic'):
        root, result = checked_audit(stage, sources)
        audits[stage] = result
        bindings[stage] = {'path': str(root), 'audit_sha256': sha(root/'audit.json')}
    for stage in ('training', 'validation', 'growth'):
        run = RAW/f'20261113-fgr3-{stage}-01/run'
        manifest = run/'manifest.json'
        assert sha(manifest) == audits[stage]['bindings']['manifest_sha256']
        assert sha(manifest) == sha(ARCHIVE/f'20261113-fgr3-{stage}-01/run/manifest.json')
        bindings[stage].update(raw=str(run), backup=audits[stage]['bindings']['backup'],
                               manifest_sha256=sha(manifest), source_commit=audits[stage]['bindings']['source_commit'])
    frozen = RAW/'20261113-fgr3-validation-01/run/frozen-model.json'
    assert sha(frozen) == audits['error-diagnostic']['model_sha256']
    figures = {}
    for key, root in [('returns', args.return_figures), ('growth', args.growth_figures)]:
        backup = ARCHIVE/root.name
        assert sha(root/'receipt.json') == sha(backup/'receipt.json')
        fig = json.loads((root/'receipt.json').read_text())
        for name, item in fig['outputs'].items():
            assert sha(root/name) == item['sha256'] == sha(backup/name)
            assert (root/name).stat().st_size == item['bytes']
        for path, digest in fig['sources'].items():
            assert sha(Path(path)) == digest
        figures[key] = {'path':str(root), 'backup':str(backup),
                        'receipt_sha256':sha(root/'receipt.json'), 'outputs':fig['outputs']}
    valroot = Path(bindings['validation']['path'])
    forecasts = json.loads((valroot/'autonomous.json').read_text())
    terminal = {}
    for mode in ('piecewise', 'nearest'):
        all_kinds = Counter(p['kind'] for r in forecasts for p in r['forecasts'][mode])
        terminal[mode] = {'step_labels': dict(all_kinds),
                          'final_labels': dict(Counter(r['forecasts'][mode][-1]['kind'] for r in forecasts)),
                          'outside_domain_abstentions': all_kinds.get('outside_domain', 0),
                          'limits':'128-cycle surrogate cap and fixed network update cap differ; these terminal counts are not fate accuracy.'}
    validation = audits['validation']
    result = {'study':'FGR3', 'status':'Scientific audits complete; publication verification tracked separately.',
              'plan':'docs/fold-return-autonomous-plan.json',
              'plan_sha256':sha(Path('docs/fold-return-autonomous-plan.json')),
              'audits':bindings, 'audit_files':sources, 'model_sha256':sha(frozen),
              'training':audits['training'],
              'validation':{k:v for k,v in validation.items() if k!='autonomous'},
              'autonomous_rollout_horizons':{k:v['horizons'] for k,v in validation['autonomous'].items()},
              'autonomous_terminal_counts':terminal,
              'input_checks':audits['input-bindings'],
              'post_test_diagnosis':audits['error-diagnostic'],
              'growth':audits['growth'], 'figures':figures,
              'limits':['One model, prefix, steering direction and parameter setting; constructed sample, not basin volume.',
                        'Capture precision/recall uses observed-departure inputs; autonomous rollouts never reset to network states.',
                        'Conditional coordinate/time errors exclude premature terminal predictions; cohorts differ from baseline.',
                        'No refitting against fresh outcomes; endpoint agreement does not prove an interval resolved.',
                        'Positive finite-window tangent growth before capture does not certify a chaotic saddle or persistent chaos.',
                        'Finite offsets independently check local JVPs for the first 128 updates only, not full-window twin exponents.']}
    write_json(args.out,result)
    print(json.dumps({'receipt':str(args.out), 'sha256':sha(args.out),
                      'complete_growth_windows':audits['growth']['complete_network_windows']},indent=2))


if __name__ == '__main__':
    main()
