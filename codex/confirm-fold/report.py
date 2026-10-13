"""Format every frozen row and retained attempt; no fitting or model calls."""
import argparse
import json
from pathlib import Path


def report(plan, audit):
    targets={t['label']:t for t in plan['targets']}
    observed={t['label']:t for t in audit['targets']}
    checked=sum(t['status']=='ready' for t in plan['targets'])
    all_rows=[r for t in audit['targets'] for r in t['rows']]
    observed_count=sum(r.get('measured_G') is not None for r in all_rows)
    success=sum(r['primary_pass'] for r in all_rows)
    arc_success=sum(r['arctan_pass'] for r in all_rows)
    lines=['# Task 5: prospective fold-passage results','',
           f'**TLDR.** Of 24 planned forecasts, {observed_count} passages were observed; '
           f'{success} meet the frozen one-G-iterate quadratic accuracy criterion and '
           f'{arc_success} meet the separate unrounded arctangent criterion. '
           f'{checked}/2 targets passed setup. Evidence integrity: **{audit["integrity_pass"]}**. '
           'These are forecasts of lingering and escape in the same fixed-prompt, continuous-feedback '
           'Qwen model. They do not establish a benefit for sampled text or behavior monitoring.','',
           f'Plan SHA-256: `{audit["plan_sha256"]}`. '
           f'Primary all-row success: **{audit["primary_all_pass"]}**; '
           f'arctangent all-row success: **{audit["arctan_all_pass"]}**. '
           'Individual scientific scores do not override a failed integrity check.','']
    for label in ['A','B']:
        t=targets[label];o=observed[label]
        lines += [f'## Target {label}: G=F^{t["period"]}', '', f'Status: {t["status"]}; result status: {o["status"]}.','']
        if t['status']=='ready':
            s=json.loads(Path(t['setup']['path']).read_text())
            lines += [f'Independent fold: alpha={t["alpha"]:.16g}, log10(T)={t["log10T"]:.16g}. '
                      f'Critical multiplier modulus={s["spectrum"]["critical_modulus"]:.12g}, '
                      f'argument={s["spectrum"]["critical_argument_turns"]:.3g} turns; '
                      f'next modulus={s["spectrum"]["next_modulus"]:.9g}. '
                      f'AD coefficients in physical `{t["coordinate"]}`: '
                      f'a={t["raw_a"]:.12g}, b={t["raw_b"]:.12g}; '
                      f'ghost direction={t["ghost_direction"]:+d}. U={t["U"]:.12g}. '
                      f'Transverse seed={t["seed"]}; max G={t["max_g"]}.','',
                      f'Left-mode norm={t["geometry"]["left_norm"]:.6g}; '
                      'The transverse directions are defined by the left eigenvector. '
                      f'Transverse kick norm={t["geometry"]["transverse_kick_norm"]:.6g}.','']
        lines += ['| epsilon | entrance | measured G | quadratic G | arctan G | quadratic error | arctan error | primary pass | arctan pass |',
                  '|---:|---|---:|---:|---:|---:|---:|:---:|:---:|']
        measured={r['id']:r for r in o['rows']}
        for row in t['rows']:
            r=measured[row['id']]
            def f(key):
                v=r.get(key)
                return '—' if v is None else f'{v:.8g}' if isinstance(v,float) else str(v)
            lines += [f'| {row["eps"]:.0e} | {row["entrance"]} | {f("measured_G")} | '
                      f'{f("quadratic_G")} | {f("arctan_G")} | {f("quadratic_error")} | '
                      f'{f("arctan_error")} | {r["primary_pass"]} | {r["arctan_pass"]} |']
        bad=[c['check'] for c in o['checks'] if not c['passed']]
        lines += ['',f'Target integrity: **{o["integrity_pass"]}**. Failed checks: '+(', '.join(bad) if bad else 'none')+'.','']
    lines += ['## Scope and interpretation','',
              'The targets were selected before new passage outcomes. A is a new point on the familiar P2 fold; '
              'B is a different period at the assigned P8 fold. Fold coordinates and composed-map nested-AD '
              'coefficients were computed independently. The exact numerical design, runtime, analysis and '
              'entrances were privately pushed before trajectories. Every planned epsilon and entrance is '
              'shown, with no fitted passage coefficient or post-outcome replacement. Counts are in complete '
              'G steps; multiply by 2 or 8 to obtain model-map calls.','',
              'The quadratic map is the primary predictor. The arctangent is a separate approximation, '
              'compared without rounding. The transverse test is one fixed direction per target, of norm U '
              'in ker(l); it is not a statistical sample of perturbations. Passing these selected deterministic '
              'cases supports their finite-range local prediction. Failures constrain that accuracy and '
              'robustness claim. The evidence level is numerical validation. Generalization to other '
              'models, prompts and sampled-token behavior requires separate tests.','',
              'The external raw store contains full states at every counted G iterate, including first exit, '
              'with hash-bound receipts. The audit independently reconstructs all coordinates, crossings, '
              'coefficient contractions and scalar predictions; it does not re-run the language model. '
              'Private Git protects source and compact records; the bulky local raw store has no verified '
              'independent backup in this task.','']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--design',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('refuse to overwrite report')
    a.output.write_text(report(json.loads(a.design.read_text()),json.loads(a.audit.read_text())))

if __name__=='__main__':main()
