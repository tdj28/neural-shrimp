"""FGR4 zoom and FGR5 fresh trajectories at the fold setting (local CPU, patched float64).

zoom  : one full-network departure-to-departure evaluation from the reconstructed natural exit state x(qU) for
        stratified uniform base points over the whole table domain, plus partners q+delta for the bases inside
        the unresolved interval M, at a ladder of delta. One replicate batch repeats sixteen bases exactly.
fresh : new fresh trajectories from the FGR1 transverse line under a frozen design (docs/fold-return-fgr5-plan.json),
        no stopping on returns.

The network step, departure detector, capture and recurrence checks, evidence layout and dual-disk backup are the
FGR3 ones (tools/fold_return_followup.py). This file only chooses the starts and the stopping rule.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from fold_global_return import sha, write_json
from fold_return_curve import reconstruct
from fold_return_followup import Engine, Evidence


def zoom_items(spec):
    """Every evaluation of the zoom, in a fixed order: bases, partners, replicate."""
    lo, hi = spec['domain']; n = spec['bases']
    rng = np.random.default_rng(spec['seed'])
    base = lo+(hi-lo)*(np.arange(n)+rng.random(n))/n
    m_lo, m_hi = spec['unresolved_interval']
    items = [{'q': float(q), 'base': i, 'delta': 0.0, 'role': 'base'} for i, q in enumerate(base)]
    inside = [i for i, q in enumerate(base) if m_lo <= q < m_hi]
    if spec.get('partners_only'):
        # Extension run: the bases themselves come from the parent zoom; delta = 0 is an exact rerun in a new
        # batch position (rounding control).
        items = []
    for d in spec['deltas']:
        items += [{'q': float(base[i]+d), 'base': i, 'delta': d, 'role': 'partner'} for i in inside]
    items += [{'q': float(base[i]), 'base': i, 'delta': 0.0, 'role': 'replicate'}
              for i in inside[:spec.get('replicates', 0)]]
    return items, inside


def zoom_batches(items, B):
    """Batches of B items sorted by q, so batch members have similar return times; replicate batch last."""
    main = sorted((k for k, it in enumerate(items) if it['role'] != 'replicate'), key=lambda k: items[k]['q'])
    rep = [k for k, it in enumerate(items) if it['role'] == 'replicate']
    batches = [main[i:i+B] for i in range(0, len(main), B)]
    return batches+[rep[i:i+B] for i in range(0, len(rep), B)]


def check_committed(files):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    for f in files:
        committed = subprocess.check_output(['git', 'show', f'{head}:{f}'], cwd=ROOT)
        if hashlib.sha256(committed).hexdigest() != sha(ROOT/f):
            raise ValueError(f'Uncommitted runtime source or plan: {f}')
    return head


def execute(args):
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    head = check_committed(['tools/fold_return_zoom.py', str(plan_path.relative_to(ROOT))])
    evidence = Evidence(args.out, args.backup)
    evidence.save('plan.json', plan)
    started = time.monotonic()
    try:
        engine = Engine(plan, evidence)
        curve = {'mid': float(engine.data['curve_mid']), 'half': float(engine.data['curve_half']),
                 'coef': engine.data['curve_coef']}
        U = engine.parameters['width']
        if args.mode == 'zoom':
            spec = plan['zoom']
            items, inside = zoom_items(spec)
            batches = zoom_batches(items, engine.B)
            mine = [b for k, b in enumerate(batches) if k % args.parts == args.part]
            evidence.save('selection.json', {'part': args.part, 'parts': args.parts, 'zoom_source_commit': head,
                                             'items': items, 'batches': batches, 'this_part': mine,
                                             'bases_inside_interval': len(inside)})
            records = []
            for k, batch in enumerate(mine):
                q = np.array([items[i]['q'] for i in batch])
                seeds = reconstruct(q*U, curve, engine.data['centre'], engine.data['right'])
                name = f'batch-{batches.index(batch):04d}'
                result = engine.run_batch(seeds, name, spec['max_updates'], True)
                for row, (i, outcome) in enumerate(zip(batch, result['outcomes'])):
                    records.append({'item': i, **items[i], 'batch': name, 'batch_row': row, **outcome})
                if time.monotonic()-started > spec['part_wall_seconds']:
                    evidence.save('records.json', records)
                    raise TimeoutError('Zoom part wall bound reached; all completed batches preserved')
            evidence.save('records.json', records)
        elif args.mode == 'fresh':
            spec = plan['fresh']
            if spec['rows'] != engine.B:
                raise ValueError('This design runs the fresh rows as one batch')
            line = json.loads((Path(plan['parent_line'])/'plan.json').read_text())
            prior = Path(line['prior_path'])
            with np.load(prior/'inputs.npz') as z:
                a, b = z['seeds'][spec['line_seed_rows']]
            n = spec['rows']
            rng = np.random.default_rng(spec['seed'])
            s = (np.arange(n)+rng.random(n))/n
            seeds = a+s[:, None]*(b-a)
            evidence.save('selection.json', {'s': s.tolist(), 'parent_inputs': str(prior/'inputs.npz'),
                                             'parent_sha256': sha(prior/'inputs.npz'), 'source_commit': head})
            result = engine.run_batch(seeds, 'trajectories', spec['max_updates'], False)
            evidence.save('result.json', result)
        else:
            raise ValueError(args.mode)
        evidence.save('wall.json', {'wall_seconds': time.monotonic()-started})
        evidence.finish()
        print(json.dumps({'completed': args.mode, 'wall_seconds': time.monotonic()-started,
                          'output': str(evidence.out)}), flush=True)
    except BaseException as exc:
        from fold_global_return import backup_file
        write_json(evidence.out/'failure.json', {'type': type(exc).__name__, 'message': str(exc),
                                               'seconds': time.monotonic()-started})
        backup_file(evidence.out/'failure.json', evidence.out, evidence.backup)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['zoom', 'fresh'])
    p.add_argument('--plan', required=True)
    p.add_argument('--out', required=True); p.add_argument('--backup', required=True)
    p.add_argument('--part', type=int, default=0); p.add_argument('--parts', type=int, default=1)
    execute(p.parse_args())
