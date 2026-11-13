"""Numerical closure, primitive-period and full float64 monodromy of FGR1's cycle."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from fold_global_return import backup_file, sha, write_json


def dense_from_row_products(product, dimension, chunk):
    if dimension % chunk:
        raise ValueError('Use equal-sized blocks to retain the same model batch')
    eye = np.eye(dimension)
    return np.concatenate([product(eye[i:i+chunk]) for i in range(0, dimension, chunk)]).T


def main(args):
    from explore_llm_soft_loop import SoftLoop
    import explore_llm_phase_map as phase
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    plan = json.loads((run / 'plan.json').read_text())
    result = json.loads((run / 'result.json').read_text())
    prior = json.loads((run / 'metadata.json').read_text())
    backup = Path(prior['backup']) / 'cycle-verification'
    candidate_ids = [i for i, r in enumerate(result['outcomes']) if r['kind'] == 'outside_recurrence_candidate']
    if not candidate_ids or any(result['outcomes'][i]['lag'] != 3 for i in candidate_ids):
        raise ValueError('This follow-up requires the observed period-three candidates')
    states = []
    for p in sorted(run.glob('updates-*.npz')):
        with np.load(p) as z:
            states.append(z['states'][:, candidate_ids])
    states = np.concatenate(states)
    ends = np.array([states[result['valid_through'][i]-1, k] for k, i in enumerate(candidate_ids)])
    x = ends[0]
    so, B, d = plan['source'], 64, len(x)
    start = time.monotonic()
    loop = SoftLoop(so['model'], so['layer'], 8, so['prompt'], so['opener'], so['window'], 'float64', 'cpu')
    torch = loop.torch
    direction = phase.load_vectors(so['vectors'])[so['concept']]
    loop.prepare(np.tile((so['alpha'] + plan['epsilon']) * direction, (B, 1)))
    T = np.full(B, 10. ** so['log10_temperature'])
    X0 = torch.tensor(np.tile(x, (B, 1))[:, None], dtype=torch.float64)
    pts, readout = [x], []
    X = X0.clone()
    for _ in range(3):
        with torch.no_grad():
            X, p = loop.step(X, T)
            values, ids = torch.topk(p[0], 8)
        readout.append([{'id': int(i), 'text': loop.sm.tok.decode([int(i)]), 'probability': float(v)}
                        for i, v in zip(ids, values)])
        pts.append(X.numpy()[0, 0].copy())
    closure = [float(np.linalg.norm(y-x) / np.linalg.norm(x)) for y in pts[1:]]
    write_json(out / 'closure.json', {'relative_residuals_lags_1_2_3': closure, 'readout': readout})
    if not (closure[2] < 1e-11 and min(closure[:2]) > 1e-4):
        raise ValueError(f'Primitive period-three closure gate failed: {closure}')
    calls = 0

    def product(vectors):
        nonlocal calls
        X = X0.clone()
        V = torch.tensor(vectors[:, None], dtype=torch.float64)
        for _ in range(3):
            X, _, V = loop.step_jvp(X, T, V)
        if float(np.max(np.abs(X.numpy()[0, 0] - pts[3]))) > 1e-12:
            raise ValueError('AD primal differs from the saved map')
        calls += 1
        print(json.dumps({'composed_tangent_batches': calls}), flush=True)
        return V.numpy()[:, 0].copy()

    M = dense_from_row_products(product, d, B)
    rng = np.random.default_rng(20261113)
    v = rng.normal(size=d); v /= np.linalg.norm(v)
    direct = product(np.tile(v, (B, 1)))[0]
    dense = M @ v
    relative_jvp_error = float(np.linalg.norm(dense-direct) / np.linalg.norm(direct))
    h = np.linalg.norm(x) * np.array([1e-3, 3e-4, 1e-4])
    samples = np.tile(x, (B, 1))
    for i in range(3):
        samples[2*i] += h[i]*v
        samples[2*i+1] -= h[i]*v
    Y = torch.tensor(samples[:, None], dtype=torch.float64)
    for _ in range(3):
        with torch.no_grad():
            Y, _ = loop.step(Y, T)
    y = Y.numpy()[:, 0]
    errors = [float(np.linalg.norm((y[2*i]-y[2*i+1])/(2*h[i])-direct) / np.linalg.norm(direct)) for i in range(3)]
    orders = [float(np.log(errors[i]/errors[i+1])/np.log(h[i]/h[i+1])) for i in range(2)]
    precision_pass = relative_jvp_error < 1e-10 and min(errors) < 1e-5 and (
        min(errors) < 1e-9 or all(1.5 < order < 2.5 for order in orders))
    precision = {'composed_dense_vs_direct_JVP_relative_error': relative_jvp_error,
                 'finite_difference_h': h.tolist(), 'finite_difference_relative_errors': errors,
                 'observed_convergence_orders': orders, 'passed': precision_pass}
    write_json(out / 'precision.json', precision)
    if not precision_pass:
        np.save(out / 'monodromy-unqualified.npy', M)
        raise ValueError('Composed derivative precision gate failed; retained, no stability claim')
    values, eigvec = np.linalg.eig(M)
    order = np.argsort(-np.abs(values)); values, eigvec = values[order], eigvec[:, order]
    residuals = np.linalg.norm(M @ eigvec - eigvec * values[None], axis=0) / np.linalg.norm(eigvec, axis=0)
    assert residuals.max() < 1e-8
    matches = [float(min(np.linalg.norm(end-y) / np.linalg.norm(y) for y in pts[:3])) for end in ends]
    assert max(matches) < 1e-9
    np.savez(out / 'orbit-and-monodromy.npz', orbit=np.array(pts[:3]), monodromy=M,
             multipliers=values, eigenpair_residuals=residuals, candidate_end_states=ends)
    record = {'status': 'numerically attracting primitive period-three orbit' if abs(values[0]) < 1 else 'non-attracting or unresolved',
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'source_sha256': sha(__file__), 'run_manifest_sha256': sha(run / 'manifest.json'),
              'plan_sha256': sha(run / 'plan.json'), 'model_revision': loop.sm.engine.revision,
              'device': 'cpu', 'dtype': str(loop.dtype), 'threads': torch.get_num_threads(),
              'primitive_closure': closure, 'candidate_rows': candidate_ids,
              'candidate_relative_distance_to_orbit': matches,
              'spectral_radius': float(abs(values[0])),
              'leading_multipliers': [[float(z.real), float(z.imag)] for z in values[:8]],
              'max_eigenpair_residual': float(residuals.max()), 'precision': precision,
              'seconds': time.monotonic()-start, 'readout': readout,
              'limitations': 'Floating-point numerical verification, not an interval-certified existence or stability proof.'}
    write_json(out / 'verification.json', record)
    receipts = [backup_file(p, out, backup) for p in sorted(out.iterdir()) if p.is_file()]
    write_json(out / 'manifest.json', receipts)
    backup_file(out / 'manifest.json', out, backup)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    main(parser.parse_args())
