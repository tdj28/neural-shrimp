"""Measure eight signed Arb JVP columns using a provenance-bound fixed prefix.

This is a one-phase feasibility experiment. It omits other columns, the orbit
closure proof and full-box derivative remainder; it cannot issue a certificate.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT/'codex/crosscheck'))
import flint
from flint import arb, ctx
import numpy as np
import torch
from qwen_map import ManualQwenSoftMap, PINNED_SHA256, PREFIX_IDS, sha256
from ball_jet import BallQwenJet
from ball_linear import DyadicWeightCache
from ball_qwen import exact
from cache_io import load_prefix
from pilot import OWNED, jsonwrite, packed


def validate_plan(plan):
    if plan['schema'] != 'codex-llm-signed-pilot-v1' or plan['precision'] != 1024:
        raise ValueError('unsupported signed pilot plan')
    if plan['direction_indices'] != list(range(8)) or plan['phase_index'] != 0:
        raise ValueError('pilot scope is phase zero, eight declared coordinate directions')
    if plan['alpha'] != .071674 or plan['log10_temperature'] != -.115:
        raise ValueError('different assigned parameters')
    for key in ('receipt_sha256', 'map_source_sha256', 'proposal_sha256'):
        value = plan[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('unbound plan hash: '+key)
    attempt = Path(plan['prior_attempt']).resolve()
    if attempt.parent != OWNED or not attempt.is_dir():
        raise ValueError('prior attempt must be an existing owned direct child')
    return attempt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(OWNED) or out == OWNED:
        raise ValueError('owned exclusive output required')
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    phase = 'initialization'

    def event(name, **fields):
        row = dict(event=name, phase=phase, elapsed=time.monotonic()-started,
                   utc=datetime.now(timezone.utc).isoformat(), **fields)
        with (out/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps(row, allow_nan=False), flush=True)

    def progress(kind, index, values):
        fields = {}
        if kind.endswith('_end'):
            scalars = [q for row in values for q in row]
            fields = dict(max_primal_radius=str(max(q.value.rad() for q in scalars)),
                          max_tangent_radius=str(max(t.rad() for q in scalars for t in q.tangent)),
                          max_tangent_abs=str(max(t.abs_upper() for q in scalars for t in q.tangent)))
        event(kind, layer=index, **fields)

    try:
        source = os.environ.get('CERT_SOURCE_COMMIT', '')
        if len(source) != 40 or any(v not in '0123456789abcdef' for v in source):
            raise ValueError('exact source commit required')
        plan = json.loads(args.plan.read_text())
        prior = validate_plan(plan)
        ctx.prec = plan['precision']; ctx.threads = 1
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        if os.getpriority(os.PRIO_PROCESS, 0) < 10:
            os.nice(10-os.getpriority(os.PRIO_PROCESS, 0))
        proposal_path = prior/'run/proposal.npz'
        if sha256(proposal_path) != plan['proposal_sha256']:
            raise ValueError('proposal hash mismatch')
        proposal = np.load(proposal_path, allow_pickle=False)['points']
        if proposal.shape != (9, 896) or proposal.dtype != np.float64 or not np.isfinite(proposal).all():
            raise ValueError('unexpected full-state proposal')
        x = proposal[0].copy()
        directions = np.eye(896, dtype=np.float64)[:8]
        jsonwrite(out/'inputs.json', dict(plan=plan, plan_sha256=sha256(args.plan),
                    source_commit=source, pid=os.getpid(), pinned_assets=PINNED_SHA256,
                    precision=ctx.prec, threads=1, python_flint=flint.__version__,
                    flint=flint.__FLINT_VERSION__, numpy=np.__version__, torch=torch.__version__,
                    statement='one phase, eight columns only; no orbit certificate'))
        phase = 'model_load'; event('load')
        model = ManualQwenSoftMap.from_pinned_files()
        weights = {name: tensor.numpy() for name, tensor in model.weights.items()}
        for weight in weights.values(): weight.flags.writeable = False
        vector = model.vector.numpy(); vector.flags.writeable = False
        ball = BallQwenJet(model.config, weights, PREFIX_IDS, vector, direction_count=8,
                    alpha=plan['alpha'], log10_temperature=plan['log10_temperature'],
                    linear_options=dict(chunk_rows=64, chunk_cols=896,
                                        cache=DyadicWeightCache(max_elements=262144)), progress=progress)
        phase = 'prefix_restore'
        binding = load_prefix(prior, ball, expected_receipt_sha256=plan['receipt_sha256'],
                    expected_map_source_sha256=plan['map_source_sha256'], expected_assets=PINNED_SHA256,
                    expected_config=model.config, expected_weights=weights, expected_vector=vector,
                    expected_steering_layer=model.steering_layer)
        jsonwrite(out/'prefix-binding.json', binding); event('prefix_restored', **binding)
        phase = 'signed_point_jvp'; began = time.monotonic()
        values, derivatives = ball.forward(x, directions)
        arb_seconds = time.monotonic()-began
        jsonwrite(out/'signed-step.json', dict(value=[packed(v) for v in values],
                    columns=[[packed(derivatives[i,k]) for i in range(896)] for k in range(8)],
                    direction_indices=list(range(8)), seconds=arb_seconds))
        event('signed_step_done', seconds=arb_seconds,
              max_primal_radius=str(max(v.rad() for v in values)),
              max_derivative_radius=str(max(derivatives[i,k].rad() for i in range(896) for k in range(8))))
        # Independent binary64 evaluation is a consistency check, never an enclosure gate.
        phase = 'float64_comparison'; began = time.monotonic()
        state = torch.tensor(x, dtype=torch.float64)
        cache = model.prefix_cache(plan['alpha'])
        floating_value = model.forward(state, plan['alpha'], plan['log10_temperature'], cache).numpy()
        floating_columns = np.array([model.jvp(state, torch.tensor(v), plan['alpha'],
                    plan['log10_temperature'], cache=cache).numpy() for v in directions])
        np.savez(out/'float64-comparison.npz', value=floating_value, columns=floating_columns)
        rows = []
        for k in range(8):
            midpoints = np.array([float(derivatives[i,k].mid()) for i in range(896)])
            difference = midpoints-floating_columns[k]
            norm = np.linalg.norm(midpoints)
            rows.append(dict(direction=k, maximum_absolute_difference=float(np.max(np.abs(difference))),
                        relative_l2_difference=float(np.linalg.norm(difference)/norm) if norm else None,
                        midpoint_l2_norm=float(norm),
                        rigorous_maximum_radius=str(max(derivatives[i,k].rad() for i in range(896)))))
        comparison = dict(columns=rows,
                    value_maximum_absolute_difference=float(np.max(np.abs(np.array([float(v.mid()) for v in values])-floating_value))),
                    seconds=time.monotonic()-began,
                    note='Exact-real and binary64 maps differ; floating values are not required to lie in the tight exact-real balls.')
        jsonwrite(out/'comparison.json', comparison)
        jsonwrite(out/'status.json', dict(status='setup_completed', certificate_passed=False,
                    reason='eight signed point columns only; 888 columns, full-box remainder and orbit gates absent',
                    arb_seconds=arb_seconds, elapsed=time.monotonic()-started,
                    max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        event('completed', comparison=comparison)
    except Exception as error:
        jsonwrite(out/'status.json', dict(status='setup_failed', certificate_passed=False,
                    phase=phase, error=repr(error), traceback=traceback.format_exc(), elapsed=time.monotonic()-started))
        raise


if __name__ == '__main__': main()
