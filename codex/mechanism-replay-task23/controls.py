"""Small manufactured controls only; no target assets or producer imports.

Run once only under the separately reviewed supervisor/card. The tiny tensor
fixture checks the new split adapter, not the already accepted full model.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch


def adjacent(name):
    spec = importlib.util.spec_from_file_location('task23_control_' + name,
                                                  Path(__file__).with_name(name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def tiny_model(qwen):
    rng = torch.Generator(device='cpu').manual_seed(107)
    config = {'hidden_size': 8, 'num_hidden_layers': 4, 'num_attention_heads': 2,
              'num_key_value_heads': 1, 'intermediate_size': 12, 'vocab_size': 19,
              'rms_norm_eps': 1e-6, 'rope_theta': 10000., 'hidden_act': 'silu',
              'tie_word_embeddings': True, 'attention_dropout': 0., 'use_sliding_window': False}
    def sample(*shape):
        return torch.randn(*shape, dtype=torch.float64, generator=rng)
    weights = {'model.embed_tokens.weight': .2 * sample(19, 8), 'model.norm.weight': 1 + .03 * sample(8)}
    for layer in range(4):
        name = f'model.layers.{layer}.'
        for ln in ('input_layernorm', 'post_attention_layernorm'):
            weights[name + ln + '.weight'] = 1 + .03 * sample(8)
        for projection, rows in [('q', 8), ('k', 4), ('v', 4)]:
            weights[name + f'self_attn.{projection}_proj.weight'] = .12 * sample(rows, 8)
            weights[name + f'self_attn.{projection}_proj.bias'] = .03 * sample(rows)
        weights[name + 'self_attn.o_proj.weight'] = .12 * sample(8, 8)
        for projection in ('gate', 'up'):
            weights[name + f'mlp.{projection}_proj.weight'] = .12 * sample(12, 8)
        weights[name + 'mlp.down_proj.weight'] = .12 * sample(8, 12)
    return qwen.ManualQwenSoftMap(config, weights, (1, 3, 5), .2 * sample(8), steering_layer=1)


def expect_failure(function, error=ValueError):
    try:
        function()
    except error:
        return
    raise AssertionError('expected refusal did not occur')


def perform(out):
    start = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    gain, qwen = adjacent('gain'), adjacent('qwen_map')
    model = tiny_model(qwen)
    alpha = .0655
    with torch.no_grad():
        cache = model.prefix_cache(alpha)
        base = model.initial_state()[None].repeat(4, 1)
        t = torch.from_numpy(10. ** np.asarray(gain.LOG_T))
        h, z, p, f = gain.split_map(model, base, alpha, t, cache)
        ref_z = model.logits(base, alpha, cache)
        assert torch.equal(z, ref_z)
        assert torch.equal(z, h @ model.embedding.T)
        assert torch.equal(f, torch.softmax(ref_z / t[:, None], -1) @ model.embedding)
        # The separately tested scalar helper uses Torch power; retain its finite difference.
        scalar = torch.stack([model.forward(base[i], alpha, gain.LOG_T[i], cache) for i in range(4)])
        assert torch.allclose(f, scalar, atol=1e-14, rtol=1e-13)
        evidence = {'split_logits_max_abs_difference': float((z-ref_z).abs().max()),
                    'batched_numpyT_vs_scalar_torchT_max_abs_difference': float((f-scalar).abs().max()),
                    'tiny_final_hidden': h.tolist()}
        checks = ['tiny split matches unchanged logits, tied head and physical-temperature readout']

        expect_failure(lambda: gain.split_map(model, base.float(), alpha, t, cache))
        expect_failure(lambda: gain.split_map(model, base, alpha, t * 0, cache))
        expect_failure(lambda: gain.split_map(model, base, alpha + 1, t, cache))
        checks.append('wrong dtype/temperature/cache alpha refuse')

        directions = gain.original_directions(8)
        rng = np.random.default_rng(0)
        complete = np.stack([rng.normal(size=(201, 8)) for _ in range(2)])
        assert np.array_equal(directions, complete[:, list(gain.SOURCE_INDICES), :])
        assert not np.array_equal(directions[0], np.random.default_rng(0).normal(size=(4, 8)))
        checks.append('two full original RNG blocks and selected source rows, not four-row redraw')

        states = gain.initial_twins(base, directions)
        for k, eps in enumerate(gain.EPS):
            actual = gain.norms(states[(k+1)*4:(k+2)*4] - base) / gain.norms(base)
            np.testing.assert_allclose(actual, eps, rtol=2e-7, atol=0)
        checks.append('both represented initial offsets follow the original normalized-direction construction')

        accumulator = gain.Accumulator(base, 5)
        tail, rows = [], []
        for step in range(5):
            h, z, p, f = gain.split_map(model, states, alpha, t.repeat(3), cache)
            row, states = gain.finite_step(states, h, f, z, p, t.repeat(3))
            accumulator.add(row)
            assert row['log_growth'].shape == (2, 4)
            np.testing.assert_allclose(row['next_dx'], np.stack([gain.norms(states[4:8]-states[:4]),
                                                                gain.norms(states[8:]-states[:4])]), rtol=0, atol=0)
            tail.append(states[:4].numpy().copy())
            rows.append({key: value.tolist() for key, value in row.items()})
        summary = accumulator.finish(states[:4], tail)
        assert all(max(abs(v) for v in cell['identity_residual']) < 1e-10 for cell in summary)
        expect_failure(lambda: accumulator.add(row))
        expect_failure(lambda: gain.Accumulator(base, 2).finish(base, tail))
        checks.append('five complete tiny steps retain both twins, clocks, residuals and actual next offsets')

        # Analytic expansion: h=2x,F=3x. Relative net/read gains vanish and boundary=log3.
        a = torch.tensor([[1., 2., 3., 4., 5., 6., 7., 8.]]).to(torch.float64).repeat(4, 1)
        x = gain.initial_twins(a, directions)
        zz = torch.tensor([[1., 1., 0.]], dtype=torch.float64).repeat(12, 1)
        pp = torch.softmax(zz, -1)
        analytic, _ = gain.finite_step(x, 2*x, 3*x, zz, pp, torch.ones(12, dtype=torch.float64))
        np.testing.assert_allclose(analytic['boundary_step'], np.log(3), rtol=0, atol=1e-15)
        np.testing.assert_allclose(analytic['log_growth'], analytic['log_g_net'] + analytic['log_g_read'] + np.log(3), atol=1e-12, rtol=0)
        assert not np.allclose(analytic['log_growth'], analytic['log_g_net'] + analytic['log_g_read'] - np.log(3))
        checks.append('analytic positive boundary sign distinguishes reversed correction')

        expect_failure(lambda: gain.initial_twins(a*0, directions), gain.UndefinedGain)
        coalesced = torch.cat([a, a, a])
        expect_failure(lambda: gain.finite_step(coalesced, 2*coalesced, 3*coalesced, zz, pp,
                                               torch.ones(12, dtype=torch.float64)), gain.UndefinedGain)
        zero_h = 2*x
        zero_h[4:8] = zero_h[:4]
        expect_failure(lambda: gain.finite_step(x, zero_h, 3*x, zz, pp,
                                               torch.ones(12, dtype=torch.float64)), gain.UndefinedGain)
        expect_failure(lambda: gain.finite_step(x, 2*x, 0*x, zz, pp,
                                               torch.ones(12, dtype=torch.float64)), gain.UndefinedGain)
        bad_h = (2*x).clone()
        bad_h[0, 0] = float('nan')
        expect_failure(lambda: gain.finite_step(x, bad_h, 3*x, zz, pp,
                                               torch.ones(12, dtype=torch.float64)), gain.UndefinedGain)
        checks.append('zero/nonfinite base or twin norms are explicit undefined refusals')

        tie_z = torch.tensor([[4., 4., -1000.]], dtype=torch.float64)
        tie_t = torch.tensor([2.], dtype=torch.float64)
        obs = gain.readout_observables(tie_z, torch.softmax(tie_z/tie_t[:, None], -1), tie_t)
        np.testing.assert_allclose(obs['entropy'], np.log(2), rtol=0, atol=1e-15)
        assert obs['gap'].tolist() == [0.] and set(obs['top2_ids'][0]) == {0, 1}
        gap_z = torch.tensor([[6., 2., -1000.]], dtype=torch.float64)
        obs2 = gain.readout_observables(gap_z, torch.softmax(gap_z/tie_t[:, None], -1), tie_t)
        assert obs2['gap'].tolist() == [2.]
        checks.append('full entropy permits underflow zeros and top-two gap divides by temperature once')

        periodic = np.asarray([[[1., 2.]], [[3., 4.]]] * 33)
        diagnostic = gain.period_diagnostic(periodic)[0]
        assert diagnostic['period'] == 2 and len(diagnostic['lag_distances']) == 64
        assert gain.period_diagnostic(periodic[-1:])[0]['period'] == 0
        checks.append('literal last-state period clock and short-profile horizon remain explicit')
    evidence.update({'step_records': rows, 'tiny_summary': summary,
                     'analytic_boundary_step': analytic['boundary_step'].tolist()})
    with (out / 'fixture-evidence.json').open('x') as stream:
        json.dump(evidence, stream, allow_nan=False, indent=2)
        stream.write('\n')
    report = {'schema': 'task23-manufactured-controls-v1', 'computational_status': 'complete',
              'all_passed': True, 'passed': len(checks), 'failed': 0, 'checks': checks,
              'seconds_before_final_report': time.perf_counter()-start,
              'observed_peak_rss_kib_before_final_report': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'target_assets_loaded': False, 'target_outcomes_read': False,
              'scope': 'Tiny tensor fixture only; does not measure target profile cost.'}
    with (out / 'report.json').open('x') as stream:
        json.dump(report, stream, allow_nan=False, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=False, exist_ok=False)
    try:
        perform(out)
    except Exception as exc:
        with (out / 'failure.json').open('x') as stream:
            json.dump({'type': type(exc).__name__, 'error': str(exc)}, stream, allow_nan=False)
        raise


if __name__ == '__main__':
    main()
