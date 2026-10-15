"""Manufactured saved-field controls only; no actual inputs or model calls."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import resource
import time


def load_core():
    spec = importlib.util.spec_from_file_location('task23_compare_controls', Path(__file__).with_name('compare.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(core):
    args = {'transient': 1000, 'record': 2000, 'eps': [1e-6, 1e-9], 'layer': 12,
            'alpha': [.0655, .0655, 1.], 'beta': [0., 0., 1], 'tgrid': [.1, .3, 201.]}
    source = {'args': args, 'cells': [{'alpha': .0655, 'beta': 0., 'log10_temperature': .1+i*.001} for i in range(201)]}
    meta = {'args': copy.deepcopy(args), 'cell_count': 201, 'selected': []}
    own = {'schema': 'task23-finite-twin-result-v1', 'computational_status': 'complete', 'stage': 'actual',
           'scientific_full_run_complete': True, 'transient_steps': 1000, 'measurement_steps': 2000,
           'cell_summaries': []}
    own_meta = {'stage': 'actual', 'source_indices': core.INDICES, 'log10_temperatures': core.LOG_T,
                'alpha': .0655, 'eps': core.EPS, 'tensor_dtypes': ['torch.float64'],
                'tensor_devices': ['cpu'], 'transient_batch': 4, 'measured_batch': 12}
    for i, (index, log_t) in enumerate(zip(core.INDICES, core.LOG_T)):
        net, read, boundary = [-.2-i*.01, -.3-i*.01], [.4+i*.01, .5+i*.01], .003
        lam = [net[k]+read[k]+boundary for k in range(2)]
        residual = [lam[k]-net[k]-read[k]-boundary for k in range(2)]
        row = {'source_index': index, 'alpha': .0655, 'beta': 0., 'log10_temperature': log_t,
               'eps': core.EPS, 'measurement_steps': 2000, 'numerically_defined': True,
               'lyapunov': lam, 'mean_log_g_net': net[0], 'mean_log_g_read': read[0],
               'boundary_term': boundary, 'both_twin_mean_log_g_net': net, 'both_twin_mean_log_g_read': read,
               'identity_residual': residual, 'identity_abs_tolerance': 1e-10,
               'mean_entropy': 3.+i, 'mean_gap': 2.-i*.1, 'period': 0, 'literal_chaotic': True}
        own['cell_summaries'].append(row)
        producer = {key: copy.deepcopy(row[key]) for key in ['alpha','beta','log10_temperature','eps','lyapunov',
                    'mean_log_g_net','mean_log_g_read','boundary_term','mean_entropy','mean_gap','period']}
        producer['chaotic'] = True
        producer['lyapunov'][0] += .03
        producer['mean_entropy'] -= .5
        source['cells'][index] = producer
        requested = [.125,.129,.119,.135][i]
        meta['selected'].append({'source_index': index, 'saved_log10_temperature': log_t,
                                'saved_alpha': .0655, 'saved_beta': 0.,
                                'requested_log10_temperature': requested,
                                'exact_requested_float_match': requested == log_t})
    return own, own_meta, source, meta


def expect_refusal(function):
    try:
        function()
    except (ValueError, KeyError, FileExistsError):
        return
    raise AssertionError('expected refusal did not occur')


def perform(out):
    start = time.perf_counter()
    core = load_core()
    data = fixture(core)
    result = core.selected_comparison(*data)
    assert len(result['rows']) == 4 and sum(len(row['comparisons']) for row in result['rows']) == 28
    assert all(row['comparisons'][0]['signed_own_minus_producer'] < 0 for row in result['rows'])
    assert all(row['comparisons'][5]['signed_own_minus_producer'] == .5 for row in result['rows'])
    assert all(row['comparisons'][0]['agreement_gate'] is None for row in result['rows'])
    assert result['numeric_disagreement_is_a_failure'] is False
    checks = ['all28 scalar values/differences retained without a producer-agreement gate']

    assert result['rows'][2]['exact_requested_float_match'] is False
    assert result['rows'][2]['saved_log10_temperature'] == .11900000000000001
    assert all(v['within_own_prospective_floating_tolerance'] and v['saved_residual_exactly_matches']
               for row in result['rows'] for v in row['own_identity'])
    assert all(row['producer_first_twin_identity']['residual_lambda_minus_net_minus_read_minus_boundary'] > .02 for row in result['rows'])
    checks.append('one-ULP coordinate and positive own boundary differ from producer descriptive residual')

    for mutate in [lambda x: x[0].update(stage='profile'),
                   lambda x: x[0].update(measurement_steps=16),
                   lambda x: x[0]['cell_summaries'].reverse(),
                   lambda x: x[0]['cell_summaries'].pop(),
                   lambda x: x[0]['cell_summaries'][2].update(log10_temperature=.119),
                   lambda x: x[2]['cells'].pop(),
                   lambda x: x[2]['cells'].__setitem__(1, copy.deepcopy(x[2]['cells'][0])),
                   lambda x: x[2]['cells'][25].pop('mean_gap')]:
        broken = copy.deepcopy(data)
        mutate(broken)
        expect_refusal(lambda: core.selected_comparison(*broken))
    checks.append('profile/wrong horizon/order/missing/duplicate/one-ULP substitution refuses')

    undefined = copy.deepcopy(data)
    undefined[2]['cells'][25]['mean_gap'] = float('nan')
    undefined[2]['cells'][25]['chaotic'] = False
    got = core.selected_comparison(*undefined)
    gap = got['rows'][0]['comparisons'][6]
    assert gap['both_finite'] is False and gap['signed_own_minus_producer'] is None
    assert got['rows'][0]['producer_label']['stored_label_matches_literal'] is False
    assert got['rows'][0]['unavailable_comparisons']
    core.save(out / 'nonfinite-fixture.json', got)
    checks.append('nonfinite producer scalar and label mismatch retained as explicit unqualified evidence')

    bad_split = copy.deepcopy(data)
    bad_split[0]['cell_summaries'][0]['both_twin_mean_log_g_net'][0] += .1
    expect_refusal(lambda: core.selected_comparison(*bad_split))
    bad_type = copy.deepcopy(data)
    bad_type[2]['cells'][25]['mean_gap'] = '2.0'
    expect_refusal(lambda: core.selected_comparison(*bad_type))
    checks.append('wrong first-twin field association and nonnumeric scalar refuse')

    test_file = out / 'pinned-fixture.txt'
    test_file.write_text('immutable fixture\n')
    pin = {'path': str(test_file), 'bytes': test_file.stat().st_size, 'sha256': core.sha(test_file)}
    core.authenticate(pin)
    expect_refusal(lambda: core.authenticate({**pin, 'bytes': pin['bytes']+1}))
    expect_refusal(lambda: core.authenticate({**pin, 'sha256': '0'*64}))
    checks.append('exact bytes/hash authentication refuses mismatch')

    existing = out / 'existing'
    existing.mkdir()
    sentinel = existing / 'sentinel.txt'
    sentinel.write_text('keep\n')
    card = out / 'manufactured-card.json'
    card.write_text(json.dumps({'schema': 'task23-comparison-card-v1', 'status': 'reviewed-for-execution',
                               'comparator_sha256': core.sha(core.__file__)}))
    expect_refusal(lambda: core.dispatch(card, existing))
    assert [p.name for p in existing.iterdir()] == ['sentinel.txt'] and sentinel.read_text() == 'keep\n'
    checks.append('exclusive output refusal leaves existing directory unchanged')
    core.save(out / 'fixture-result.json', result)
    core.save(out / 'report.json', {'schema': 'task23-comparison-controls-v1', 'computational_status': 'complete',
                                   'all_passed': True, 'passed': len(checks), 'failed': 0, 'checks': checks,
                                   'seconds_before_final_report': time.perf_counter()-start,
                                   'observed_peak_rss_kib_before_final_report': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                   'serialized_bytes_before_final_report': sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
                                   'scope': 'manufactured JSON/scalars only; no original target or model access'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=False, exist_ok=False)
    try:
        perform(out)
    except Exception as exc:
        (out / 'failure.json').write_text(json.dumps({'type': type(exc).__name__, 'error': str(exc)})+'\n')
        raise


if __name__ == '__main__':
    main()
