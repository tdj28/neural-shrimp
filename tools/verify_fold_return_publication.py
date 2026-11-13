"""Verify the new return-study prose/figures against published compact receipts.

This dependency-free check is also installed in the blog bundle and run by its
normal provenance check. It verifies the added findings, not all older results.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def verify(blog):
    r = read(blog / 'fold-return-mechanism.json')
    provenance = read(blog / 'provenance.json')
    extension = next(x for x in provenance['extensions'] if x['id'] == 'fold-global-return-20261113')
    for name, digest in extension['receipts'].items():
        assert sha(blog / name) == digest, name
    expected = {
        'line_starts': sum(r['line_scan']['counts'].values()),
        'horizon': 4096,  # Fixed design; independently checked in the figure generator.
        'captured': r['line_scan']['counts']['reference_p3'],
        'unresolved': r['line_scan']['counts']['unresolved_at_horizon'],
        'returned_at_least_once': r['line_scan']['returned_at_least_once'],
        'returned_then_captured': r['line_scan']['returned_then_P3'],
        'heldout_routes': r['scalar_reconstruction']['test_pairs'],
        'heldout_returning_pairs': r['complete_return_cycle']['heldout_returning_pairs'],
        'heldout_P3_pairs': r['scalar_reconstruction']['original_P3_test_pairs'],
        'max_entry_error_U': r['scalar_reconstruction']['max_return_v_over_U_error'],
        'max_next_departure_error_U': r['complete_return_cycle']['max_next_departure_v_over_U_error'],
        'max_cycle_time_error': r['complete_return_cycle']['max_complete_cycle_time_error'],
        'cycle_duration_range': r['complete_return_cycle']['complete_cycle_updates_range'],
        'max_initial_relative_reconstruction_error': r['scalar_reconstruction']['max_test_initial_relative_state_error'],
        'constant_entry_rmse_U': r['simple_prediction_comparisons']['models']['constant']['equal_row_RMSE'][0],
        'interpolation_entry_rmse_U': r['simple_prediction_comparisons']['models']['v_linear_interpolation']['equal_row_RMSE'][0],
    }
    assert expected['max_next_departure_error_U'] < 0.0014
    assert expected['max_entry_error_U'] < 4.6e-5
    assert expected['max_initial_relative_reconstruction_error'] < 6.4e-14
    assert expected['max_cycle_time_error'] == 0
    assert f"{expected['interpolation_entry_rmse_U']:.4f}" == '0.2142'
    assert f"{expected['constant_entry_rmse_U']:.4f}" == '0.2645'
    assert r['simple_prediction_comparisons']['return_test_exits'] == 777
    prose = (blog / 'index.md').read_text()
    for stem in ('fig-fold-return-map', 'fig-fold-return-fates'):
        figure = read(blog / (stem + '.receipt.json'))
        assert figure['claims'] == expected, stem
        for name, record in figure['outputs'].items():
            if name.endswith('.png'):
                assert sha(blog / name) == record['sha256'], name
                assert (blog / name).stat().st_size == record['bytes'], name
        assert figure['alt_text'] in prose, stem
    for number in extension['new_numbers']:
        current = read(blog / number['receipt'])
        for key in number['field']:
            current = current[key]
        assert current == number['value'], number['id']
        assert number['appears_as'] in prose, number['id']
    for required in ('both through the original network', 'not an autonomous scalar simulator',
                     'do not alone demonstrate', '777 returned test exits'):
        assert required in prose, required
    print('FGR2 publication: receipt-derived claims, bounds, figure hashes, alt text and limits verified.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--blog', type=Path, default=Path(__file__).resolve().parent)
    verify(p.parse_args().blog)
