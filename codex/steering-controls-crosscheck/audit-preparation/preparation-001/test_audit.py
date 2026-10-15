"""Tiny controls for the new independent checker; synthetic values only."""
import json
from pathlib import Path
import tempfile
import unittest
import audit as A


def row(l, q, complete=True, alpha=0., i=0, j=0):
    return dict(row=i, column=j, alpha=alpha, log10_temperature=float(i),
                **A.classification(l, q, 8, complete))


def synthetic():
    # Uneven completed prefix; later tile selects row2 and preserves the earlier
    # completed overlap, with an intentionally unequal physical coordinate.
    t, a = [0., 1., 2.], [-.01, .01]
    segments = [dict(path='lower', shape=[3, 2], layout='partial', expected_done=5,
        row_offset=0, selected_local_flat_range=[0, 4], temperature_hex=[x.hex() for x in t],
        alpha_hex=[x.hex() for x in a], sha256='a' * 64, original_attempt_outcome={'status': 124}, outcome_authority={'role': 'lower'}),
        dict(path='top', shape=[1, 2], layout='full', expected_done=2, row_offset=2,
        selected_local_flat_range=[0, 2], temperature_hex=[(2.0000000000000004).hex()], alpha_hex=[x.hex() for x in a],
        sha256='b' * 64, original_attempt_outcome={'status': 0}, outcome_authority={'role': 'top'})]
    spec = dict(id='tiny', shape=[3, 2], max_period=8, axis_segment=0,
                ordinal_amendment='c' * 64, metadata={}, segments=segments)
    arrays = [(t, a, [.002, .001, float('nan'), .2, .4, 0.], [0, 0, 2, 3, 0, 0], 5),
              ([2.0000000000000004], a, [.5, -.1], [0, 4], 2)]
    return spec, arrays


class Controls(unittest.TestCase):
    def test_strict_threshold_and_positive_period(self):
        self.assertEqual(A.classification(.001, 0, 8, True)['disposition'], 'no_period_not_above_threshold')
        self.assertEqual(A.classification(.0010000000000000002, 0, 8, True)['disposition'], 'chaotic')
        self.assertEqual(A.classification(.2, 3, 8, True)['disposition'], 'detected_period')

    def test_invalid_fields_and_unfinished(self):
        for l, q, issue in [(float('nan'), 0, 'exponent_NaN'), (float('inf'), 1, 'exponent_+Inf'),
                            (-float('inf'), 1, 'exponent_-Inf'), (0., float('nan'), 'period_NaN'),
                            (0., 1.5, 'period_invalid_integer_or_range'), (0., 9, 'period_invalid_integer_or_range')]:
            self.assertIn(issue, A.classification(l, q, 8, True)['issues'])
        unfinished = A.classification(float('nan'), -1, 8, False)
        self.assertFalse(unfinished['completed'])
        self.assertIsNone(unfinished['lyapunov'])
        self.assertEqual(unfinished['issues'], [])

    def test_summary_marginal_and_median(self):
        rows = [row(.2, 0), row(.001, 0), row(.2, 2), row(-.2, 3), row(float('nan'), 4), row(0, 0, False)]
        out = A.aggregate(rows)
        self.assertEqual((out['expected'], out['completed'], out['valid'], out['chaotic']), (6, 5, 4, 1))
        self.assertEqual(out['median_detected_period'], 2.5)
        self.assertEqual(out['period_histogram_including_zero'], {'0': 2, '2': 1, '3': 1})
        self.assertEqual(out['period_histogram_all_completed_valid_period_fields'], {'0': 2, '2': 1, '3': 1, '4': 1})
        self.assertEqual(out['detected_period_above_exponent_threshold'], 1)
        self.assertIsNone(A.aggregate([])['chaotic_share_valid'])

    def test_bins_exact_zero_and_signed_nearest(self):
        alphas = [-.13, -.12, -.01, -.001, .001, .01, .12, .13]
        rows = [row(.2, 0, alpha=a, j=j) for j, a in enumerate(alphas)]
        out = A.expected_subsets(rows, [0.], alphas)
        self.assertEqual([b['all']['expected'] for b in out['absolute_alpha_bins']], [2, 2, 0, 0, 2])
        self.assertEqual(out['outside_bins']['expected'], 2)
        self.assertEqual(out['zero_and_signed_nearest']['exact_zero']['column_indices'], [])
        self.assertEqual(out['zero_and_signed_nearest']['nearest_negative']['alpha'], [-.001])
        self.assertEqual(out['zero_and_signed_nearest']['nearest_positive']['alpha'], [.001])

    def test_prefix_and_overlaps(self):
        spec, arrays = synthetic()
        cells, sources, summary = A.independent_records(spec, arrays)
        self.assertEqual(len(cells), 6)
        self.assertEqual(len(sources), 8)
        self.assertEqual(sources[5]['disposition'], 'unfinished')
        self.assertIsNone(sources[5]['period'])
        self.assertEqual(cells[4]['source_segment'], 1)
        self.assertEqual(cells[4]['allocated_segments'], [0, 1])
        self.assertEqual(len(summary['nonselected_completed_observations']), 1)
        overlap = summary['nonselected_completed_observations'][0]
        self.assertEqual(overlap['exponent_selected_minus_nonselected'], .5 - .4)
        self.assertEqual(overlap['temperature_difference'], 2.0000000000000004 - 2.)
        self.assertEqual(summary['segments'][0]['original_attempt_outcome']['status'], 124)
        self.assertEqual(summary['all_source_prefix_summaries'][0]['completed'], 5)

    def test_unfinished_selected_partial_row(self):
        spec, arrays = synthetic()
        spec['segments'] = spec['segments'][:1]
        spec['segments'][0]['selected_local_flat_range'] = [0, 6]
        cells, _, report = A.independent_records(spec, arrays[:1])
        self.assertEqual(cells[4]['disposition'], 'chaotic')
        self.assertEqual(cells[5]['disposition'], 'unfinished')
        self.assertEqual(report['summary']['completed'], 5)

    def test_duplicate_selection_refuses(self):
        spec, arrays = synthetic()
        spec['segments'][0]['selected_local_flat_range'] = [0, 6]
        with self.assertRaisesRegex(ValueError, 'duplicate selected'):
            A.independent_records(spec, arrays)

    def test_coordinate_and_prefix_identity(self):
        spec, arrays = synthetic()
        spec['ordinal_amendment'] = None
        with self.assertRaisesRegex(ValueError, 'literal unamended'):
            A.independent_records(spec, arrays)
        spec, arrays = synthetic()
        spec['segments'][1]['temperature_hex'] = [2.0.hex()]
        with self.assertRaisesRegex(ValueError, 'frozen axes'):
            A.independent_records(spec, arrays)

    def test_every_nonzero_and_types_are_retained(self):
        checks = A.Checks()
        checks.same({'x': .1 + .2, 'flag': 1, 'extra': True}, {'x': .3, 'flag': True}, '/case')
        self.assertEqual(checks.failed, 3)
        self.assertEqual(checks.numeric_nonzero, 1)
        self.assertEqual(len(checks.examples), 3)
        self.assertGreater(checks.max_abs_difference, 0)

    def test_json_duplicate_and_nonfinite_refusal(self):
        for text in ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}']:
            with self.assertRaises(ValueError):
                A.decode(text)
        self.assertEqual(A.pointer({'a/b': [False]}, '/a~1b/0'), False)

    def test_jsonl_missing_extra_and_field_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'rows.jsonl'
            path.write_text('{"value":2}\n')
            checks = A.Checks()
            A.compare_lines(path, [{'value': 1}, {'value': 3}], checks, '/rows')
            self.assertEqual(checks.failed, 2)
            with self.assertRaisesRegex(ValueError, 'extra record'):
                A.compare_lines(path, [], A.Checks(), '/rows')

    def test_literal_common_grid_not_nearest(self):
        all_rows = {}
        for name in A.EXPECTED:
            all_rows[name] = [row(.2, 0, alpha=.01)]
        all_rows['q05-random3'][0]['log10_temperature'] = 1e-16
        groups = A.expected_groups(all_rows)
        self.assertEqual(groups[0]['common_valid_exact_coordinates'], 0)
        self.assertEqual(groups[1]['common_valid_exact_coordinates'], 1)
        self.assertEqual(groups[2]['common_valid_exact_coordinates'], 1)


if __name__ == '__main__':
    unittest.main()
