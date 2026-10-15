"""Focused synthetic controls for Task21a; no original input access."""
from decimal import Decimal as D
from fractions import Fraction as F
import itertools
import math
import unittest
import arithmetic as A


def level(p, x, bracket=None, kind='superstable'):
    return dict(attempt='synthetic', period=p, parameter=D(x), clean_bracket=bracket,
                kind=kind, disposition='superstable_point_estimate' if kind == 'superstable' else 'accepted_flip')


class Controls(unittest.TestCase):
    def test_exact_scalar_and_outward(self):
        self.assertEqual(A.exact(D('0.10000000000000001')), F(10000000000000001, 10**17))
        self.assertEqual(A.outward(F(-1, 3), 4, False), '-0.3334')
        self.assertEqual(A.outward(F(-1, 3), 4, True), '-0.3333')
        self.assertEqual(A.outward(F(1, 3), 4, False), '0.3333')
        self.assertEqual(A.outward(F(1, 3), 4, True), '0.3334')

    def test_ratio_reproduction_and_rounding_difference(self):
        rec = A.ratio_record([D('.1'), D('.2'), D('.22')], D('5'))
        self.assertEqual(rec['exact_decimal_input_ratio']['decimal_60'], '5')
        self.assertNotEqual(rec['saved_minus_independent_binary64'], 0)
        with self.assertRaises(ValueError): A.ratio(1, 1, 2)

    def test_coupled_envelope_all_corners(self):
        boxes = [(F(0), F(1)), (F(3), F(4)), (F(6), F(8))]
        values = [A.ratio(*v) for v in itertools.product(*boxes)]
        out = A.coupled_ratio_envelope(boxes)['envelope']
        get = lambda r:F(int(r['numerator']), int(r['denominator']))
        self.assertEqual(get(out['exact_lower']), min(values))
        self.assertEqual(get(out['exact_upper']), max(values))
        self.assertIsNone(A.coupled_ratio_envelope([(0, 3), (2, 4), (6, 8)])['envelope'])

    def test_level_gap_and_failure_not_skipped(self):
        levels = [level(6,'0'), level(24,'1'), level(48,'2')]
        self.assertIn('nonconsecutive', A.ratio_sequence(levels)[0]['disposition'])
        levels = [level(6,'0'), level(12,'1'), level(24,'2')]
        levels[1].update(disposition='refused', parameter=None)
        self.assertIn('refused', A.ratio_sequence(levels)[0]['disposition'])

    def test_pf1_failure_and_missing_are_distinct(self):
        self.assertEqual(A.pf1(D('14.9'), None)['conjunctive_prediction'], 'fails')
        self.assertIsNone(A.pf1(D('14.9'), None)['ratio_lower_at_minus_156'])
        self.assertEqual(A.pf1(D('2'), None)['conjunctive_prediction'], 'cannot check')
        self.assertEqual(A.pf1(D('2'), D('2.2'))['conjunctive_prediction'], 'passes')

    def test_pf2_two_sided_and_no_flip_substitution(self):
        # Each of the two ratios is exactly5: above4.90245, despite exceeding4.436.
        levels = [level(6,'0'), level(12,'25'), level(24,'30'), level(48,'31')]
        self.assertFalse(A.pf2(levels,'synthetic cutoff',48)['point_estimate_pass'])
        levels[0]['kind']='flip'
        with self.assertRaisesRegex(ValueError, 'never flip'):
            A.pf2(levels,'synthetic cutoff',48)

    def test_pf2_point_vs_bracket_and_cutoff(self):
        levels = [level(6,'0',('-2','2')), level(12,'21.16',('20','22')),
                  level(24,'25.76',('25','26')), level(48,'26.76',('26.5','27'))]
        result=A.pf2(levels,'synthetic cutoff',48)
        self.assertTrue(result['point_estimate_pass'])
        self.assertFalse(result['entire_numerical_bracket_envelope_pass'])
        with self.assertRaisesRegex(ValueError, 'latest completed'):
            A.pf2(levels,'synthetic cutoff',96)

    def test_secant_orientation_and_refusal(self):
        self.assertEqual(A.secant('1','3','2','-2'), 2)
        self.assertEqual(A.secant('3','1','-2','2'), 2)
        with self.assertRaises(ValueError): A.secant(1,3,2,2)

    def test_centered_exact_line_and_scatter(self):
        rows=[{'lambda':D('0.0650000000000000')+D(k)*D('1e-15'),'g':D(k-2)*D('2e-6')}for k in range(5)]
        fit=A.ols_root(rows)
        self.assertEqual(fit['root']['decimal_60'], '0.065000000000002')
        self.assertEqual(fit['maximum_absolute_line_residual']['numerator'], '0')
        self.assertEqual(D(fit['delta_method_root_standard_error']),0)
        rows[1]['g']+=D('1e-7')
        fit=A.ols_root(rows)
        self.assertGreater(D(fit['delta_method_root_standard_error']),0)
        self.assertEqual(len(fit['residuals']),5)
        with self.assertRaises(ValueError): A.ols_root([{'lambda':0,'g':k}for k in range(3)])

    def test_saved_multiplier_role_and_original_threshold(self):
        rec={'lambda':D('.1'),'critical_multiplier':[D('-1.000002'),D('0'),D('1.000002'),D('.5')],
             'multipliers':[[D('-1.000002'),D('0')],[D('.2'),D('0')]],'next_modulus':D('.2'),
             'newton_residuals':[D('1e-5'),D('1e-13')],'ritz_residual':D('1e-12')}
        strict=A.check_evaluation(rec,12,D('1e-12'),D('1e-6'),True)
        loose=A.check_evaluation(rec,12,D('1e-12'),D('3e-4'),True)
        self.assertFalse(strict['best_evaluation_g_gate'])
        self.assertTrue(loose['saved_flip_gates_pass'])
        self.assertEqual(strict['critical_index_from_saved_pairs'],0)
        self.assertIn('not saved',strict['half_period_distance'])
        rec['newton_residuals'][-1]=D('1e-7')
        self.assertFalse(A.check_evaluation(rec,12,D('1e-12'),D('3e-4'),True)['saved_flip_gates_pass'])


if __name__ == '__main__':
    unittest.main()
