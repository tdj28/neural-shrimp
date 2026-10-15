import unittest
from decimal import Decimal as D
from fractions import Fraction as F
import kaplan_yorke as k


def val(record): return F(int(record['numerator']),int(record['denominator']))

class Controls(unittest.TestCase):
    def test_crossing_and_stable_original_order(self):
        r=k.estimate([D('-.3'),D('.1'),D('0')],leading_spectrum=True)
        self.assertEqual(r['sorted_to_original_index'],[1,2,0]);self.assertEqual(r['j'],2)
        self.assertEqual(val(r['dimension']),F(7,3))
        self.assertEqual(r['original_values'],['-0.3','0.1','0'])
    def test_all_negative(self):
        r=k.estimate(['-2','-1'],leading_spectrum=True)
        self.assertEqual(r['j'],0);self.assertEqual(val(r['dimension']),0)
    def test_zero_top_with_negative_tail(self):
        r=k.estimate(['0','0','-1'],leading_spectrum=True)
        self.assertEqual(r['j'],2);self.assertEqual(val(r['dimension']),2)
    def test_truncated_nonnegative_bound(self):
        r=k.estimate(['1','-.2'],leading_spectrum=True,ambient_dimension=8)
        self.assertIsNone(r['dimension']);self.assertEqual(val(r['conditional_lower_bound']),2)
    def test_zero_denominator_avoided_by_maximal_j(self):
        r=k.estimate(['0','0'],leading_spectrum=True)
        self.assertEqual(r['j'],2);self.assertIsNone(r['dimension']);self.assertIsNone(r['denominator'])
        full=k.estimate(['0','0'],leading_spectrum=False,complete_spectrum=True,ambient_dimension=2)
        self.assertEqual(val(full['dimension']),2)
    def test_no_tail_authority(self):
        r=k.estimate(['.1','-.2'],leading_spectrum=False)
        self.assertIsNone(r['dimension']);self.assertEqual(val(r['diagnostic_from_retained_estimates']),F(3,2))
    def test_invalid_retained(self):
        for x in ([],['NaN'],['Infinity'],[True]):
            self.assertEqual(k.estimate(x,leading_spectrum=True)['disposition'],'invalid saved spectrum')
        self.assertIsNone(k.estimate(['1'],leading_spectrum=True,complete_spectrum=True,ambient_dimension=2)['dimension'])
    def test_equal_values_stable_and_scale_invariance(self):
        a=k.estimate(['.1','.1','-.4'],leading_spectrum=True)
        b=k.estimate(['1','1','-4'],leading_spectrum=True)
        self.assertEqual(a['sorted_to_original_index'],[0,1,2]);self.assertEqual(val(a['dimension']),val(b['dimension']))
    def test_ranges_use_actual_extrema_and_outward_rounding(self):
        rows=[k.estimate(x,leading_spectrum=True) for x in (['.1','-.3'],['.2','-.3'],['1'])]
        r=k.dimension_range(rows,3)
        self.assertEqual(r['outward_display_min'],'1.333');self.assertEqual(r['outward_display_max'],'1.667')
        self.assertEqual(r['min_original_row_indices'],[0]);self.assertEqual(r['max_original_row_indices'],[1])
        self.assertEqual(r['point_estimate_count'],2);self.assertEqual(len(r['other_rows']),1)
    def test_exact_zero_cumulative_boundary(self):
        r=k.estimate(['.1','-.1','-.2'],leading_spectrum=True)
        self.assertEqual(r['j'],2);self.assertEqual(val(r['dimension']),2)

if __name__=='__main__':unittest.main()
