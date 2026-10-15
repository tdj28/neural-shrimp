import unittest
from decimal import Decimal as D
import adapter as A

class Controls(unittest.TestCase):
    def test_native_vs_sorted_and_all_arithmetic_variants(self):
        s=A.spectrum([D('-.3'),D('.1'),D('0')])
        self.assertFalse(s['native_first_positive']);self.assertTrue(s['sorted_leading_positive'])
        self.assertEqual(s['positive_count_native'],1);self.assertEqual(s['literal_binary64']['j'],2)
        self.assertIsNotNone(s['literal_minus_exact_binary_dimension'])
        self.assertEqual(A.fraction(s['native_qr_order_ky_diagnostic']['candidate_dimension']),0)
        self.assertFalse(s['native_qr_order_ky_diagnostic']['agrees_with_required_descending_order'])
    def test_cell_complete_checkpoint_and_identity(self):
        r={'index':2,'k':2,'log10_temperature':D('.1'),'alpha':D('.2'),'exponents':[D('.1'),D('-.3')],
           'checkpoints':[{'steps':10,'exponents':[D('.2'),D('-.3')]},{'steps':20,'exponents':[D('.1'),D('-.3')]}], 'entropy_mean':D('1')}
        kwargs=dict(expected_index=2,expected_coordinates=['.1','.2'],k=2,expected_checkpoint_steps=[10,20],record_steps=20)
        c=A.cell(r,**kwargs);self.assertTrue(c['checkpoints'][1]['final_step_checkpoint_exactly_matches'])
        self.assertGreater(A.fraction(c['checkpoints'][0]['dimension_minus_final']),0)
        c['key']='synthetic';g=A.group([c])
        self.assertEqual(A.fraction(g['checkpoint_dimension_differences_by_steps']['20']['maximum_absolute_difference']),0)
        self.assertGreater(A.fraction(g['checkpoint_dimension_differences_by_steps']['10']['maximum_absolute_difference']),0)
        r['index']=3
        with self.assertRaises(ValueError):A.cell(r,**kwargs)
    def test_group_preserves_nonpositive_and_bounds(self):
        rows=[{'key':str(i),'final':A.spectrum(v),'checkpoints':[]} for i,v in enumerate((['.1','-.3'],['-.1','-.2'],['1','1']))]
        g=A.group(rows)
        self.assertEqual(g['total_completed_rows'],3);self.assertEqual(g['positive_leading_count'],2)
        self.assertEqual(g['positive_leading_dimension_range']['point_estimate_count'],1)
        self.assertEqual(len(g['positive_leading_dimension_range']['other_rows']),1)
        self.assertEqual(g['nonpositive_or_invalid_row_keys'],['1'])
    def test_invalid_spectrum_is_retained(self):
        r=A.spectrum(['NaN','-.2']);self.assertFalse(r['finite'])
        self.assertIsNone(r['saved_decimal']['dimension'])

if __name__=='__main__':unittest.main()
