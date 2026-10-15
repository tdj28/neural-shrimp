import unittest
from decimal import Decimal as D
from run import cascade,superstable,command_options,literal_ratios


def record(lam,critical=-1,leading=None):
    vals=[[critical,0],[.2,0]] if leading is None else [[leading,0],[critical,0]]
    return {'lambda':D(lam),'critical_multiplier':[critical,0,abs(critical),.5 if critical<0 else 0],
        'multipliers':vals,'next_modulus':.2 if leading is None else leading,
        'newton_residuals':[1e-4,1e-14],'ritz_residual':1e-10}

class AdapterTests(unittest.TestCase):
    def test_command_and_negative_value(self):
        self.assertEqual(command_options('case 90 --period 3 --log-temperature -0.156 --points 6:-.156:.1:.2 12:-.156:.3:.4'),
            {'period':'3','log_temperature':'-0.156','points':['6:-.156:.1:.2','12:-.156:.3:.4']})
    def test_cascade_lengths_periods(self):
        d={'args':{'period':3,'flips':2,'tol':D('1e-12')},'flips':[{'period':3,'flip':record('.1'),'evaluations':[]}],
            'flip_parameters':[D('.1')],'intervals':[],'ratios':[]}
        self.assertTrue(cascade('x',d,{'period':'3','flips':'2'})['levels'][0]['checks']['saved_flip_gates_pass'])
        d['flips'][0]['period']=6
        with self.assertRaisesRegex(ValueError,'period sequence'):cascade('x',d,{'period':'3','flips':'2'})
    def test_ratio_list_mismatch(self):
        with self.assertRaisesRegex(ValueError,'ratio-list length'):literal_ratios([D('.1'),D('.2'),D('.25')],[])
    def test_superstable_positive_endpoint_and_failed_band(self):
        band={'period':6,'log10_temperature':D('-.156'),'bracket':[D('.1'),D('.9')],'detected_period':6,
            'superstable':D('.5'),'clean_bracket':[D('.2'),D('.8')],'mu':[D('.02'),D('-.02')],
            'record':record('.2',-.001,.02),'evaluations':[{'lambda':D('.5'),'mu':D('0'),'clean':False,'next_modulus':D('.001')}]}
        fail={'period':12,'log10_temperature':D('-.156'),'bracket':[D('.91'),D('.99')],'detected_period':6,'error':'wrong period'}
        doc={'args':{'tol':D('1e-12')},'bands':[band,fail],'superstable':[D('.5')],'ratios':[]}
        out=superstable('x',doc,{'points':['6:-.156:.1:.9','12:-.156:.91:.99']})
        self.assertEqual(out['levels'][0]['disposition'],'superstable_point_estimate')
        self.assertTrue(out['original_success_only_ratio_array_skips_failed_bands'])
        self.assertFalse(out['band_checks'][0]['root_was_directly_evaluated'])
        self.assertIsNone(out['levels'][1]['parameter'])
        self.assertEqual(out['band_checks'][0]['positive_endpoint_record']['critical_real_difference'],0)
        self.assertEqual(out['band_checks'][0]['evaluation_checks'][0]['claimed_clean'],False)

if __name__=='__main__':unittest.main()
