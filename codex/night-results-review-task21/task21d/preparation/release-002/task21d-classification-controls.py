"""Only the requested new scalar diagnostic; manufactured data, no target input."""
import importlib.util
import json
import os
import resource
import unittest
spec=importlib.util.spec_from_file_location('h','/tmp/task21d-inputs.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


class Diagnostic(unittest.TestCase):
    def test_priority_and_preserved_mismatch(self):
        c={'period':4,'class':'hidden chaos','lyapunov':[.2,.3],'lead_period':7}
        d=h.classification_diagnostic(c,0)
        self.assertEqual(d['reconstructed_class'],'periodic');self.assertFalse(d['matches'])
        self.assertEqual(c['class'],'hidden chaos');self.assertTrue(d['saved_class_used_for_window_selection'])
    def test_exact_threshold_classes_and_range_are_diagnostics(self):
        cases=[(0,[.002,.003],1,'hidden chaos'),(0,[.002,.003],0,'visible chaos'),
               (0,[.001,.003],0,'irregular, exponent not positive at both offsets'),
               (0,[float('nan'),.003],1,'irregular, exponent not positive at both offsets'),
               (65,[-.1],0,'periodic')]
        for i,(p,ls,lead,cl) in enumerate(cases):
            d=h.classification_diagnostic({'period':p,'class':cl,'lyapunov':ls,'lead_period':lead},i)
            self.assertTrue(d['matches']);self.assertEqual(d['period_in_source_range_0_to_64'],p<=64)
            if i==3:self.assertFalse(d['lyapunov_all_finite'])


if __name__=='__main__':
    if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3));resource.setrlimit(resource.RLIMIT_CPU,(5,5))
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Diagnostic))
    with open('/tmp/task21d-classification-controls-001.json','x') as f:json.dump({'synthetic_only':True,'tests':r.testsRun,'failures':len(r.failures),'errors':len(r.errors)},f);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
