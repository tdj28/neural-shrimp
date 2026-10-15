"""Only the metadata-resolved prefix check; no target data."""
import copy
import importlib.util
import json
import os
import resource
import unittest
spec=importlib.util.spec_from_file_location('previous','/tmp/task21d-admission-controls.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)


class Prefix(unittest.TestCase):
    def test_literal_resolved_prefix_and_refusals(self):
        l,d,n,c,p=old.fixture()
        n['source'].update(prompt='open',opener='Hello!')
        for cell in l['cells']:cell.update(lyapunov=[.2],lead_period=1)
        r=old.h.admit_line(l,d,n,c,p)
        self.assertEqual(r['effective_prefix_checked'],{'prompt':'open','opener':'Hello!'})
        for key,value in [('prompt','custom'),('opener',None),('opener','Hi!')]:
            bad=copy.deepcopy(n);bad['source'][key]=value
            with self.assertRaisesRegex(ValueError,'prefix mismatch'):old.h.admit_line(l,d,bad,c,p)


if __name__=='__main__':
    if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3));resource.setrlimit(resource.RLIMIT_CPU,(5,5))
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Prefix))
    with open('/tmp/task21d-prefix-control-001.json','x') as f:json.dump({'synthetic_only':True,'tests':r.testsRun,'failures':len(r.failures),'errors':len(r.errors)},f);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
