"""Manufactured-data controls only; never reads original scalar records."""
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
import unittest

spec=importlib.util.spec_from_file_location('audit','/tmp/task21c-analyze.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def fixture(period=2, sign=1):
    cs=[6.,4.,3.,2.]; cs=[c*sign for c in cs]; hs=[.01,.003,.001,.0003]
    k=1; c=(cs[k]+cs[k+1])/2
    offsets=[2.5e-5,5e-5,1e-4,2e-4,4e-4]
    af=.075; side=-1 if c>0 else 1
    r={'args':{'period':period,'side':-1,'alpha_flip':af,'log_temperature':.0319,'tol':1e-12,'offsets':offsets},
       'normal_form':{'c_steps':[{'h_relative':h,'c':v,'c_direct':v*.25,'c_centre_manifold':v*.75,'h2_norm':2.} for h,v in zip(hs,cs)],
                      'c':c,'c_plateau_pair':hs[k:k+2],'c_plateau_relative_spread':abs(cs[k+1]-cs[k])/abs(c),
                      'crossing':{'multiplier':[-1.,0.],'modulus':1.,'argument_turns':.5,'next_modulus':.2}},
       'kind':'supercritical' if c>0 else 'subcritical','offset_side':side,'flip_lowrank_critical':[-1.,0.],'offsets':[],'seconds':1.}
    for s in offsets:
        mp=-1-sign*s*10; beta=-mp-1; eta=math.sqrt(beta/c); pred=1-4*beta
        ratio=1+s*10
        r['offsets'].append({'offset':s,'alpha':af+side*s,'mu_p':[mp,0.],'beta':beta,'next_modulus_p':.2,
                            'eta_predicted':eta,'eta_measured':eta*ratio,'ratio':ratio,'centre_offset':s*s,
                            'mu_2p':[pred+4*beta*beta,0.],'mu_2p_predicted':pred,'next_modulus_2p':.3,'newton_2p':1e-14})
    return r


def expected(p=2): return {'id':'FNF2b' if p==2 else 'FNF4','period':p,'c_step_count':4,'offset_count':5}


class Controls(unittest.TestCase):
    def test_sum_selection_first_absolute_tie(self):
        r=m.reconstruct(fixture(),expected())
        self.assertTrue(r['schema_complete'])
        self.assertEqual(r['normal_form']['selected_pair_index'],1)
        self.assertEqual(r['normal_form']['selected_pair'],[.003,.001])
        self.assertEqual(r['normal_form']['c']['absolute_difference'],0.)
    def test_negative_c_and_offset_side(self):
        r=m.reconstruct(fixture(sign=-1),expected())
        self.assertTrue(r['schema_complete']); self.assertEqual(r['normal_form']['offset_side'],1)
        self.assertTrue(all(x['amplitude_prediction_domain'] for x in r['offset_rows']))
    def test_complex_discrepancy_and_beta_squared(self):
        r=fixture(); r['offsets'][-1]['mu_2p'][1]=.207
        a=m.reconstruct(r,expected())['offset_rows'][-1]
        self.assertGreater(a['mu_comparison']['complex_absolute_difference'],.207)
        self.assertEqual(a['mu_comparison']['signed_imaginary_difference'],.207)
        self.assertAlmostEqual(a['mu_error_over_beta_squared']['signed_real'],4.,places=9)
        self.assertGreater(a['mu_error_over_beta_squared']['complex_absolute'],a['mu_error_over_beta_squared']['absolute_real'])
    def test_finite_trend_not_asymptotic_pass(self):
        f=m.reconstruct(fixture(),expected())['finite_confirmation_summary']
        self.assertEqual(f['adjacent_errors_nondecreasing_as_offset_increases'],[True]*4)
        self.assertTrue(f['smallest_offset_within_10_percent']); self.assertFalse(f['asymptotic_big_O_verified'])
    def test_notes_errors_retained(self):
        r=fixture(); r['offsets'][0]={'offset':r['offsets'][0]['offset'],'alpha':r['offsets'][0]['alpha'],'error':'original refusal'}
        r['offsets'][1].update(mu_p=[-.9,0.],beta=-.1,note='no amplitude')
        a=m.reconstruct(r,expected()); self.assertTrue(a['schema_complete'])
        self.assertEqual(len(a['offset_rows']),5); self.assertEqual(a['offset_rows'][0]['disposition'],'producer_error')
        self.assertIsNone(a['finite_confirmation_summary']['smallest_offset_within_10_percent'])
    def test_missing_and_nonfinite_preserved(self):
        r=fixture(); del r['offsets'][0]['newton_2p']; r['normal_form']['c_steps'][2]['c_direct']=float('nan')
        a=m.reconstruct(r,expected()); self.assertFalse(a['schema_complete'])
        self.assertEqual(len(a['offset_rows']),5); self.assertEqual(len(a['coefficient_rows']),4)
        encoded=json.dumps(m.json_safe(a),allow_nan=False)
        self.assertIn('nonfinite_original',encoded)
    def test_failed_crossing_gate_is_not_schema_failure(self):
        r=fixture(); r['normal_form']['crossing']['next_modulus']=.95
        a=m.reconstruct(r,expected()); self.assertTrue(a['schema_complete'])
        self.assertFalse(a['normal_form']['original_crossing_gates']['other_moduli_below_0.95'])
    def test_newton_strict_and_parent_complex_guard(self):
        r=fixture(); r['offsets'][0]['newton_2p']=1e-12; r['offsets'][0]['mu_p'][1]=1e-7
        a=m.reconstruct(r,expected())['offset_rows'][0]
        self.assertFalse(a['source_newton_2p_gate']['strictly_below'])
        self.assertFalse(a['parent_source_complex_guard_accepts_saved_mu'])
    def test_count_and_order_dispositions(self):
        r=fixture(); r['offsets']=r['offsets'][::-1]
        a=m.reconstruct(r,expected()); self.assertFalse(a['offset_rows'][0]['offset_order_matches_args'])
        r['offsets'].pop(); self.assertFalse(m.reconstruct(r,expected())['schema_complete'])
    def test_duplicate_keys_and_prepared_cli_refusal(self):
        with self.assertRaises(ValueError): m.strict_json('{"a":1,"a":2}')
        with tempfile.TemporaryDirectory(prefix='task21c-prepared-control-') as d:
            p=Path(d); (p/'card.json').write_text(json.dumps({'schema':m.SCHEMA,'admission':'prepared','limits':m.LIMITS}))
            cp=subprocess.run([sys.executable,'-I','/tmp/task21c-analyze.py','--card',str(p/'card.json'),'--out',str(p/'out')],capture_output=True,text=True,timeout=3)
            self.assertNotEqual(cp.returncode,0); self.assertIn('refusing',cp.stderr); self.assertFalse((p/'out').exists())


def serialize_measurement():
    tic=time.perf_counter()
    result={'records':[m.reconstruct(fixture(p),expected(p)) for p in (2,4)],'synthetic_only':True}
    payload=json.dumps(m.json_safe(result),indent=2,allow_nan=False).encode()+b'\n'
    p=Path('/tmp/task21c-synthetic-output-001.json')
    with p.open('xb') as f: f.write(payload)
    return {'synthetic_only':True,'full_two_records':2,'coefficient_rows':8,'offset_rows':10,
            'through_serialization_write_seconds':time.perf_counter()-tic,
            'observed_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'output_bytes':len(payload),
            'output_sha256':hashlib.sha256(payload).hexdigest(),'source_sha256':m.digest('/tmp/task21c-analyze.py')}


if __name__=='__main__':
    if os.getpriority(os.PRIO_PROCESS,0)<10: os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3)); resource.setrlimit(resource.RLIMIT_CPU,(5,5))
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
    measurement=serialize_measurement() if result.wasSuccessful() else {'synthetic_only':True,'measurement_not_run':True}
    measurement.update(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors))
    with open('/tmp/task21c-controls-001.json','x') as f: json.dump(measurement,f,indent=2); f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
