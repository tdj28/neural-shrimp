"""Small manufactured-only controls and a finite typical/stress serialization measure."""
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import time
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('own','/tmp/task21d-score.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def cell(x,p,cl='periodic'):
    return {'log10_temperature':x,'alpha':x,'period':p,'class':cl,'lyapunov':.2}


def windows(n):
    out=[]
    for p in range(n+2,2,-1):
        d=.04/(p+.4)**2
        out.append({'period':p,'laps':float(p),'edge':d,'d':d,'d_centre':d+1e-8,'cells':2})
    return out


class Controls(unittest.TestCase):
    def test_divisor_base_and_no_lyapunov_gate(self):
        cs=[cell(-2,2),cell(-1,2),cell(.1,7),cell(.2,7)]
        ws,side,l=m.collect(cs,0.,4,'log10_temperature')
        self.assertEqual(side,1.);self.assertEqual([w['period'] for w in ws],[7]);self.assertEqual(l['selected_base_run'],0)
    def test_singleton_adjacent_doubling_and_wrong_side(self):
        cs=[cell(-3,5),cell(-2.9,5),cell(-2,1),cell(-1,1),cell(.1,3),cell(.2,3),cell(.3,6),cell(.4,6),cell(.5,0,'irregular'),cell(.6,9)]
        ws,_,l=m.collect(cs,0.,1,'log10_temperature')
        self.assertEqual([w['period'] for w in ws],[3])
        reasons=[r['exclusion_reasons'] for r in l['run_dispositions']]
        self.assertTrue(any('single_cell_unresolved' in r for r in reasons))
        self.assertTrue(any('adjacent_half_period_run' in r for r in reasons))
        self.assertTrue(any('not_beyond_fold_by_run_centre' in r for r in reasons))
    def test_duplicate_coordinates_and_repeated_period(self):
        cs=[cell(-2,1),cell(-1,1),cell(.1,5),cell(.1,5),cell(.3,0,'chaotic'),cell(.4,5),cell(.5,5)]
        ws,_,l=m.collect(cs,0.,1,'log10_temperature')
        self.assertEqual(ws[0]['edge'],.1);self.assertEqual(l['duplicate_coordinates'][0]['input_indices'],[2,3])
        self.assertEqual(l['run_dispositions'][-1]['period_choice'],'farther_or_equal_repeated_period')
    def test_classification_priority_preserved(self):
        cs=[cell(-2,1),cell(-1,1),cell(.1,7,'chaotic'),cell(.2,7,'chaotic'),cell(.3,5),cell(.4,5)]
        ws,_,l=m.collect(cs,0.,1,'log10_temperature')
        self.assertEqual([w['period'] for w in ws],[5]);self.assertEqual(l['runs'][1]['period'],0)
    def test_original_and_amended_fans(self):
        w=[{'period':p,'d':d} for p,d in [(20,1.),(17,2.),(16,3.),(12,4.),(8,5.)]]
        self.assertEqual([x['period'] for x in m.fan_windows(w,4,'original_stop')],[20])
        self.assertEqual([x['period'] for x in m.fan_windows(w,4,'amended_skip')],[20,16,12,8])
    def test_first_grid_tie_and_literal_counts(self):
        a=m.grid_fit([{'laps':3.,'d':2.}]*4)
        self.assertEqual(a['grid_index'],0);self.assertEqual(a['delta'],-2.999)
        self.assertEqual(a['audit']['candidate_count'],22002);self.assertEqual(a['audit']['first_minimum_tie_count'],22002)
    def test_grid_scalar_oracle(self):
        w=windows(4);f=m.grid_fit(w);x=np.array([v['laps'] for v in w]);d=np.array([v['d'] for v in w])
        grid=np.r_[np.linspace(-min(x)+.001,10.,20001),np.linspace(-min(x)+.001,-min(x)+1.001,2001)]
        best=None
        for i,g in enumerate(grid):
            v=d*(x+g)**2;s=float((max(v)-min(v))/np.mean(v))
            if best is None or s<best[0]:best=(s,i,float(np.mean(v)),float(g))
        self.assertEqual(f['grid_index'],best[1]);self.assertEqual(f['spread'],best[0]);self.assertEqual(f['C'],best[2])
        self.assertNotEqual(m.grid_fit(w,upper=1.)['delta'],f['delta'])
    def test_empty_and_undefined_dispositions(self):
        s,a=m.score_variant([],0.,1.,1,1.,1.,'amended_skip')
        self.assertFalse(s['law_holds']);self.assertEqual(s['fits'],[])
        z=m.grid_fit([{'laps':p,'d':0.} for p in [3.,4.,5.,6.]])
        self.assertFalse(z['audit']['defined']);self.assertEqual(z['grid_index'],0)
        self.assertEqual(len(z['audit']['nonfinite_candidate_indices']),22002)
    def test_leaf_comparison_retains_small_delta_and_missing(self):
        a=m.compare_leaves({'a':1.,'b':[True,2.]},{'a':1.+2**-52,'b':[True],'c':3})
        self.assertEqual(a[0]['independent_minus_original'],2**-52)
        self.assertTrue(any(x.get('disposition')=='missing_index' for x in a))
        self.assertTrue(any(x.get('disposition')=='missing_key' for x in a))
    def test_nonfinite_input_refusal(self):
        with self.assertRaises(ValueError):m.collect([cell(float('nan'),2)],0.,1,'alpha')
        with self.assertRaises(ValueError):m.collect([cell(1.,2.5)],0.,1,'alpha')
        with self.assertRaises(ValueError):m.collect([cell(1.,3)],0.,1,'alpha')


def measure():
    tic=time.perf_counter(); results=[]
    for n in [8,7,32]:
        t=time.perf_counter(); obj={}
        for mode in ('original_stop','amended_skip'):
            score,audit=m.score_variant(windows(n),0.,1.,1,3.,4.,mode);obj[mode]={'score':score,'audit':audit}
        results.append({'manufactured_window_count':n,'both_variant_seconds':time.perf_counter()-t,'result':obj})
    payload=json.dumps({'synthetic_only':True,'cases':results},indent=2,allow_nan=False).encode()+b'\n'
    with open('/tmp/task21d-synthetic-output-001.json','xb') as f:f.write(payload)
    return {'synthetic_only':True,'seconds_through_serialization_write':time.perf_counter()-tic,
            'observed_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'bytes':len(payload),
            'cases':[{'windows':r['manufactured_window_count'],'both_variant_seconds':r['both_variant_seconds']} for r in results],
            'scope':'Two typical sizes and32-window stress; not a worst-case theorem or target scoring.'}


if __name__=='__main__':
    if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3));resource.setrlimit(resource.RLIMIT_CPU,(20,20))
    test=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
    report=measure() if test.wasSuccessful() else {'synthetic_only':True,'measurement_skipped':True}
    report.update(tests=test.testsRun,failures=len(test.failures),errors=len(test.errors))
    with open('/tmp/task21d-controls-001.json','x') as f:json.dump(report,f,indent=2);f.write('\n')
    raise SystemExit(0 if test.wasSuccessful() else 1)
