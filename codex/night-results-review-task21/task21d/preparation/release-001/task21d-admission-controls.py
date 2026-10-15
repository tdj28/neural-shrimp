"""Manufactured-only admission/helper controls; no original target file reads."""
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
import numpy as np


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
h=load('helper','/tmp/task21d-inputs.py');s=load('score','/tmp/task21d-score.py')


def fixture(n=20,q=4):
    coords=np.linspace(-1.,1.,n).tolist();alpha=.05
    args={'mode':'cells','clamp_entropy':None,'model':'synthetic','layer':2,'vectors':'own','concept':'test','window':1}
    periodlist=[q if x<0 else q+3 for x in coords]
    line={'args':args,'cells':[{'log10_temperature':x,'alpha':alpha,'period':p,'class':'periodic'} for x,p in zip(coords,periodlist)]}
    definition={'cells':[[x,alpha] for x in coords]}
    run='synthetic/curve'
    nf={'args':{'param':'temperature','index':0,'nearest_fold':False,'run':run,'steps':[.01,.003,.001,.0003,.0001]},
        'source':{**{k:args[k] for k in ['model','layer','vectors','concept','window']},'period':q,'alpha':alpha,'log10_temperature':0.,'index':0,'run':run},
        'a':2.,'b':2.5,'ab':5.,'b_steps':[{'h_relative':v,'b':b} for v,b in zip([.01,.003,.001,.0003,.0001],[8.,5.,3.,2.,1.])],
        'b_plateau_pair':[.001,.0003],'multiplier':[1.,0.],'orbit_residual':1e-14,'second_modulus':.2,
        'left_right_cosine':.8,'pi_over_sqrt_ab':float(np.pi/np.sqrt(5.))}
    curve={'period':q,'points':[{'log10_temperature':0.,'alpha':alpha}]}
    spec={'declared_line_args':copy.deepcopy(args),'expected_cells':n,'base_period':q,
          'normal_form_source_curve':{'original_path':'/data2/x/explore/'+run+'/curve.json'}}
    return line,definition,nf,curve,spec


class Controls(unittest.TestCase):
    def test_complete_association_and_nf_first_tie(self):
        l,d,n,c,p=fixture();r=h.admit_line(l,d,n,c,p)
        self.assertEqual(r['cell_count'],20)
        f=h.normal_form(n,{'source':{'period':4},'args':{'param':'temperature'},'coordinate':'log10_temperature'})
        self.assertEqual(f['selected_pair_index'],2);self.assertEqual(f['b']['absolute_difference'],0.)
        self.assertTrue(f['crossing_from_saved_fields']['real_plus_one_within_0_01'])
    def test_missing_and_duplicate_coordinate_refusal(self):
        for change in ['missing','duplicate']:
            l,d,n,c,p=fixture()
            if change=='missing':l['cells'].pop()
            else:d['cells'][1]=d['cells'][0]
            with self.assertRaises(ValueError):h.admit_line(l,d,n,c,p)
    def test_nonfinite_and_ordinal_mismatch(self):
        for v in [float('nan'),1e-99]:
            l,d,n,c,p=fixture();l['cells'][0]['log10_temperature']=v
            with self.assertRaises(ValueError):h.admit_line(l,d,n,c,p)
    def test_wrong_model_alpha_and_card(self):
        for change in ['model','alpha','card']:
            l,d,n,c,p=fixture()
            if change=='model':n['source']['model']='other'
            elif change=='alpha':n['source']['alpha']=.06
            else:p['declared_line_args']['model']='other'
            with self.assertRaises(ValueError):h.admit_line(l,d,n,c,p)
    def test_wrong_curve_index_and_parameter(self):
        l,d,n,c,p=fixture();n['source']['index']=1
        with self.assertRaises(ValueError):h.admit_line(l,d,n,c,p)
        n['args']['param']='alpha'
        with self.assertRaises(ValueError):h.normal_form(n,{'source':{},'args':{},'coordinate':'log10_temperature'})
    def test_failed_saved_classification_is_reported(self):
        _,_,n,_,_=fixture();n['multiplier']=[1.02,2e-9]
        f=h.normal_form(n,{'source':{},'args':{},'coordinate':'log10_temperature'})
        self.assertFalse(f['crossing_from_saved_fields']['imaginary_within_1e-9'])
        self.assertFalse(f['crossing_from_saved_fields']['real_plus_one_within_0_01'])
    def test_prepared_driver_refuses_before_ownership(self):
        with tempfile.TemporaryDirectory(prefix='task21d-refusal-') as td:
            p=Path(td);(p/'card').write_text('{"admission":"prepared"}')
            cp=subprocess.run([sys.executable,'-I','/tmp/task21d-run.py','--card',str(p/'card'),'--out',str(p/'out')],capture_output=True,text=True,timeout=3)
            self.assertNotEqual(cp.returncode,0);self.assertIn('refusing',cp.stderr);self.assertFalse((p/'out').exists())


def measure():
    tic=time.perf_counter();result=[];hashbytes=0
    for n,q in [(1700,4),(1000,2)]:
        l,d,nf,c,sp=fixture(n,q)
        encoded=json.dumps([l,d,nf,c,sp],allow_nan=False).encode();hashbytes+=len(encoded)
        before=hashlib.sha256(encoded).hexdigest()
        row={'admission':h.admit_line(l,d,nf,c,sp),'normal_form':h.normal_form(nf,{'source':{},'args':{},'coordinate':'log10_temperature'}),
             'rescore':s.rescore(l['cells'],nf,q),'source_before_after_equal':before==hashlib.sha256(encoded).hexdigest()}
        result.append(row)
    raw=json.dumps(result,indent=2,allow_nan=False).encode()+b'\n'
    with open('/tmp/task21d-admission-synthetic-001.json','xb') as f:f.write(raw)
    return {'synthetic_only':True,'full_manufactured_cell_count':2700,'admission_hash_ledger_serialize_write_seconds':time.perf_counter()-tic,
            'output_bytes':len(raw),'input_hash_bytes_per_pass':hashbytes,'observed_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'numpy_version':np.__version__,'scope':'Full cell-count admission/hash/ledger serialization; two single-window synthetic fans, grid cost separately measured in controls001.'}


if __name__=='__main__':
    if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3));resource.setrlimit(resource.RLIMIT_CPU,(10,10))
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
    o=measure() if r.wasSuccessful() else {'measurement_not_run':True}
    o.update(tests=r.testsRun,failures=len(r.failures),errors=len(r.errors))
    with open('/tmp/task21d-admission-controls-001.json','x') as f:json.dump(o,f,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
