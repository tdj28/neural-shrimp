import importlib.util
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pytest

def module():
 p=Path(__file__).parents[1]/'tools/review_tiny_lm_response.py';s=importlib.util.spec_from_file_location('lm_response_review',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_independent_compensated_dot_and_norm():
 m=module();assert m.dot([1e16,1,-1e16],[1,1,1])==1
 assert m.norm(np.array([3.,4.]))==5
 with pytest.raises(ValueError):m.norm([float('nan')])

def test_complete_comparators_and_signed_functional_slopes():
 m=module();u=np.array([1.,0.]);response=np.array([0.,2.]);row=m.response_row('gradient',4,response,response,np.array([8.,0.]),response,u,2.,[-6.,-6.],np.array([0.,-3.]))
 assert row['resolved']and row['training_ce_slope_resolved']and row['response_gain']==2
 assert row['frozen_vector_error']==pytest.approx(np.sqrt(68)/2)
 assert row['persistence_vector_error']==pytest.approx(np.sqrt(5)/2)
 assert row['scalar_gain']==16 and row['scalar_gain_relative_error']==7
 assert row['transported_training_ce_slope']==-6
 bad=m.response_row('gradient',4,response,response*1.02,response,response,u,2.,[-6.,-6.],np.array([0.,-3.]));assert not bad['resolved']

def test_every_operation_and_global_case_count():
 m=module();ops=m.expected_operations();assert Counter(ops)==dict(gradient=580,hvp=256,ce=96)
 assert ops[:17]==['gradient']*17 and ops[17:81]==['hvp']*64 and ops[81:83]==['gradient','ce']

def test_inventory_rejects_changed_bytes_and_extra_file(tmp_path):
 m=module();(tmp_path/'x').write_text('abc');v={'files':{'x':{'bytes':3,'sha256':m.sha(tmp_path/'x')}}};(tmp_path/'manifest.json').write_text(json.dumps(v));pin=m.sha(tmp_path/'manifest.json');assert m.inventory(tmp_path,pin)['files_including_manifest']==2
 (tmp_path/'x').write_text('abd')
 with pytest.raises(ValueError,match='identity'):m.inventory(tmp_path,pin)
 (tmp_path/'x').write_text('abc');(tmp_path/'extra').write_text('x')
 with pytest.raises(ValueError,match='complete inventory'):m.inventory(tmp_path,pin)

def test_report_comparison_does_not_accept_false_gate():
 m=module();m.same(1+1e-12,1.)
 with pytest.raises(ValueError):m.same(False,True)
 with pytest.raises(ValueError):m.same(1.01,1.)
