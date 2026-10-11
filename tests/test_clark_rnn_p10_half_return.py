import importlib.util
from pathlib import Path
import numpy as np
s=importlib.util.spec_from_file_location('half',Path(__file__).parents[1]/'tools/diagnose_clark_rnn_p10_symmetry.py');M=importlib.util.module_from_spec(s);s.loader.exec_module(M)
def test_repeated_noncommuting_half_products_square_in_native_order():
 j=np.array([[[1.,1.],[0.,1.]],[[1.,0.],[.2,1.]],[[.8,0.],[0.,.9]],[[1.,.1],[0.,1.]],[[.7,0.],[0.,.8]]]);js=np.r_[j,j];h=np.eye(2)
 for x in j:h=x@h
 c=np.arange(10).reshape(5,2);cycle=np.r_[c,-c]
 r=M.half_metrics(cycle,js,h@h)
 assert r['antipodal_defect']==0 and r['square_relative_defect']==0
 assert r['native_product_relative_error']<1e-15
def test_simple_native_minus_one_is_not_a_real_repeated_half():
 js=np.tile(np.eye(2),(10,1,1));js[-1]=np.diag([-1.,.2]);c=np.ones((10,2));r=M.half_metrics(c,js,np.diag([-1.,.2]))
 assert r['square_relative_defect']>1 and r['antipodal_defect']>0
