import importlib.util
from pathlib import Path
import numpy as np
s=importlib.util.spec_from_file_location('half',Path(__file__).parents[1]/'tools/diagnose_clark_rnn_n12_half_return.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_native_product_order_and_signed_half_spectrum():
    a=np.array([[1.,2.],[0.,1.]]);b=np.array([[2.,0.],[1.,1.]]);c=np.diag([.5,.25]);phase=np.array([a,b,c,a,b,c]);half=c@b@a;cycle=np.array([[1,2],[2,3],[4,5],[-1,-2],[-2,-3],[-4,-5]],float)
    d=m.context(cycle,phase,half@half)
    assert d['half_wave_relative_error']==0 and d['square_relative_error']==0 and d['full_product_relative_error']==0
    np.testing.assert_array_equal(d['signed_half_jacobian'],-half)
    assert d['signed_half_eigen_residual']<1e-14 and min(d['proper_divisor_separations'].values())>0

def test_without_half_wave_or_repeated_derivatives_no_square_identity():
    phase=np.array([np.eye(2)]*6);phase[3]=np.eye(2)*2;cycle=np.arange(12.).reshape(6,2)
    d=m.context(cycle,phase,2*np.eye(2));assert d['half_wave_relative_error']>0 and d['half_derivative_relative_error']>0 and d['square_relative_error']>0 and d['full_product_relative_error']==0
