import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pytest
from neural_shrimp.second_unit_flip import seed_from_signed

def test_nonnormal_complex_realification_and_literal_negative_mode():
 n=13;angle=.117;c,s=np.cos(angle),np.sin(angle);B=np.array([[c,-s/4],[4*s,c]])*1.001;M=np.diag([.2]*n);M[:2,:2]=B;M[2,2]=-1.;w=np.eye(n)[2];point=dict(z=np.r_[np.arange(n)/100,72.,w],hidden=72.,output=.845)
 y,r=seed_from_signed(point,M);q=y[15:28];v=y[28:41];lam=complex(*r['eigenvalue']);u=q+1j*np.sqrt(y[41])*v
 np.testing.assert_allclose(M@u,lam*u,atol=1e-12);assert abs(q@q-1)<1e-12 and abs(q@v)<1e-12
 np.testing.assert_array_equal(y[42:],w);assert y[13]==72 and y[14]==.845
 M[:2,:2]*=.9
 with pytest.raises(ValueError,match='unstable'):seed_from_signed(point,M)

def test_competing_real_leading_mode_is_not_called_unit_pair():
 M=np.diag([1.1,-1,.2]);point=dict(z=np.r_[np.zeros(3),72.,[0,1,0]],hidden=72.,output=.845)
 with pytest.raises(ValueError,match='complex'):seed_from_signed(point,M)
