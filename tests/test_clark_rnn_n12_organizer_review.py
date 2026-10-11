import importlib.util
from pathlib import Path
import numpy as np
import pytest
p=Path(__file__).parents[1]/'tools/review_clark_rnn_n12_organizer.py';s=importlib.util.spec_from_file_location('phase_review',p);A=importlib.util.module_from_spec(s);s.loader.exec_module(A)

def fixture(k):
 angle=np.pi/3;r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);j=np.kron(np.eye(6),r);x=np.arange(12.)+.2;q=np.eye(12)[0];cycle=np.array([np.linalg.matrix_power(j,i)@x for i in range(6)]);t=np.linalg.matrix_power(j,k);rev=np.linalg.matrix_power(j,6-k);qk=t@q;qk/=np.linalg.norm(qk);back=rev@qk;back/=np.linalg.norm(back);z=np.r_[cycle[k],.8,qk]
 return dict(original_cycle=cycle,phase_jacobians=np.repeat(j[None],6,axis=0),original_physical=np.r_[x,.8,q],physical=z,continued_physical=z.copy(),continued_cycle=np.roll(cycle,-k,axis=0),cycle=np.roll(cycle,-k,axis=0),transport=t,transport_norm=np.array(np.linalg.norm(t@q)),reverse_transport=rev,reverse_q=back)

def test_literal_phase4_and_mode_orientation():
 a=fixture(4);assert A.phase_arithmetic(a,4)['joined'];a['continued_physical'][13:]*=-1;assert A.phase_arithmetic(a,4)['joined']

@pytest.mark.parametrize('field',['physical','transport','continued_physical'])
def test_wrong_literal_state_product_or_matching_mode_rejected(field):
 a=fixture(4)
 if field=='physical':a[field][0]*=-1
 elif field=='transport':a[field][0,0]+=.1
 else:a[field][13:]=np.eye(12)[3]
 with pytest.raises(AssertionError):A.phase_arithmetic(a,4)
