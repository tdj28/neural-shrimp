import copy
import numpy as np
import pytest
from flint import arb
from neural_shrimp.rigorous_orbits import ball_json
from neural_shrimp.generalized_chain_proposal import from_accepted_chart


def fixture(delta=-.02):
    center=[arb(0)]*470;center[13]=arb(1);center[-2:]=[arb(1),arb(1)]
    chart={'passed':True,'gates':dict.fromkeys(('uniform_normalized_eigenpair_locus','primitive_period','anchor_identified'),True),'period':18,'dimension':13,'sigma':1,'shooting_dimension':470,'center':[ball_json(v)for v in center]}
    matrix=np.diag([1,1+delta]+[.1+i*.03 for i in range(11)]);matrix[0,1]=1
    values,vectors=np.linalg.eig(matrix)
    spec={'monodromy_midpoint':matrix.tolist(),'eigenvalues':[[float(v.real),float(v.imag)]for v in values],'eigenvectors':[[[float(v.real),float(v.imag)]for v in row]for row in vectors],'relative_residuals':[0.]*13,'critical_nearest_index':0}
    return chart,spec


def test_near_jordan_seed_recovers_generalized_direction_and_native_order():
    chart,spec=fixture();seed,diag=from_accepted_chart(chart,spec)
    assert len(seed)==41 and all(v.is_exact()for v in seed)
    assert seed[13]==seed[14]==seed[15]==1 and seed[29].contains(1)
    assert all(abs(float(seed[i]))<1e-12 for i in range(28,41)if i!=29)
    assert diag['near_chain_residual']<1e-12 and diag['gap']<.03

@pytest.mark.parametrize('kind',['unqualified','mode','rate','shape','residual','complex','gap','eigenvector'])
def test_bad_seed_is_not_silently_used(kind):
    chart,spec=fixture()
    if kind=='unqualified':chart['passed']=False
    elif kind=='mode':chart['center'][13]=ball_json(arb(2))
    elif kind=='rate':chart['center'][-1]=ball_json(arb(-1))
    elif kind=='shape':spec['monodromy_midpoint']=[]
    elif kind=='residual':spec['relative_residuals'][1]=1e-4
    elif kind=='complex':spec['eigenvalues'][1][1]=.001
    elif kind=='eigenvector':spec['eigenvectors'][3][3]=[0,0]
    else:spec['eigenvalues'][1][0]=.9
    with pytest.raises(ValueError):from_accepted_chart(chart,spec)
