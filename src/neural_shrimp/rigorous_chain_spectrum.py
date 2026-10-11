"""Conditional complementary spectrum of a validated length-two Jordan chain.

At a proved root Mq=q, Mr=r+q, a frame [q,r,E] conjugates M to an
upper block triangular matrix with Jordan block [[1,1],[0,1]]. The lower
block supplies all remaining eigenvalues. This routine validates the frame
and lower block enclosures, conditional on those exact root identities.
"""
import itertools
import numpy as np
from flint import arb,arb_mat
from .rigorous_eigenpair_locus import require
from .rigorous_cusps import spectral_exclusion
from .rigorous_orbits import matrix_json


def chain_complement(monodromy,q,r):
    n=monodromy.nrows()
    require(n>=3 and monodromy.ncols()==n and len(q)==len(r)==n,'chain/frame dimensions')
    require(all(v.is_finite()for v in q+r) and all(monodromy[i,j].is_finite()for i in range(n)for j in range(n)),'finite chain/frame inputs')
    # A floating minor ranking only chooses fixed coordinate columns. The
    # actual Arb inverse must enclose an inverse for every input frame.
    qmid=np.array([float(v.mid())for v in q]);rmid=np.array([float(v.mid())for v in r])
    require(np.isfinite(qmid).all() and np.isfinite(rmid).all(),'finite frame proposal')
    pair=max(itertools.combinations(range(n),2),key=lambda p:abs(qmid[p[0]]*rmid[p[1]]-qmid[p[1]]*rmid[p[0]]))
    remaining=[i for i in range(n)if i not in pair]
    frame=arb_mat([[q[i],r[i]]+[arb(int(i==j))for j in remaining]for i in range(n)])
    inverse=frame.inv();require(all(inverse[i,j].is_finite()for i in range(n)for j in range(n)),'validated finite frame inverse')
    transformed=inverse*monodromy*frame
    expected=[[1,1],[0,1]]
    require(all(transformed[i,j].contains(expected[i][j]if i<2 else 0)for i in range(n)for j in range(2)),'conditional exact Jordan identities are enclosed')
    block=arb_mat([[transformed[i,j]for j in range(2,n)]for i in range(2,n)])
    spectral=spectral_exclusion(block)
    attracting=bool(spectral['passed'] and all(row['strict_inside']for row in spectral['discs']))
    return {'schema':'neural-shrimp-conditional-chain-spectrum-v1','passed':spectral['passed'],'dimension':n,'coordinate_pivots':list(pair),'coordinate_columns':remaining,'frame':matrix_json(frame),'validated_frame_inverse':matrix_json(inverse),'transformed':matrix_json(transformed),'quotient':matrix_json(block),'spectral_exclusion':spectral,'complement_attracting':attracting,'statement':'Conditional on a rigorously proved same-root Mq=q and Mr=r+q: all remaining multipliers avoid the unit circle when passed, so +1 has exact algebraic multiplicity two and one length-two Jordan block. Complement attraction requires all quotient discs strictly inside. Generic unfolding and branch connection are not established.'}
