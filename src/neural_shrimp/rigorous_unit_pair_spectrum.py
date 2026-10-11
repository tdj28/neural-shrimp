"""Conditional periodic-frame quotient of a proved native unit-pair tube.

Every exact native root must already satisfy its normalized shooting equations.
Containment sanity checks below do not supply that hypothesis. Local frame
inverses remove the known invariant two-plane before the quotient product,
avoiding propagation of the two critical directions through the full monodromy.
"""
import itertools
import numpy as np
from flint import arb,arb_mat
from .rigorous_unit_pair import entire_pair
from .rigorous_eigenpair_locus import require,finite
from .rigorous_cusps import spectral_exclusion
from .rigorous_orbits import eye,matrix_json


def invariant_frame(q,r):
    n=len(q);require(n>=2 and len(r)==n and all(v.is_finite()for v in q+r),'finite frame coordinates')
    qm=np.array([float(v.mid())for v in q]);rm=np.array([float(v.mid())for v in r])
    require(np.isfinite(qm).all()and np.isfinite(rm).all(),'finite minor proposal')
    pivots=max(itertools.combinations(range(n),2),key=lambda p:abs(qm[p[0]]*rm[p[1]]-qm[p[1]]*rm[p[0]]))
    remaining=[i for i in range(n)if i not in pivots]
    frame=arb_mat([[q[i],r[i]]+[arb(int(i==j))for j in remaining]for i in range(n)])
    inverse=frame.inv();require(finite(inverse),'validated frame inverse')
    return frame,inverse,{'coordinate_pivots':list(pivots),'coordinate_columns':remaining,'frame':matrix_json(frame),'inverse':matrix_json(inverse)}


def native_complement(blocks,flat,period,n,t):
    """Conditional uniform complementary multiplier enclosure on the whole tube.

    `blocks[i]` is the whole native n*n state derivative DF_i; `flat` has all
    (x_i,q_i,r_i), followed by the actual rates. All inputs must be bound to
    the same rigorously proved full704-domain certificate by the caller.
    """
    require(type(period)is int and 1<=period<=18 and type(n)is int and 2<=n<=13,'native dimensions')
    require(len(blocks)==period and len(flat)==3*n*period+2,'complete native inputs')
    require(all(b.nrows()==b.ncols()==n and finite(b)for b in blocks),'finite full native derivative blocks')
    require(all(v.is_finite()for v in flat)and all(v>0 for v in flat[-2:]),'finite positive-rate native box')
    c,e,d=entire_pair(t);frames=[];records=[]
    for k in range(period):
        offset=3*n*k
        v,w,record=invariant_frame(flat[offset+n:offset+2*n],flat[offset+2*n:offset+3*n])
        frames.append((v,w));records.append(record)
    quotients=[];transformed=[];product=eye(n-2)if n>2 else None
    for k,A in enumerate(blocks):
        T=frames[(k+1)%period][1]*A*frames[k][0]
        expected=arb_mat([[c,e],[-d,c]])if k==period-1 else eye(2)
        require(all(T[i,j].contains(0)for i in range(2,n)for j in range(2)),'conditional invariant plane enclosed')
        require(all(T[i,j].overlaps(expected[i,j])for i in range(2)for j in range(2)),'conditional native critical identities enclosed')
        transformed.append(matrix_json(T))
        if n>2:
            Q=arb_mat([[T[i,j]for j in range(2,n)]for i in range(2,n)])
            product=Q*product;quotients.append(matrix_json(Q))
    if n>2:
        require(finite(product),'finite quotient monodromy')
        spectral=spectral_exclusion(product)
        attracting=bool(spectral['passed']and all(v['strict_inside']for v in spectral['discs']))
    else:spectral={'passed':True,'discs':[]};attracting=True
    return {'schema':'neural-shrimp-conditional-native-unit-pair-spectrum-v1','passed':spectral['passed'],
        'complement_attracting':attracting,'dimension':n,'period':period,'frames':records,'transformed_native_blocks':transformed,
        'quotient_blocks':quotients,'quotient_monodromy':None if product is None else matrix_json(product),'spectral_exclusion':spectral,
        'scope':'Conditional on the same proved native normalized unit-pair tube and matching whole-domain derivative blocks. Every local exact quotient preserves its critical two-plane; the quotient product contains all n-2 remaining multipliers. Attraction requires every quotient disc strictly inside the unit circle. No finite-interval NS criticality, stable-side or complete-window claim.'}
