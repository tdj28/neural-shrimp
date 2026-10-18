"""Independent rational row-norm check for the disclosed powers four/eight."""
from fractions import Fraction
import witness_audit as w

def audit(attr, n):
    w.require([r['power'] for r in attr['powers']] == [4,8], 'declared powers differ')
    w.require(all(w.point(x)==1 for x in w.vector(attr['weights'],n)), 'unweighted norm required')
    flags=[];count=0
    for row in attr['powers']:
        mat=w.matrix(row['matrix'],n)
        saved=[w.point(x,nonnegative=True) for x in w.vector(row['weighted_row_upper'],n)]
        for i,entries in enumerate(mat):
            exact=sum((w.absolute(v) for v in entries),Fraction(0));count+=n
            w.require(saved[i]>=exact,'understated row norm')
        upper=w.point(row['induced_infinity_upper'],nonnegative=True);margin=w.point(row['margin'])
        w.require(upper>=max(saved) and margin<=1-upper,'understated bound')
        w.require(row['passed'] is (margin>0),'pass flag differs');flags.append(margin>0)
    witness=next((p for p,ok in zip((4,8),flags) if ok),None)
    w.require(attr['passed'] is any(flags) and attr['witness_power']==witness,'aggregate differs')
    return dict(passed=any(flags),witness_power=witness,matrix_entries_checked=count,
        arithmetic='independent exact rational row sums and strict margins',dense_products_recomputed=False)
