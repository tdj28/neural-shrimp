"""Saved-only interval bounds comparing fixed and polynomial inverse defects."""
from fractions import Fraction as F
from flint import arb,arb_mat

def require(ok,message):
    if not ok:raise ValueError(message)
def exact(v):
    q=F(v);require(q.denominator&(q.denominator-1)==0,'exact dyadic endpoint');return arb(q.numerator)/arb(q.denominator)
def interval_power(left,right,degree):
    a,b=F(left),F(right);require(a<=b and type(degree)is int and 0<=degree<=5,'bounded integer polynomial power')
    if degree==0:return arb(1)
    endpoints=[a**degree,b**degree];lo,hi=min(endpoints),max(endpoints)
    if degree%2==0 and a<=0<=b:lo=F(0)
    return arb(exact((lo+hi)/2),exact((hi-lo)/2))
def polynomial(matrices,left,right):
    require(1<=len(matrices)<=6 and all(m.nrows()==matrices[0].nrows()and m.ncols()==matrices[0].ncols()for m in matrices),'same bounded coefficient shape')
    return sum((m*interval_power(left,right,k)for k,m in enumerate(matrices)),arb_mat(matrices[0].nrows(),matrices[0].ncols()))
def weighted_norm(matrix,weights):
    n=len(weights);require(matrix.nrows()==matrix.ncols()==n and all(w.is_exact()and w.is_finite()and w>0 for w in weights),'positive exact weighted square norm')
    rows=[]
    for i in range(n):
        row=sum((abs(matrix[i,j]).upper()*weights[j]for j in range(n)),arb(0))/weights[i];require(row.is_finite(),'finite complete defect');rows.append(row.upper())
    return max(rows),rows
