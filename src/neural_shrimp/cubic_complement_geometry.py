"""Exact generic cubic predictor/closed-cell geometry for a known root family.

This is bookkeeping and polynomial evaluation only. The caller authenticates the
original full705 root, residual Taylor coefficients, inverse family and domains.
"""
from fractions import Fraction as F
from .rigorous_connection import require


def predictor(c,s,a,b,d):
    require(1<=len(c)<=705 and len(c)==len(s)==len(a)==len(b),'full matched cubic predictor')
    d2=d*d;d3=d2*d
    return [c[i]+s[i]*d+a[i]*d2/2+b[i]*d3 for i in range(len(c))]


def tangent(s,a,b,d):
    require(1<=len(s)<=705 and len(s)==len(a)==len(b),'full matched cubic tangent')
    return [s[i]+a[i]*d+3*b[i]*d*d for i in range(len(s))]


def residual_derivative(coefficients):
    """Taylor for R' itself: final coefficient bounds R''''/3!, not a derivative of an interval remainder."""
    require(len(coefficients)==5 and all(v.nrows()==coefficients[0].nrows() and v.ncols()==1 for v in coefficients),'full quartic residual coefficients')
    return [(i+1)*coefficients[i+1]for i in range(4)]


def domains(left,right,count):
    l,r=F(left),F(right)
    require(type(count)is int and 1<=count<=8 and l<r,'bounded exact derivative partition')
    h=(r-l)/count
    return [dict(index=i+1,endpoints=[str(l+i*h),str(l+(i+1)*h)])for i in range(count)]


def cell(rows,domain,depth=0,index=0):
    require(type(domain)is int and 1<=domain<=len(rows) and type(depth)is int and 0<=depth<=3 and type(index)is int and 0<=index<2**depth,'fixed depth3 binary cell')
    a,b=map(F,rows[domain-1]['endpoints']);width=(b-a)/2**depth;l=a+index*width;r=l+width
    return dict(domain=domain,depth=depth,index=index,endpoints=[str(l),str(r)],center=str((l+r)/2),halfwidth=str((r-l)/2))


def children(rows,row,outcome):
    require(row==cell(rows,row['domain'],row['depth'],row['index']),'literal closed cell')
    require(outcome['completed']and outcome['native_calls']==126 and outcome['dimension']==705 and outcome['complement_dimension']==11 and outcome['passed']is False,'only completed negative full11 metric splits')
    require(row['depth']<3,'fixed maximum refinement depth')
    return [cell(rows,row['domain'],row['depth']+1,2*row['index']+i)for i in(0,1)]


def complete(rows,accepted):
    require(accepted and len(accepted)<=8*len(rows),'finite accepted-cell inventory')
    spans=[]
    for out in accepted:
        row=out['cell'];require(row==cell(rows,row['domain'],row['depth'],row['index'])and out['passed']and out['completed']and out['native_calls']==126 and out['dimension']==705 and out['complement_dimension']==11,'all complete full11 leaves')
        spans.append(tuple(map(F,row['endpoints'])))
    spans.sort();ends=[rows[0]['endpoints'][0],rows[-1]['endpoints'][1]]
    require(spans[0][0]==F(ends[0])and spans[-1][1]==F(ends[1])and all(a[1]==b[0]for a,b in zip(spans,spans[1:])),'exact closed union without duplicate interiors or gaps')
    return dict(passed=True,endpoints=ends,accepted_cells=len(accepted),domain_count=len(rows))
