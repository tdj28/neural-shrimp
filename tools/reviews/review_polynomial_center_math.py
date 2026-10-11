"""Independent polynomial known-root/tangent/point reductions; no native calls."""
from fractions import Fraction as F
from flint import arb,arb_mat
import review_native705_complement_math as H
R,M,need=H.R,H.M,H.need
ball_json,matrix_json,vector_json=H.ball_json,H.matrix_json,H.vector_json


def polynomial(coeff,a,b):
 a,b=F(a),F(b);need(a<=b and 1<=len(coeff)<=6,'finite same-parameter polynomial')
 out=arb_mat(coeff[0].nrows(),coeff[0].ncols())
 for k,A in enumerate(coeff):
  ends=[a**k,b**k];lo,hi=min(ends),max(ends)
  if k and k%2==0 and a<=0<=b:lo=F(0)
  out+=A*arb(R.exact((lo+hi)/2),R.exact((hi-lo)/2))
 return out


def defect(ds,weights,a,b,tick=lambda:None):
 D=polynomial(ds,a,b);n=len(weights);need(D.nrows()==D.ncols()==n and H.finite(D)and all(w.is_exact()and w>0 for w in weights),'full finite weighted defect')
 q=max((sum((abs(D[i,j]).upper()*weights[j]for j in range(n)),arb(0))/weights[i]).upper()for i in range(n));need(q.is_finite()and 0<=q<1,'strict independently recomputed weighted Neumann premise');tick();return D,q


def root(ds,forcing,weights,original,image,a,b,tick=lambda:None):
 D,q=defect(ds,weights,a,b,tick);rhs=polynomial(forcing,a,b);init=arb_mat([[original[i,0].intersection(image[i,0])]for i in range(len(weights))]);ans=H.H.enclosure(D,rhs,weights,q,init,iterations=6);tick();return ans,dict(passed=True,q=ball_json(q),error=matrix_json(ans))


def tangent(inverse,ds,rp,mixed,pprime,weights,a,b,tick=lambda:None):
 n=len(weights);need(len(inverse)==2 and all(x.is_exact()for A in inverse for row in A.tolist()for x in row),'same exact affine inverse coefficients');D,q=defect(ds,weights,a,b,tick)
 forced=[arb_mat(n,1)for _ in range(len(inverse)+len(rp)-1)]
 for i,C in enumerate(inverse):
  for j,v in enumerate(rp):forced[i+j]-=C*v
 rhs=polynomial(forced,a,b)-polynomial(inverse,a,b)*mixed;err=H.H.enclosure(D,rhs,weights,q,iterations=6);answer=[pprime[i]+err[i,0]for i in range(n)];tick();return answer,dict(passed=True,q=ball_json(q),rhs=matrix_json(rhs),tangent=vector_json(answer))


def point(inverse,ds,residual,x,known,weights,d,tick=lambda:None):
 n=len(weights);need(len(x)==len(known)==n and all(v.is_exact()for v in x),'entire exact fresh point');D,q=defect(ds,weights,d,d,tick);rhs=-polynomial(inverse,d,d)*residual;init=arb_mat([[known[i]-x[i]]for i in range(n)]);err=H.H.enclosure(D,rhs,weights,q,init,iterations=6);ans=[known[i].intersection(x[i]+err[i,0])for i in range(n)];tick();return ans,dict(passed=True,q=ball_json(q),box=vector_json(ans),rhs=matrix_json(rhs))


def cover(old,new,label):
 need(len(old)==len(new),'full-coordinate coverage '+label)
 for a,b in zip(old,new,strict=True):H.H.contains(a,b)


def original(error,predictor,box):
 need(len(error)==len(predictor)==len(box),'full original point/known segment')
 for e,p,x in zip(error,predictor,box,strict=True):
  lo,hi=M.interval(e);q=M.point(ball_json(p));a,b=M.interval(ball_json(x));need(q+lo<=a<=b<=q+hi,'literal original correction domain')
