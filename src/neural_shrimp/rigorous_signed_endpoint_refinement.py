"""Conditional known-root sharpening using a proved weighted contraction.

C is the same point value of the certified polynomial inverse, q bounds its
weighted I-CJ defect on every mean-value segment from point to the known root.
Caller authenticates those source/domain/same-gauge premises. No existence,
new native derivatives, or continuation join follows from this helper alone.
"""
from fractions import Fraction as F
from flint import arb,arb_mat
from neural_shrimp.rigorous_eigenpair_locus import require,finite
from neural_shrimp.rigorous_orbits import ball_json,vector_json

def exact_interval(v):
 return F(int(v['mid'][0]))*F(2)**int(v['mid'][1])-F(int(v['rad'][0]))*F(2)**int(v['rad'][1]),F(int(v['mid'][0]))*F(2)**int(v['mid'][1])+F(int(v['rad'][0]))*F(2)**int(v['rad'][1])

def fixed_parameter_domain(chart,domain,known,point,parameter):
 size=len(point);d=F(parameter)-F(chart['parameter_center']);require(size==len(known)==len(domain['error'])==len(chart['center'])==len(chart['slope'])==len(chart['curvature']),'complete actual fixed parameter box')
 lo,hi=map(F,chart['parameter_endpoints']);require(lo<=F(parameter)<=hi,'exact covered parameter')
 for i in range(size):
  p=F(0)
  for factor,key in [(F(1),'center'),(d,'slope'),(d*d/2,'curvature')]:
   a,b=exact_interval(chart[key][i]);require(a==b,'exact predictor coefficient');p+=factor*a
  a,b=exact_interval(domain['error'][i]);lower,upper=p+a,p+b
  for v in [known[i],point[i]]:
   x,y=exact_interval(v);require(lower<=x<=y<=upper,'whole original fixed-parameter correction domain')
 return True

def refine(C,q,weights,point,residual,known):
 n=len(point);require(1<=n<=470 and len(weights)==len(residual)==len(known)==n and C.nrows()==C.ncols()==n and finite(C),'entire finite conditional point system')
 require(q.is_finite()and 0<=q and q<1 and all(w.is_finite()and w>0 for w in weights),'proved strict weighted contraction and positive scales')
 require(all(v.is_finite()and v.is_exact()for v in point)and all(v.is_finite()for v in residual+known),'exact finite point and known-root enclosure')
 force=-C*arb_mat([[v]for v in residual]);require(finite(force),'finite full residual forcing');norm=max((abs(force[i,0]).upper()/weights[i]).upper()for i in range(n));bound=(norm/(1-q)).upper();out=[]
 for i in range(n):
  radius=(bound*weights[i]).upper();enclosure=point[i]+arb(0,radius);lo=max(enclosure.lower(),known[i].lower());hi=min(enclosure.upper(),known[i].upper());require(lo<=hi,'conditional known-root intersection');out.append(arb((lo+hi)/2,(hi-lo)/2))
 return out,dict(passed=True,coordinates=n,weighted_forcing=ball_json(norm),weighted_radius=ball_json(bound),box=vector_json(out),scope='Conditional same-known-root weighted Neumann enclosure under authenticated original fixed-parameter domain and polynomial-inverse contraction. No new existence or branch join.')
