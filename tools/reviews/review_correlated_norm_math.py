"""Independent 384-bit saved derivative-to-norm-cocycle reduction.

No producer norm-cocycle routines, map calls or floating proposals. The caller
binds exact model/root/derivative domains and source/native identities.
"""
from fractions import Fraction
from flint import arb,arb_mat,ctx
import review_localization_math as R
from neural_shrimp.rigorous_orbits import ball_json


def need(x,message):
 if not x:raise ValueError(message)
def identity(n):return arb_mat([[int(i==j) for j in range(n)] for i in range(n)])
def midpoint(a):return arb_mat([[a[i,j].mid() for j in range(a.ncols())] for i in range(a.nrows())])
def finite(a):return all(x.is_finite() for row in a.tolist() for x in row)
def frobenius(a):return sum((abs(x).upper()*abs(x).upper() for row in a.tolist() for x in row),arb(0)).sqrt().upper()
def infnorm(a):return max(sum((abs(a[i,j]).upper() for j in range(a.ncols())),arb(0)).upper() for i in range(a.nrows()))
def convolution(a,b):
 n=a[0].nrows();out=[arb_mat(n,n) for _ in range(len(a)+len(b)-1)]
 for i,x in enumerate(a):
  for j,y in enumerate(b):out[i+j]+=x*y
 return out

def polynomial(coefficients):
 """Independent range enclosure for common t in[-1,1], through degree4."""
 need(1<=len(coefficients)<=5,'fixed degree')
 out=coefficients[0]+arb_mat(coefficients[0].nrows(),coefficients[0].ncols())
 for k in range(1,len(coefficients)):
  domain=arb(0,1) if k%2 else arb(0).union(arb(1))
  out+=coefficients[k]*domain
 return out

def bound_product(previous,phase):
 a,ea,ga=phase;b,eb,gb=previous
 need(ea>=0 and eb>=0 and ga>=0 and gb>=0,'nonnegative scalar bounds')
 full=convolution(a,b)
 anorm=(ga+frobenius(a[1])+frobenius(a[2])).upper()
 bnorm=(gb+frobenius(b[1])+frobenius(b[2])).upper()
 remainder=(anorm*eb+ea*bnorm+ea*eb+frobenius(full[3])+frobenius(full[4])).upper()
 return full[:3],remainder,(ga*gb).upper()

def strict_metric(coeff,epsilon,gamma,metric,positive):
 n=coeff[0].nrows();P=R.original_matrix(metric['P'])
 need(P.nrows()==P.ncols()==n and all(P[i,j].is_exact() and P[i,j]==P[j,i] for i in range(n) for j in range(n)),'exact symmetric saved P')
 pp=positive(P,R.original_matrix(metric['P_positive']['basis']))
 need(pp['passed'],'strict independent P positivity')
 terms=convolution([x.transpose()*P for x in coeff],coeff)
 d=[-x for x in terms];d[0]+=P
 qnorm=(gamma+frobenius(coeff[1])+frobenius(coeff[2])).upper()
 penalty=(infnorm(P)*(2*qnorm*epsilon+epsilon*epsilon)).upper()
 decrease=polynomial(d)-identity(n)*penalty
 need(finite(decrease),'finite full decrease')
 dp=positive(decrease,R.original_matrix(metric['complete_D_positive']['basis']))
 need(dp['passed'],'strict independent complete decrease')
 return {'passed':True,'P_positive':pp,'complete_D_positive':dp,'remainder_norm_upper':ball_json(epsilon),
  'polynomial_norm_upper':ball_json(qnorm),'penalty_upper':ball_json(penalty)}

def audit(phases,frames,norm_phases,metric,halfwidth,*,positive=None,tick=lambda:None):
 if positive is None:
  from review_endpoint_attraction import positive
 half=Fraction(halfwidth);need(half>0,'positive exact halfwidth')
 p=len(phases);need(1<=p<=18 and len(norm_phases)==p and len(frames['witnesses'])==p and frames['cyclic_last_index']==0,'complete cyclic phase/frame inventory')
 with ctx.workprec(384):
  delta=arb(half.numerator)/half.denominator;need(delta.is_exact(),'exact dyadic halfwidth')
  n=len(phases[0]['center_J']);need(1<=n<=13,'native dimension cap');basis=[];inverse=[];frame_bounds=[]
  for j,row in enumerate(frames['witnesses']):
   need(row['phase']==j,'native frame order');b=R.original_matrix(row['B']);need(b.nrows()==b.ncols()==n and finite(b) and all(v.is_exact() for line in b.tolist() for v in line),'exact fixed frames')
   bi=b.inv();C=midpoint(bi);defect=infnorm(identity(n)-C*b);need(defect.is_finite() and defect<1,'independently invertible frame')
   basis.append(b);inverse.append(bi);frame_bounds.append(ball_json(defect));tick()
  previous=([identity(n),arb_mat(n,n),arb_mat(n,n)],arb(0),arb(1));rows=[]
  for j,(phase,saved_norm) in enumerate(zip(phases,norm_phases,strict=True)):
   need(phase['phase']==saved_norm['phase']==j,'native derivative/norm order')
   J=R.original_matrix(phase['center_J']);Jprime=R.original_matrix(phase['whole_total_Jprime'])
   need(J.nrows()==J.ncols()==Jprime.nrows()==Jprime.ncols()==n and finite(J) and finite(Jprime),'complete native derivatives')
   constant=midpoint(J);scaled=Jprime*delta;slope=midpoint(scaled)
   # Proposal slope can differ from producer rounding: defect is retained in
   # full, so the independent polynomial remains a valid enclosure.
   error=(J-constant)+(scaled-slope)*arb(0,1)
   left=basis[(j+1)%p];right=inverse[j]
   coeff=[left*constant*right,left*slope*right,arb_mat(n,n)];E=left*error*right
   gamma=R.original_ball(saved_norm['operator_bound']['gamma']);need(gamma.is_finite() and gamma.is_exact() and gamma>0,'fixed positive proposed gamma')
   test=gamma*gamma*identity(n)-coeff[0].transpose()*coeff[0]
   check=positive(test,R.original_matrix(saved_norm['operator_bound']['positive']['basis']))
   need(check['passed'],'independent strict phase norm bound')
   previous=bound_product(previous,(coeff,frobenius(E),gamma));need(all(finite(x) for x in previous[0]) and previous[1].is_finite(),'finite full product and remainder')
   rows.append({'phase':j,'gamma':ball_json(gamma),'phase_remainder_norm':ball_json(frobenius(E)),'product_remainder_norm':ball_json(previous[1]),'strict_operator_bound':True});tick()
  final=strict_metric(*previous,metric,positive)
 return {'passed':True,'precision':384,'native_phases':p,'dimension':n,'halfwidth':str(half),'frame_inverse_defects':frame_bounds,'phases':rows,
  'metric':final,'scope':'Independent saved384 cyclic similarity, phase proposal-defect enclosure, noncommuting quadratic convolution, full scalar2 remainder and strict congruence decrease. Conditional on authenticated native root/domain/model derivatives; no model calls, new floating proposals or neutral-fold/whole-branch claim.'}
