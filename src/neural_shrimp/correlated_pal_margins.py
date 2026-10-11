"""Floating saved-coefficient margin diagnosis only; no interval proof gate."""
import math
from fractions import Fraction
from .rigorous_eigenpair_locus import require

def upper(row):
 require(all(isinstance(row[k],list)and len(row[k])==2 and all(type(v)is int for v in row[k])for k in ['mid','rad'])and row['rad'][0]>=0,'exact saved dyadic shape')
 v=abs(math.ldexp(float(row['mid'][0]),row['mid'][1]))+math.ldexp(float(row['rad'][0]),row['rad'][1])
 require(math.isfinite(v),'finite floating absolute proposal');return v

def margins(proof,check=lambda:None):
 N=proof['shooting_dimension'];require(type(N)is int and 1<=N<=705 and proof['passed']is True and all(proof['gates'].values()),'passed full source proof only')
 b=proof['bound'];require(b['passed']is True and b['dimension']==N and len(b['coordinate_inclusions'])==N and all(b['coordinate_inclusions']),'complete strict root evidence')
 w=[float(Fraction(v))for v in proof['weights']];r=[upper(v)for v in b['radii']];h=2.**proof['settings']['halfwidth_exponent']
 require(len(w)==len(r)==N and all(math.isfinite(v)and v>0 for v in w+r)and h>0,'positive saved scales')
 D=b['defect_coefficients'];F=b['forcing_coefficients'];require(len(D)==4 and len(F)==5 and all(len(m)==N and all(len(row)==N for row in m)for m in D)and all(len(m)==N and all(len(row)==1 for row in m)for m in F),'full actual coefficient shapes')
 dn=[]
 for k,m in enumerate(D):
  rows=[]
  for i,row in enumerate(m):
   rows.append(math.fsum(upper(v)*w[j]/w[i]for j,v in enumerate(row)));check()
  dn.append(max(rows))
 fn=[max(upper(m[i][0])/r[i]for i in range(N))for m in F]
 image=[upper(row[0])/r[i]for i,row in enumerate(b['whole_image'])];require(len(image)==N,'complete image shape')
 forecasts=[]
 for multiple in [1,2,4]:
  z=h*multiple;forecasts.append({'width_multiple':multiple,'sum_individual_defect_norms':math.fsum(dn[k]*z**k for k in range(4)),'sum_individual_forcing_radius_ratios':math.fsum(fn[k]*z**k for k in range(5))})
 return {'dimension':N,'current_halfwidth':h,'current_max_image_radius_ratio':max(image),'current_max_coordinate':image.index(max(image)),'defect_coefficient_norms':dn,'current_defect_terms':[dn[k]*h**k for k in range(4)],'forcing_coefficient_radius_norms':fn,'current_forcing_terms':[fn[k]*h**k for k in range(5)],'frozen_coefficient_forecasts':forecasts,'scope':'Floating diagnosis of saved enclosures only. The coefficient domains are the original fixed parameter/error boxes. Width or radius changes invalidate those domain hypotheses and require fresh full acquisitions; extrapolations are not bounds or proof gates.'}
