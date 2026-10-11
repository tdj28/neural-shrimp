"""Saved-only floating diagnostic; no theorem, new derivative, or gate repair."""
import math
import numpy as np
from .rigorous_eigenpair_locus import require
from .rigorous_eigenpair_march import decode

def values(rows):
 v=np.array([float(decode(x).mid())for x in rows]);require(np.isfinite(v).all(),'finite saved midpoint vector');return v

def exact_matrix(rows):
 a=np.array([[math.ldexp(int(v[0]),int(v[1]))for v in row]for row in rows]);require(np.isfinite(a).all(),'finite exact inverse float proposal');return a

def diagnose(proof,proposals):
 n=proof['shooting_dimension'];require(n==len(proof['weights'])and len(proposals)==3,'complete fixed source dimensions/proposals')
 w=np.array([float(x)if '/'not in x else float(__import__('fractions').Fraction(x))for x in proof['weights']]);require(np.isfinite(w).all()and np.all(w>0),'positive finite exact weights')
 rho=2.**proof['settings']['radius_exponent'];h=2.**proof['settings']['halfwidth_exponent'];v=values(proof['slope']);a=values(proof['curvature']);b=values(proof['cubic']);normal=np.array([float(__import__('fractions').Fraction(x))for x in proof['normal']]);center=values(proof['center'])
 C=[exact_matrix(x)for x in proof['inverse_polynomial_exact']];r=[values([row[0]for row in coeff])for coeff in proof['residual_coefficients']];rr=[np.array([float(decode(row[0]).rad())for row in coeff])for coeff in proof['residual_coefficients']]
 steps=[]
 for row in proposals:
  correction=values([v[0]for v in row['correction']]);g=values([v[0]for v in row['value']]);small=np.array([[float(decode(v).mid())for v in line]for line in row['condensed_border']]);sv=np.linalg.svd(small,compute_uv=False)
  steps.append({'iteration':row['iteration'],'weighted_correction_in_radii':float(np.max(np.abs(correction)/(rho*w))),'residual_inf':float(np.max(np.abs(g))),'small_singular_values':sv.tolist(),'condensed_condition':float(sv[0]/sv[-1])if sv[-1]>0 else None})
 order=[]
 for k in range(5):
  terms=[]
  for j in range(2):
   mid=C[j]@r[k];spread=np.abs(C[j])@rr[k];factor=h**(j+k)
   terms.append({'inverse_degree':j,'total_degree':j+k,'scaled_midpoint_max':float(np.max(np.abs(mid)*factor/(rho*w))),'scaled_radius_max':float(np.max(spread*factor/(rho*w)))})
  order.append({'residual_degree':k,'residual_mid_inf':float(np.max(np.abs(r[k]))),'residual_radius_inf':float(np.max(rr[k])),'forcing_proposals':terms})
 # No true Jacobian or inverse theorem follows from these finite norm summaries.
 cnorm=[float(np.max(np.sum(np.abs(x)*w[None,:],axis=1)/w))for x in C]
 tangent_length=float(np.linalg.norm(v));alignment=float(abs(normal@v)/(np.linalg.norm(normal)*tangent_length))
 return {'scope':'Noncertified saved numerical diagnosis only. Old failed gates unchanged; no shortened-domain or new-root result. Absolute interval-radius product proposals overestimate but floating arithmetic is not rigorous.', 'dimension':n,'parameter_center':proof['parameter_center'],'physical_center':center[-3:].tolist(),'physical_slope':v[-3:].tolist(),'physical_curvature':a[-3:].tolist(),'physical_cubic_coefficient':b[-3:].tolist(),'tangent_length':tangent_length,'gauge_alignment_cosine':alignment,'curvature_displacement_fraction':float(np.linalg.norm(a)*h/(2*tangent_length)),'cubic_displacement_fraction':float(np.linalg.norm(b)*h*h/tangent_length),'weighted_inverse_inf_proposals':cnorm,'newton_steps':steps,'forcing_orders':order,'old_contraction':proof['bound']['contraction'],'old_strict_inclusions':sum(proof['bound']['coordinate_inclusions']),'old_max_image_ratio':max(float(decode(x).mid())for x in proof['bound']['image_radius_ratio_upper'])}
