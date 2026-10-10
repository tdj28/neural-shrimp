"""Fixed equal-pulse/equal-update paired comparator; no dynamics calls."""
import math
import numpy as np

def paired(g,rates,hidden=9):
 g=np.asarray(g);rates=np.asarray(rates)
 if g.ndim!=1 or g.dtype!=np.float64 or rates.shape!=(2,) or not np.isfinite(g).all() or not np.isfinite(rates).all() or min(rates)<=0 or not 0<hidden<len(g):raise ValueError('finite nondegenerate shape/rates')
 s=np.r_[rates[0]*g[:hidden],rates[1]*g[hidden:]];H=math.hypot(*s[:hidden]);O=math.hypot(*s[hidden:]);norm=math.hypot(H,O)
 if not all(math.isfinite(x) and x>0 for x in (H,O,norm)):raise ValueError('positive finite both block norms')
 kappa=1/8;cap=1/4;tau=(1/16)*min(H/O,O/H)
 v=np.r_[tau*(O/H)*s[:hidden],-tau*(H/O)*s[hidden:]]
 plus=kappa*s+v;minus=kappa*s-v;uplus=-s+plus;uminus=-s+minus
 fractions=np.array([[kappa+tau*O/H,kappa-tau*H/O],[kappa-tau*O/H,kappa+tau*H/O]])
 dot=lambda a,b:math.fsum(float(x)*float(y) for x,y in zip(a,b,strict=True))
 pulse=[math.hypot(*x) for x in (plus,minus)];update=[math.hypot(*x) for x in (uplus,uminus)]
 tol=2e-13*norm;orth=dot(s,v);km=1-math.hypot(1-kappa,tau)
 if not all(np.isfinite(x).all() for x in (s,v,plus,minus,uplus,uminus,fractions)):raise ValueError('finite derived arrays')
 qualified=bool(tau>0 and np.all((fractions>=0)&(fractions<=cap)) and abs(pulse[0]-pulse[1])<=tol and abs(update[0]-update[1])<=tol and abs(orth)<=2e-13*norm*math.hypot(*v) and 0<=km<=cap)
 return {'qualified':qualified,'kappa':kappa,'cap':cap,'tau':tau,'block_norms':[H,O],'block_fractions':fractions,'scaled_gradient':s,'orthogonal_component':v,'delta_plus':plus,'delta_minus':minus,'update_plus':uplus,'update_minus':uminus,'paired_pulse_norms':pulse,'paired_update_norms':update,'absolute_matching_tolerance':tol,'s_dot_v':orth,'uniform_equal_update_fraction':km,'uniform_equal_update_pulse_norm':km*norm,'update_direction_angle_degrees':2*math.atan2(tau,1-kappa)*180/math.pi,'first_order_loss_difference_plus_minus':2*dot(g,v),'gradient_alignment_v':dot(g,v)/(math.hypot(*g)*math.hypot(*v)),'scope':'Algebraic comparator feasibility and ordinary first-order predictor only; no applied intervention, response, geometry utility or rigorous interval claim.'}
