"""Conditional subset-radius repair of a completed native full51 tube.

No acquisition is performed here. The unchanged coefficient arrays must come
from the authenticated original full E. A smaller correction box can reuse
those derivative bounds; the caller separately qualifies its native orbit.
"""
from fractions import Fraction as F
from flint import arb,arb_mat
import clark_subset_radius_math as S
from neural_shrimp import rigorous_clark_correlated_tube as T
from neural_shrimp.rigorous_orbits import ball_json,vector_json,matrix_json
need=S.need

def pair(v):
 m,e=v;return F(m)*F(2)**e

def interval(v):
 p=ball_json(v);m,r=pair(p['mid']),pair(p['rad']);return m-r,m+r

def exact(v):
 lo,hi=interval(v);need(lo==hi,'exact finite dyadic input');return lo

def bound(v):
 need(v.is_finite(),'finite interval envelope');lo,hi=interval(v);return max(abs(lo),abs(hi))

def dyadic(v):
 need(type(v)is F and v.denominator&(v.denominator-1)==0,'dyadic proposal');r=arb((v.numerator,-(v.denominator.bit_length()-1)));need(exact(r)==v,'exact proposal retained at working precision');return r

def serial(v):
 if isinstance(v,F):return str(v)
 if isinstance(v,dict):return {k:serial(x)for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [serial(x)for x in v]
 return v

def envelopes(data,radii,known,known_b,c,s,a,b,h):
 n=len(radii);need(1<=n<=51 and all(len(v)==n for v in(known,c,s,a)),'all root/known coordinates')
 rr=[exact(v)for v in radii];cc,ss,aa=[[exact(x)for x in v]for v in(c,s,a)];d=exact(known_b)-exact(b);hh=exact(h)
 need(hh>0 and -hh<=d<=hh and all(v>0 for v in rr),'known scalar and positive radii')
 anchor=[max(abs(lo-cc[i]-ss[i]*d-aa[i]*d*d/2),abs(hi-cc[i]-ss[i]*d-aa[i]*d*d/2))/rr[i]for i,x in enumerate(known)for lo,hi in[interval(x)]]
 A=[[bound(data['whole_defect'][i,j])*rr[j]/rr[i]for j in range(n)]for i in range(n)];beta=[bound(data['residual'][i,0])/rr[i]for i in range(n)]
 return A,beta,anchor

def restrict(C,J,R,original,radii,known,known_b,c,s,a,b,h,*,observe=lambda *_:None,tick=lambda:None):
 """Exact subset proposal followed by a second outward full Krawczyk check.

Failure of the floating solver never establishes infeasibility. The entire
known box enters the strict join, and no correction coordinate may enlarge.
"""
 need(not original['passed'],'fallback only after a completed original root-bound failure')
 A,beta,anchor=envelopes(original,radii,known,known_b,c,s,a,b,h)
 observe('bounds',serial(dict(A=A,beta=beta,anchor=anchor,original_radii=[exact(v)for v in radii],known=vector_json(known),known_parameter=ball_json(known_b))))
 tick();proposal=S.propose(A,beta,anchor,observe=lambda p:observe('proposal',p));tick();verification=proposal['verification'];observe('verification',serial(dict(verification=verification,scope=proposal.get('scope','Exact Fraction inequalities decide; no new native call.'))))
 if not verification or not verification['passed']:return None
 new=[dyadic(exact(r)*z)for r,z in zip(radii,verification['proposal'],strict=True)];need(all(0<x<=y for x,y in zip(new,radii,strict=True)),'positive componentwise subset of actual full E')
 data=T.reduce(C,J,R,new,h,tick=tick);join=T.join(known,known_b,c,s,a,b,h,new);d=arb(0,h)
 whole=[c[i]+s[i]*d+a[i]*d*d/2+arb(0,new[i])for i in range(len(c))]
 ends=[[c[i]+s[i]*sign*h+a[i]*h*h/2+data['image'][i,0]for i in range(len(c))]for sign in(-1,1)]
 proof=dict(T.serialize(data),center=vector_json(c),slope=vector_json(s),curvature=vector_json(a),parameter=ball_json(b),halfwidth=ball_json(h),radii=vector_json(new),actual_error=vector_json([arb(0,r)for r in radii]),acquisition_radii=vector_json(radii),whole=vector_json(whole),inverse_coefficients=[matrix_json(m)for m in C],residual_coefficients=[matrix_json(m)for m in R],jacobian_coefficients=[matrix_json(m)for m in J],endpoints=[vector_json(v)for v in ends],subset_verification=serial(verification),known_join=join,scope='Same acquired full E bounds reused on a strictly positive componentwise subset; no derivative domain enlarged. Exact Fraction and outward interval root tests plus complete known-root join all mandatory. Native primitive/phase/complement gates are separate fresh acquisitions.')
 observe('reduced',proof)
 if not(data['passed']and join['passed']and join['coordinates']==len(c)):return None
 return whole,ends,data,proof,new
