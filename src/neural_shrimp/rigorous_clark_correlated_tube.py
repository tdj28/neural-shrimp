"""Full51 common-parameter tube reduction; native/root identity is separate.

F(p(d)+e,b0+d) uses the same explicit fold equations as the MixedJet code.
An affine finite inverse cancels common parameter dependence before interval
bounding. All error coordinates are retained in J0/J1 and whole-domain J2.
"""
from flint import arb,arb_mat
from .rigorous_orbits import eye,midpoint_matrix,ball_json,vector_json,matrix_json
from .rigorous_clark_amplitude_fold import require,finite
from . import rigorous_clark_correlated_path as P


def col(v):return arb_mat([[x]for x in v])
def vec(v):return [v[i,0]for i in range(v.nrows())]

def powers(h):
 require(h.is_finite()and h.is_exact()and h>0,'positive exact halfwidth')
 h2=h*h;h3=h2*h;h4=h2*h2
 return [arb(1),arb(0,h),arb(h2/2,h2/2),arb(0,h3),arb(h4/2,h4/2)]

def reduce(C,J,R,radii,h,*,tick=lambda:None):
 """Conditional coefficient reduction with full noncommuting products.

Caller authenticates J0(E),J1(E),J2(I,E) and R0,R1,R2,R3(I) for the
same predictor, actual full E and explicit external b. C is finite exact.
"""
 n=len(radii);require(1<=n<=51 and len(C)==2 and len(J)==3 and len(R)==4,'bounded complete coefficient lists')
 require(all(m.nrows()==m.ncols()==n and finite(m)for m in C+J)and all(m.nrows()==n and m.ncols()==1 and finite(m)for m in R),'complete finite coefficient arrays')
 require(all(v.is_finite()and v.is_exact()and v>0 for v in radii)and all(x.is_exact()for m in C for row in m.tolist()for x in row),'exact positive radii and finite inverse proposal')
 D=[eye(n)-C[0]*J[0],-C[0]*J[1]-C[1]*J[0],-C[0]*J[2]-C[1]*J[1],-C[1]*J[2]];tick()
 forcing=[arb_mat(n,1)for _ in range(5)]
 for i in range(2):
  for k in range(4):forcing[i+k]-=C[i]*R[k]
 ds=powers(h);defect=sum((D[k]*ds[k]for k in range(4)),arb_mat(n,n));residual=sum((forcing[k]*ds[k]for k in range(5)),arb_mat(n,1));error=col([arb(0,r)for r in radii]);image=residual+defect*error;tick()
 require(all(finite(x)for x in(defect,residual,image)),'finite full correlated image')
 q=max((sum((abs(defect[i,k]).upper()*radii[k]for k in range(n)),arb(0))/radii[i]).upper()for i in range(n));included=[bool(abs(image[i,0]).upper()<radii[i])for i in range(n)]
 return dict(passed=bool(q<1 and all(included)),dimension=n,contraction=q,coordinate_inclusions=included,defect_coefficients=D,forcing_coefficients=forcing,whole_defect=defect,residual=residual,image=image)


def serialize(data):
 return dict(passed=data['passed'],dimension=data['dimension'],contraction=ball_json(data['contraction']),coordinate_inclusions=data['coordinate_inclusions'],defect_coefficients=[matrix_json(m)for m in data['defect_coefficients']],forcing_coefficients=[matrix_json(m)for m in data['forcing_coefficients']],whole_defect=matrix_json(data['whole_defect']),preconditioned_residual=matrix_json(data['residual']),image_correction=matrix_json(data['image']))


def tube(j,c,s,a,b,h,radii,*,acquire=P.coefficients,observe=lambda *_:None,tick=lambda:None):
 """Four authenticated acquisitions; residual E=0, Jacobian full same E.

Exact c,s,a are finite proposals, never derivative existence authority.
The native caller still proves organizer/previous-root and all phase gates.
"""
 n=len(c);require(n==4*len(j)+3 and n<=51 and all(len(x)==n for x in(s,a,radii)),'complete actual fold predictor')
 powers(h);require(all(x.is_finite()and x.is_exact()for x in c+s+a+[b])and all(r.is_finite()and r.is_exact()and r>0 for r in radii),'exact finite tube data')
 d=arb(0,h);zero=[arb(0)]*n;E=[arb(0,r)for r in radii]
 fc,jc=acquire(j,c,s,a,zero,b,arb(0),tick=tick,observe=observe)
 fw,_=acquire(j,c,s,a,zero,b,d,tick=tick,observe=observe)
 _,j0=acquire(j,c,s,a,E,b,arb(0),tick=tick,observe=observe)
 _,jw=acquire(j,c,s,a,E,b,d,tick=tick,observe=observe)
 C0=midpoint_matrix(midpoint_matrix(jc[0]).inv());C1=midpoint_matrix(-C0*midpoint_matrix(jc[1])*C0)
 R=[col([fc[i,k]if k<3 else fw[i,3]for i in range(n)])for k in range(4)];J=[j0[0],j0[1],jw[2]];data=reduce([C0,C1],J,R,radii,h,tick=tick)
 image=data['image'];whole=[c[i]+s[i]*d+a[i]*d*d/2+E[i]for i in range(n)];ends=[[c[i]+s[i]*sign*h+a[i]*h*h/2+image[i,0]for i in range(n)]for sign in(-1,1)]
 report=dict(serialize(data),center=vector_json(c),slope=vector_json(s),curvature=vector_json(a),parameter=ball_json(b),halfwidth=ball_json(h),radii=vector_json(radii),actual_error=vector_json(E),whole=vector_json(whole),inverse_coefficients=[matrix_json(C0),matrix_json(C1)],residual_coefficients=[matrix_json(m)for m in R],jacobian_coefficients=[matrix_json(m)for m in J],endpoints=[vector_json(v)for v in ends],scope='Uniform full51 root inequalities only on the authenticated native curved correction tube. Strict all51 organizer/predecessor projection, native primitive6/simple+1/complement and normalized partner/phase identity remain separate caller gates. No H_U invertibility or generic fold follows merely from full51 invertibility.')
 observe('correlated_explicit_tube_return',report);return whole,ends,data,report


def join(known,known_b,c,s,a,b,h,radii):
 n=len(c);require(1<=n<=51 and all(len(v)==n for v in(known,s,a,radii)),'complete all-coordinate curved join')
 require(all(x.is_finite()and x.is_exact()for x in c+s+a+[b,h])and h>0 and all(r.is_finite()and r.is_exact()and r>0 for r in radii),'finite exact curved join geometry')
 require(known_b.is_finite()and all(v.is_finite()for v in known),'finite caller-authenticated known root')
 delta=known_b-b;inside=bool(b-h<=known_b and known_b<=b+h);error=[known[i]-c[i]-s[i]*delta-a[i]*delta*delta/2 for i in range(n)];checks=[bool(abs(error[i]).upper()<radii[i])for i in range(n)]
 return dict(passed=bool(inside and all(checks)),parameter_inside=inside,coordinate_inclusions=checks,coordinates=sum(checks),error=vector_json(error),known_parameter=ball_json(known_b),scope='Conditional strict complete-coordinate identity of an authenticated known root with the unique curved tube root; the caller must separately establish the known root for the same full equations and exact native map/phase.')
