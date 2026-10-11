"""Full51 critical tubes; generic physical fold is a separate endpoint gate.

The exact explicit family supplies H=0,H_U v+H_b=0,v_leak=0. A full51
inverse proves only that square family, not H_U invertibility. Native
primitive6/simple+1/complement may be proved conditionally from these exact
identities. Generic physical fold curvature requires the separate validated
paired inverse at a strictly positive amplitude endpoint. b=0 is degenerate.
"""
from flint import arb,arb_mat
from .rigorous_orbits import eye,midpoint_matrix,norm_inf_upper,ball_json,vector_json,matrix_json
from .rigorous_clark_amplitude_fold import require,finite
from . import rigorous_clark_amplitude_fold as A
from . import rigorous_clark_amplitude_tube as T


def col(v):return arb_mat([[x]for x in v])
column=col

def vec(v):return [v[i,0]for i in range(v.nrows())]


def tube(evaluate,center,parameter,halfwidth,radii,*,observe=lambda *_:None,tick=lambda:None):
 """Uniform full root for general F(z,b), retaining its explicit b derivative.

Along p(d)=c+v*d, F(p(d),b0+d)=F0+(Jp*v+Fp_b)*d by the
mean-value theorem. Jp/Fp_b use the complete predictor parameter domain;
Jw uses predictor+E at that same full parameter domain.
"""
 N=len(center);require(2<=N<=51 and len(radii)==N and all(x.is_exact()and x.is_finite()for x in center+[parameter,halfwidth]+radii)and halfwidth>0 and all(x>0 for x in radii),'exact complete amplitude tube geometry')
 f0,J0,h0,_=evaluate(center,parameter);tick();require(J0.nrows()==J0.ncols()==N and f0.nrows()==h0.nrows()==N and f0.ncols()==h0.ncols()==1 and finite(f0)and finite(J0)and finite(h0),'complete center F/J/Fb')
 C=midpoint_matrix(midpoint_matrix(J0).inv());v=[x.mid()for x in vec(-C*h0)];d=arb(0,halfwidth);p=[center[i]+v[i]*d for i in range(N)];E=[arb(0,r)for r in radii];whole=[p[i]+E[i]for i in range(N)]
 _,Jp,hp,_=evaluate(p,parameter+d);tick();_,Jw,hw,data=evaluate(whole,parameter+d);tick()
 require(Jp.nrows()==Jp.ncols()==Jw.nrows()==Jw.ncols()==N and hp.nrows()==N and hp.ncols()==1 and all(finite(m)for m in(Jp,hp,Jw,hw)),'complete predictor and whole F/J/Fb')
 D=eye(N)-C*Jw;residual=-C*f0-C*(Jp*col(v)+hp)*d;image=residual+D*col(E)
 q=max((sum((abs(D[i,k]).upper()*radii[k]for k in range(N)),arb(0))/radii[i]).upper()for i in range(N));checks=[bool(abs(image[i,0]).upper()<radii[i])for i in range(N)]
 passed=bool(all(checks)and q.is_finite()and q<1 and finite(image));ends=[[center[i]+v[i]*sign*halfwidth+image[i,0]for i in range(N)]for sign in(-1,1)]
 report=dict(passed=passed,dimension=N,center=vector_json(center),slope=vector_json(v),parameter=ball_json(parameter),halfwidth=ball_json(halfwidth),radii=vector_json(radii),actual_parameter_box=ball_json(parameter+d),actual_error=vector_json(E),predictor=vector_json(p),whole=vector_json(whole),center_value=matrix_json(f0),center_jacobian=matrix_json(J0),center_parameter_derivative=matrix_json(h0),predictor_jacobian=matrix_json(Jp),predictor_parameter_derivative=matrix_json(hp),whole_jacobian=matrix_json(Jw),whole_parameter_derivative=matrix_json(hw),preconditioner=matrix_json(C),defect=matrix_json(D),preconditioned_residual=matrix_json(residual),image_correction=matrix_json(image),contraction=ball_json(q),coordinate_inclusions=checks,endpoints=[vector_json(x)for x in ends],scope='Uniform unique full augmented root for each exact squared amplitude in the closed interval. Mathematical existence requires passed=true; exact organizer/previous-root joins remain separate full-coordinate authority.')
 observe('explicit_amplitude_tube_return',report);return whole,v,data,Jw,hw,ends,report


def join(box,known_parameter,center,slope,parameter,halfwidth,radii):
 """Same-gauge full-root projection; b is external, not the gain coordinate."""
 N=len(center);require(2<=N<=51 and all(len(v)==N for v in(box,slope,radii)),'complete root projection')
 offset=known_parameter-parameter;inside=bool(parameter-halfwidth<=known_parameter and known_parameter<=parameter+halfwidth);error=[box[i]-center[i]-slope[i]*offset for i in range(N)];checks=[bool(abs(error[i]).upper()<radii[i])for i in range(N)]
 return dict(passed=bool(inside and all(checks)),parameter_inside=inside,coordinates=sum(checks),coordinate_inclusions=checks,parameter=ball_json(known_parameter),error=vector_json(error),scope='Conditional full-coordinate identity of the caller-authenticated root with the uniform tube root. Closed parameter endpoints are allowed; all coordinate error inclusions are strict.')


def root_enclosure(evaluate,seed,*,newton_steps=4,radius_exponent=-96,observe=lambda kind,row:None,tick=lambda:None):
 n=len(seed);require(1<=n<=51 and type(newton_steps)is int and 0<=newton_steps<=8 and type(radius_exponent)is int and -512<=radius_exponent<=-16,'fixed bounded augmented root settings')
 c=[arb(v).mid()for v in seed];require(all(v.is_finite()for v in c),'finite seed')
 for k in range(newton_steps):
  tick();h,j,_=evaluate(c);require(h.nrows()==n and h.ncols()==1 and j.nrows()==j.ncols()==n and finite(h)and finite(j),'complete Newton proposal')
  step=midpoint_matrix(j).inv()*h;new=[(c[i]-step[i,0]).mid()for i in range(n)];observe('newton_proposal',dict(index=k,before=vector_json(c),value=matrix_json(h),jacobian=matrix_json(j),step=matrix_json(step),after=vector_json(new)));c=new
 tick();h0,j0,_=evaluate(c);C=midpoint_matrix(j0.inv());r=arb((1,radius_exponent));box=[arb(v,r)for v in c];h,J,data=evaluate(box);tick()
 require(h0.nrows()==h.nrows()==n and h0.ncols()==h.ncols()==1 and J.nrows()==J.ncols()==n and finite(h0)and finite(J),'complete finite root enclosure')
 D=eye(n)-C*J;kappa=norm_inf_upper(D);delta=column([b-v for b,v in zip(box,c)]);image=-C*h0+D*delta;inclusions=[bool(box[i].contains_interior(c[i]+image[i,0]))for i in range(n)];passed=bool(all(inclusions)and kappa.is_finite()and kappa<1)
 report=dict(passed=passed,dimension=n,newton_steps=newton_steps,radius_exponent=radius_exponent,center=vector_json(c),box=vector_json(box),center_value=matrix_json(h0),whole_value=matrix_json(h),whole_jacobian=matrix_json(J),preconditioner=matrix_json(C),defect=matrix_json(D),actual_box_delta=matrix_json(delta),image_correction=matrix_json(image),coordinate_inclusions=inclusions,contraction_norm_upper=ball_json(kappa),scope='Strict full augmented Krawczyk/Banach root existence and local uniqueness only; nondegeneracy, primitive orbit and same-root spectral claims remain separate.')
 observe('explicit_fixed_gain_root',report);return box,J,data,report



def fixed_gain(evaluate,unknown,gain,n):
 """Unknown=(U,v,b), gain fixed; preserve every explicit derivative column."""
 K=2*n+1;N=K+1;dim=2*K+1
 require(1<=n<=12 and len(unknown)==dim,'complete fixed-gain explicit unknown')
 full=list(unknown[:K])+[gain]+list(unknown[K:-1]);b=unknown[-1]
 f,J,h,data=evaluate(full,b);columns=list(range(K))+list(range(N,dim))
 require(f.nrows()==J.nrows()==J.ncols()==h.nrows()==dim and len(columns)==dim-1,'complete fixed-gain original equations')
 jac=arb_mat([[J[i,k]for k in columns]+[h[i,0]]for i in range(dim)])
 return f,jac,dict(unknown=full,z=full[:N],b=b,paired_tangent=col(full[N:]),family_data=data)


def conditional_endpoint_fold(j,unknown,b,observe=lambda *_:None):
 """Caller supplies a proved explicit root at a strictly positive fixed b.

Unlike full51 invertibility, the following independent paired inverse is
actually sufficient to identify v as U_b at fixed gain. The old implicit
augmented equations then differentiate leak_b to obtain physical leak_bb.
Its zero-containment tests only check the already proved common identity.
"""
 n=len(j);N=2*n+2;require(len(unknown)==4*n+3 and b>0,'positive explicit endpoint root')
 z=unknown[:N];f,J,h,v=A.augmented(j,z,b,observe)
 require(all(x.contains(0)for row in f.tolist()for x in row)and all((v[i,0]-unknown[N+i]).contains(0)for i in range(N-1)),'same exact explicit paired solution and uniquely identified tangent consistency')
 result=T.fold_curvature(J,h,v,n,b)
 result['scope']='Conditional positive explicit51 endpoint with separately validated H_U inverse and actual nonzero fixed-gain leak_bb; b0 degeneracy and uniform branch nondegeneracy remain separate.'
 observe('explicit_endpoint_fold',result);return result
