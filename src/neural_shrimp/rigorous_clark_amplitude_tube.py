"""Conditional squared-amplitude tubes and native daughter-fold certificates.

Exact organizer/root authority belongs to the caller. Zero containment is
only a consistency check; all returned joins compare complete coordinates.
"""
from flint import arb,arb_mat
from .rigorous_orbits import eye,midpoint_matrix,ball_json,vector_json,matrix_json,norm_inf_upper
from .rigorous_clark_generalized_flip_proof import inverse
from .rigorous_clark_amplitude_fold import require,finite


def col(v):return arb_mat([[x]for x in v])
def vec(v):return [v[i,0]for i in range(v.nrows())]


def tube(evaluate,center,parameter,halfwidth,radii,*,observe=lambda *_:None,tick=lambda:None):
 """Uniform full root for general F(z,b), retaining its explicit b derivative.

Along p(d)=c+v*d, F(p(d),b0+d)=F0+(Jp*v+Fp_b)*d by the
mean-value theorem. Jp/Fp_b use the complete predictor parameter domain;
Jw uses predictor+E at that same full parameter domain.
"""
 N=len(center);require(2<=N<=26 and len(radii)==N and all(x.is_exact()and x.is_finite()for x in center+[parameter,halfwidth]+radii)and halfwidth>0 and all(x>0 for x in radii),'exact complete amplitude tube geometry')
 f0,J0,h0,_=evaluate(center,parameter);tick();require(J0.nrows()==J0.ncols()==N and f0.nrows()==h0.nrows()==N and f0.ncols()==h0.ncols()==1 and finite(f0)and finite(J0)and finite(h0),'complete center F/J/Fb')
 C=midpoint_matrix(midpoint_matrix(J0).inv());v=[x.mid()for x in vec(-C*h0)];d=arb(0,halfwidth);p=[center[i]+v[i]*d for i in range(N)];E=[arb(0,r)for r in radii];whole=[p[i]+E[i]for i in range(N)]
 _,Jp,hp,_=evaluate(p,parameter+d);tick();_,Jw,hw,data=evaluate(whole,parameter+d);tick()
 require(Jp.nrows()==Jp.ncols()==Jw.nrows()==Jw.ncols()==N and hp.nrows()==N and hp.ncols()==1 and all(finite(m)for m in(Jp,hp,Jw,hw)),'complete predictor and whole F/J/Fb')
 D=eye(N)-C*Jw;residual=-C*f0-C*(Jp*col(v)+hp)*d;image=residual+D*col(E)
 q=max((sum((abs(D[i,k]).upper()*radii[k]for k in range(N)),arb(0))/radii[i]).upper()for i in range(N));checks=[bool(abs(image[i,0]).upper()<radii[i])for i in range(N)]
 passed=bool(all(checks)and q.is_finite()and q<1 and finite(image));ends=[[center[i]+v[i]*sign*halfwidth+image[i,0]for i in range(N)]for sign in(-1,1)]
 report=dict(passed=passed,dimension=N,center=vector_json(center),slope=vector_json(v),parameter=ball_json(parameter),halfwidth=ball_json(halfwidth),radii=vector_json(radii),actual_parameter_box=ball_json(parameter+d),actual_error=vector_json(E),predictor=vector_json(p),whole=vector_json(whole),center_value=matrix_json(f0),center_jacobian=matrix_json(J0),center_parameter_derivative=matrix_json(h0),predictor_jacobian=matrix_json(Jp),predictor_parameter_derivative=matrix_json(hp),whole_jacobian=matrix_json(Jw),whole_parameter_derivative=matrix_json(hw),preconditioner=matrix_json(C),defect=matrix_json(D),preconditioned_residual=matrix_json(residual),image_correction=matrix_json(image),contraction=ball_json(q),coordinate_inclusions=checks,endpoints=[vector_json(x)for x in ends],scope='Uniform unique full augmented root for each exact squared amplitude in the closed interval. Mathematical existence requires passed=true; exact organizer/previous-root joins remain separate full-coordinate authority.')
 observe('amplitude_tube_return',report);return whole,v,data,Jw,hw,ends,report


def join(box,known_parameter,center,slope,parameter,halfwidth,radii):
 """Same-gauge full-root projection; b is external, not the gain coordinate."""
 N=len(center);require(2<=N<=26 and all(len(v)==N for v in(box,slope,radii)),'complete root projection')
 offset=known_parameter-parameter;inside=bool(parameter-halfwidth<=known_parameter and known_parameter<=parameter+halfwidth);error=[box[i]-center[i]-slope[i]*offset for i in range(N)];checks=[bool(abs(error[i]).upper()<radii[i])for i in range(N)]
 return dict(passed=bool(inside and all(checks)),parameter_inside=inside,coordinates=sum(checks),coordinate_inclusions=checks,parameter=ball_json(known_parameter),error=vector_json(error),scope='Conditional full-coordinate identity of the caller-authenticated root with the uniform tube root. Closed parameter endpoints are allowed; all coordinate error inclusions are strict.')


def fixed_gain(evaluate,unknown,gain):
 """Change only which physical variable is external: unknown=(U,b), gain fixed."""
 z=list(unknown[:-1])+[gain];b=unknown[-1];f,J,h,v=evaluate(z,b);N=len(z)
 return f,arb_mat([[J[i,k]for k in range(N-1)]+[h[i,0]]for i in range(N)]),dict(z=z,b=b,paired_tangent=v,original_jacobian=J,parameter_derivative=h)


def fold_curvature(jac,parameter_derivative,paired_tangent,n,b):
 N=2*n+2;require(jac.nrows()==jac.ncols()==N and parameter_derivative.nrows()==N and parameter_derivative.ncols()==1 and paired_tangent.nrows()==N-1 and paired_tangent.ncols()==1,'complete physical fold curvature')
 # The last augmented equation is leak_b. Along H(U,g,b)=0 at fixed g,
 # its total derivative is leak_bb; no differentiation of a finite inverse.
 bb=parameter_derivative[N-1,0]+sum((jac[N-1,k]*paired_tangent[k,0]for k in range(N-1)),arb(0));aa=4*b*bb
 return dict(passed=bool(b>0 and bb.is_finite()and not bb.contains(0)),leak_bb=ball_json(bb),leak_aa=ball_json(aa),scope='Conditional on an exact paired fold root: total leak_bb at fixed physical gain. For b>0, leak_aa=4*b*leak_bb is nonzero; this does not establish primitive period or complementary stability.')


def conditional_native(j,z,b,v,*,max_power=512,observe=lambda *_:None):
 """Full primitive6/+1/complement from an authenticated exact paired fold.

Exact H=0 gives G(x+)=x- and G(x-)=x+. Oddness of F permits two
three-phase native paths and exact sign reboxing. At leak_b=0, their
amplitude derivatives are an exact +1 mode of F6. No identity is inferred
from the subsequent interval consistency checks.
"""
 n=len(j);require(2<=n<=12 and len(z)==2*n+2 and v.nrows()==2*n+1 and v.ncols()==1 and b>=0 and type(max_power)is int and 1<=max_power<=4096,'complete bounded paired native context')
 m,d,q,g=z[:n],z[n],z[n+1:-1],z[-1];require(g>0 and 0<d<1 and all(len(row)==n for row in j),'physical paired map');a=b.sqrt()
 require((sum((x*x for x in q),arb(0))-1).contains(0)and v[n,0].contains(0),'known normalization and fold consistency')
 starts=[[m[i]+sign*a*q[i]for i in range(n)]for sign in(1,-1)]
 modes=[[sign*(q[i]+2*b*v[n+1+i,0])+2*a*v[i,0]for i in range(n)]for sign in(1,-1)]
 paths=[];blocks=[]
 for partner,x in enumerate(starts):
  path=[x];js=[]
  for phase in range(3):
   t=[u.tanh()for u in path[-1]];J=arb_mat([[(1-d)*int(i==k)+d*g*j[i][k]*(1-t[k]*t[k])for k in range(n)]for i in range(n)]);end=[(1-d)*path[-1][i]+d*g*sum((j[i][k]*t[k]for k in range(n)),arb(0))for i in range(n)];path.append(end);js.append(J)
   observe('paired_native_phase',dict(partner=partner,phase=phase,input=vector_json(path[-2]),state=vector_json(end),jacobian=matrix_json(J)))
  paths.append(path);blocks.append(js)
 require(all((paths[s][3][i]+starts[1-s][i]).contains(0)for s in(0,1)for i in range(n)),'exact paired native F3 identity consistency')
 transport=[]
 for s in(0,1):
  P=blocks[s][2]*blocks[s][1]*blocks[s][0];r=vec(P*col(modes[s])+col(modes[1-s]));require(all(x.contains(0)for x in r),'literal native F3 partner mode consistency');transport.append(vector_json(r))
 contexts=[];orbits=[]
 for s in(0,1):
  orbit=paths[s][:3]+[[-x for x in phase]for phase in paths[1-s][:3]];js=blocks[s]+blocks[1-s];orbits.append(orbit)
  for shift in(0,1):
   cycle=orbit[shift:]+orbit[:shift];native=js[shift:]+js[:shift];mode=modes[s]if shift==0 else vec(js[0]*col(modes[s]));normalization=sum((x*x for x in mode),arb(0)).sqrt();require(normalization.is_finite()and normalization>0,'positive native mode normalization');mode=[x/normalization for x in mode]
   M=eye(n)
   for J in native:M=J*M
   require(all(x.contains(0)for x in vec(M*col(mode)-col(mode))),'known native critical +1 consistency')
   pivot=max(range(n),key=lambda k:float(abs(mode[k].mid())));cols=[k for k in range(n)if k!=pivot];frame=arb_mat([[mode[i]]+[arb(int(i==k))for k in cols]for i in range(n)]);R=inverse(frame,'paired-native-frame',observe);T=R*M*frame
   require((T[0,0]-1).contains(0)and all(T[i,0].contains(0)for i in range(1,n)),'known exact native critical plane consistency');Q=arb_mat([[T[i,k]for k in range(1,n)]for i in range(1,n)]);power=Q;bounds=[];witness=None;k=1
   while k<=max_power:
    bound=norm_inf_upper(power);bounds.append(dict(power=k,bound=ball_json(bound)))
    if bound.is_finite()and bound<1:witness=k;break
    if not bound.is_finite():break
    power=power*power;k*=2
   divisors=[]
   for k in(1,2,3):
    differences=[cycle[k][i]-cycle[0][i]for i in range(n)];ids=[i for i,x in enumerate(differences)if x.is_finite()and not x.contains(0)];divisors.append(dict(divisor=k,excluded=bool(ids),coordinates=ids,differences=vector_json(differences)))
   contexts.append(dict(partner=s,shift=shift,passed=bool(witness is not None and all(r['excluded']for r in divisors)),orbit=[vector_json(x)for x in cycle],mode=vector_json(mode),normalization=ball_json(normalization),native_monodromy=matrix_json(M),frame=matrix_json(frame),frame_inverse=matrix_json(R),quotient=matrix_json(Q),power_bounds=bounds,witness_power=witness,divisors=divisors))
 gap=4*b*sum((x*x for x in q),arb(0));orientation=[sum((modes[s][i]*(-1 if s else 1)*q[i]for i in range(n)),arb(0))for s in(0,1)]
 out=dict(passed=bool(all(c['passed']for c in contexts)),positive_amplitude=bool(b>0 and gap>0),squared_partner_distance=ball_json(gap),partner_orientation=vector_json(orientation),partner_transport_consistency=transport,contexts=contexts,complement_dimension=n-1,primitive_period=6,critical_native_multiplier=1,scope='Conditional exact native6 fold contexts for both literal G partners and their adjacent native phases. All proper divisors excluded, single +1 critical mode and all other modes strictly interior. Amplitude zero remains the known organizer, not a distinct daughter pair; no native12 assertion.')
 observe('paired_native_result',out);return out
