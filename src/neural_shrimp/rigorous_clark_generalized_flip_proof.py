"""Bounded exact-real root and conditional generalized-flip proof arithmetic.

The conditional coefficient/spectrum routines require an exact root of all
augmented equations in their input box. Consistency tests do not supply it.
No target inputs or model selection live in this module.
"""
from flint import arb,arb_mat
from . import rigorous_clark_generalized_flip as A
from .rigorous_orbits import eye,midpoint_matrix,norm_inf_upper,ball_json,vector_json,matrix_json
require=A.require


def finite(matrix):return all(v.is_finite()for row in matrix.tolist()for v in row)
def column(v):return arb_mat([[x]for x in v])
def vec(v):return [v[i,0]for i in range(v.nrows())]


def inverse(matrix,tag,observe=lambda kind,row:None):
 n=matrix.nrows();require(n==matrix.ncols()and 1<=n<=34 and finite(matrix),'bounded finite square inverse')
 pre=midpoint_matrix(matrix.inv());defect=eye(n)-pre*matrix;bound=norm_inf_upper(defect)
 require(bound.is_finite()and bound<1,'strict validated inverse '+tag);answer=(pre*matrix).inv()*pre
 identity=answer*matrix
 require(finite(answer)and all(identity[i,j].contains(int(i==j))for i in range(n)for j in range(n)),'validated inverse identity '+tag)
 observe('validated_inverse',dict(tag=tag,matrix=matrix_json(matrix),preconditioner=matrix_json(pre),defect=matrix_json(defect),bound=ball_json(bound),inverse=matrix_json(answer)));return answer


def root_enclosure(evaluate,seed,*,newton_steps=4,radius_exponent=-96,observe=lambda kind,row:None,tick=lambda:None):
 n=len(seed);require(1<=n<=34 and type(newton_steps)is int and 0<=newton_steps<=8 and type(radius_exponent)is int and -512<=radius_exponent<=-16,'fixed bounded augmented root settings')
 c=[arb(v).mid()for v in seed];require(all(v.is_finite()for v in c),'finite seed')
 for k in range(newton_steps):
  tick();h,j,_=evaluate(c);require(h.nrows()==n and h.ncols()==1 and j.nrows()==j.ncols()==n and finite(h)and finite(j),'complete Newton proposal')
  step=midpoint_matrix(j).inv()*h;new=[(c[i]-step[i,0]).mid()for i in range(n)];observe('newton_proposal',dict(index=k,before=vector_json(c),value=matrix_json(h),jacobian=matrix_json(j),step=matrix_json(step),after=vector_json(new)));c=new
 tick();h0,j0,_=evaluate(c);C=midpoint_matrix(j0.inv());r=arb((1,radius_exponent));box=[arb(v,r)for v in c];h,J,data=evaluate(box);tick()
 require(h0.nrows()==h.nrows()==n and h0.ncols()==h.ncols()==1 and J.nrows()==J.ncols()==n and finite(h0)and finite(J),'complete finite root enclosure')
 D=eye(n)-C*J;kappa=norm_inf_upper(D);delta=column([b-v for b,v in zip(box,c)]);image=-C*h0+D*delta;inclusions=[bool(box[i].contains_interior(c[i]+image[i,0]))for i in range(n)];passed=bool(all(inclusions)and kappa.is_finite()and kappa<1)
 report=dict(passed=passed,dimension=n,newton_steps=newton_steps,radius_exponent=radius_exponent,center=vector_json(c),box=vector_json(box),center_value=matrix_json(h0),whole_value=matrix_json(h),whole_jacobian=matrix_json(J),preconditioner=matrix_json(C),defect=matrix_json(D),actual_box_delta=matrix_json(delta),image_correction=matrix_json(image),coordinate_inclusions=inclusions,contraction_norm_upper=ball_json(kappa),scope='Strict full augmented Krawczyk/Banach root existence and local uniqueness only; nondegeneracy, primitive orbit and same-root spectral claims remain separate.')
 observe('root_enclosure',report);return box,J,data,report


def multiply(a,b):
 return [sum((a[i]*b[k-i]for i in range(k+1)),arb(0))for k in range(6)]


def conditional_normal_form(x,mat,q,p,evaluate,observe=lambda kind,row:None):
 """At a known exact c3=0 simple flip, bound its odd-gauge quintic."""
 n=len(x);require(1<=n<=16 and mat.nrows()==mat.ncols()==n and len(q)==len(p)==n,'full conditional normal form')
 require(all(v.contains(0)for v in vec(mat*column(q)+column(q))+vec(mat.transpose()*column(p)+column(p)))and (sum((p[i]*q[i]for i in range(n)),arb(0))-1).contains(0),'consistent known normalized critical identities')
 even=inverse(eye(n)-mat,'normal-even',observe);border=arb_mat([[mat[i,j]+int(i==j)for j in range(n)]+[-q[i]]for i in range(n)]+[p+[arb(0)]]);odd=inverse(border,'normal-odd',observe)
 h=[list(x),list(q)]+[[arb(0)for _ in range(n)]for _ in range(4)]
 def path():
  out=evaluate(h);require(len(out)==6 and all(len(v)==n and all(a.is_finite()for a in v)for v in out),'complete fifth-order path');observe('normal_form_path',dict(input=[vector_json(v)for v in h],output=[vector_json(v)for v in out]));return out
 h[2]=vec(even*column(path()[2]));third=odd*column([-v for v in path()[3]]+[arb(0)]);h[3]=vec(third)[:-1];c3=third[n,0];require(c3.contains(0),'known exact cubic-zero consistency')
 # Exact augmented root fixes c3=0, so no uncertain off-root cubic is inserted.
 h[4]=vec(even*column(path()[4]));fifth=odd*column([-v for v in path()[5]]+[arb(0)]);h[5]=vec(fifth)[:-1];c5=fifth[n,0]
 actual=path();reduced=[arb(0),arb(-1),arb(0),arb(0),arb(0),c5];power=[arb(1)]+[arb(0)]*5;expected=[[arb(0)for _ in range(n)]for _ in range(6)]
 for k in range(6):
  expected=[[expected[d][i]+power[d]*h[k][i]for i in range(n)]for d in range(6)];power=multiply(power,reduced)
 defect=[[actual[k][i]-expected[k][i]for i in range(n)]for k in range(6)];gauges=[sum((p[i]*h[k][i]for i in range(n)),arb(0))for k in(3,5)]
 consistent=all(v.contains(0)for row in defect for v in row)and all(v.contains(0)for v in gauges);nonzero=c5.is_finite()and not c5.contains(0)
 out=dict(passed=bool(consistent and nonzero),conditional_on_exact_cubic_zero=True,cubic_consistency=ball_json(c3),quintic=ball_json(c5),parameterization=[vector_json(v)for v in h],reduced=vector_json(reduced),invariance_defect=[vector_json(v)for v in defect],odd_gauges=vector_json(gauges),scope='Ordinary Taylor convention h(u)=x+qu+..., f(u)=-u+c5*u^5+O(u^6), p.h3=p.h5=0 at the supplied exact cubic-zero root. Validated homological inverses; consistency tests alone are not root existence.')
 observe('quintic_result',out);return out


def conditional_unfolding(J,mat,p,observe=lambda kind,row:None):
 """Physical (gain,leak) determinant at the same exact augmented root."""
 n=mat.nrows();N=2*n+2;require(1<=n<=16 and J.nrows()==J.ncols()==N and len(p)==n,'full same-root unfolding derivatives')
 critical=arb_mat([[J[i,j]for j in range(N-1)]for i in range(N-1)]);slope=-inverse(critical,'critical-gain-curve',observe)*column([J[i,N-1]for i in range(N-1)]);dc3=J[N-1,N-1]+sum((J[N-1,j]*slope[j,0]for j in range(N-1)),arb(0))
 xdelta=-inverse(mat-eye(n),'fixed-root-leak',observe)*column([J[i,n]for i in range(n)])
 lambda_delta=sum((p[i]*(J[n+i,n]+sum((J[n+i,j]*xdelta[j,0]for j in range(n)),arb(0)))for i in range(n)),arb(0));det=-lambda_delta*dc3
 passed=all(v.is_finite()and not v.contains(0)for v in (dc3,lambda_delta,det))
 out=dict(passed=passed,critical_gain_slope=matrix_json(slope),fixed_root_leak_slope=matrix_json(xdelta),critical_cubic_gain_derivative=ball_json(dc3),multiplier_leak_derivative=ball_json(lambda_delta),physical_determinant=ball_json(det),parameter_order=['gain','leak'],function_order=['lambda+1','c3'],scope='Conditional same-root physical two-parameter transversality. Determinant=-lambda_leak*(dc3/dgain along the exact simple-flip curve).')
 observe('unfolding_result',out);return out


def conditional_native_context(j,box,mat,q,*,max_power=512,observe=lambda kind,row:None):
 n=len(j);require(2<=n<=16 and len(box)==2*n+2 and type(max_power)is int and 1<=max_power<=4096,'bounded full native context');x,d,g=box[:n],box[n],box[-1];require(g>0 and 0<d<1,'positive actual physical parameters');orbit=[list(x)];product=eye(n);js=[]
 for phase in range(6):
  t=[v.tanh()for v in orbit[-1]];J=arb_mat([[(1-d)*int(i==k)+d*g*j[i][k]*(1-t[k]*t[k])for k in range(n)]for i in range(n)]);end=[(1-d)*orbit[-1][i]+d*g*sum((j[i][k]*t[k]for k in range(n)),arb(0))for i in range(n)];product=J*product;js.append(J);orbit.append(end);observe('native_context_phase',dict(phase=phase,state=vector_json(end),jacobian=matrix_json(J),product=matrix_json(product)))
 divisors=[]
 for k in(1,2,3):
  differences=[orbit[k][i]-x[i]for i in range(n)];ids=[i for i,v in enumerate(differences)if v.is_finite()and not v.contains(0)];divisors.append(dict(divisor=k,excluded=bool(ids),coordinates=ids,differences=vector_json(differences)))
 half=all((orbit[3][i]+x[i]).contains(0)for i in range(n));closed=all((orbit[6][i]-x[i]).contains(0)for i in range(n));square=product-mat*mat
 require(half and closed and all(v.contains(0)for row in square.tolist()for v in row),'known odd half-wave and native-square consistency')
 # A coordinate complement is merely an exact proposal; validate full frame.
 pivot=max(range(n),key=lambda i:float(abs(q[i].mid())));cols=[i for i in range(n)if i!=pivot];frame=arb_mat([[q[i]]+[arb(int(i==k))for k in cols]for i in range(n)]);inv=inverse(frame,'critical-frame',observe);T=inv*mat*frame
 require((T[0,0]+1).contains(0)and all(T[i,0].contains(0)for i in range(1,n)),'known exact critical plane consistency');quotient=arb_mat([[T[i,k]for k in range(1,n)]for i in range(1,n)]);power=quotient;bounds=[];witness=None;k=1
 while k<=max_power:
  bound=norm_inf_upper(power);bounds.append(dict(power=k,bound=ball_json(bound)))
  if bound.is_finite()and bound<1:witness=k;break
  if not bound.is_finite():break
  power=power*power;k*=2
 out=dict(passed=bool(all(v['excluded']for v in divisors)and witness is not None),primitive_period=6,divisors=divisors,orbit_enclosures=[vector_json(v)for v in orbit],native_jacobians=[matrix_json(v)for v in js],native_monodromy=matrix_json(product),signed_monodromy=matrix_json(mat),square_consistency=matrix_json(square),frame=matrix_json(frame),frame_inverse=matrix_json(inv),quotient=matrix_json(quotient),power_bounds=bounds,witness_power=witness,complement_dimension=n-1,scope='Conditional on exact G=-F^3 fixed root with critical -1. Exact odd F implies F^6(x)=x and DF^6=(DG)^2; proper divisors excluded and all signed complementary modes strictly interior. Native critical multiplier is +1, not a native flip.')
 observe('native_context_result',out);return out
