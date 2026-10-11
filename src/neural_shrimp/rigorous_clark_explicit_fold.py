"""Explicit paired tangent equations for native Clark daughter folds.

The inverse-free family has 4n+3 unknowns (z=(m,leak,q,gain),v=U_b),
external b, and equations H, H_U v+H_b, v_leak. No broad-box inverse of
H_U is required to evaluate this system. Full root and native fold gates
remain caller obligations; no claim follows merely from a finite Jacobian.
"""
from flint import arb,arb_mat
from . import rigorous_clark_amplitude_fold as A
from .rigorous_orbits import eye,midpoint_matrix,norm_inf_upper,vector_json,matrix_json,ball_json
require=A.require

def augment(family,unknown,b,n,observe=lambda *_:None):
 N=2*n+2;K=N-1;dim=N+K
 require(1<=n<=12 and len(unknown)==dim and all(v.is_finite()for v in unknown+[b]),'complete finite explicit paired tangent system')
 z=unknown[:N];v=unknown[N:];values=z+[b];direction=v+[arb(0),arb(1)]
 jets=[A.MixedJet.variable(x,N+1,i,direction[i])for i,x in enumerate(values)]
 h=family(jets[:-1],jets[-1]);require(len(h)==K and all(len(x.g)==len(x.m)==N+1 for x in h),'full base and mixed derivative columns')
 f=arb_mat([[x.f]for x in h]+[[x.d]for x in h]+[[v[n]]])
 jac=arb_mat([[x.g[k]for k in range(N)]+[arb(0)]*K for x in h]+[[x.m[k]for k in range(N)]+[x.g[k]for k in range(K)]for x in h]+[[arb(0)]*N+[arb(int(k==n))for k in range(K)]])
 hb=arb_mat([[x.g[N]]for x in h]+[[x.m[N]]for x in h]+[[arb(0)]])
 require(all(A.finite(x)for x in(f,jac,hb)),'finite complete explicit equations')
 data=dict(z=z,paired_tangent=arb_mat([[x]for x in v]),paired_jacobian=arb_mat([[x.g[k]for k in range(K)]for x in h]),paired_parameter=arb_mat([[x.g[N]]for x in h]))
 observe('explicit_amplitude_fold_return',dict(input=vector_json(unknown),squared_amplitude=ball_json(b),value=matrix_json(f),jacobian=matrix_json(jac),parameter_derivative=matrix_json(hb),paired_jacobian=matrix_json(data['paired_jacobian']),paired_tangent=matrix_json(data['paired_tangent'])))
 return f,jac,hb,data

def augmented(j,unknown,b,observe=lambda *_:None):
 n=len(j);N=2*n+2;require(len(unknown)==4*n+3 and unknown[N-1]>0 and 0<unknown[n]<=1 and all(v.is_exact()and v.is_finite()for row in j for v in row),'exact fixed physical map')
 return augment(lambda z,a:A.paired_family(j,z,a,observe),unknown,b,n,observe)

def conditional_organizer_anchor(family,z,n,lambda_leak,observe=lambda *_:None):
 """Enclose v only at an independently proved c3=0 signed-flip organizer.

Caller must authenticate that exact root, same phase/q normalization, and
its actual nonzero total lambda_leak. At b=0, projecting differentiated
paired equations on the normalized left mode gives c3+lambda_leak*v_leak=0.
Thus this known root has v_leak=0. Interval zero tests below are consistency,
not identity authority. All other v coordinates retain inverse uncertainty.
"""
 N=2*n+2;K=N-1;require(len(z)==N and lambda_leak.is_finite()and not lambda_leak.contains(0),'nonzero physical organizer crossing authority')
 values=z+[arb(0)];jets=[A.MixedJet.variable(x,N+1,i)for i,x in enumerate(values)];h=family(jets[:-1],jets[-1]);require(len(h)==K,'all organizer paired equations')
 matrix=arb_mat([[x.g[k]for k in range(K)]for x in h]);rhs=arb_mat([[-x.g[N]]for x in h]);C=midpoint_matrix(midpoint_matrix(matrix).inv());D=eye(K)-C*matrix
 require(A.finite(D)and norm_inf_upper(D)<1,'point organizer paired inverse bound');inv=(C*matrix).inv()*C;v=inv*rhs
 require(A.finite(v)and all(x.f.contains(0)for x in h)and v[n,0].contains(0),'conditional known organizer consistency')
 known=list(z)+[v[i,0]for i in range(K)]
 report=dict(known=vector_json(known),paired_jacobian=matrix_json(matrix),rhs=matrix_json(rhs),inverse=matrix_json(inv),inverse_defect=matrix_json(D),inverse_bound=ball_json(norm_inf_upper(D)),lambda_leak=ball_json(lambda_leak),scope='Conditional same proved c3=0 organizer with actual normalized phase and nonzero total lambda_leak. Exact identity follows c3+lambda_leak*v_leak=0; zero containment alone is insufficient.')
 observe('explicit_organizer_anchor',report);return known,report
