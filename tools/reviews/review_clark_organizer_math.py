"""Independent interval reductions for one exact signed Clark organizer.

No producer proof helper is imported. The caller supplies authenticated shared
augmented values/Jacobians; native orbit and fifth-order paths are reconstructed
here via cosh derivatives and exp/reciprocal Taylor composition respectively.
"""
from flint import arb,arb_mat


def need(ok,message):
 if not ok:raise ValueError(message)
def eye(n):return arb_mat([[int(i==j)for j in range(n)]for i in range(n)])
def col(v):return arb_mat([[x]for x in v])
def vec(v):return [v[i,0]for i in range(v.nrows())]
def midpoint(m):return arb_mat([[x.mid()for x in row]for row in m.tolist()])
def norm(m):return max(sum((abs(x).upper()for x in row),arb(0)).upper()for row in m.tolist())
def finite(m):return all(x.is_finite()for row in m.tolist()for x in row)
def decode(v):return arb(arb(tuple(v['mid'])),arb(tuple(v['rad'])))
def matrix(v):return arb_mat([[decode(x)for x in row]for row in v])


def inverse(a,emit=lambda *_:None):
 n=a.nrows();need(n==a.ncols()and n<=34 and finite(a),'finite bounded inverse')
 R=midpoint(midpoint(a).inv());D=eye(n)-R*a;rho=norm(D);need(rho.is_finite()and rho<1,'inverse Neumann defect')
 inv=(R*a).inv()*R;left=inv*a;right=a*inv
 need(finite(inv)and all(left[i,j].contains(int(i==j))and right[i,j].contains(int(i==j))for i in range(n)for j in range(n)),'two-sided inverse validation')
 emit('inverse',dict(matrix=a,proposal=R,defect=D,bound=rho,inverse=inv));return inv


def krawczyk(center,box,value,jacobian,emit=lambda *_:None,exact_bounds=None):
 n=len(center);need(n==len(box)==jacobian.nrows()==jacobian.ncols()and value.nrows()==n and value.ncols()==1,'complete Krawczyk shape')
 need(all(b.contains(c)for b,c in zip(box,center))and finite(value)and finite(jacobian),'finite center/domain')
 C=midpoint(midpoint(jacobian).inv());D=eye(n)-C*jacobian;delta=col([b-c for b,c in zip(box,center)]);image=-C*value+D*delta;rho=norm(D)
 if exact_bounds is None:exact_bounds=[(b.lower(),b.upper())for b in box]
 need(len(exact_bounds)==n and all(lo.is_exact()and hi.is_exact()and lo<=c<=hi for (lo,hi),c in zip(exact_bounds,center)),'exact actual acquired box endpoints')
 strict=[bool(lo<(c+image[i,0]).lower()and(c+image[i,0]).upper()<hi)for i,((lo,hi),c)in enumerate(zip(exact_bounds,center))]
 report=dict(passed=bool(all(strict)and rho.is_finite()and rho<1),center=center,box=box,exact_domain_endpoints=exact_bounds,preconditioner=C,defect=D,image=image,coordinate_inclusions=strict,contraction=rho)
 emit('krawczyk',report);need(report['passed'],'all strict independent root inclusions');return report


def tanh_series(x):
 """tanh(x)=1-2/(1+exp(2x)); ordinary coefficients, independent recurrence."""
 n=len(x);e=[(2*x[0]).exp()]+[arb(0)]*(n-1)
 for k in range(1,n):e[k]=sum((2*i*x[i]*e[k-i]for i in range(1,k+1)),arb(0))/k
 den=list(e);den[0]+=1;inv=[1/den[0]]+[arb(0)]*(n-1)
 for k in range(1,n):inv[k]=-sum((den[i]*inv[k-i]for i in range(1,k+1)),arb(0))/den[0]
 return [1-2*inv[0]]+[-2*v for v in inv[1:]]


def signed_path(coeff,j,g,d,emit=lambda *_:None):
 n=len(j);need(len(coeff)==6 and all(len(v)==n for v in coeff),'full order5 native path');out=[list(v)for v in coeff]
 for phase in range(3):
  a=[tanh_series([out[k][i]for k in range(6)])for i in range(n)]
  out=[[(1-d)*out[k][i]+d*g*sum((j[i][l]*a[l][k]for l in range(n)),arb(0))for i in range(n)]for k in range(6)]
  emit('independent_native_path',dict(phase=phase,coefficients=out))
 return [[-v for v in row]for row in out]


def native(j,box,emit=lambda *_:None):
 n=len(j);x=box[:n];d=box[n];g=box[-1];need(g>0 and 0<d<1,'actual positive physical rates');orbit=[x];M=eye(n);products=[]
 for phase in range(6):
  state=orbit[-1];t=[v.tanh()for v in state];s=[1/(v.cosh()**2)for v in state];DF=arb_mat([[(1-d)*int(i==k)+d*g*j[i][k]*s[k]for k in range(n)]for i in range(n)])
  nextx=[(1-d)*state[i]+d*g*sum((j[i][k]*t[k]for k in range(n)),arb(0))for i in range(n)];M=DF*M;products.append(M);orbit.append(nextx);emit('independent_native_phase',dict(phase=phase,state=nextx,jacobian=DF,product=M))
 A=-products[2];q=box[n+1:-1]
 need(all((orbit[3][i]+x[i]).contains(0)and (orbit[6][i]-x[i]).contains(0)for i in range(n)),'known-root half-wave/closure consistency')
 need(all(v.contains(0)for row in(M-A*A).tolist()for v in row),'odd-map square consistency')
 divisors={str(k):[orbit[k][i]-x[i]for i in range(n)]for k in(1,2,3)};need(all(any(v.is_finite()and not v.contains(0)for v in row)for row in divisors.values()),'all native6 proper divisors')
 border=arb_mat([[A[k,i]+int(i==k)for k in range(n)]+[q[i]]for i in range(n)]+[q+[arb(0)]]);left=inverse(border,emit)*col([arb(0)]*n+[arb(1)]);p=vec(left)[:n]
 need(left[n,0].contains(0)and all(v.contains(0)for v in vec(A*col(q)+col(q)))and(sum((p[i]*q[i]for i in range(n)),arb(0))-1).contains(0),'known exact normalized critical identities')
 pivot=max(range(n),key=lambda i:float(abs(q[i].mid())));basis=[k for k in range(n)if k!=pivot];frame=arb_mat([[q[i]]+[arb(int(i==k))for k in basis]for i in range(n)]);T=inverse(frame,emit)*A*frame
 need((T[0,0]+1).contains(0)and all(T[i,0].contains(0)for i in range(1,n)),'exact invariant critical direction consistency')
 Q=arb_mat([[T[i,k]for k in range(1,n)]for i in range(1,n)]);power=Q;bounds=[];witness=None
 for k in(1,2,4,8,16,32,64,128,256,512):
  bound=norm(power);bounds.append(dict(power=k,bound=bound))
  if bound.is_finite()and bound<1:witness=k;break
  if not bound.is_finite():break
  power=power*power
 need(witness is not None,'strict complementary power contraction');report=dict(orbit=orbit,monodromy=M,signed_monodromy=A,divisors=divisors,frame=frame,quotient=Q,power_bounds=bounds,witness_power=witness,complement_dimension=n-1,primitive_period=6,left=p)
 emit('native_result',report);return A,p,report


def normal_form(x,A,q,p,path,emit=lambda *_:None):
 n=len(x);even=inverse(eye(n)-A,emit);B=arb_mat([[A[i,j]+int(i==j)for j in range(n)]+[-q[i]]for i in range(n)]+[p+[arb(0)]]);odd=inverse(B,emit)
 h=[list(x),list(q)]+[[arb(0)]*n for _ in range(4)]
 y=path(h);h[2]=vec(even*col(y[2]));y=path(h);sol=odd*col([-v for v in y[3]]+[arb(0)]);h[3]=vec(sol)[:n];c3=sol[n,0];need(c3.contains(0),'exact augmented c3 identity consistency')
 y=path(h);h[4]=vec(even*col(y[4]));y=path(h);sol=odd*col([-v for v in y[5]]+[arb(0)]);h[5]=vec(sol)[:n];c5=sol[n,0];y=path(h)
 # At the previously established exact c3=0, h(-u+c5*u^5) through fifth order.
 expect=[h[0],[-v for v in h[1]],h[2],[-v for v in h[3]],h[4],[q[i]*c5-h[5][i]for i in range(n)]]
 defect=[[y[k][i]-expect[k][i]for i in range(n)]for k in range(6)];gauges=[sum((p[i]*h[k][i]for i in range(n)),arb(0))for k in(3,5)]
 need(c5.is_finite()and not c5.contains(0)and all(v.contains(0)for row in defect for v in row)and all(v.contains(0)for v in gauges),'nonzero quintic/full invariance/odd gauges')
 report=dict(quintic=c5,cubic_consistency=c3,parameterization=h,invariance_defect=defect,odd_gauges=gauges);emit('normal_form_result',report);return report


def unfolding(J,A,p,emit=lambda *_:None):
 n=A.nrows();N=2*n+2;need(J.nrows()==J.ncols()==N,'full physical augmented Jacobian');critical=arb_mat([[J[i,k]for k in range(N-1)]for i in range(N-1)]);slope=-inverse(critical,emit)*col([J[i,N-1]for i in range(N-1)])
 cprime=J[N-1,N-1]+sum((J[N-1,i]*slope[i,0]for i in range(N-1)),arb(0));xleak=inverse(eye(n)-A,emit)*col([J[i,n]for i in range(n)]);lprime=sum((p[i]*(J[n+i,n]+sum((J[n+i,k]*xleak[k,0]for k in range(n)),arb(0)))for i in range(n)),arb(0));det=-lprime*cprime
 need(all(v.is_finite()and not v.contains(0)for v in(cprime,lprime,det)),'strict physical two-parameter transversality');report=dict(critical_gain_slope=slope,fixed_root_leak_slope=xleak,critical_cubic_gain_derivative=cprime,multiplier_leak_derivative=lprime,physical_determinant=det);emit('unfolding_result',report);return report
