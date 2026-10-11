"""Independent saved51 polynomial root, known-root, and native reductions.

The caller authenticates native Taylor coefficients with their actual complete
error/parameter domains. These routines import no producer fold/path theorem.
C coefficients are arbitrary finite exact proposals. An organizer anchor and
all subsequent same-map/phase identities remain explicit caller prerequisites.
"""
from flint import arb,arb_mat
import review_clark_organizer_math as M
need=M.need


def power(x,k):
 need(type(k)is int and k>=0,'nonnegative integer power');v=arb(1)
 for _ in range(k):v*=x
 return v


def monomials(h,count):
 need(h.is_finite()and h.is_exact()and h>0 and 1<=count<=8,'positive exact bounded scalar interval')
 out=[arb(1)]
 for k in range(1,count):
  v=power(h,k);out.append(arb(0,v)if k%2 else arb(v/2,v/2))
 return out


def evaluate(coeff,weights):
 need(len(coeff)==len(weights)>0,'complete coefficient contraction')
 return sum((a*b for a,b in zip(coeff,weights,strict=True)),arb_mat(coeff[0].nrows(),coeff[0].ncols()))


def reduction(C,J,R,radii,h):
 n=len(radii);need(1<=n<=51 and len(C)==2 and len(J)==3 and len(R)==4,'full finite polynomial degrees')
 need(all(a.nrows()==a.ncols()==n and M.finite(a)for a in C+J)and all(a.nrows()==n and a.ncols()==1 and M.finite(a)for a in R),'full authenticated arrays')
 need(all(x.is_finite()and x.is_exact()and x>0 for x in radii)and all(x.is_exact()for a in C for row in a.tolist()for x in row),'exact finite proposals and radii')
 defect=[arb_mat(n,n)for _ in range(4)];forcing=[arb_mat(n,1)for _ in range(5)];defect[0]=M.eye(n)
 for i in range(2):
  for k in range(3):defect[i+k]-=C[i]*J[k]
  for k in range(4):forcing[i+k]-=C[i]*R[k]
 D=evaluate(defect,monomials(h,4));f=evaluate(forcing,monomials(h,5));image=f+D*M.col([arb(0,x)for x in radii]);need(M.finite(D)and M.finite(f)and M.finite(image),'finite independent image')
 rows=[sum((abs(D[i,k]).upper()*radii[k]for k in range(n)),arb(0))/radii[i]for i in range(n)];q=max(x.upper()for x in rows);inside=[bool(abs(image[i,0]).upper()<radii[i])for i in range(n)]
 return dict(passed=bool(q<1 and all(inside)),dimension=n,defect_coefficients=defect,forcing_coefficients=forcing,defect=D,forcing=f,image=image,weighted_rows=rows,contraction=q,inclusions=inside,radii=radii,halfwidth=h)


def join(known,scalar,c,s,a,parameter,h,radii):
 n=len(c);need(1<=n<=51 and all(len(v)==n for v in(known,s,a,radii)),'all known-root coordinates')
 need(all(x.is_finite()and x.is_exact()for x in c+s+a+radii+[parameter,h,scalar])and h>0 and all(x>0 for x in radii),'exact phase-preserving curved geometry')
 d=scalar-parameter;inside=bool(-h<=d<=h);e=[known[i]-c[i]-s[i]*d-a[i]*d*d/2 for i in range(n)];checks=[bool(x.is_finite()and abs(x).upper()<radii[i])for i,x in enumerate(e)]
 return dict(passed=bool(inside and all(checks)),inside=inside,coordinates=sum(checks),inclusions=checks,error=e)


def endpoint(data,c,s,a,sign):
 """One same-root endpoint enclosure, no new existence assertion."""
 n=len(c);need(data['passed']and n==data['dimension']and sign in(-1,1),'proved complete tube endpoint')
 h=data['halfwidth'];d=sign*h;D=evaluate(data['defect_coefficients'],[power(d,k)for k in range(4)]);rhs=evaluate(data['forcing_coefficients'],[power(d,k)for k in range(5)]);box=M.col([arb(0,v)for v in data['radii']])
 for _ in range(6):
  image=rhs+D*box;box=arb_mat([[box[i,0].intersection(image[i,0])]for i in range(n)])
 need(M.finite(box),'finite exact endpoint correction')
 return [c[i]+s[i]*d+a[i]*d*d/2+box[i,0]for i in range(n)]


def anchor(A,rhs,z,lambda_leak):
 """Conditional on independently proved normalized c3=0 organizer."""
 n=len(z);need(n%2==0 and 4<=n<=26 and A.nrows()==A.ncols()==n-1 and rhs.nrows()==n-1 and rhs.ncols()==1 and lambda_leak.is_finite()and not lambda_leak.contains(0),'same complete organizer authority')
 v=M.inverse(A)*rhs;leak=(n-2)//2;need(v[leak,0].contains(0),'known c3+lambda_leak*v_leak identity consistency')
 return z+M.vec(v)


def native_context(cycle,blocks,mode,max_power=512):
 """Conditional exact native6/+1; proper divisors and complement are proved."""
 n=len(mode);need(type(max_power)is int and 1<=max_power<=4096,'bounded quotient power');need(2<=n<=12 and len(cycle)==len(blocks)==6 and all(len(x)==n for x in cycle)and all(a.nrows()==a.ncols()==n and M.finite(a)for a in blocks),'complete source-bound native phases')
 length=sum((x*x for x in mode),arb(0)).sqrt();need(length.is_finite()and length>0,'nonzero actual critical mode');q=[x/length for x in mode];monodromy=M.eye(n)
 for A in blocks:monodromy=A*monodromy
 need(all(x.contains(0)for x in M.vec(monodromy*M.col(q)-M.col(q))),'known exact critical+1 consistency')
 pivot=max(range(n),key=lambda i:float(abs(q[i].mid())));columns=[i for i in range(n)if i!=pivot];frame=arb_mat([[q[i]]+[arb(int(i==k))for k in columns]for i in range(n)]);R=M.inverse(frame);T=R*monodromy*frame
 need((T[0,0]-1).contains(0)and all(T[i,0].contains(0)for i in range(1,n)),'same invariant critical line consistency');Q=arb_mat([[T[i,k]for k in range(1,n)]for i in range(1,n)]);A=Q;bounds=[];witness=None;k=1
 while k<=max_power:
  bound=M.norm(A);bounds.append((k,bound))
  if bound.is_finite()and bound<1:witness=k;break
  if not bound.is_finite():break
  A=A*A;k*=2
 divisors=[]
 for d in(1,2,3):
  difference=[cycle[d][i]-cycle[0][i]for i in range(n)];ids=[i for i,x in enumerate(difference)if x.is_finite()and not x.contains(0)];divisors.append(dict(divisor=d,coordinates=ids,difference=difference))
 return dict(passed=bool(witness and all(x['coordinates']for x in divisors)),mode=q,normalization=length,monodromy=monodromy,frame=frame,frame_inverse=R,quotient=Q,bounds=bounds,witness=witness,divisors=divisors)
