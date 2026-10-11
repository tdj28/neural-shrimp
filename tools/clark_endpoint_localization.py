"""Conditional finite-precision endpoint restriction for a proved full51 tube.

Caller binds the original polynomial C/J/R, full acquisition E and same root.
Each outward matrix image contains that already known root. Exact dyadic
intersections are monotone; outward serialization must still fit old box/E.
"""
from fractions import Fraction
from flint import arb,arb_mat

def need(v,m):
 if not v:raise ValueError(m)
def dyad(v):
 m,e=v;return Fraction(int(m))*Fraction(2)**int(e)
def bounds(v):
 need(v.is_finite(),'finite interval');m=dyad(v.mid().man_exp());r=dyad(v.rad().man_exp());return m-r,m+r
def contains(a,b):return a[0]<=b[0]<=b[1]<=a[1]
def meet(a,b):
 z=max(a[0],b[0]),min(a[1],b[1]);need(z[0]<=z[1],'known-root nonempty intersection');return z
def restore(z):
 m=(z[0]+z[1])/2;r=(z[1]-z[0])/2
 def exact(v):
  need(v.denominator&(v.denominator-1)==0,'dyadic coefficient interval');return arb((v.numerator,-(v.denominator.bit_length()-1)))
 out=arb(exact(m),exact(r))+arb(0);need(contains(bounds(out),z),'outward endpoint rounding');return out

def localize(data,c,s,a,radii,h,old,*,iterations=16,tick=lambda:None):
 n=len(c);need(data['passed']and data['dimension']==n and 1<=n<=51 and all(len(v)==n for v in(s,a,radii,old)),'complete already proved tube')
 need(type(iterations)is int and 1<=iterations<=16,'finite endpoint intersections')
 need(all(x.is_finite()and x.is_exact()for x in c+s+a+radii+[h])and h>0 and all(x>0 for x in radii),'exact positive tube geometry')
 Ds=data['defect_coefficients'];fs=data['forcing_coefficients'];need(len(Ds)==4 and len(fs)==5 and all(m.nrows()==n and m.ncols()==n for m in Ds)and all(m.nrows()==n and m.ncols()==1 for m in fs),'complete coefficient degrees')
 p=[c[i]+s[i]*h+a[i]*h*h/2 for i in range(n)];need(all(x.is_exact()for x in p),'exact polynomial endpoint')
 powers=[arb(1)]
 for _ in range(4):powers.append(powers[-1]*h)
 D=sum((m*powers[k]for k,m in enumerate(Ds)),arb_mat(n,n));f=sum((m*powers[k]for k,m in enumerate(fs)),arb_mat(n,1))
 need(all(x.is_finite()for m in(D,f,data['image'])for row in m.tolist()for x in row),'finite actual point coefficients')
 offsets=[bounds(x)[0]for x in p];E=[(-bounds(r)[0],bounds(r)[0])for r in radii];old_bounds=[bounds(x)for x in old];X=[]
 for i in range(n):
  old_error=tuple(v-offsets[i]for v in old_bounds[i]);image=bounds(data['image'][i,0]);need(contains(old_error,image),'recorded endpoint covers proved uniform image');X.append(meet(old_error,meet(E[i],image)))
 history=[]
 for _ in range(iterations):
  rounded=arb_mat([[restore(x)]for x in X]);image=f+D*rounded;Y=[meet(X[i],bounds(image[i,0]))for i in range(n)];need(all(contains(X[i],Y[i])and contains(E[i],Y[i])for i in range(n)),'exact monotone same-root restriction');X=Y
  history.append(dict(rounded_input=[rounded[i,0]for i in range(n)],point_image=[image[i,0]for i in range(n)],intersection=[restore(x)for x in X]));tick()
 proposal=[restore(tuple(v+offsets[i]for v in X[i]))for i in range(n)]
 checks=[contains(old_bounds[i],bounds(proposal[i]))and contains(E[i],tuple(v-offsets[i]for v in bounds(proposal[i])))for i in range(n)]
 # Rounding can prevent a representable subset at an unchanged boundary. Do
 # not clip or infer a sharper root; preserve the original proved endpoint.
 restricted=all(checks);endpoint=proposal if restricted else old
 need(all(contains(old_bounds[i],bounds(endpoint[i]))and contains(E[i],tuple(v-offsets[i]for v in bounds(endpoint[i])))for i in range(n)),'final endpoint inside original known box and E')
 return dict(passed=True,restricted=restricted,iterations=iterations,endpoint=endpoint,proposal=proposal,serialized_subset_checks=checks,point_defect=D,point_forcing=f,history=history,scope='Conditional restriction of the same already proved endpoint; original fullE retained. No new existence, native phase or generic-fold theorem.')
