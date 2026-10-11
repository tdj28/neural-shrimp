"""Conditional exact fixed-parameter localization of an already proved tube root.

No new acquisition or existence inference. Interval coefficients retain the
original full error domain. Exact rational intersections keep the known root
inside every iterate; outward Arb serialization is checked against that domain.
"""
from fractions import Fraction as F
from flint import arb,arb_mat
import review_clark_correlated_math as H
M,need=H.M,H.need

def dyad(v):
 m,e=v;return F(int(m))*F(2)**int(e)
def bounds(x):
 need(x.is_finite(),'finite interval coefficient');m=dyad(x.mid().man_exp());r=dyad(x.rad().man_exp());return m-r,m+r
def add(x,y):return x[0]+y[0],x[1]+y[1]
def mul(x,y):
 v=[a*b for a in x for b in y];return min(v),max(v)
def cover(x,y):return x[0]<=y[0]<=y[1]<=x[1]
def intersection(x,y):
 z=max(x[0],y[0]),min(x[1],y[1]);need(z[0]<=z[1],'nonempty known-root intersection');return z
def restore(x):
 m=(x[0]+x[1])/2;r=(x[1]-x[0])/2
 def a(v):
  need(v.denominator&(v.denominator-1)==0,'dyadic endpoint arithmetic');return arb((v.numerator,-(v.denominator.bit_length()-1)))
 z=arb(a(m),a(r));need(cover(bounds(z),x),'outward restored rational interval');return z

def localize(data,c,s,a,old_endpoint,*,iterations=16):
 n=len(c);need(data['passed']and n==data['dimension']and 1<=n<=51 and len(s)==len(a)==len(old_endpoint)==n,'already proved complete root')
 need(type(iterations)is int and 1<=iterations<=16,'fixed bounded intersections');h=data['halfwidth'];need(h.is_finite()and h.is_exact()and h>0,'exact right endpoint')
 p=[c[i]+s[i]*h+a[i]*h*h/2 for i in range(n)];need(all(x.is_exact()for x in p),'exact polynomial endpoint at working precision')
 D=H.evaluate(data['defect_coefficients'],[H.power(h,k)for k in range(4)]);f=H.evaluate(data['forcing_coefficients'],[H.power(h,k)for k in range(5)])
 rr=[bounds(r)[0]for r in data['radii']];need(all(r.is_exact()and r>0 for r in data['radii']),'exact proved correction radii');E=[(-r,r)for r in rr]
 # Check the old recorded box covers the independently recomputed whole image.
 initial=[]
 for i in range(n):
  pi=bounds(p[i])[0];old=tuple(x-pi for x in bounds(old_endpoint[i]));fresh=bounds(data['image'][i,0]);need(cover(old,fresh),'original endpoint covers own rigorous image');initial.append(intersection(old,intersection(E[i],fresh)))
 db=[[bounds(D[i,j])for j in range(n)]for i in range(n)];fb=[bounds(f[i,0])for i in range(n)];box=initial;history=[]
 for _ in range(iterations):
  image=[add(fb[i],sum_intervals([mul(db[i][j],box[j])for j in range(n)]))for i in range(n)]
  new=[intersection(box[i],image[i])for i in range(n)];need(all(cover(box[i],new[i])and cover(E[i],new[i])for i in range(n)),'monotone full-domain restriction');box=new;history.append(box)
 endpoint=[restore(tuple(v+bounds(p[i])[0]for v in box[i]))for i in range(n)]
 need(all(cover(bounds(old_endpoint[i]),bounds(endpoint[i]))and cover(E[i],tuple(v-bounds(p[i])[0]for v in bounds(endpoint[i])))for i in range(n)),'serialized endpoint remains in original known box/full E')
 rows=[dict(index=i,original_width=bounds(old_endpoint[i])[1]-bounds(old_endpoint[i])[0],new_width=box[i][1]-box[i][0],ratio=float((box[i][1]-box[i][0])/(bounds(old_endpoint[i])[1]-bounds(old_endpoint[i])[0]))if bounds(old_endpoint[i])[1]>bounds(old_endpoint[i])[0]else 0.)for i in range(n)]
 return dict(passed=True,endpoint=endpoint,point_defect=D,point_forcing=f,initial=initial,history=history,exact_correction=box,rows=rows,scope='Same unique root at exact right parameter; sixteen monotone exact rational intersections on unchanged full-error coefficient bounds. No new derivative or existence assertion.')
def sum_intervals(v):
 z=(F(0),F(0))
 for x in v:z=add(z,x)
 return z
