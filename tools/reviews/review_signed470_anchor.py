"""Independent exact-rational negative-mode transport, conditional on known55."""
from fractions import Fraction as F
from flint import arb,ctx
from neural_shrimp.rigorous_orbits import ball_json
import review_joined_period18_math as M
need=M.need

def add(x,y):return x[0]+y[0],x[1]+y[1]
def mul(x,y):
 v=[a*b for a in x for b in y];return min(v),max(v)
def dot(row,v):
 ans=(F(0),F(0))
 for a,b in zip(row,v,strict=True):ans=add(ans,mul(a,b))
 return ans

def enclosing_ball(v):
 a,b=v;mid=(a+b)/2;rad=(b-a)/2;out=arb(mid.numerator)/mid.denominator+arb(0,arb(rad.numerator)/rad.denominator);q=ball_json(out);lo,hi=M.interval(q);need(lo<=a<=b<=hi,'outward exact rational interval conversion');return q

def negative_lift(root,unit,returns,*,n=13,p=18,tick=lambda:None):
 need(len(root)==4*n+3 and len(unit)==3*n*p+3 and len(returns)==2*p,'complete native known domains')
 rates=root[n:n+2];w=list(map(M.interval,root[3*n+3:]));w0=list(w);flat=[]
 need(unit[:3*n]==root[:n]+root[n+2:3*n+2]and unit[-3:]==rates+[root[3*n+2]],'same entire normalized root and actual rates')
 for k in range(p):
  x,q,r=[unit[3*n*k+off:3*n*k+off+n]for off in (0,n,2*n)];a,b=returns[2*k:2*k+2]
  for direction,mode,row in [('q',q,a),('r',r,b)]:
   need(row['phase']==k and row['direction']==direction and row['input']==x+mode+rates,'literal saved native domain');need(len(row['block'])==2*n and all(len(v)==2*n for v in row['block'])and len(row['value'])==2*n,'entire native DF/values')
  A=[v[:n]for v in a['block'][:n]];need(A==[v[:n]for v in b['block'][:n]]and a['value'][:n]==b['value'][:n],'identical direction-independent DF and state')
  if k+1<p:need([v[0]for v in a['value'][:n]]==unit[3*n*(k+1):3*n*(k+1)+n],'actual propagated whole-state chain')
  flat+=x+[enclosing_ball(v)for v in w];w=[dot([M.interval(v)for v in row],w)for row in A];tick()
 need(all(a<=0<=b for a,b in[add(x,y)for x,y in zip(w,w0)]),'conditional exact signed closure compatible with enclosure')
 return flat+rates
