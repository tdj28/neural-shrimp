"""Independent assembly from authenticated saved native path/gradient returns.

No map, native derivative, or producer augmented-system code is imported.
The analytic per-phase native acquisitions remain a stated shared dependency.
"""
from fractions import Fraction as F
from flint import arb,arb_mat
import review_clark_correlated_math as H
M=H.M;need=H.need


def dyadic(x):
 m,e=x;need(type(m)is int and type(e)is int,'integer dyadic encoding');return F(m)*F(2)**e

def interval(x):
 need(isinstance(x,dict)and 'mid'in x and 'rad'in x,'finite serialized interval');m=dyadic(x['mid']);r=dyadic(x['rad']);need(r>=0,'nonnegative radius');return m-r,m+r

def point(x):
 lo,hi=interval(x);need(lo==hi,'literal exact scalar');return lo

def covers(x,pair):
 lo,hi=interval(x);return lo<=pair[0]<=pair[1]<=hi

def quadratic(c,s,a,d,e):
 lo,hi=d;values=[c+s*t+a*t*t/2 for t in(lo,hi)]
 if a and lo<=-s/a<=hi:values.append(c-s*s/(2*a))
 return min(values)+e[0],max(values)+e[1]

def linear(c,s,d):
 v=[c+s*x for x in d];return min(v),max(v)

def ring(raw):
 need(len(raw)==2 and all(len(x)==4 for x in raw),'complete path/mode ring');return [[M.decode(x)for x in row]for row in raw]

def const(x):return [[arb(x),arb(0),arb(0),arb(0)],[arb(0)]*4]
def scale(x,v):return [[a*v for a in row]for row in x]
def add(x,y):return [[a+b for a,b in zip(u,v,strict=True)]for u,v in zip(x,y,strict=True)]
def product(x,y):
 return [[sum((x[0][i]*y[0][k-i]for i in range(k+1)),arb(0))for k in range(4)],[sum((x[0][i]*y[1][k-i]+x[1][i]*y[0][k-i]for i in range(k+1)),arb(0))for k in range(4)]]

def initial(request,row):
 """Exact rational coverage of every base/path/mode/gradient input."""
 c,s,a=[[point(v)for v in request[k]]for k in('center','slope','curvature')];N=(len(c)+1)//2;K=N-1;n=(N-2)//2;dim=N+K;need(2<=n<=12 and dim==len(c)==len(s)==len(a)==len(request['error']),'complete actual dimensions');d=interval(request['d']);b=point(request['parameter']);error=[interval(v)for v in request['error']];need(all(lo<=0<=hi for lo,hi in error),'full zero-containing error')
 def path_domain(i):return [quadratic(c[i],s[i],a[i],d,error[i]),linear(s[i],a[i],d),(a[i]/2,a[i]/2),(F(0),F(0))]
 fields=row['even']+[row['leak']]+row['odd']+[row['gain']]
 need(row['phase']==0 and len(row['even'])==len(row['odd'])==n and len(fields)==N,'actual initial native phase')
 for i,x in enumerate(fields):
  need(len(x['value'])==2 and len(x['gradient'])==N+1,'full actual native value/base gradient')
  for mode in(0,1):
   expected=path_domain(i)if mode==0 else path_domain(N+i)if i<K else [(F(0),F(0))]*4
   need(len(x['value'][mode])==4 and all(covers(z,p)for z,p in zip(x['value'][mode],expected,strict=True)),'full actual predictor/error/path/mode coverage')
  for k,g in enumerate(x['gradient']):
   for mode in(0,1):
    need(len(g[mode])==4 and all(point(v)==int(mode==0 and j==0 and i==k)for j,v in enumerate(g[mode])),'actual independent base basis')
 x=row['squared_amplitude'];need(len(x['value'])==2 and len(x['gradient'])==N+1,'full external amplitude jet')
 expected=[[(b+d[0],b+d[1]),(F(1),F(1)),(F(0),F(0)),(F(0),F(0))],[(F(1),F(1))]+[(F(0),F(0))]*3]
 need(all(len(x['value'][m])==4 and all(covers(v,p)for v,p in zip(x['value'][m],expected[m],strict=True))for m in(0,1)),'actual nonlinear external b path/mode')
 for k,g in enumerate(x['gradient']):need(all(len(g[m])==4 and all(point(v)==int(m==0 and j==0 and k==N)for j,v in enumerate(g[m]))for m in(0,1)),'actual external b base basis')
 return n,N,K,fields

def assemble_path(request,inputs,returns):
 need(len(inputs)==len(returns)==3,'all three actual native paths');n,N,K,first=initial(request,inputs[0]);dim=N+K
 for phase in range(3):
  x,y=inputs[phase],returns[phase];need(x['phase']==y['phase']==phase and len(y['even'])==len(y['odd'])==n,'ordered actual phase return')
  need(all(x[k]==inputs[0][k]for k in('leak','gain','squared_amplitude')),'unchanged native physical/path inputs')
  if phase:need(x['even']==returns[phase-1]['even']and x['odd']==returns[phase-1]['odd'],'literal native phase chain')
 last=returns[-1]['even']+returns[-1]['odd'];old=inputs[0]['even']+inputs[0]['odd'];values=[];gradients=[]
 for i,(y,x)in enumerate(zip(last,old,strict=True)):
  sign=-1 if i<n else 1;need(len(y['gradient'])==N+1,'complete actual base gradient return');values.append(add(scale(ring(y['value']),-1),scale(ring(x['value']),sign)));gradients.append([add(scale(ring(v),-1),scale(ring(w),sign))for v,w in zip(y['gradient'],x['gradient'],strict=True)])
 norm=const(-1);g=[const(0)for _ in range(N+1)]
 for x in inputs[0]['odd']:
  q=ring(x['value']);norm=add(norm,product(q,q))
  for k in range(N+1):g[k]=add(g[k],product(q,ring(x['gradient'][k])))
 values.append(scale(norm,arb(1)/2));gradients.append(g)
 # The last equation is the independent v_leak coordinate, including its path.
 c,s,a=[[M.decode(v)for v in request[k]]for k in('center','slope','curvature')];d=M.decode(request['d']);e=M.decode(request['error'][N+n]);lastrow=[c[N+n]+s[N+n]*d+a[N+n]*d*d/2+e,s[N+n]+a[N+n]*d,a[N+n]/2,arb(0)]
 value=arb_mat([x[0]for x in values]+[x[1]for x in values]+[lastrow]);J=[arb_mat(dim,dim)for _ in range(3)]
 for k in range(3):
  for i,g in enumerate(gradients):
   for u in range(N):J[k][i,u]=g[u][0][k];J[k][K+i,u]=g[u][1][k]
   for u in range(K):J[k][K+i,N+u]=g[u][0][k]
  J[k][dim-1,N+n]=int(k==0)
 need(M.finite(value)and all(M.finite(x)for x in J),'finite independent raw native assembly');return value,J
