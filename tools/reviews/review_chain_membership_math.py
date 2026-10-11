"""Independent exact-rational native-domain and quadratic-tube membership checks.

Conditional on an independently proved normalized chain root and authenticated
analytic phase acquisitions. No producer lift/membership routine or Arb used.
"""
from fractions import Fraction as F

def need(test,message):
 if not test:raise ValueError(message)
def dyadic(x):
 need(isinstance(x,list)and len(x)==2 and all(type(i)is int for i in x),'integer dyadic pair')
 need(abs(x[1])<=100000,'bounded dyadic exponent')
 return F(x[0])*F(2)**x[1]
def ball(v):
 need(isinstance(v,dict)and 'mid'in v and 'rad'in v and v.get('finite',True)is True,'finite encoded interval')
 m,r=dyadic(v['mid']),dyadic(v['rad']);need(r>=0,'nonnegative radius');return(m-r,m+r)
def point(s):
 need(isinstance(s,str),'exact rational constant');q=F(s);return(q,q)
def add(a,b):return(a[0]+b[0],a[1]+b[1])
def sub(a,b):return(a[0]-b[1],a[1]-b[0])
def mul(a,b):
 c=[x*y for x in a for y in b];return(min(c),max(c))
def square(a):return(F(0)if a[0]<=0<=a[1]else min(a[0]**2,a[1]**2),max(a[0]**2,a[1]**2))
def contains(a,b):return a[0]<=b[0]and b[1]<=a[1]
def dump(a):return[str(x)for x in a]
def exact_ball(v):
 a=ball(v);need(a[0]==a[1],'exact tube predictor');return a

def check_phase_domains(root_box,lift,phases,n=13,period=18):
 need(type(n)is int and type(period)is int and n>0 and period>0,'native integer dimensions')
 need(len(root_box)==3*n+2 and len(phases)==period,'full chain and phase count')
 need(lift['schema']=='neural-shrimp-conditional-chain-lift-v1','conditional lift schema')
 need((lift['dimension'],lift['period'],lift['sigma'],lift['shooting_dimension'])==(n,period,1,2*n*period+2),'native lift dimensions')
 root=[ball(v)for v in root_box];rates=root[n:n+2];need(all(v[0]>0 for v in rates),'positive original rates')
 expected=root[:n]+root[n+2:2*n+2];flat=[]
 for k,row in enumerate(phases):
  need(row['index']==k and len(row['input'])==2*n+2,'ordered actual native phase')
  domain=[ball(v)for v in row['input']]
  need(all(contains(a,b)for a,b in zip(domain,expected+rates)),'actual phase domain contains preceding values and same root rates')
  values=row['value'];need(len(values)==2*n and all(len(v)==1 for v in values),'full native phase output')
  expected=[ball(v[0])for v in values];flat.extend(domain[:2*n])
 flat.extend(rates);saved=[ball(v)for v in lift['box']];need(len(saved)==len(flat),'all native lift coordinates')
 # Serialization may widen a propagated value; the retained lift must cover
 # the full actual acquisition domains, including normalized root phase zero.
 need(all(contains(a,b)for a,b in zip(saved,flat)),'saved native lift covers actual phase domains')
 return {'passed':True,'phase_domains':period,'native_coordinates':len(flat),'shared_analytic_acquisitions':True}

def check_membership(lift,chart):
 for k in ('period','dimension','sigma','shooting_dimension','normal','origin'):need(lift[k]==chart[k],'identical native convention '+k)
 n=chart['shooting_dimension'];need(type(n)is int and n>0,'positive native size')
 box=[ball(v)for v in lift['box']];normal=[point(s)for s in chart['normal']];origin=[point(s)for s in chart['origin']]
 center=[exact_ball(v)for v in chart['center']];slope=[exact_ball(v)for v in chart['slope']];curve=[exact_ball(v)for v in chart['curvature']]
 need(all(len(v)==n for v in (box,normal,origin,center,slope,curve,chart['weights'])),'all-coordinate geometry')
 need(any(v!=(0,0)for v in normal),'nonzero native gauge')
 parameter=(F(0),F(0))
 for a,x,o in zip(normal,box,origin):parameter=add(parameter,mul(a,sub(x,o)))
 sc=point(chart['parameter_center']);half=F(2)**chart['settings']['halfwidth_exponent'];need(half>0,'positive halfwidth')
 endpoints=[F(s)for s in chart['parameter_endpoints']];need(endpoints==[sc[0]-half,sc[0]+half],'exact symmetric chart endpoints')
 inside=endpoints[0]<parameter[0]and parameter[1]<endpoints[1];d=sub(parameter,sc);d2=square(d);errors=[];inclusions=[];radii=[]
 for x,c,s,h,w in zip(box,center,slope,curve,chart['weights']):
  radius=F(2)**chart['settings']['radius_exponent']*point(w)[0];need(radius>0,'strictly positive native radius')
  error=sub(sub(sub(x,c),mul(s,d)),mul(h,mul((F(1,2),F(1,2)),d2)))
  errors.append(dump(error));radii.append(str(radius));inclusions.append(-radius<error[0]and error[1]<radius)
 return {'passed':inside and all(inclusions),'parameter_inside':inside,'parameter':dump(parameter),'coordinate_inclusions':inclusions,'predictor_errors':errors,'coordinate_radii':radii,'arithmetic':'exact rational intervals; exact square range','scope':'Conditional authenticated root and native phase lift membership in independently proved full-coordinate uniqueness tube. No generic unfolding claim.'}
