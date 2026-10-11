import mpmath as mp
import pytest
from flint import arb,ctx
from neural_shrimp.rigorous_gradient_path import GradientPathJet,jacobian_path_coefficients
from neural_shrimp.rigorous_eigenpair_path import PathJet

class PolynomialMap:
 n=2
 def cusp_step(self,x,a,b):return [x[0]+a*x[1]*x[1]+b*x[0]*x[1],x[1]+a*x[0]*x[0]*x[0]+b*x[1]]


def fixture():
 c=[arb(3)/32,-arb(1)/8,arb(1),arb(1)/4,-arb(1)/4,arb(1),arb(1)/16,arb(3)/32,arb(1)/64]
 s=[arb(i-4)/64 for i in range(9)];a=[arb(i%3-1)/128 for i in range(9)];n=[arb((-1)**i*(i+1))/32 for i in range(9)]
 return c,s,a,n


def independent_jacobian(z,normal):
 x,y,q0,q1,r0,r1,a,b,t=z;q=[q0,q1];r=[r0,r1];M=mp.matrix(9)
 D=mp.matrix([[1+b*y,2*a*y+b*x],[3*a*x*x,1+b]])
 H=[mp.matrix([[0,b],[b,2*a]]),mp.matrix([[6*a*x,0],[0,0]])]
 da=[y*y,x*x*x];db=[x*y,y]
 Da=mp.matrix([[0,2*y],[3*x*x,0]]);Db=mp.matrix([[y,x],[0,1]])
 root=mp.sqrt(t);cc=mp.cos(root);ee=mp.sin(root)/root;dd=t*ee
 cp=-mp.sin(root)/(2*root);ep=(root*mp.cos(root)-mp.sin(root))/(2*root**3);dp=ee+t*ep
 for i in range(2):
  for j in range(2):
   M[i,j]=D[i,j]-int(i==j)
   M[2+i,j]=sum(H[i][j,k]*q[k]for k in range(2));M[4+i,j]=sum(H[i][j,k]*r[k]for k in range(2))
   M[2+i,2+j]=D[i,j]-cc*int(i==j);M[2+i,4+j]=dd*int(i==j)
   M[4+i,2+j]=-ee*int(i==j);M[4+i,4+j]=D[i,j]-cc*int(i==j)
  M[i,6]=da[i];M[i,7]=db[i]
  for col,rate in [(6,Da),(7,Db)]:
   M[2+i,col]=sum(rate[i,k]*q[k]for k in range(2));M[4+i,col]=sum(rate[i,k]*r[k]for k in range(2))
  M[2+i,8]=-cp*q[i]+dp*r[i];M[4+i,8]=-ep*q[i]-cp*r[i]
  M[6,2+i]=q[i];M[7,2+i]=r[i];M[7,4+i]=q[i]
 for j in range(9):M[8,j]=normal[j]
 return M


def compare(value,reference):
 ref=arb(mp.nstr(reference,85))+arb(0,arb('1e-60'))
 assert (value-ref).contains(0),(value,reference)


def test_all_native_angle_rate_mode_and_normalization_coefficients():
 with ctx.workprec(256),mp.workdps(90):
  c,s,a,n=fixture();count=[0];events=[]
  def tick():count[0]+=1
  J=jacobian_path_coefficients(PolynomialMap(),c,s,a,[arb(0)]*9,arb(0),1,n,tick=tick,observe=lambda k,v:events.append(k))
  assert count[0]==2 and events.count('gradient_path_input')==events.count('gradient_path_return')==2
  cc,ss,aa,nn=[[mp.mpf(float(v))for v in row]for row in [c,s,a,n]]
  def value(d,i,j):return independent_jacobian([cc[k]+ss[k]*d+aa[k]*d*d/2 for k in range(9)],nn)[i,j]
  for k in range(3):
   for i in range(9):
    for j in range(9):compare(J[k][i,j],mp.diff(lambda d:value(d,i,j),mp.mpf(0),k)/mp.factorial(k))


def test_whole_remainder_includes_nonzero_correction_and_parameter():
 with ctx.workprec(256),mp.workdps(90):
  c,s,a,n=fixture();error=[arb(0,arb((1,-18)))]*9;h=arb((1,-4))
  J=jacobian_path_coefficients(PolynomialMap(),c,s,a,error,arb(0,h),1,n)
  cc,ss,aa,nn=[[mp.mpf(float(v))for v in row]for row in [c,s,a,n]]
  for d,e in [(mp.mpf(1)/16,mp.mpf(1)/2**18),(-mp.mpf(1)/16,-mp.mpf(1)/2**18)]:
   def value(v,i,j):return independent_jacobian([cc[k]+ss[k]*v+aa[k]*v*v/2+e for k in range(9)],nn)[i,j]
   for i,j in [(0,0),(2,1),(3,6),(4,7),(2,8),(4,8),(6,3),(7,4),(8,8)]:compare(J[2][i,j],mp.diff(lambda v:value(v,i,j),d,2)/2)


def test_tanh_full_gradient_and_mixed_mode_through_cubic_path():
 with ctx.workprec(256),mp.workdps(90):
  x=GradientPathJet(PathJet([arb(1)/8,arb(1)/16,arb(1)/32,0],[arb(1)/4,arb(1)/8,arb(1)/64,0]),[1]);y=x.tanh()
  def base(t):return mp.mpf(1)/8+t/16+t*t/32
  def mode(t):return mp.mpf(1)/4+t/8+t*t/64
  def da(t):return 1-mp.tanh(base(t))**2
  def db(t):return -2*mp.tanh(base(t))*da(t)*mode(t)
  for k in range(4):
   compare(y.g[0].a[k],mp.diff(da,mp.mpf(0),k)/mp.factorial(k));compare(y.g[0].b[k],mp.diff(db,mp.mpf(0),k)/mp.factorial(k))


def test_bad_domains_and_gradient_dimensions_fail_closed():
 c,s,a,n=fixture()
 with pytest.raises(ValueError):jacobian_path_coefficients(PolynomialMap(),c,s,a,[arb(1)]*9,arb(0),1,n)
 with pytest.raises(ValueError):jacobian_path_coefficients(PolynomialMap(),c,s,a,[arb(0)]*9,arb(0),1,n[:-1])
 with pytest.raises(ValueError):GradientPathJet(0,[0])*GradientPathJet(0,[0,0])
 c[-1]=arb(1)
 with pytest.raises(ValueError):jacobian_path_coefficients(PolynomialMap(),c,s,a,[arb(0)]*9,arb(0),1,n)


def test_two_phase_native_assembly_matches_independently_reviewed_original_jacobian():
 from neural_shrimp.rigorous_unit_pair_pal import assemble
 with ctx.workprec(256):
  c,s,a,n=fixture();c=c[:6]+[x+arb(1)/32 for x in c[:6]]+c[-3:];N=len(c)
  v=[arb(i-7)/256 for i in range(N)];acc=[arb(0)]*N;normal=[arb(i+1)/64 for i in range(N)];error=[arb(0)]*N
  got=jacobian_path_coefficients(PolynomialMap(),c,v,acc,error,arb(0),2,normal)[0]
  _,old,_=assemble(PolynomialMap(),c,2,normal,[arb(0)]*N,arb(0))
  assert got.nrows()==15
  assert all((got[i,j]-old[i,j]).contains(0)for i in range(N)for j in range(N))
