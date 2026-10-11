"""Independent uniform26D critical-family and native phase-join reductions.

Only source-authenticated family acquisitions are accepted by the caller.
No producer proof or native derivative helper is imported here.
"""
from flint import arb,arb_mat
import review_clark_organizer_math as M
need=M.need

def endpoints(raw):
 m=arb(tuple(raw['mid']));r=arb(tuple(raw['rad']));lo,hi=m-r,m+r
 need(r>=0 and lo.is_exact()and hi.is_exact(),'exact serialized native domain endpoints');return lo,hi

def covered(raw,value):
 lo,hi=endpoints(raw);return bool(lo<=value.lower()and value.upper()<=hi)

def uniform(t,emit=lambda *_:None):
 c,v,r=[[M.decode(z)for z in t[k]]for k in ['center','slope','radii']];N=len(c);s=M.decode(t['parameter_center']);h=M.decode(t['halfwidth'])
 need(2<=N<=34 and len(v)==len(r)==N and all(z.is_exact()and z.is_finite()for z in c+v+r+[s,h])and all(z>0 for z in r)and h>0 and c[-1]==s and v[-1]==1,'exact complete finite family geometry')
 d=arb(0,h);E=[arb(0,z)for z in r];p=[c[i]+v[i]*d for i in range(N)];whole=[p[i]+E[i]for i in range(N)]
 for key,values in [('actual_error',E),('predictor_box',p),('whole_box',whole)]:need(len(t[key])==N and all(covered(a,b)for a,b in zip(t[key],values)),'actual acquired '+key+' coverage')
 H0=M.matrix(t['center_value']);J0=M.matrix(t['center_jacobian']);Jp=M.matrix(t['predictor_jacobian']);Jw=M.matrix(t['whole_jacobian'])
 need(H0.nrows()==N and H0.ncols()==1 and all(A.nrows()==A.ncols()==N and M.finite(A)for A in [J0,Jp,Jw]),'full finite authenticated family derivatives')
 C=M.midpoint(M.inverse(J0,emit));D=M.eye(N)-C*Jw;e=[arb(0)]*(N-1)+[arb(1)]
 # Independent mean-value reduction for p(d)=c+v*d and actual gain s+d.
 residual=-(C*H0)-C*(Jp*M.col(v)-M.col(e))*d;image=residual+D*M.col(E)
 row=[sum((abs(D[i,k]).upper()*r[k]for k in range(N)),arb(0))/r[i]for i in range(N)];q=max(z.upper()for z in row);checks=[bool(abs(image[i,0]).upper()<r[i])for i in range(N)]
 report=dict(passed=bool(q<1 and all(checks)),dimension=N,preconditioner=C,defect=D,forcing=residual,image=image,weighted_rows=row,contraction=q,coordinate_inclusions=checks,center=c,slope=v,radii=r,parameter=s,halfwidth=h,whole=whole)
 emit('independent_uniform',report);need(report['passed'],'independent full uniform strict root tube');return report

def join(box,t,emit=lambda *_:None):
 n=len(t['center']);need(len(box)==n,'complete known-root box');d=box[-1]-t['parameter'];errors=[box[i]-t['center'][i]-t['slope'][i]*d for i in range(n)];checks=[bool(abs(x).upper()<t['radii'][i])for i,x in enumerate(errors)]
 row=dict(passed=bool(abs(d).upper()<t['halfwidth']and all(checks)),parameter_inside=bool(abs(d).upper()<t['halfwidth']),coordinates=sum(checks),coordinate_inclusions=checks,error=errors)
 emit('independent_join',row);return row

def both_orientations(box,t):
 n=(len(box)-2)//2;rows=[]
 for sign in(1,-1):
  z=list(box);z[n+1:-1]=[sign*v for v in z[n+1:-1]];rows.append(dict(sign=sign,join=join(z,t)))
 return rows

def return_box(box,native):
 n=(len(box)-2)//2;q=M.vec(native['monodromy']*M.col(box[n+1:-1]));norm=sum((v*v for v in q),arb(0)).sqrt();need(norm>0 and norm.is_finite(),'positive native6 normalization')
 return native['orbit'][6]+[box[n]]+[v/norm for v in q]+[box[-1]],norm

def choose(direct,returned):
 a=[v['sign']for v in direct if v['join']['passed']and v['join']['coordinates']==26];b=[v['sign']for v in returned if v['join']['passed']and v['join']['coordinates']==26]
 need(len(a)==len(b)==1 and a==b,'unique same full-native mode orientation');return a[0]
