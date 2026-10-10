"""Independent interval arithmetic for one same-root unit-pair/flip proof."""
from fractions import Fraction as F
from flint import arb, arb_mat, acb, acb_mat

def need(v,s):
    if not v: raise ValueError(s)

def rat(v):
    need(len(v)==2 and all(type(x)is int for x in v) and abs(v[1])<=10000,'bounded dyadic');return F(v[0])*F(2)**v[1]

def ends(v):
    need(v.get('finite',True),'finite ball');m,r=rat(v['mid']),rat(v['rad']);need(r>=0,'nonnegative radius');return m-r,m+r

def decode(v):
    ends(v);return arb(arb(tuple(v['mid'])),arb(tuple(v['rad'])))

def mat(rows):return arb_mat([[decode(v)for v in row]for row in rows])
def eye(n):return arb_mat([[int(i==j)for j in range(n)]for i in range(n)])
def exact(v):return arb(v.numerator)/v.denominator
def norm(m):return max(sum((abs(m[i,j])for j in range(m.ncols())),arb(0)).upper()for i in range(m.nrows()))

def root_arithmetic(root):
    n,p=root['dimension'],root['period'];size=4*n+3
    need(root['augmented_dimension']==size and len(root['box'])==len(root['center'])==size,'complete root dimensions')
    box=[decode(v)for v in root['box']];radii=[]
    for a,b in zip(root['center'],root['box']):
        need(rat(a['rad'])==0 and rat(a['mid'])==rat(b['mid']),'exact shared center');radii.append(rat(b['rad']))
    need(all(v>0 for v in radii),'positive full box')
    Y,J,f=mat(root['preconditioner']),mat(root['whole_jacobian']),mat(root['value'])
    need(Y.nrows()==Y.ncols()==J.nrows()==J.ncols()==size and f.nrows()==size and f.ncols()==1,'complete matrices')
    need(all(rat(v['rad'])==0 for row in root['preconditioner']for v in row),'exact fixed inverse')
    D=eye(size)-Y*J;correction=-Y*f+D*arb_mat([[arb(0,exact(v))]for v in radii])
    checks=[bool(abs(correction[i,0]).upper()<exact(radii[i]))for i in range(size)]
    need(all(checks)and norm(D)<1,'strict root contraction')
    need(len(root['orbit_enclosures'])==p+1 and root['orbit_enclosures'][0]==root['box'][:n],'exact initial orbit domain')
    divisors=[d for d in range(1,p)if p%d==0]
    for d in divisors:
        need(len(root['orbit_enclosures'][d])==n and any(not(decode(v)-box[i]).contains(0)for i,v in enumerate(root['orbit_enclosures'][d])),'proper divisor excluded')
    need(box[n]>0 and box[n+1]>0 and box[3*n+2]>0 and box[3*n+2]<arb(1)/4,'positive rates and nonreal unit pair')
    return dict(inclusions=len(checks),contraction=norm(D),divisors=divisors)

def validated_solve(A,b):
    """Exact midpoint preconditioner; validated inverse of the interval product."""
    n=A.nrows();need(n==A.ncols()and b.nrows()==n,'square interval system')
    center=arb_mat([[A[i,j].mid()for j in range(n)]for i in range(n)])
    raw=center.inv();R=arb_mat([[raw[i,j].mid()for j in range(n)]for i in range(n)])
    inverse=(R*A).inv()*R
    need(all(inverse[i,j].is_finite()for i in range(n)for j in range(n)),'validated preconditioned inverse')
    residual=A*inverse-eye(n)
    need(all(residual[i,j].contains(0)for i in range(n)for j in range(n)),'inverse identity enclosure')
    return inverse*b

def spectrum_crossing(root,proposal):
    n=root['dimension'];k=3*n+2;box=[decode(v)for v in root['box']]
    M,J=mat(root['monodromy']),mat(root['whole_jacobian']);need(M.nrows()==M.ncols()==n,'full native monodromy')
    q,r,w,t=box[n+2:2*n+2],box[2*n+2:k],box[k+1:],box[k]
    columns=proposal['coordinate_columns'];need(len(columns)==len(set(columns))==n-3 and all(type(i)is int and 0<=i<n for i in columns),'integer coordinate basis')
    frame=arb_mat([[q[i],r[i],w[i]]+[arb(int(i==j))for j in columns]for i in range(n)])
    transformed=frame.inv()*M*frame;need(all(transformed[i,j].is_finite()for i in range(n)for j in range(n)),'validated frame inverse')
    # Direct trig is independent of the producer's entire-series enclosure.
    angle=t.sqrt();c=angle.cos();e=angle.sin()/angle;d=t*e
    critical=arb_mat([[c,e,0],[-d,c,0],[0,0,-1]])
    need(all(transformed[i,j].contains(0)for i in range(3,n)for j in range(3)),'conditional critical invariance')
    need(all(transformed[i,j].overlaps(critical[i,j])for i in range(3)for j in range(3)),'conditional critical block')
    # Exact invariance follows from the SAME proved augmented root, not overlap.
    Q=acb_mat([[acb(transformed[i,j])for j in range(3,n)]for i in range(3,n)])
    rows=proposal['spectral_exclusion']['proposal_basis'];need(len(rows)==n-3 and all(len(row)==n-3 for row in rows),'full complex basis')
    need(all(rat(v[p]['rad'])==0 for row in rows for v in row for p in ['real','imag']),'exact complex proposal')
    V=acb_mat([[acb(decode(v['real']),decode(v['imag']))for v in row]for row in rows]);T=V.inv()*Q*V
    bounds=[sum((abs(T[i,j])for j in range(n-3)),arb(0)).upper()for i in range(n-3)]
    need(all(v.is_finite()and v<1 for v in bounds),'all complementary eigenvalues attract')
    section=arb_mat([[J[i,k]if j==n else J[i,j]for j in range(k)]for i in range(k)])
    tangent=validated_solve(section,arb_mat([[-J[i,n]]for i in range(k)]));physical=[arb(1)if i==n else tangent[i,0]for i in range(n+2)]
    border=arb_mat([[M[j,i]+int(i==j)if i<n and j<n else w[i]if i<n else w[j]if j<n else arb(0)for j in range(n+1)]for i in range(n+1)])
    left=validated_solve(border,arb_mat([[int(i==n)]for i in range(n+1)]))
    derivative=sum((left[i,0]*sum((J[k+i,j]*physical[j]for j in range(n+2)),arb(0))for i in range(n)),arb(0))
    need(derivative.is_finite()and not derivative.contains(0),'nonzero total physical crossing')
    return dict(complement_bounds=bounds,crossing=derivative)

def strict_nested(outer,inner):
    need(len(outer)==len(inner)==55,'full55 identity');checks=[]
    for a,b in zip(outer,inner):
        al,ah=ends(a);bl,bh=ends(b);checks.append(al<bl<=bh<ah)
    need(all(checks),'strict fresh-in-original containment');return checks
