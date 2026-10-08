"""Independent expanded-polynomial and cubic-conjugacy review of scalar NS.

Never imports the producer's Jet, derivative, inclusion or coefficient code
for the independent proof path. Arb/Acb remains shared arithmetic trust.
The official verifier is imported only for separately labelled replay/tamper
checks. All outputs, including tampered copies, live in an exclusive directory.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
from flint import arb, acb, arb_mat, acb_mat, ctx
import flint

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
ZERO=(0,0,0)


def canonical(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def sha(data):return hashlib.sha256(data).hexdigest()
def exact(value):
    f=Fraction(value);return arb(f.numerator)/arb(f.denominator)
def enc(value):
    assert value.is_finite()
    return {'mid':[int(x) for x in value.mid().man_exp()],
            'rad':[int(x) for x in value.rad().man_exp()], 'display':value.str(25)}
def decode(value):return arb(arb(tuple(value['mid'])),arb(tuple(value['rad'])))
def cenc(value):return {'real':enc(value.real),'imag':enc(value.imag)}
def write(path,value):
    with path.open('xb') as h:h.write(canonical(value));h.flush();os.fsync(h.fileno())


class Polynomial:
    """Complete rational polynomial; no Taylor truncation or interval jets."""
    def __init__(self,terms):self.terms={a:Fraction(v) for a,v in terms.items() if v}
    @staticmethod
    def coerce(other):return other if isinstance(other,Polynomial) else Polynomial({ZERO:other})
    def __add__(self,other):
        other=self.coerce(other);out=self.terms.copy()
        for a,v in other.terms.items():out[a]=out.get(a,Fraction(0))+v
        return Polynomial(out)
    __radd__=__add__
    def __neg__(self):return Polynomial({a:-v for a,v in self.terms.items()})
    def __sub__(self,other):return self+-self.coerce(other)
    def __rsub__(self,other):return self.coerce(other)+-self
    def __mul__(self,other):
        other=self.coerce(other);out={}
        for a,x in self.terms.items():
            for b,y in other.terms.items():
                k=tuple(a[i]+b[i] for i in range(3));out[k]=out.get(k,Fraction(0))+x*y
        return Polynomial(out)
    __rmul__=__mul__
    def diff(self,*axes):
        result=self
        for axis in axes:
            out={}
            for a,v in result.terms.items():
                if a[axis]:
                    k=list(a);k[axis]-=1;out[tuple(k)]=v*a[axis]
            result=Polynomial(out)
        return result
    def evaluate(self,z):
        out=arb(0)
        for a,v in self.terms.items():
            term=arb(v.numerator)/arb(v.denominator)
            for i,power in enumerate(a):term*=z[i]**power
            out+=term
        return out
    def serial(self):return [{'powers':a,'coefficient':str(v)} for a,v in sorted(self.terms.items())]


def build(inputs):
    u,v,e=[Polynomial({tuple(int(i==j) for i in range(3)):1}) for j in range(3)]
    d,l,r=[Fraction(inputs[k]) for k in ('target','ridge','ratio')]
    # Independently expanded gradient of (uv-d)^2/2 + l*(u^2+v^2)/2.
    f=(u-e*l*u-e*u*v*v+e*d*v, v-r*e*l*v-r*e*u*u*v+r*e*d*u)
    a,b=f
    g=(a-e*l*a-e*a*b*b+e*d*b, b-r*e*l*b-r*e*a*a*b+r*e*d*a)
    det=g[0].diff(0)*g[1].diff(1)-g[0].diff(1)*g[1].diff(0)
    return f,g,(g[0]-u,g[1]-v,det-1),det


def evaluate_derivatives(g,z):
    A=[[g[i].diff(j).evaluate(z) for j in range(2)] for i in range(2)]
    B=[[[g[i].diff(j,k).evaluate(z) for k in range(2)] for j in range(2)] for i in range(2)]
    C=[[[[g[i].diff(j,k,l).evaluate(z) for l in range(2)] for k in range(2)] for j in range(2)] for i in range(2)]
    return A,B,C


def independent_proof(inputs,certificate,precision,radius_exponent):
    f,g,H,D=build(inputs)
    with ctx.workprec(precision):
        center=[exact(v) for v in certificate['result']['inclusion']['center_rational']]
        box=[arb(v,arb((1,radius_exponent))) for v in center]
        original=[decode(v) for v in certificate['result']['inclusion']['box']]
        nested=all(original[i].contains_interior(box[i]) for i in range(3))
        J0=arb_mat([[H[i].diff(j).evaluate(center) for j in range(3)] for i in range(3)])
        proposal_inverse=J0.inv()
        R=arb_mat([[proposal_inverse[i,j].mid() for j in range(3)] for i in range(3)])
        JB=arb_mat([[H[i].diff(j).evaluate(box) for j in range(3)] for i in range(3)])
        defect=arb_mat([[int(i==j) for j in range(3)] for i in range(3)])-R*JB
        q=max(sum((abs(defect[i,j]).upper() for j in range(3)),arb(0)).upper() for i in range(3))
        residual=arb_mat([[H[i].evaluate(center)] for i in range(3)])
        # Use the actual outward-rounded box radii, not merely requested radii.
        displacement=arb_mat([[arb(0,box[i].rad())] for i in range(3)])
        image=-R*residual+defect*displacement
        inclusions=[box[i].contains_interior(center[i]+image[i,0]) for i in range(3)]
        differences=[f[i].evaluate(box)-box[i] for i in range(2)]
        primitive=any(not value.contains(0) for value in differences)
        A,B,C=evaluate_derivatives(g,box)
        trace=A[0][0]+A[1][1];det=D.evaluate(box)
        radicand=4*det-trace*trace
        nonresonance=all(not (trace-x).contains(0) for x in (-2,-1,0,2))
        branch_matrix=arb_mat([[A[i][j]-int(i==j) for j in range(2)] for i in range(2)])
        parameter=arb_mat([[g[i].diff(2).evaluate(box)] for i in range(2)])
        branch=-branch_matrix.solve(parameter)
        slope=D.diff(2).evaluate(box)+sum((D.diff(i).evaluate(box)*branch[i,0] for i in range(2)),arb(0))
        mu=acb(trace,radicand.sqrt())/2
        rawq=[acb(A[0][1]),mu-A[0][0]]
        norm=sum(((w.conjugate()*w).real for w in rawq),arb(0)).sqrt()
        right=[w/norm for w in rawq]
        # Independent direct transpose eigenvector, not inverse[q,qbar].
        rawp=[acb(A[1][0]),mu.conjugate()-A[0][0]]
        overlap=sum((rawp[i].conjugate()*right[i] for i in range(2)),acb(0))
        assert not overlap.contains(0)
        left=[w/overlap.conjugate() for w in rawp]
        pairing=lambda values:sum((left[i].conjugate()*values[i] for i in range(2)),acb(0))
        qb=[w.conjugate() for w in right]
        bil=lambda x,y:[sum((B[i][j][k]*x[j]*y[k] for j in range(2) for k in range(2)),acb(0)) for i in range(2)]
        tri=lambda x,y,z:[sum((C[i][j][k][l]*x[j]*y[k]*z[l] for j in range(2) for k in range(2) for l in range(2)),acb(0)) for i in range(2)]
        g20,g11,g02,g21=pairing(bil(right,right)),pairing(bil(right,qb)),pairing(bil(qb,qb)),pairing(tri(right,right,qb))
        denominators=[mu*mu-mu,mu*mu.conjugate()-mu,mu.conjugate()**2-mu]
        assert all(not value.contains(0) for value in denominators)
        h20,h11,h02=(g20/2/denominators[0],g11/denominators[1],g02/2/denominators[2])
        # Direct cubic coefficient of f o (id+h2) = (id+h2) o N,
        # where N has no quadratic terms. No resolvent or planar ell formula.
        cubic=g21/2+g20*h11+g11*(h11.conjugate()+h20)+g02*h02.conjugate()
        ell=(mu.conjugate()*cubic).real
        producer_coefficient=certificate['result']['local']['coefficient']
        overlaps={name:(ell-decode(producer_coefficient[name])).contains(0) for name in ('ell','planar_ell')}
        right_res=[sum((A[i][j]*right[j] for j in range(2)),acb(0))-mu*right[i] for i in range(2)]
        left_res=[sum((A[j][i]*left[j] for j in range(2)),acb(0))-mu.conjugate()*left[i] for i in range(2)]
        normalization=pairing(right)-1
        eigenframe=all(v.contains(0) for v in right_res+left_res+[normalization])
        gates={'nested_in_producer_box':bool(nested),'augmented_inclusion':bool(all(inclusions) and q<1),
          'positive_rate':bool(box[2]>0),'primitive_period_two':bool(primitive),
          'complete_simple_unit_pair_at_root':bool(radicand>0 and trace>-2 and trace<2),
          'nonresonance_orders_1_to_4':bool(nonresonance),'positive_modulus_slope':bool(slope>0),
          'negative_cubic_conjugacy_coefficient':bool(ell<0),'independent_eigenframe_checks':bool(eigenframe),
          'overlap_with_both_producer_formulas':bool(all(overlaps.values()))}
        return {'passed':all(gates.values()),'gates':gates,'precision':precision,'requested_radius_exponent':radius_exponent,
          'box':[enc(v) for v in box],'contraction_upper':enc(q),'inclusion_coordinates':[bool(v) for v in inclusions],
          'primitive_differences':[enc(v) for v in differences],'trace':enc(trace),'determinant':enc(det),
          'determinant_slope':enc(slope),'modulus_slope_at_root':enc(slope/2),'ell_via_cubic_conjugacy':enc(ell),
          'right':[cenc(v) for v in right],'adjoint':[cenc(v) for v in left],
          'coefficient_formula_overlaps':{k:bool(v) for k,v in overlaps.items()},
          'polynomial_terms':[len(p.terms) for p in g],'polynomial_sha256':sha(canonical([p.serial() for p in g])),
          'scope':'same locally unique augmented root in a strictly nested box; exact dyadic coefficients, unknown real eta; no finite parameter tube or explicit invariant curve'}


def artifact_controls(driver,raw,folder):
    folder.mkdir()
    results={}
    for kind in ('relocated_unchanged','rehash_changed_gate','rehash_changed_ell','rehash_changed_provenance','rehash_nondyadic_input'):
        out=folder/kind;shutil.copytree(raw,out)
        out.chmod(0o755)
        for p in out.iterdir():p.chmod(0o644)
        receipt=json.loads((out/'receipt.json').read_text());receipt['external_root']=str(out)
        certificate=json.loads((out/'certificate.json').read_text());inputs=json.loads((out/'inputs.json').read_text())
        if kind=='rehash_changed_gate':certificate['result']['gates']['negative_criticality']=False
        if kind=='rehash_changed_ell':certificate['result']['local']['coefficient']['ell']['mid'][0]*=-1
        if kind=='rehash_changed_provenance':inputs['provenance']['seed_sha256']='0'*64
        if kind=='rehash_nondyadic_input':inputs['target']='6/5'
        (out/'inputs.json').write_bytes(canonical(inputs));certificate['inputs_sha256']=sha(canonical(inputs))
        (out/'certificate.json').write_bytes(canonical(certificate))
        for name in receipt['files']:
            data=(out/name).read_bytes();receipt['files'][name]={'bytes':len(data),'sha256':sha(data)}
        (out/'receipt.json').write_bytes(canonical(receipt))
        try:driver.verify(out/'receipt.json');accepted=True;error=None
        except Exception as exc:accepted=False;error=type(exc).__name__+': '+str(exc)
        results[kind]={'accepted':accepted,'error':error,'expected_acceptance':kind=='relocated_unchanged'}
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    record={'review_status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
       'review_source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'review_source_sha256':sha(Path(__file__).read_bytes()),'pid':os.getpid(),'nice':os.getpriority(os.PRIO_PROCESS,0),
       'python':platform.python_version(),'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__,
       'receipt':str(args.receipt),'receipt_sha256':sha(args.receipt.read_bytes())}
    write(args.output/'runtime-start.json',record)
    try:
        receipt=json.loads(args.receipt.read_text());raw=Path(receipt['external_root'])
        certificate=json.loads((raw/'certificate.json').read_text());inputs=json.loads((raw/'inputs.json').read_text())
        assert receipt['source_commit']=='0652d49529265565078b63360ec76bd9dd5f21a8'
        # Exact target/seed association independently checked, including all proposal dyadics.
        seed_path=Path(inputs['provenance']['seed_path']);seed_bytes=seed_path.read_bytes();seed=json.loads(seed_bytes)
        assert sha(seed_bytes)==inputs['provenance']['seed_sha256']=='cbef5ea46ee1a0253dccbe93fbc4e36523a47f9363533861c8bce09e4381215e'
        for key in ('target','ridge','ratio'):
            assert Fraction(inputs[key])==Fraction.from_float(seed[key])
        assert [Fraction(v) for v in inputs['approximation']]==[Fraction.from_float(v) for v in seed['ns_root']['point']+[seed['ns_root']['eta']]]
        assert inputs['provenance']['case']==15 and Fraction(inputs['ratio'])==4
        spec=importlib.util.spec_from_file_location('official_scalar_ns_reviewed',ROOT/'tools/certify_scalar_ns.py')
        driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
        record['official_replay']=driver.verify(args.receipt)
        record['independent']=independent_proof(inputs,certificate,512,-48)
        record['mutation_controls']=artifact_controls(driver,raw,args.output/'mutations')
        record['review_status']='recomputed'
    except BaseException as error:
        record['review_status']='failed_or_interrupted';record['error']=type(error).__name__+': '+str(error);raise
    finally:
        record['elapsed_seconds']=time.monotonic()-start
        record['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write(args.output/'review.json',record)
        files={str(p.relative_to(args.output)):{'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in args.output.rglob('*') if p.is_file()}
        write(args.output/'manifest.json',files)
    print(json.dumps({'status':record['review_status'],'independent_passed':record['independent']['passed'],
       'mutation_controls':record['mutation_controls'],'elapsed_seconds':record['elapsed_seconds']},indent=2))


if __name__=='__main__':main()
