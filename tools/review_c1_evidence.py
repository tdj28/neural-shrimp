"""Independent C1 review: provenance, stored witnesses, higher precision, and AD.

The scalar objective is differentiated with second-order Arb jets, independently
of the producer's hand-coded gradient/Hessian. This does not formalize Arb.
"""
import argparse
from dataclasses import asdict
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from flint import arb, arb_mat, ctx


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rational(value):
    v = Fraction(value)
    return arb(v.numerator) / arb(v.denominator)


def stored(value):
    return arb(arb(tuple(value['mid'])), arb(tuple(value['rad'])))


def matrix(values):
    return arb_mat([[stored(v) for v in row] for row in values])


class Jet:
    def __init__(self, value, n, index=None):
        self.v = value if isinstance(value, arb) else arb(value)
        self.g = [arb(int(i == index)) for i in range(n)]
        self.h = [[arb(0) for _ in range(n)] for _ in range(n)]
        self.n = n

    def coerce(self, other):
        return other if isinstance(other, Jet) else Jet(other, self.n)

    def __add__(self, other):
        b = self.coerce(other)
        out = Jet(self.v + b.v, self.n)
        out.g = [a + c for a, c in zip(self.g, b.g)]
        out.h = [[self.h[i][j] + b.h[i][j] for j in range(self.n)] for i in range(self.n)]
        return out

    __radd__ = __add__

    def __mul__(self, other):
        b = self.coerce(other)
        out = Jet(self.v * b.v, self.n)
        out.g = [self.g[i] * b.v + self.v * b.g[i] for i in range(self.n)]
        out.h = [[self.h[i][j] * b.v + self.v * b.h[i][j]
                  + self.g[i] * b.g[j] + b.g[i] * self.g[j]
                  for j in range(self.n)] for i in range(self.n)]
        return out

    __rmul__ = __mul__

    def tanh(self):
        out = Jet(self.v.tanh(), self.n)
        first = 1 - out.v * out.v
        second = -2 * out.v * first
        out.g = [first * a for a in self.g]
        out.h = [[first * self.h[i][j] + second * self.g[i] * self.g[j]
                  for j in range(self.n)] for i in range(self.n)]
        return out


def objective_ad(inputs, theta):
    spec = inputs['network']
    d, w, bias, n = spec['d'], spec['width'], spec['bias'], len(theta)
    if not bias:
        raise ValueError('reviewer AD is scoped to the supplied biased S2 maps')
    ins, out, scale, ridge = [rational(spec[k]) for k in ('in_scale', 'out_scale', 'loss_scale', 'ridge')]
    variables = [Jet(t, n, i) for i, t in enumerate(theta)]
    loss = Jet(0, n)
    for xs, ys in zip(inputs['x'], inputs['y']):
        prediction = variables[-1]
        for h in range(w):
            activation = variables[d * w + h]
            for k in range(d):
                activation = activation + ins * rational(xs[k]) * variables[k * w + h]
            prediction = prediction + out * variables[d * w + w + h] * activation.tanh()
        residual = prediction + (-rational(ys))
        loss = loss + scale / len(inputs['x']) * residual * residual
    for variable in variables:
        loss = loss + ridge / 2 * variable * variable
    rates = [rational(inputs['lr_hidden'])] * (d * w + w) + [rational(inputs['lr_output'])] * (w + 1)
    return ([theta[i] - rates[i] * loss.g[i] for i in range(n)],
            arb_mat([[int(i == j) - rates[i] * loss.h[i][j] for j in range(n)] for i in range(n)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--producer-root', type=Path, required=True)
    parser.add_argument('--producer-commit', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.producer_root
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != args.producer_commit:
        raise ValueError('producer tip changed')
    sys.path.insert(0, str(root / 'src'))
    from neural_shrimp.rigorous_orbits import BallTanhMap, check_orbit
    from neural_shrimp.trainability import SYSTEMS, array_sha256
    reports = []
    for row_index in (35, 121):
        receipt_path = root / f'certificates/c1-row{row_index}-receipt.json'
        receipt = json.loads(receipt_path.read_text())
        completed = subprocess.run([sys.executable, str(root / 'tools/verify_certificates.py'), str(receipt_path)],
                                   cwd=root, capture_output=True, text=True, timeout=30, check=True)
        replay = json.loads(completed.stdout)[0]
        raw = Path(receipt['external_root'])
        inputs = json.loads((raw / 'inputs.json').read_text())
        result = json.loads((raw / 'certificate.json').read_text())['result']
        provenance = inputs['provenance']
        for key in ('certificate', 'plan'):
            assert sha(Path(provenance[key + '_file']).read_bytes()) == provenance[key + '_sha256']
        historical = json.loads(Path(provenance['certificate_file']).read_text())[row_index]
        assert inputs['period'] == historical['period']
        for key in ('lr_hidden', 'lr_output'):
            assert Fraction(inputs[key]) == Fraction(historical[key])
        assert [Fraction(x) for x in inputs['approximation']] == [Fraction(x) for x in historical['certificate']['orbit'][0]]
        plan = json.loads(Path(provenance['plan_file']).read_text())
        spec = next(s for s in plan['maps'] if s['map_id'] == provenance['map_id'].removesuffix('-zoom'))
        net, x, y, init = SYSTEMS[spec['system']](spec['seed'])
        assert asdict(net) == spec['network']
        assert {'x': array_sha256(x), 'y': array_sha256(y), 'theta0': array_sha256(init)} == spec['data_sha256'] == provenance['data_sha256']
        assert [[Fraction(v) for v in r] for r in inputs['x']] == [[Fraction(float(v)) for v in r] for r in x]
        assert [Fraction(v) for v in inputs['y']] == [Fraction(float(v)) for v in y]
        for key, value in asdict(net).items():
            assert (Fraction(inputs['network'][key]) == Fraction(value)) if isinstance(value, float) else inputs['network'][key] == value
        with ctx.workprec(384):
            box, center = [stored(v) for v in result['box']], [stored(v) for v in result['center']]
            krawczyk = [stored(v) for v in result['krawczyk_image']]
            assert all(b.contains_interior(k) for b, k in zip(box, krawczyk))
            defect = matrix(result['defect'])
            contraction = max(sum(abs(defect[i,j]).upper() for j in range(len(box))).upper() for i in range(len(box)))
            assert contraction < 1
            power = matrix(result['monodromy'])
            for _ in range(6):
                power = power * power
            norm = max(sum(abs(power[i,j]).upper() for j in range(len(box))).upper() for i in range(len(box)))
            assert norm < 1
            orbit = [[stored(v) for v in state] for state in result['orbit_enclosures']]
            for divisor in (1, 2, 5):
                assert any(not (orbit[divisor][j] - box[j]).contains(0) for j in range(len(box)))
            # Independent jet differentiation at center and an interior point.
            model = BallTanhMap(inputs)
            enclosure, jacobian = model.step(box)
            for sign in (0, 1):
                sample = [c + sign * (-1) ** i * arb((1, -82)) for i, c in enumerate(center)]
                v, j = objective_ad(inputs, sample)
                assert all(enclosure[i].contains(v[i]) for i in range(len(box)))
                assert all(jacobian[i,k].contains(j[i,k]) for i in range(len(box)) for k in range(len(box)))
            stronger = check_orbit(inputs, precision=384, radius_exponent=-90, newton_steps=6, max_power=512)
            assert stronger['passed']
            assert all(b.contains_interior(stored(s)) for b, s in zip(box, stronger['box']))
        reports.append({'row': row_index, 'receipt_sha256': sha(receipt_path.read_bytes()),
                        'certificate_sha256': replay['certificate_sha256'], 'producer_source_commit': receipt['source_commit'],
                        'producer_source_sha256': receipt['source_sha256'], 'authoritative_replay': 'passed',
                        'exact_input_and_frozen_plan_binding': 'passed', 'stored_inclusion_contraction_divisors_M64': 'passed',
                        'independent_Arb_second_order_AD_samples': 2, 'precision_384_radius_2_neg90_nested_proof': 'passed',
                        'stronger_result_sha256': sha(json.dumps(stronger, sort_keys=True).encode())})
    result = {'schema': 'neural-shrimp-c1-independent-review-v1',
              'reviewed_producer_tip': args.producer_commit,
              'reviewer_source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'reviewer_code_sha256': sha(Path(__file__).read_bytes()),
              'verdict': 'no_blocking_findings_for_stated_exact_map_claims', 'blocking_findings': [],
              'checks': reports,
              'scope': 'Unique local primitive period-10 attracting orbit of exact-real GD at each of two specified exact-dyadic S2 rate settings.',
              'trusted': ['Banach contraction theorem', 'correctness of Arb/FLINT enclosures and Python bindings', 'analytic chain and product rules', 'rho(M)^64 <= norm_inf(M^64)'],
              'not_established': ['floating-point training dynamics', 'surrounding chaos', 'shrimp geometry', 'invariant circles', 'training utility', 'formal proof-assistant verification'],
              'review_limits': 'Independent reasoning and second-order AD of the objective; same Arb library. Sampled AD agreement supplements mathematical code inspection and does not by itself prove implementation equivalence.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'verdict': result['verdict'], 'rows': len(reports), 'output': str(args.out)}))


if __name__ == '__main__':
    main()
