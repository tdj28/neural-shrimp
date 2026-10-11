"""Exact saved gauge containment only; no new interval/native acquisition."""
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path

ROOT = Path('/data2/neural-shrimp-data/audits/native705-correlated-independent/attempt002')
SOURCE = 'd70eaaf9abd34ea65f1946c7727912b89a57ce55'
OUTER = 'db0c695ee52bd019239f683a7158701ad82be9728fcbd0af99359af8c9b66b79'
WORKER = '12157fcd7d23f7aff29c2a8a317ec2a7999348f6162b3347f5fcf8af1537332c'
MEMBERSHIP = 'cfbc127d130f098e31976db4647bff6ccb10e352a09a9406b2e24c0e0774c197'


def digest(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def check():
    read = lambda p: json.loads(p.read_bytes())
    assert digest(ROOT / 'manifest.json')['sha256'] == OUTER
    assert digest(ROOT / 'run/manifest.json')['sha256'] == WORKER
    outer = read(ROOT / 'manifest.json')['files']
    worker = read(ROOT / 'run/manifest.json')['files']
    assert outer['run/manifest.json'] == digest(ROOT / 'run/manifest.json')
    for name in ['known-root-membership.json', 'runtime.json', 'input-binding.json', 'outcome.json']:
        assert worker[name] == outer['run/' + name] == digest(ROOT / 'run' / name)
    rt = read(ROOT / 'run/runtime.json')
    assert rt['source_commit'] == SOURCE
    assert read(ROOT / 'run/outcome.json')['passed']
    known = read(ROOT / 'run/known-root-membership.json')
    assert digest(ROOT / 'run/known-root-membership.json')['sha256'] == MEMBERSHIP
    assert known['passed'] and known['parameter_inside'] and known['coordinates'] == 705
    z = known['projection']
    dyadic = lambda x: F(x[0]) * F(2) ** x[1]
    mid, rad = dyadic(z['mid']), dyadic(z['rad'])
    assert rad >= 0
    lo, hi = mid - rad, mid + rad
    first, second = (F(-49, 2048), F(-17, 2048)), (F(-17, 2048), F(15, 2048))
    contains = lambda bounds: bounds[0] <= lo <= hi <= bounds[1]
    assert not contains(first) and contains(second) and first[1] == second[0]
    return dict(
        schema='exact-saved-chart30-to-chart31-stable-overlap-v1',
        native_calls=0, root=str(ROOT), source=SOURCE, outer=OUTER, worker=WORKER,
        known_membership=digest(ROOT / 'run/known-root-membership.json'),
        previous_independent=read(ROOT / 'run/input-binding.json')['card']['independent'],
        projection=z, exact_projection_endpoints=[str(lo), str(hi)],
        first_quarter=list(map(str, first)), second_quarter=list(map(str, second)),
        first_quarter_contains_previous_endpoint=False,
        second_quarter_contains_previous_endpoint=True, quarters_exactly_adjacent=True,
        authority='The previously independently proved full705 anchor membership stores absolute normal·(U−origin); its helper subtracts the center only afterwards.',
        scope='First quarter is a separately proved stable arc on the same known root curve. Continuous complementary stability from the first30 charts requires covering the second quarter, which contains the complete previous endpoint projection. This exact metadata check supplies no new spectral proof.'
    )


if __name__ == '__main__':
    print(json.dumps(check(), indent=2, sort_keys=True))
