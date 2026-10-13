"""Pure conditional derivative coverage over externally authenticated records.

No files, receipts, checkpoints or models are loaded. The caller MUST first
verify original shard artifacts, successful receipts, source-bundle inventory,
map/assets/cache construction, precision/backend, and frozen candidate design.
Hashes here cross-bind that external obligation; they cannot establish it.

Every producer input is checked using its ORIGINAL packed dyadic endpoints,
before any decoding. Output decoding may widen but never narrow. This module
neither evaluates a derivative nor proves a root, primitive period or attraction.
Inputs must stay immutable for the duration of collection.
"""
from __future__ import annotations

import json

from flint import arb, arb_mat, ctx
import phase_domain as domain
import shooting_gates as gates

ROLE = 'candidate_final_shooting_boxes'
SCHEMA = 'codex-conditional-derivative-coverage-v1'


def _hash(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(v not in '0123456789abcdef' for v in value)):
        raise ValueError('external lowercase SHA256 required: '+name)
    return value


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError('integer required: '+name)
    return value


def _records(values, n, name, *, exact=False):
    if not isinstance(values, list) or len(values) != n:
        raise ValueError('full ordered shape required: '+name)
    result=[]
    for record in values:
        mid,rad=domain._record_parts(record,exact=exact)
        # One spelling of each dyadic prevents display-equivalent hash aliases.
        if record != dict(mid=domain._pair(mid),rad=domain._pair(rad)):
            raise ValueError('canonical packed dyadic required: '+name)
        result.append((mid,rad))
    return result


def _decode(record):
    """Check original endpoints, not an equality-sensitive radius round trip."""
    mid,rad=domain._record_parts(record)
    value=arb(tuple(record['mid']),tuple(record['rad']))
    if not value.is_finite():
        raise ValueError('finite decoded derivative/primal required')
    am,ar=domain._record_parts(domain.pack_ball(value))
    if am-ar>mid-rad or am+ar<mid+rad:
        raise ValueError('decoded enclosure narrowed original endpoints')
    return value


def _nominal(centers,radii,p,n):
    points,rs,mixed_hash=domain._nominal_domain(centers,radii,p,n)
    for row in centers:_records(row,n,'exact nominal centers',exact=True)
    _records(radii,p,'exact nominal radii',exact=True)
    c=[[arb(tuple(v['mid'])) for v in row] for row in centers]
    r=[arb(tuple(v['mid'])) for v in radii]
    # Tuple construction is exact even when midpoint mantissas exceed ctx.prec.
    for row,original in zip(c,centers):
        if [domain.pack_ball(v) for v in row]!=original:
            raise ValueError('nominal center decoding lost exact data')
    if [domain.pack_ball(v) for v in r]!=radii:
        raise ValueError('nominal radius decoding lost exact data')
    return points,rs,mixed_hash,gates.domain_sha256(c,r)


def _matrix_hash(matrix):
    """Same matrix hashing convention as the downstream shooting gate."""
    return gates._matrix_sha256(matrix)


def collect_derivatives(centers,radii,shards,*,period,dimension,
        expected_design_sha256,expected_map_sha256,expected_evaluator_sha256,
        expected_backend_sha256,expected_precision):
    """Assemble exactly one signed e_j column per phase/state coordinate.

    ``expected_evaluator_sha256`` identifies the complete independently pinned
    evaluator SOURCE BUNDLE, not just a convenient top-level file. Expected
    hashes are caller trust anchors, never read back from an untrusted shard.
    ``expected_precision`` is producer precision; current ``ctx.prec`` is the
    possibly different assembly precision. Each shard is a dictionary with:
      role; design/map/evaluator/backend_sha256; precision; phase_index;
      direction_indices; phase_domain_sha256; shooting_domain_sha256;
      actual_inputs; actual_inputs_sha256; directions; derivatives; primal.
    All numerical records use canonical ``{mid:[m,e],rad:[m,e]}`` dyadics.
    directions and derivatives are full dimension-by-shard_width row arrays.
    Extra provenance fields are retained by the shard-content hash.

    Receipt authentication is deliberately outside this small pure API. A
    returned result is conditional on that authentication and on evaluator
    correctness. Existing prototype records have neither the required role nor
    this complete domain/binding contract and are refused.
    """
    p=_integer(period,'period',2);n=_integer(dimension,'dimension')
    precision=_integer(expected_precision,'producer precision',16)
    expected={name:_hash(value,name) for name,value in (
        ('design_sha256',expected_design_sha256),('map_sha256',expected_map_sha256),
        ('evaluator_sha256',expected_evaluator_sha256),('backend_sha256',expected_backend_sha256))}
    points,rs,mixed_hash,shooting_hash=_nominal(centers,radii,p,n)
    if isinstance(shards,(dict,str,bytes)):
        raise ValueError('iterable of whole shard records required')
    seen=[set() for _ in range(p)];matrices=[None]*p;audits=[]
    primal_intersections=[None]*p
    for shard_index,shard in enumerate(shards):
        if not isinstance(shard,dict) or shard.get('role')!=ROLE:
            raise ValueError('prototype/non-final shard role refused')
        if any(shard.get(name)!=value for name,value in expected.items()):
            raise ValueError('mixed design/map/source-bundle/backend binding')
        if type(shard.get('precision')) is not int or shard['precision']!=precision:
            raise ValueError('different producer precision')
        if (shard.get('phase_domain_sha256')!=mixed_hash
                or shard.get('shooting_domain_sha256')!=shooting_hash):
            raise ValueError('nominal exact-domain hash cross-binding mismatch')
        phase=shard.get('phase_index')
        if type(phase) is not int or not 0<=phase<p:
            raise ValueError('phase index outside exact domain')
        columns=shard.get('direction_indices')
        if (not isinstance(columns,list) or not columns
                or any(type(j) is not int or not 0<=j<n for j in columns)
                or columns!=sorted(set(columns))):
            raise ValueError('ordered unique signed basis indices required')
        if seen[phase].intersection(columns):
            raise ValueError('duplicate derivative coverage')
        k=len(columns)
        original=shard.get('actual_inputs')
        boxes=_records(original,n,'original producer state box')
        if shard.get('actual_inputs_sha256')!=domain.canonical_hash(original):
            raise ValueError('original actual-input hash mismatch')
        witnesses=[]
        for coordinate,((mid,rad),c) in enumerate(zip(boxes,points[phase])):
            lower=c-rs[phase]-(mid-rad);upper=(mid+rad)-(c+rs[phase])
            if lower<0 or upper<0:
                raise ValueError('ORIGINAL producer box omits nominal domain')
            witnesses.append(dict(coordinate=coordinate,lower_margin=domain._pair(lower),upper_margin=domain._pair(upper)))
        directions=shard.get('directions');derivatives=shard.get('derivatives')
        if not isinstance(directions,list) or len(directions)!=n:
            raise ValueError('full direction row shape required')
        if not isinstance(derivatives,list) or len(derivatives)!=n:
            raise ValueError('full derivative row shape required')
        for i,row in enumerate(directions):
            vals=_records(row,k,'signed basis direction',exact=True)
            if any(v!=(int(i==j),0) for v,j in zip(vals,columns)):
                raise ValueError('direction is not exact positive coordinate basis')
        primal_parts=_records(shard.get('primal'),n,'full primal output')
        intervals=[(mid-rad,mid+rad) for mid,rad in primal_parts]
        if primal_intersections[phase] is not None:
            intervals=[(max(lo,oldlo),min(hi,oldhi))
                       for (lo,hi),(oldlo,oldhi) in zip(intervals,primal_intersections[phase])]
        if any(lo>hi for lo,hi in intervals):
            raise ValueError('disjoint ORIGINAL primal outputs on shared nominal box')
        primal_intersections[phase]=intervals
        for record in shard['primal']:_decode(record)
        if matrices[phase] is None:matrices[phase]=arb_mat(n,n)
        for i,row in enumerate(derivatives):
            _records(row,k,'full derivative outputs')
            for j,record in zip(columns,row):
                value=_decode(record);matrices[phase][i,j]=value
                # Check the actual native matrix used downstream, not just the
                # transient decoded scalar; assignments must not narrow either.
                m,r=domain._record_parts(record)
                am,ar=domain._record_parts(domain.pack_ball(matrices[phase][i,j]))
                if am-ar>m-r or am+ar<m+r:
                    raise ValueError('assembled derivative narrowed original output')
        seen[phase].update(columns)
        audits.append(dict(shard_index=shard_index,phase_index=phase,direction_indices=columns,
            original_shard_sha256=domain.canonical_hash(shard),
            original_actual_inputs_sha256=shard['actual_inputs_sha256'],
            original_endpoint_witnesses=witnesses,
            original_endpoints_contain_nominal=True,full_primal_retained_in_shard=True))
    missing=[dict(phase_index=i,missing_columns=sorted(set(range(n))-seen[i])) for i in range(p) if len(seen[i])!=n]
    if missing:
        raise ValueError('incomplete phase/direction coverage: '+json.dumps(missing,separators=(',',':')))
    primal_records=[[dict(lower=domain._pair(lo),upper=domain._pair(hi))
                     for lo,hi in phase] for phase in primal_intersections]
    report=dict(schema=SCHEMA,certificate_passed=False,coverage_complete=True,
        original_primal_intersections=primal_records,
        original_primal_intersections_sha256=domain.canonical_hash(primal_records),
        primal_intersection_scope='Original-endpoint consistency only; not a separately authenticated point-image artifact.',
        external_authentication_performed=False,
        conditional_obligation='Caller authenticates complete successful receipts and evaluator source/map/assets/backend; reviewed evaluator encloses each derivative over its original input box. Metadata is not proof.',
        period=p,dimension=n,phase_column_counts=[len(s) for s in seen],total_columns=p*n,
        phase_domain_sha256=mixed_hash,shooting_domain_sha256=shooting_hash,
        source_precision=precision,assembly_precision=ctx.prec,**expected,
        phase_matrix_sha256=[_matrix_hash(m) for m in matrices],shards=audits,
        point_images_supplied=False,root_gate_evaluated=False,
        primitive_period_gate_evaluated=False,attraction_gate_evaluated=False)
    return matrices,report
