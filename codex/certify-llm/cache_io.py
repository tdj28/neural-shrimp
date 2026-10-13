"""Reuse exact saved prefix enclosures only through an explicit provenance chain.

This is not independent proof of the original evaluator. The caller pins the
reviewed evaluator's source and receipt hashes, and trusts its Arb arithmetic.
The decoded values keep their full radii; state tangents are exactly zero.
Receiver construction must independently bind the pinned assets to the config,
weight dictionary and steering vector passed here. This module cannot establish
a cryptographic file-hash-to-array relationship from a metadata declaration.
"""
import gzip
import hashlib
import json
from pathlib import Path
import flint
from flint import arb,fmpq,ctx


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def unpack(data):
    if not isinstance(data,dict) or set(data)!= {'mid','rad'}:
        raise ValueError('exact midpoint/radius encoding required')
    for key in ['mid','rad']:
        v=data[key]
        if not isinstance(v,list) or len(v)!=2 or any(type(x) is not int for x in v):
            raise ValueError('exact integer mantissa/exponent pairs required')
    if data['rad'][0]<0:raise ValueError('negative radius')
    value=arb(tuple(data['mid']),tuple(data['rad']))
    if not value.is_finite():raise ValueError('nonfinite decoded ball')
    return value


def load_prefix(attempt,model,*,expected_receipt_sha256,expected_map_source_sha256,expected_assets,
                expected_config,expected_weights,expected_vector,expected_steering_layer):
    """Install a checked prefix in an unchanged-parameter BallQwen-compatible map.

    The expected hashes are scientific inputs fixed by the next run card, not
    values guessed from the bundle being validated. Model source compatibility
    must be independently reviewed if its scalar algebra has changed.

    The four expected construction inputs must come from the independently
    hash-checked loader, not be inferred from the receiving map. Equal config
    contents and identical weights/vector objects establish receiver consistency
    with that construction; they do not themselves authenticate array contents.
    These inputs and their aliases must remain immutable during cache use.
    This importer currently supports the reviewed original pilot's fixed layer
    12 only; the source binding is what establishes that historical choice.
    """
    if model.config!=expected_config:raise ValueError('different receiving config')
    if model.weights is not expected_weights:raise ValueError('different receiving weight dictionary')
    if model.vector is not expected_vector:raise ValueError('different receiving steering vector')
    if (type(expected_steering_layer) is not int or expected_steering_layer!=12
            or type(model.steering_layer) is not int or model.steering_layer!=expected_steering_layer):
        raise ValueError('receiving steering layer must equal the reviewed original layer 12')
    attempt=Path(attempt).resolve();receipt_path=attempt/'receipt.json'
    if digest(receipt_path)!=expected_receipt_sha256:raise ValueError('receipt hash mismatch')
    receipt=json.loads(receipt_path.read_text());inventory={v['path']:v for v in receipt['artifacts']}
    if len(inventory)!=len(receipt['artifacts']):raise ValueError('duplicate artifact paths')
    def checked(relative):
        path=(attempt/relative).resolve()
        if not path.is_relative_to(attempt) or relative not in inventory:raise ValueError('unbound artifact path')
        entry=inventory[relative]
        if path.stat().st_size!=entry['bytes'] or digest(path)!=entry['sha256']:raise ValueError('artifact binding mismatch: '+relative)
        return path
    inputs=json.loads(checked('run/inputs.json').read_text())
    source=checked('source/codex/certify-llm/ball_qwen.py')
    if digest(source)!=expected_map_source_sha256:raise ValueError('unreviewed source graph')
    if inputs['source_commit']!=receipt['source_commit']:raise ValueError('source commit mismatch')
    if inputs['pinned_assets']!=expected_assets:raise ValueError('different pinned assets')
    if tuple(inputs['prefix_ids'])!=model.prefix_ids:raise ValueError('different prefix')
    for field,value in [('alpha_ratio',model.alpha),('log10_temperature_ratio',model.log10_temperature)]:
        pair=inputs[field]
        if len(pair)!=2 or any(type(v) is not int for v in pair) or pair[1]<=0 or not arb(fmpq(*pair))==value:
            raise ValueError('different exact parameter: '+field)
    if inputs['precision']!=ctx.prec or model.precision!=ctx.prec:raise ValueError('different precision')
    versions=inputs['versions']
    if versions['python_flint']!=flint.__version__ or versions['flint']!=flint.__FLINT_VERSION__:
        raise ValueError('different arithmetic backend')
    if inputs['temperature']!='exp(log(10)*exact_dyadic_log10T)' or inputs['rotary']!='real base^(-2j/head_dim), real sin/cos':
        raise ValueError('different exact-real constants convention')
    cache_path=checked('run/prefix-cache.json.gz')
    with gzip.open(cache_path,'rt') as f:data=json.load(f)
    if not isinstance(data,list) or len(data)!=model.layers:raise ValueError('wrong layer count')
    cache=[]
    for layer in data:
        if not isinstance(layer,list) or len(layer)!=2:raise ValueError('K/V pair required')
        fields=[]
        for field in layer:
            if not isinstance(field,list) or len(field)!=len(model.prefix_ids):raise ValueError('wrong prefix length')
            tokens=[]
            for token in field:
                if not isinstance(token,list) or len(token)!=model.kv_heads:raise ValueError('wrong KV head count')
                heads=[]
                for head in token:
                    if not isinstance(head,list) or len(head)!=model.hdim:raise ValueError('wrong head dimension')
                    heads.append([model.scalar(unpack(v)) for v in head])
                tokens.append(heads)
            fields.append(tokens)
        cache.append(tuple(fields))
    # No midpoint projection, dropped radius, or imported floating cache.
    model.cache=cache
    return dict(source_commit=receipt['source_commit'],receipt_sha256=expected_receipt_sha256,
                graph_sha256=expected_map_source_sha256,prefix_sha256=digest(cache_path),precision=ctx.prec,
                receiver_construction_checked=True,steering_layer=expected_steering_layer,
                hashes_to_arrays_authenticated_by_this_loader=False)
