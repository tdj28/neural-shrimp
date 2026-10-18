"""Prepared original-receipt prefix importer; explicit2048-to1024 enclosure conversion."""
import json,gzip
from pathlib import Path

def checked_receipt(path, expected, digest):
    if digest(path) != expected:
        raise ValueError('receipt hash mismatch')
    r = json.loads(Path(path).read_text())
    if (r['returncode'] != 0 or r['failure'] is not None or not r['worker_reaped']
            or not r['worker_group_session_absent']):
        raise ValueError('input producer not successfully closed')
    inventory = {v['path']: v for v in r['output_files']}
    if len(inventory) != len(r['output_files']):
        raise ValueError('duplicate output paths')
    def check(p):
        p = str(Path(p).resolve())
        e = inventory[p]
        if Path(p).stat().st_size != e['bytes'] or digest(p) != e['sha256']:
            raise ValueError('original artifact mismatch')
        return Path(p)
    return r, check


def install_prefix(card, ball, assets, digest, unpack):
    import flint
    from flint import arb, fmpq, ctx
    receipt, check = checked_receipt(card['prefix_receipt'],card['prefix_receipt_sha256'],digest)
    if receipt['source_commit'] != card['prefix_source_commit']:
        raise ValueError('unreviewed prefix producer source')
    admission_path=Path(card['prefix_admission_path'])
    if digest(admission_path)!=card['prefix_admission_sha256']:
        raise ValueError('original prefix admission hash mismatch')
    admission=json.loads(admission_path.read_text())
    if admission['source_commit']!=receipt['source_commit']:
        raise ValueError('prefix admission source mismatch')
    blobs={v['relative_path']:v['sha256'] for v in admission['source_blobs']}
    if len(blobs)!=len(admission['source_blobs']) or blobs.get('codex/certify-llm/ball_qwen.py')!=card['prefix_map_source_sha256']:
        raise ValueError('unreviewed original prefix map source')
    root = Path(card['prefix_output'])
    meta = json.loads(check(root/'inputs.json').read_text())
    if meta['source_commit'] != receipt['source_commit'] or meta['pinned_assets'] != assets:
        raise ValueError('different prefix source/assets')
    if meta['prefix_ids'] != list(ball.prefix_ids) or meta['steering_layer'] != ball.steering_layer:
        raise ValueError('different prefix/layer')
    if meta['precision'] != 2048 or ctx.prec not in (1024,2048) or ball.precision != ctx.prec:
        raise ValueError('different precision')
    for name, v in [('alpha_ratio',ball.alpha),('log10_temperature_ratio',ball.log10_temperature)]:
        pair=meta[name]
        if len(pair)!=2 or any(type(x) is not int for x in pair) or pair[1]<=0 or arb(fmpq(*pair))!=v:
            raise ValueError('different exact parameter')
    if meta['versions'] != {'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__}:
        raise ValueError('different backend')
    if (meta['temperature']!='exp(log(10)*exact_dyadic_log10T)' or
            meta['rotary']!='real base^(-2j/head_dim), real sin/cos'):
        raise ValueError('different real constants')
    # Original producer source/plan/admission pins are additional fixed card inputs.
    # Receiver was just constructed from the hash-checked immutable loader arrays.
    with gzip.open(check(root/'prefix-cache.json.gz'),'rt') as f:
        data=json.load(f)
    if len(data)!=ball.layers:
        raise ValueError('layer count')
    cache=[]
    for layer in data:
        if len(layer)!=2:raise ValueError('K/V pair')
        fields=[]
        for field in layer:
            if len(field)!=len(ball.prefix_ids):raise ValueError('token count')
            tokens=[]
            for token in field:
                if len(token)!=ball.kv_heads:raise ValueError('head count')
                heads=[]
                for head in token:
                    if len(head)!=ball.hdim:raise ValueError('head dimension')
                    heads.append([ball.scalar(unpack(v)) for v in head])
                tokens.append(heads)
            fields.append(tokens)
        cache.append(tuple(fields))
    ball.cache=cache

