"""Isolated period2/3 centre proposal; fixed dyadic A/B and rigorous point images.

No certificate gates are asserted. Original supervisor receipts remain the
input authority; they are never relabelled as historical P8 launcher receipts.
"""
import argparse, gzip, json, sys, time
from pathlib import Path
import numpy as np


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
    root = Path(card['prefix_output'])
    meta = json.loads(check(root/'inputs.json').read_text())
    if meta['source_commit'] != receipt['source_commit'] or meta['pinned_assets'] != assets:
        raise ValueError('different prefix source/assets')
    if meta['prefix_ids'] != list(ball.prefix_ids) or meta['steering_layer'] != ball.steering_layer:
        raise ValueError('different prefix/layer')
    if meta['precision'] != ctx.prec or ball.precision != ctx.prec:
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


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--card',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();card=json.loads(Path(args.card).read_text());root=Path(card['repository'])
    sys.path[:0]=[str(root/'codex/certify-llm'),str(root/'codex/crosscheck')]
    import torch
    from flint import arb_mat,ctx
    from ball_qwen import BallQwen,exact
    from ball_linear import DyadicWeightCache
    from cache_io import digest,unpack
    from shooting_correction import correct
    from qwen_map import ManualQwenSoftMap,PREFIX_IDS,PINNED_SHA256,SNAPSHOT,VECTOR_FILE
    p=card['period']
    if type(p) is not int or p not in (2,3) or card['precision']!=2048 or card['corrections']!=1:
        raise ValueError('unreviewed period/precision/pass count')
    if (card['alpha'],card['log10T'])!=(.06468,-.148):raise ValueError('different cell')
    for e in card['inputs']:
        if digest(e['path'])!=e['sha256']:raise ValueError('input mismatch')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);ctx.prec=2048;ctx.threads=1
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def write(name,obj):
        (out/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    def pack(v):
        if not v.is_finite():raise ValueError('nonfinite ball')
        return {'mid':list(map(int,v.mid().man_exp())),'rad':list(map(int,v.rad().man_exp()))}
    _,check=checked_receipt(card['proposal_receipt'],card['proposal_receipt_sha256'],digest)
    with np.load(check(card['preconditioner']),allow_pickle=False) as z:
        if set(z.files)!={'A','B','centers','parameters','product','left_inverse_defect','right_inverse_defect'}:
            raise ValueError('preconditioner schema')
        arrays={k:z[k].copy() for k in z.files}
    if (arrays['A'].shape!=(p,896,896) or arrays['B'].shape!=(896,896)
            or arrays['centers'].shape!=(p,896) or not np.array_equal(arrays['parameters'],[.06468,-.148])
            or any(not np.isfinite(v).all() for v in arrays.values())):
        raise ValueError('invalid proposals')
    centers=[[exact(v) for v in row] for row in arrays['centers']]
    if card.get('centers_input'):
        # Fixed hash must be in the card; each successive attempt remains separate.
        e=card['centers_input']
        if digest(e['path'])!=e['sha256']:raise ValueError('centre hash')
        centers=[[unpack(v) for v in row] for row in json.loads(Path(e['path']).read_text())]
    if len(centers)!=p or any(len(row)!=896 or any(not v.is_exact() for v in row) for row in centers):
        raise ValueError('exact full centres required')
    matrices=[arb_mat(896,896,[exact(v) for v in A.flat]) for A in arrays['A']]
    B=arb_mat(896,896,[exact(v) for v in arrays['B'].flat])
    model=ManualQwenSoftMap.from_pinned_files();weights={k:t.numpy() for k,t in model.weights.items()}
    for w in weights.values():w.flags.writeable=False
    vector=model.vector.numpy();vector.flags.writeable=False
    ball=BallQwen(model.config,weights,PREFIX_IDS,vector,alpha=card['alpha'],log10_temperature=card['log10T'],
                  linear_options={'chunk_rows':64,'chunk_cols':896,'cache':DyadicWeightCache(max_elements=262144)})
    install_prefix(card,ball,PINNED_SHA256,digest,unpack)
    summaries=[]
    for label in ('before','after'):
        images=[]
        write(label+'-centers.json',[[pack(v) for v in row] for row in centers])
        for i,c in enumerate(centers):
            tick=time.monotonic();values,tangent=ball.forward(c)
            if len(values)!=896 or len(tangent)!=896 or any(not v.is_finite() for v in values) or any(not v.is_exact() or v!=0 for v in tangent):
                raise ValueError('invalid full point image')
            write(f'{label}-phase-{i}.json',dict(value=[pack(v) for v in values],tangent=[pack(v) for v in tangent],seconds=time.monotonic()-tick))
            images.append(values)
        residuals=[[images[i][j]-centers[(i+1)%p][j] for j in range(896)] for i in range(p)]
        upper=max(v.abs_upper() for row in residuals for v in row)
        record=dict(label=label,residual_inf_upper=pack(upper),residuals=[[pack(v) for v in row] for row in residuals])
        write(label+'-residuals.json',record);summaries.append(dict(label=label,residual_inf_upper=pack(upper)))
        if label=='before':
            proposal=correct(centers,images,matrices,B);write('correction-witnesses.json',proposal['record']);centers=proposal['centers']
    for e in card['inputs']:
        if digest(e['path'])!=e['sha256']:raise ValueError('postrun input mismatch')
    for name,h in PINNED_SHA256.items():
        if digest(VECTOR_FILE if name=='vectors.npz' else SNAPSHOT/name)!=h:raise ValueError('postrun asset mismatch')
    write('report.json',dict(period=p,precision=ctx.prec,summaries=summaries,seconds=time.monotonic()-start,certificate_passed=False))

if __name__=='__main__':main()
