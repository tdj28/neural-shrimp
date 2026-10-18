"""Task34 exact-cell 2048-bit prefix; original radii retained for later centre work."""
import argparse, gzip, hashlib, json, resource, subprocess, sys, time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def pack(value):
    if not value.is_finite():
        raise ValueError('nonfinite prefix enclosure')
    return {'mid': list(map(int, value.mid().man_exp())),
            'rad': list(map(int, value.rad().man_exp()))}


def encode_prefix(prefix):
    return [[[[[pack(q.value) for q in head] for head in token]
               for token in field] for field in layer] for layer in prefix]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--card', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    card = json.loads(Path(args.card).read_text())
    allowed={'cascade-p3':(.0650,-.14798),'cascade-p6':(.065693,-.14798),'cascade-p12':(.0659717,-.14798),'band-p6':(.0700,-.1404),'stripe-p10':(.06776,-.115)}
    if card.get('cell') not in allowed or (card['alpha'],card['log10T']) != allowed[card['cell']] or card['precision'] != 2048:
        raise ValueError('unreviewed cell or precision')
    for entry in card['inputs']:
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('input changed')
    root = Path(card['repository'])
    sys.path[:0] = [str(root/'codex/crosscheck'), str(root/'codex/certify-llm')]
    import torch, flint
    from flint import ctx
    from qwen_map import ManualQwenSoftMap, PREFIX_IDS, PINNED_SHA256, SNAPSHOT, VECTOR_FILE
    from ball_qwen import BallQwen
    from ball_linear import DyadicWeightCache
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    ctx.prec = 2048
    ctx.threads = 1
    start = time.monotonic()
    model = ManualQwenSoftMap.from_pinned_files()
    weights = {k: t.numpy() for k, t in model.weights.items()}
    for value in weights.values():
        value.flags.writeable = False
    vector = model.vector.numpy()
    vector.flags.writeable = False
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    source = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    inputs = dict(source_commit=source, cell=card['cell'], pinned_assets=PINNED_SHA256,
                  prefix_ids=list(PREFIX_IDS), precision=ctx.prec,
                  alpha_ratio=list(card['alpha'].as_integer_ratio()),
                  log10_temperature_ratio=list(card['log10T'].as_integer_ratio()),
                  versions={'python_flint': flint.__version__, 'flint': flint.__FLINT_VERSION__},
                  temperature='exp(log(10)*exact_dyadic_log10T)',
                  rotary='real base^(-2j/head_dim), real sin/cos', steering_layer=12)
    (out/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
    layers = []
    with (out/'layers.jsonl').open('x') as events:
        def progress(kind, index, hidden):
            if kind != 'prefix_end':
                return
            values = [q.value for row in hidden for q in row]
            row = dict(layer=index, seconds=time.monotonic()-start,
                       maximum_radius=pack(max(v.rad() for v in values)))
            layers.append(index)
            events.write(json.dumps(row)+'\n')
            events.flush()
        ball = BallQwen(model.config, weights, PREFIX_IDS, vector,
                        alpha=card['alpha'], log10_temperature=card['log10T'],
                        linear_options={'chunk_rows':64, 'chunk_cols':896,
                                        'cache':DyadicWeightCache(max_elements=262144)},
                        progress=progress)
        tick = time.monotonic()
        prefix = ball.prefix_cache()
        prefix_seconds = time.monotonic()-tick
    if layers != list(range(24)) or len(prefix) != 24:
        raise ValueError('incomplete prefix')
    with gzip.open(out/'prefix-cache.json.gz','wt') as f:
        json.dump(encode_prefix(prefix),f)
    for entry in card['inputs']:
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('postrun input changed')
    for name, expected in PINNED_SHA256.items():
        if sha(VECTOR_FILE if name=='vectors.npz' else SNAPSHOT/name) != expected:
            raise ValueError('postrun asset changed')
    (out/'report.json').write_text(json.dumps(dict(completed=True, precision=2048,
        prefix_seconds=prefix_seconds, seconds=time.monotonic()-start,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        layers=layers, scope='Exact-cell prefix enclosures only; no orbit certificate.'),indent=2)+'\n')

if __name__ == '__main__':
    main()
