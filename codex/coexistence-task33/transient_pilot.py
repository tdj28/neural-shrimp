"""Bounded uncertain-input fixed-start pilot. No midpoint reset or certificate."""
import argparse
import json
from pathlib import Path
import sys
import time


def pack(v):
    if not v.is_finite():
        raise ValueError('nonfinite enclosure')
    return {'mid': list(map(int, v.mid().man_exp())),
            'rad': list(map(int, v.rad().man_exp()))}


def propagate(initial, forward, steps, save):
    """Pass the original live output balls to the next call without serialization."""
    if type(steps) is not int or not 1 <= steps <= 20:
        raise ValueError('pilot requires 1..20 steps')
    state = list(initial)
    dimension = len(state)
    if not dimension or any(not v.is_finite() for v in state):
        raise ValueError('invalid initial enclosure')
    save('state-000.json', {'step': 0, 'state': [pack(v) for v in state]})
    for step in range(1, steps + 1):
        tick = time.monotonic()
        values, tangent = forward(state)
        if len(values) != dimension or len(tangent) != dimension:
            raise ValueError('incomplete output')
        if any(not v.is_finite() for v in values):
            raise ValueError('nonfinite image')
        if any(not v.is_exact() or v != 0 for v in tangent):
            raise ValueError('unexpected tangent')
        record = {'step': step, 'state': [pack(v) for v in values],
                  'max_radius': pack(max(v.rad() for v in values)),
                  'seconds': time.monotonic() - tick}
        save(f'state-{step:03d}.json', record)
        state = values
    return state


def validate_card(card):
    if (card['precision'] != 2048 or card['steps'] != 20
            or card['alpha'] != .06468 or card['log10T'] != -.148
            or card['initial_token_id'] != 198
            or card['initial_state'] != 'exact checkpoint embedding of last prefix token'):
        raise ValueError('unreviewed transient definition')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--card', required=True); ap.add_argument('--out', required=True)
    args = ap.parse_args(); card = json.loads(Path(args.card).read_text())
    validate_card(card)
    root = Path(card['repository'])
    sys.path[:0] = [str(Path(__file__).resolve().parent),
                   str(root/'codex/certify-llm'), str(root/'codex/crosscheck')]
    import torch
    from flint import ctx
    from ball_qwen import BallQwen, exact
    from ball_linear import DyadicWeightCache
    from cache_io import digest, unpack
    from prefix_import import install_prefix
    from qwen_map import ManualQwenSoftMap, PREFIX_IDS, PINNED_SHA256, SNAPSHOT, VECTOR_FILE
    def verify():
        for item in card['inputs']:
            if digest(item['path']) != item['sha256']:
                raise ValueError('pinned input mismatch')
        for name, expected in PINNED_SHA256.items():
            if digest(VECTOR_FILE if name == 'vectors.npz' else SNAPSHOT/name) != expected:
                raise ValueError('asset mismatch')
    verify()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    ctx.prec = 2048; ctx.threads = 1
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (out/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    start = time.monotonic()
    model = ManualQwenSoftMap.from_pinned_files()
    weights = {k: t.numpy() for k, t in model.weights.items()}
    for value in weights.values(): value.flags.writeable = False
    vector = model.vector.numpy(); vector.flags.writeable = False
    ball = BallQwen(model.config, weights, PREFIX_IDS, vector,
                   alpha=card['alpha'], log10_temperature=card['log10T'],
                   linear_options={'chunk_rows':64, 'chunk_cols':896,
                                   'cache':DyadicWeightCache(max_elements=262144)})
    if PREFIX_IDS[-1] != card['initial_token_id'] or ball.d != 896 or ball.vocab != 151936:
        raise ValueError('different initial state or map dimensions')
    install_prefix(card, ball, PINNED_SHA256, digest, unpack)
    initial = [exact(v) for v in weights['model.embed_tokens.weight'][PREFIX_IDS[-1]]]
    if any(not v.is_exact() for v in initial): raise ValueError('initial state must be exact')
    try:
        propagate(initial, ball.forward, card['steps'], save)
        verify()
    except Exception as error:
        save('failure.json', {'type':type(error).__name__, 'message':str(error),
                             'seconds':time.monotonic()-start, 'certificate_passed':False})
        raise
    save('report.json', {'steps':card['steps'], 'precision':ctx.prec,
                        'seconds':time.monotonic()-start,
                        'uncertainty_preserved':True, 'certificate_passed':False,
                        'reachability_test_performed':False})

if __name__ == '__main__': main()
