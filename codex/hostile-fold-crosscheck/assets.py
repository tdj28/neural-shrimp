"""Task7 asset boundary; reuse the unchanged independent Qwen architecture."""
import hashlib
import json
from pathlib import Path

VECTOR = Path('/data2/neural-shrimp-data/explore/llm-phase-map/vectors-l12-p1b/vectors.npz')
VECTOR_SHA256 = 'd18395239dba63c7deacb8aa9431d768269617c9cc44a98f89cc1e22423b5958'
VECTOR_KEY = 'hostility-vs-kindness'
SEED = Path('/data2/neural-shrimp-data/explore/llm-soft-loop/pod64/hnormal/curve-p2-fold/states.npy')
SEED_SHA256 = 'd574d0740da3d026276c508527b6001e642c35daace57da7755ec48fd4036123'
PARAMETER = -.04624653570443442
ALPHA = .083


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()


def seed_and_vector():
    """Pinned producer state is a disclosed initial guess, never proof input."""
    import numpy as np
    if sha(SEED) != SEED_SHA256 or sha(VECTOR) != VECTOR_SHA256:
        raise ValueError('Task7 original seed/vector hash mismatch')
    states = np.load(SEED, allow_pickle=False)
    if states.shape != (3,896) or states.dtype != np.float64 or not np.isfinite(states).all():
        raise ValueError('three full finite binary64 producer seed states required')
    with np.load(VECTOR, allow_pickle=False) as archive:
        vector = archive[VECTOR_KEY].copy()
    if vector.shape != (896,) or vector.dtype != np.float64 or not np.isfinite(vector).all():
        raise ValueError('full finite binary64 hostility contrast vector required')
    return states[0].copy(), vector


def load_model(vector):
    """Only call inside an admitted bounded model run; no producer tool import."""
    import numpy as np
    import torch
    from safetensors import safe_open
    from codex.crosscheck.qwen_map import ManualQwenSoftMap, SNAPSHOT, PINNED_SHA256, PREFIX_IDS
    if vector.shape != (896,) or vector.dtype != np.float64 or not np.isfinite(vector).all():
        raise ValueError('admitted full finite vector required')
    for name in ('config.json','model.safetensors'):
        if sha(SNAPSHOT/name) != PINNED_SHA256[name]:
            raise ValueError('original independent model asset mismatch: '+name)
    config = json.loads((SNAPSHOT/'config.json').read_text())
    with safe_open(SNAPSHOT/'model.safetensors', framework='pt', device='cpu') as source:
        weights = {name: source.get_tensor(name).to(torch.float64) for name in source.keys()}
    # The independent constructor checks the complete architecture/tensor shapes.
    return ManualQwenSoftMap(config, weights, PREFIX_IDS, vector, steering_layer=12)
