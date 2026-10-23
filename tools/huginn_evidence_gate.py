"""Per-setting evidence barrier. Acknowledgments contain hashes, never commands."""
import json
import time
from pathlib import Path


def wait_for_ack(root, payload, deadline, *, clock=time.monotonic, sleep=time.sleep):
    path = Path(root)/'.retrieval-acks'/f"{payload['index']:04d}.json"
    while clock() < deadline:
        if path.exists():
            if json.loads(path.read_text()) != payload:
                raise ValueError('retrieval acknowledgment does not match exact setting evidence')
            return
        sleep(min(1., max(0., deadline-clock())))
    raise TimeoutError('evidence barrier reached original compute deadline')


def setting_payload(root, index, source, digest):
    root=Path(root)
    names=[p.name for p in root.iterdir() if p.is_file() and
           (p.name.startswith(f'states-{index:04d}-') or p.name in
            [f'initial-{index:04d}.npz',f'chunk-{index:04d}.npz',
             'plan.json','runtime.json','steering-direction.npy','benchmark.npz'])]
    if f'chunk-{index:04d}.npz' not in names:
        raise ValueError('cannot acknowledge an incomplete setting')
    return dict(index=index,source=source,files={n:digest(root/n) for n in sorted(names)})
