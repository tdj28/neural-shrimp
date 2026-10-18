"""Prepared Task33 original returned-shard operational admission; no proof claim."""
import hashlib,json
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''): h.update(block)
    return h.hexdigest()

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def same(a, b, label):
    if canonical(a) != canonical(b): raise ValueError(label+' differs')

def read(path, maximum=16*1024**2):
    path = Path(path)
    if path.is_symlink() or path.resolve() != path or path.stat().st_size > maximum:
        raise ValueError('original bounded regular JSON required')
    def unique(pairs):
        answer = {}
        for k, v in pairs:
            if k in answer: raise ValueError('duplicate JSON key')
            answer[k] = v
        return answer
    return json.loads(path.read_bytes(), object_pairs_hook=unique)

def digest(value, length=64):
    if type(value) is not str or len(value) != length or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('literal lowercase source/hash required')

def common(plan):
    return {k: v for k, v in plan.items() if k not in ('phase_index', 'start_column', 'purpose', 'basis')}

def inventory(attempt, expected_receipt=None):
    """Whole byte inventory only; no derivative-array parsing or evaluation."""
    receipt_path = attempt/'receipt.json'
    if expected_receipt is not None and sha(receipt_path) != expected_receipt: raise ValueError('receipt pin differs')
    receipt = read(receipt_path); files = {}
    for row in receipt['artifacts']:
        if set(row) != {'path', 'bytes', 'sha256'}: raise ValueError('inventory row malformed')
        rel = Path(row['path']); path = attempt/rel; digest(row['sha256'])
        if rel.is_absolute() or '..' in rel.parts or str(rel) in files or str(rel) == 'receipt.json' or type(row['bytes']) is not int:
            raise ValueError('inventory path/size invalid')
        if path.is_symlink() or path.resolve() != path or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('original artifact bytes differ')
        files[str(rel)] = path
    observed = set()
    for path in attempt.rglob('*'):
        if path.is_symlink(): raise ValueError('symlink in attempt')
        if path.is_file(): observed.add(str(path.relative_to(attempt)))
    if observed != set(files)|{'receipt.json'}: raise ValueError('whole attempt inventory differs')
    return receipt, files

def completed(attempt, row, c, source=None, expected_receipt=None):
    if type(c.get('period')) is not int or c['period'] not in (2,3):raise ValueError('Task33 period2/3 required')
    if type(row['phase_index']) is not int or not 0<=row['phase_index']<c['period']:raise ValueError('invalid phase')
    if type(row['start_column']) is not int or row['start_column'] not in range(0,896,64):raise ValueError('invalid column block')
    receipt, files = inventory(attempt, expected_receipt)
    if type(receipt.get('exit_code')) is not int or receipt['exit_code'] != 0 or receipt.get('timed_out') is not False:
        raise ValueError('shard failed or timed out; no retry')
    if source is not None: same(receipt['source_commit'], source, 'captured campaign source')
    manifest = read(files['manifest.json']); inputs = read(files['run/inputs.json']); status = read(files['run/status.json'])
    same(manifest['source_commit'], receipt['source_commit'], 'source'); same(inputs['source_commit'], receipt['source_commit'], 'input source')
    same(inputs['plan_sha256'], row['plan']['sha256'], 'actual committed shard plan')
    same(sha(files['source/codex/coexistence-task33/'+row['plan']['name']]), row['plan']['sha256'], 'archived actual plan bytes')
    same(common(inputs['plan']), c, 'actual same domain/evaluator/backend')
    same([inputs['plan']['phase_index'], inputs['plan']['start_column']], [row['phase_index'], row['start_column']], 'actual assignment')
    for k, v in {'status': 'final_derivative_shard_completed', 'certificate_passed': False, 'phase_index': row['phase_index'],
                 'direction_indices': list(range(row['start_column'], row['start_column']+64)), 'columns_completed': 64,
                 'full_state_dimension': 896, 'full_vocabulary': 151936, 'precision': 1024}.items(): same(status[k], v, 'complete shard '+k)
    same(status['shard_sha256'], sha(files['run/derivative-shard.json']), 'complete saved shard')
    for rel, h in c['source_sha256'].items():
        if sha(files['source/'+rel]) != h: raise ValueError('actual captured evaluator differs')
    return dict(receipt_sha256=sha(attempt/'receipt.json'), source_commit=receipt['source_commit'],
                elapsed_seconds=receipt['elapsed_seconds'], max_rss_kib=status['max_rss_kib'],
                bytes=sum(p.stat().st_size for p in files.values())+(attempt/'receipt.json').stat().st_size)
