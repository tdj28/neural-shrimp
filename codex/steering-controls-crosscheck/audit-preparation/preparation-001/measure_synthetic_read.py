"""Bounded synthetic JSONL read/field-check measurement, never target data."""
import hashlib
import json
from pathlib import Path
import resource
import tempfile
import time
import audit

rows = []
for i in range(4096):
    rows.append(dict(dataset='synthetic', row=i // 64, column=i % 64,
        log10_temperature=(i // 64) / 64., canonical_temperature=(i // 64) / 64.,
        alpha=(i % 64 - 32) / 256., source_segment=0, allocated_segments=[0],
        **audit.classification(.002 if i % 3 else -.002, 0 if i % 5 else 3, 8, i < 4001)))
with tempfile.TemporaryDirectory(prefix='task16-audit-synthetic-') as temp:
    path = Path(temp) / 'rows.jsonl'
    with open(path, 'w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')
    started = time.monotonic()
    before = audit.sha(path)
    checks = audit.Checks()
    audit.compare_lines(path, rows, checks, '/synthetic')
    after = audit.sha(path)
    elapsed = time.monotonic() - started
    result = dict(scope='synthetic JSONL hashing, parsing and exact field checks only; no target data',
        records=len(rows), bytes=path.stat().st_size, checks=checks.total, failures=checks.failed,
        elapsed_seconds=elapsed, sha256=before, unchanged=before == after,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256=audit.sha(Path(audit.__file__)),
        prediction_limitation='Does not measure actual target archive decoding, total bookkeeping, cold disk or source/authority IO.')
print(json.dumps(result, indent=2))
