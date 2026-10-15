"""Run a committed Task 5b (PT2) source snapshot under exclusive, bounded ownership (adapted from confirm-fold)."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT = Path('/data2/neural-shrimp-data/codex/task5b-fresh-prediction')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(part)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--attempt', required=True)
    p.add_argument('--stage', choices=['setup', 'passage'], required=True)
    p.add_argument('--seconds', type=int, required=True)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--memory-gib', type=int, default=18)
    p.add_argument('--freeze')
    p.add_argument('arguments', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if not a.attempt or '/' in a.attempt or a.attempt.startswith('.'):
        raise ValueError('exclusive simple attempt name required')
    if not (1 <= a.threads <= 6 and 1 <= a.seconds <= 21600 and 8 <= a.memory_gib <= 20):
        raise ValueError('resource envelope exceeded')
    if a.stage == 'passage' and not a.freeze:
        raise ValueError('a pushed freeze is required for passages')
    repo = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    if a.freeze and a.freeze != head:
        raise ValueError('run from exact freeze commit')
    paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', head, 'codex/confirm-fold-2', 'codex/crosscheck/qwen_map.py', 'codex/crosscheck/numerics.py'], cwd=repo, text=True).splitlines()
    blobs = {}
    for name in paths:
        if not name.endswith(('.py', '.json', '.md')):
            continue
        data = subprocess.check_output(['git', 'show', head+':'+name], cwd=repo)
        if (repo/name).read_bytes() != data:
            raise RuntimeError('uncommitted source: '+name)
        blobs[name] = data
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=repo, text=True).split()[0]
    # A later private main may descend from the freeze; preserve this ancestry evidence.
    subprocess.run(['git', 'fetch', 'origin', 'main'], cwd=repo, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['git', 'merge-base', '--is-ancestor', head, 'origin/main'], cwd=repo, check=True)
    verified = datetime.now(timezone.utc).isoformat()
    attempt = ROOT/a.attempt
    attempt.mkdir(parents=True, exist_ok=False)
    source = attempt/'source'
    for name, data in blobs.items():
        dest = source/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    script = 'fold_setup.py' if a.stage == 'setup' else 'passage.py'
    arguments = a.arguments[1:] if a.arguments and a.arguments[0] == '--' else a.arguments
    command = [sys.executable, '-I', '-B', '-u', str(source/'codex/confirm-fold-2'/script), '--output', str(attempt/'run'), '--threads', str(a.threads), *arguments]
    if a.stage == 'passage':
        command += ['--freeze', a.freeze, '--pushed-at', verified]
    manifest = dict(source_commit=head, remote_main=remote, remote_ancestry_verified_at=verified,
                    stage=a.stage, command=command, threads=a.threads, memory_gib=a.memory_gib,
                    timeout_seconds=a.seconds, source=[dict(path=n, sha256=hashlib.sha256(b).hexdigest(), bytes=len(b)) for n,b in blobs.items()])
    (attempt/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads), OPENBLAS_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
               PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TASK5_SOURCE_COMMIT=head)
    if a.stage == 'passage':
        env['TASK5_PLAN_SHA256'] = sha(source/'codex/confirm-fold-2/design.json')
        env['TASK5_PUSH_VERIFIED_AT'] = verified
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    def limits():
        os.nice(10)
        resource.setrlimit(resource.RLIMIT_AS, (a.memory_gib*1024**3,)*2)
    start = time.monotonic()
    with (attempt/'stdout.log').open('wb') as out, (attempt/'stderr.log').open('wb') as err:
        child = subprocess.Popen(command, cwd=source, env=env, stdout=out, stderr=err,
                                 preexec_fn=limits, start_new_session=True)
        manifest['owned_pid'] = child.pid
        (attempt/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(json.dumps(dict(attempt=str(attempt), pid=child.pid, source_commit=head)), flush=True)
        timeout = False
        try:
            rc = child.wait(timeout=a.seconds)
        except subprocess.TimeoutExpired:
            timeout = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                rc = child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                rc = child.wait()
    receipt = dict(source_commit=head, exit_code=rc, timed_out=timeout, elapsed_seconds=time.monotonic()-start,
                   artifacts=[dict(path=str(f.relative_to(attempt)), bytes=f.stat().st_size, sha256=sha(f)) for f in sorted(attempt.rglob('*')) if f.is_file()])
    (attempt/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    for f in attempt.rglob('*'):
        if f.is_file():
            f.chmod(0o444)
    print(json.dumps({k:v for k,v in receipt.items() if k!='artifacts'}), flush=True)
    sys.exit(rc if rc >= 0 else 1)

if __name__ == '__main__':
    main()
