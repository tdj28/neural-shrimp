"""Snapshot committed task code, run one bounded CPU worker, retain all outcomes."""
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

ROOT = Path('/data2/neural-shrimp-data/codex/task2-independent-crosscheck')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--attempt', required=True)
    p.add_argument('--stage', required=True)
    p.add_argument('--seconds', type=int, required=True)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--chunk', type=int, default=16)
    p.add_argument('--memory-gib', type=int, default=16)
    args = p.parse_args()
    if '/' in args.attempt or args.attempt.startswith('.') or not args.attempt:
        raise ValueError('simple exclusive attempt name required')
    if not (1 <= args.threads <= 8 and 1 <= args.seconds <= 21600 and 8 <= args.memory_gib <= 20):
        raise ValueError('resource envelope rejected')
    repo = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    files = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', head, 'codex/crosscheck'], cwd=repo, text=True).splitlines()
    selected = [f for f in files if f.endswith(('.py', '.json', '.md'))]
    if not selected:
        raise RuntimeError('no committed source')
    blobs = {}
    for name in selected:
        data = subprocess.check_output(['git', 'show', head+':'+name], cwd=repo)
        if (repo/name).read_bytes() != data:
            raise RuntimeError('uncommitted task source: '+name)
        blobs[name] = data
    attempt = ROOT / args.attempt
    attempt.mkdir(exist_ok=False)
    source = attempt/'source'
    source.mkdir()
    for name, data in blobs.items():
        destination = source/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    manifest = dict(source_commit=head, stage=args.stage, threads=args.threads,
                    memory_gib=args.memory_gib, timeout_seconds=args.seconds,
                    chunk=args.chunk, started_utc=datetime.now(timezone.utc).isoformat(),
                    source=[dict(path=n, sha256=hashlib.sha256(b).hexdigest(), bytes=len(b)) for n,b in blobs.items()])
    (attempt/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
               OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
               CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1')
    def limits():
        os.nice(10)
        resource.setrlimit(resource.RLIMIT_AS, (args.memory_gib*1024**3,)*2)
    command = [sys.executable, '-u', '-m', 'codex.crosscheck.run', '--stage', args.stage,
               '--output', str(attempt/'run'), '--threads', str(args.threads), '--chunk', str(args.chunk)]
    started = time.monotonic()
    with (attempt/'stdout.log').open('wb') as out, (attempt/'stderr.log').open('wb') as err:
        child = subprocess.Popen(command, stdout=out, stderr=err, env=env, cwd=source,
                                 preexec_fn=limits, start_new_session=True)
        manifest['owned_pid'] = child.pid
        (attempt/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(json.dumps(dict(attempt=str(attempt), owned_pid=child.pid, source_commit=head)), flush=True)
        timed_out = False
        try:
            code = child.wait(timeout=args.seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                code = child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait()
    files = sorted(path for path in attempt.rglob('*') if path.is_file())
    receipt = dict(exit_code=code, timed_out=timed_out, elapsed_seconds=time.monotonic()-started,
                   source_commit=head, artifacts=[dict(path=str(path.relative_to(attempt)), bytes=path.stat().st_size,
                                                      sha256=digest(path)) for path in files])
    (attempt/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    for path in attempt.rglob('*'):
        if path.is_file():
            path.chmod(0o444)
    print(json.dumps(dict(exit_code=code, timed_out=timed_out, elapsed_seconds=receipt['elapsed_seconds'])), flush=True)
    sys.exit(code if code >= 0 else 1)


if __name__ == '__main__':
    main()
