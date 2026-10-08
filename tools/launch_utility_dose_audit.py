"""Source-bound single launch; root must authorize after independent review.

GNU timeout supplies the 10 s TERM / 5 s KILL bound. This wrapper retains
process identity, bounded stdout and immutable evidence even on child failure.
It never retries. Review SHA is provenance, not programmatic authorization.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from neural_shrimp.period2_branch_io import canonical, digest, file_entry
from neural_shrimp.utility_dose_audit import LAUNCHER_BYTES, STORE_BYTES, TOTAL_BYTES, need as require
PYTHON = '/data2/neural-shrimp/.venv/bin/python'
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
LOG_LIMIT = 128 * 1024
SUPERVISOR_SECONDS = 15


def command(out, review_commit, execution_commit, raw_manifest, launcher_manifest):
    return ['timeout', '--signal=TERM', '--kill-after=5s', '10s', 'nice', '-n', '10', PYTHON,
            'tools/audit_utility_dose_feasibility.py', '--out', str(out), '--review-commit', review_commit,
            '--execution-commit', execution_commit, '--raw-manifest', raw_manifest, '--launcher-manifest', launcher_manifest]


def write(path, value):
    data = value if isinstance(value, bytes) else canonical(value)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('xb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def committed_sources():
    spec = importlib.util.spec_from_file_location('dose_auditor_source_check', ROOT / 'tools/audit_utility_dose_feasibility.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner.sources()


def launch(out, launcher, source_commit, review_commit, execution_commit, raw_manifest, launcher_manifest):
    import re
    require(all(re.fullmatch('[0-9a-f]{40}', s) for s in (source_commit, review_commit, execution_commit)), 'full commit identities required')
    require(all(re.fullmatch('[0-9a-f]{64}', s) for s in (raw_manifest,launcher_manifest)), 'full external manifest identities')
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == source_commit, 'execution HEAD mismatch')
    subprocess.run(['git', 'cat-file', '-e', review_commit + '^{commit}'], cwd=ROOT, check=True)
    source_inventory = committed_sources()
    out, launcher = out.resolve(), launcher.resolve()
    require(out != launcher and out not in launcher.parents and launcher not in out.parents, 'separate artifact directories')
    require(not out.exists() and not launcher.exists(), 'exclusive attempt directories required')
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'launcher nice10 required')
    launcher.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update({k: '1' for k in THREADS})
    env.update(PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1')
    cmd = command(out, review_commit, execution_commit, raw_manifest, launcher_manifest)
    runtime = dict(schema='utility-dose-audit-launcher-v1', source_commit=source_commit, source_review_commit=review_commit,
        sources=source_inventory, producer_execution_commit=execution_commit, raw_manifest_sha256=raw_manifest, producer_launcher_manifest_sha256=launcher_manifest, command=cmd, cwd=str(ROOT), output=str(out), launcher=str(launcher),
        started_utc=datetime.now(timezone.utc).isoformat(), launcher_pid=os.getpid(), child_pid=None,
        nice=os.getpriority(os.PRIO_PROCESS, 0), thread_env={k: env[k] for k in THREADS},
        load_average=os.getloadavg(), external_seconds=10, kill_grace_seconds=5,
        launcher_bytes=LAUNCHER_BYTES, store_bytes=STORE_BYTES, total_raw_bytes=TOTAL_BYTES)
    write(launcher / 'launcher-source.py', Path(__file__).read_bytes())
    write(launcher / 'runtime-start.json', runtime)
    proc = None
    started = time.monotonic()
    receipt = dict(status='launch_pending', process_exit=None, log_truncated=False, failure=None)
    try:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        runtime['child_pid'] = proc.pid
        write(launcher / 'runtime-start.json', runtime)
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        logged = 0
        with (launcher / 'stdout.log').open('xb') as log:
            while selector.get_map():
                for key, _ in selector.select(.2):
                    data = os.read(key.fd, 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    chunk = data[:max(0, LOG_LIMIT-logged)]
                    log.write(chunk)
                    log.flush()
                    os.fsync(log.fileno())
                    logged += len(chunk)
                    if len(chunk) != len(data) and not receipt['log_truncated']:
                        receipt['log_truncated'] = True
                        os.killpg(proc.pid, signal.SIGTERM)
                # An exited leader may leave a descendant holding this pipe open.
                # The owned process group and pipe must still be bounded.
                if time.monotonic()-started >= SUPERVISOR_SECONDS:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    receipt['failure'] = 'supervisor_timeout_or_inherited_pipe'
                    for key in list(selector.get_map().values()):
                        selector.unregister(key.fileobj)
                    proc.stdout.close()
            selector.close()
        receipt.update(process_exit=proc.wait(), status='process_finished')
    except BaseException as error:
        receipt.update(status='launcher_failure', failure=type(error).__name__ + ': ' + str(error))
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            receipt['process_exit'] = proc.wait()
        raise
    finally:
        receipt.update(elapsed_seconds=time.monotonic()-started, finished_utc=datetime.now(timezone.utc).isoformat(),
                       source_commit=source_commit, source_review_commit=review_commit, producer_execution_commit=execution_commit, raw_manifest_sha256=raw_manifest, producer_launcher_manifest_sha256=launcher_manifest, child_pid=None if proc is None else proc.pid)
        entries = {} if not out.exists() else {p.name: file_entry(p) for p in sorted(out.iterdir()) if p.is_file()}
        receipt.update(raw_file_count=len(entries), raw_bytes=sum(v['bytes'] for v in entries.values()),
                       raw_inventory_sha256=digest(canonical(entries)),
                       raw_manifest=entries.get('manifest.json'), raw_result=entries.get('result.json'),
                       stdout=file_entry(launcher / 'stdout.log') if (launcher / 'stdout.log').exists() else None)
        receipt['within_store_cap'] = receipt['raw_bytes'] <= STORE_BYTES
        write(launcher / 'receipt.json', receipt)
        files = {p.name: file_entry(p) for p in launcher.iterdir()}
        write(launcher / 'manifest.json', dict(files=files, status=receipt['status']))
        require(sum(p.stat().st_size for p in launcher.iterdir()) <= LAUNCHER_BYTES, 'launcher storage cap')
        for folder in (out, launcher):
            if folder.exists():
                for path in folder.iterdir():
                    path.chmod(0o444)
                folder.chmod(0o555)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--launcher', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--review-commit', required=True)
    parser.add_argument('--execution-commit',required=True)
    parser.add_argument('--raw-manifest',required=True)
    parser.add_argument('--launcher-manifest',required=True)
    args = parser.parse_args()
    receipt = launch(args.out, args.launcher, args.source_commit, args.review_commit, args.execution_commit, args.raw_manifest, args.launcher_manifest)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt['process_exit'] == 0 and not receipt['log_truncated'] and receipt['within_store_cap'] and receipt['failure'] is None else 2

if __name__ == '__main__':
    raise SystemExit(main())
