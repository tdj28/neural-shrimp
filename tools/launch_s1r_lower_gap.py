"""One owned process group; separate review and root launch authorization required.

External GNU timeout: 20 s TERM +5 s KILL. The supervisor independently bounds
inherited pipes/leader exit and monitors group RSS. RSS polling is not a hard
kernel memory limit; transient overshoot is possible. No retry is performed.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/data2/neural-shrimp/.venv/bin/python'
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
STORE_BYTES, LAUNCHER_BYTES, RSS_BYTES = 7*1024**2, 1024**2, 2*1024**3
LOG_LIMIT, SUPERVISOR_SECONDS = 128*1024, 25.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def entry(path):
    data = path.read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def write(path, value):
    data = value if isinstance(value, bytes) else (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
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


def command(out, review_commit):
    return ['timeout', '--signal=TERM', '--kill-after=5s', '20s', 'nice', '-n', '10', PYTHON,
            'tools/diagnose_s1r_lower_gap.py', '--out', str(out), '--review-commit', review_commit]


def committed_sources():
    spec = importlib.util.spec_from_file_location('lower_gap_source_check', ROOT / 'tools/diagnose_s1r_lower_gap.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sources()


def group_status(group):
    rows = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            # comm can contain spaces/parentheses; fields after the last ')' start at state.
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == group:
                rss = int(fields[21]) * os.sysconf('SC_PAGE_SIZE')
                rows.append(dict(pid=int(path.name), state=fields[0], rss_bytes=max(0, rss)))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return rows


def kill_owned(group, sig):
    try:
        os.killpg(group, sig)
    except ProcessLookupError:
        pass


def launch(out, launcher, source_commit, review_commit):
    require(all(isinstance(s, str) and re.fullmatch('[0-9a-f]{40}', s) for s in (source_commit, review_commit)), 'full commit identities required')
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == source_commit, 'execution HEAD mismatch')
    subprocess.run(['git', 'cat-file', '-e', review_commit + '^{commit}'], cwd=ROOT, check=True)
    inventory = committed_sources()
    out, launcher = out.resolve(), launcher.resolve()
    require(out != launcher and out not in launcher.parents and launcher not in out.parents, 'separate artifact directories required')
    require(not out.exists() and not launcher.exists(), 'exclusive attempt directories required')
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'launcher nice>=10 required')
    launcher.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update({k: '1' for k in THREADS})
    env.update(PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1')
    cmd = command(out, review_commit)
    runtime = dict(schema='s1r-lower-gap-launcher-v1', source_commit=source_commit, review_commit=review_commit,
        sources=inventory, command=cmd, cwd=str(ROOT), out=str(out), launcher=str(launcher),
        started_utc=datetime.now(timezone.utc).isoformat(), launcher_pid=os.getpid(), group_pid=None,
        thread_env={k: env[k] for k in THREADS}, nice=os.getpriority(os.PRIO_PROCESS, 0),
        load_average=os.getloadavg(), memory_info=Path('/proc/meminfo').read_text(),
        gpu_used=False, external_seconds=20, kill_grace_seconds=5, supervisor_seconds=SUPERVISOR_SECONDS,
        rss_cap_bytes=RSS_BYTES, rss_enforcement='monitored process-group RSS, not kernel-hard; polling 0.1 s',
        store_bytes=STORE_BYTES, launcher_bytes=LAUNCHER_BYTES, log_bytes=LOG_LIMIT)
    write(launcher / 'launcher-source.py', Path(__file__).read_bytes())
    write(launcher / 'runtime-start.json', runtime)
    proc, selector = None, None
    started = time.monotonic()
    receipt = dict(status='launch_pending', process_exit=None, failure=None, log_truncated=False,
                   max_group_rss_bytes=0, observed_owned_pids=[], owned_live_pids_remaining=[])
    seen = set()
    try:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        runtime['group_pid'] = proc.pid
        write(launcher / 'runtime-start.json', runtime)
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        logged = 0
        with (launcher / 'stdout.log').open('xb') as log:
            while selector.get_map() or proc.poll() is None:
                for key, _ in selector.select(.1):
                    data = os.read(key.fd, 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    chunk = data[:max(0, LOG_LIMIT - logged)]
                    log.write(chunk); log.flush(); os.fsync(log.fileno())
                    logged += len(chunk)
                    if len(chunk) != len(data):
                        receipt.update(log_truncated=True, failure='log_limit')
                        kill_owned(proc.pid, signal.SIGKILL)
                rows = group_status(proc.pid)
                seen.update(row['pid'] for row in rows)
                rss = sum(row['rss_bytes'] for row in rows)
                receipt['max_group_rss_bytes'] = max(receipt['max_group_rss_bytes'], rss)
                if rss > RSS_BYTES:
                    receipt['failure'] = 'monitored_group_rss_limit'
                    kill_owned(proc.pid, signal.SIGKILL)
                if time.monotonic() - started >= SUPERVISOR_SECONDS:
                    receipt['failure'] = 'supervisor_timeout_or_inherited_pipe'
                    kill_owned(proc.pid, signal.SIGKILL)
                    for key in list(selector.get_map().values()):
                        selector.unregister(key.fileobj)
                    proc.stdout.close()
                    break
            receipt.update(process_exit=proc.wait(timeout=1), status='process_finished')
    except BaseException as error:
        receipt.update(status='launcher_failure', failure=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        if proc is not None:
            # This is the group created by this launcher, including inherited-pipe descendants.
            kill_owned(proc.pid, signal.SIGKILL)
            if proc.poll() is None:
                proc.wait(timeout=1)
            receipt['process_exit'] = proc.returncode
            receipt['owned_live_pids_remaining'] = [r['pid'] for r in group_status(proc.pid) if r['state'] != 'Z']
        if selector is not None:
            selector.close()
        receipt.update(elapsed_seconds=time.monotonic()-started, finished_utc=datetime.now(timezone.utc).isoformat(),
            source_commit=source_commit, review_commit=review_commit, observed_owned_pids=sorted(seen),
            group_pid=None if proc is None else proc.pid)
        files = {} if not out.exists() else {p.name: entry(p) for p in out.iterdir() if p.is_file()}
        receipt.update(raw_file_count=len(files), raw_bytes=sum(v['bytes'] for v in files.values()), raw_inventory=files)
        receipt['within_store_cap'] = receipt['raw_bytes'] <= STORE_BYTES
        receipt['raw_manifest_verified'] = False
        receipt['scientific_complete'] = False
        try:
            if out.exists() and (out / 'manifest.json').is_file():
                expected = json.loads((out / 'manifest.json').read_text())['files']
                receipt['raw_manifest_verified'] = (all(p.is_file() and not p.is_symlink() for p in out.iterdir()) and
                    expected == {k: v for k, v in files.items() if k != 'manifest.json'})
            if out.exists() and (out / 'result.json').is_file():
                result = json.loads((out / 'result.json').read_text())
                receipt['scientific_complete'] = (result['status'] == 'diagnostic_complete' and
                    result['accepted_orbit'] is False and result['applied_correction'] is False and
                    result['fold_verified'] is False)
        except (ValueError, KeyError, TypeError) as error:
            receipt['raw_validation_error'] = type(error).__name__ + ': ' + str(error)
        receipt['successful'] = bool(receipt['process_exit'] == 0 and receipt['failure'] is None and
            not receipt['log_truncated'] and receipt['within_store_cap'] and receipt['raw_manifest_verified'] and
            receipt['scientific_complete'] and not receipt['owned_live_pids_remaining'])
        write(launcher / 'receipt.json', receipt)
        write(launcher / 'manifest.json', dict(files={p.name: entry(p) for p in launcher.iterdir()}))
        require(sum(p.stat().st_size for p in launcher.iterdir()) <= LAUNCHER_BYTES, 'launcher byte cap')
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
    args = parser.parse_args()
    receipt = launch(args.out, args.launcher, args.source_commit, args.review_commit)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt['successful'] else 2

if __name__ == '__main__':
    raise SystemExit(main())
