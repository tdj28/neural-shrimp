"""Verified local snapshot of the external data root on a separate physical disk (docs/data-storage.md).

Copies /data2/neural-shrimp-data (minus re-downloadable model caches and rebuildable environments by default) and
the Git-ignored private files into /archive/neural-shrimp-backups/<UTC stamp>-<label>/, then verifies every regular
file by SHA-256 against its source, records symlinks by target, and flags any source file whose size or mtime changed
during the copy (not verified; to be caught by the next snapshot). Writes manifest.json and receipt.json next to the
copy and a compact receipt to docs/backups/. A Git push is not a backup of these files; this snapshot is a
same-machine copy on another disk, not an offsite or continuous mirror.

Usage:
  python tools/backup_snapshot.py --label llm-dynamics-13oct            # rsync + verification, idle I/O priority
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/data2/neural-shrimp-data")
DEST_PARENT = Path("/archive/neural-shrimp-backups")
EXCLUDE = ("models", "envs")               # pinned downloads and rebuildable virtual environments
PRIVATE = (ROOT / "cost-ledger.md", ROOT / "checkpoint.md", Path("/data2/Handoff.md"))   # never .env (secrets)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def pair_hash(args):
    src, dst = args
    try:
        a = sha256(src)
        b = sha256(dst)
    except OSError as err:
        return src, None, None, str(err)
    return src, a, b, None


def inventory(base, exclude):
    """Regular files (path -> (size, mtime_ns)) and symlinks (path -> target) under base, relative paths."""
    files, links = {}, {}
    for dirpath, dirnames, filenames in os.walk(base):
        rel_dir = os.path.relpath(dirpath, base)
        if rel_dir == ".":
            dirnames[:] = [d for d in dirnames if d not in exclude]
        for name in dirnames + filenames:
            p = os.path.join(dirpath, name)
            rel = os.path.normpath(os.path.join(rel_dir, name))
            if os.path.islink(p):
                links[rel] = os.readlink(p)
            elif name in filenames and os.path.isfile(p):
                st = os.stat(p)
                files[rel] = (st.st_size, st.st_mtime_ns)
    return files, links


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="inventory only")
    parser.add_argument("--no-link", action="store_true", help="copy everything instead of hard-linking unchanged files")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = DEST_PARENT / f"{stamp}-{args.label}"
    tic = time.time()
    before, links = inventory(SOURCE, EXCLUDE)
    print(json.dumps({"files": len(before), "symlinks": len(links), "bytes": sum(s for s, _ in before.values())}),
          flush=True)
    if args.dry_run:
        return
    if dest.exists():
        raise SystemExit(f"{dest} exists; snapshots are never overwritten")
    (dest / "raw-data").mkdir(parents=True)
    os.chmod(DEST_PARENT, 0o700)
    excludes = [f"--exclude=/{e}" for e in EXCLUDE]
    # unchanged files are hard-linked to the newest earlier full snapshot (same content, no extra space); every file
    # is still verified by SHA-256 below
    earlier = sorted(p for p in DEST_PARENT.iterdir() if p != dest and (p / "receipt.json").exists()
                     and json.loads((p / "receipt.json").read_text()).get("excluded_roots") is not None)
    link = [f"--link-dest={earlier[-1] / 'raw-data'}"] if earlier and not args.no_link else []
    rs = subprocess.run(["rsync", "-a", "--no-inc-recursive", *link, *excludes, f"{SOURCE}/", f"{dest}/raw-data/"],
                        capture_output=True, text=True)
    (dest / "private-handoff").mkdir()
    for p in PRIVATE:
        if p.exists():
            subprocess.run(["cp", "-p", str(p), str(dest / "private-handoff" / p.name)], check=True)
    jobs = [(str(SOURCE / rel), str(dest / "raw-data" / rel)) for rel in before]
    ok, mismatch, errors = 0, [], []
    with ProcessPoolExecutor(args.workers) as pool:
        for src, a, b, err in pool.map(pair_hash, jobs, chunksize=256):
            rel = os.path.relpath(src, SOURCE)
            if err:
                errors.append({"path": rel, "error": err})
            elif a != b:
                mismatch.append(rel)
            else:
                ok += 1
                before[rel] = before[rel] + (a,)
    after, _ = inventory(SOURCE, EXCLUDE)
    changed = sorted(rel for rel, v in before.items() if rel not in after or after[rel][:2] != v[:2])
    new = sorted(set(after) - set(before))
    # a file written or removed while the snapshot ran is listed as changed, not as a copy failure
    mismatch = [rel for rel in mismatch if rel not in set(changed)]
    errors = [e for e in errors if e["path"] not in set(changed)]
    manifest = {"files": {rel: {"bytes": v[0], "sha256": v[2]} for rel, v in before.items() if len(v) == 3},
                "symlinks": links}
    mtext = json.dumps(manifest, sort_keys=True)
    (dest / "manifest.json").write_text(mtext)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    receipt = {"status": "verified" if not mismatch and not errors and rs.returncode == 0 else "incomplete",
               "started_utc": stamp, "completed_utc": datetime.now(timezone.utc).isoformat(),
               "source": str(SOURCE), "destination": str(dest), "excluded_roots": list(EXCLUDE),
               "rsync_returncode": rs.returncode, "rsync_stderr_tail": rs.stderr[-2000:],
               "link_dest": link[0].split("=", 1)[1] if link else None,
               "file_count": len(before), "verified_files": ok, "bytes": sum(v[0] for v in before.values()),
               "symlink_count": len(links), "hash_mismatches": mismatch[:100], "errors": errors[:100],
               "changed_during_copy": changed[:200], "changed_during_copy_count": len(changed),
               "new_during_copy_count": len(new),
               "private_files": [p.name for p in PRIVATE if p.exists()],
               "manifest_sha256": hashlib.sha256(mtext.encode()).hexdigest(),
               "elapsed_seconds": round(time.time() - tic, 1), "source_commit_at_copy": head,
               "verification": "complete SHA-256 comparison of every regular source file with its copy; source sizes "
                               "and mtimes re-inventoried after hashing (changed files listed, not claimed verified)",
               "physical_storage": {"source": "nvme1n1", "destination": "sda",
                                    "basis": "lsblk, 13 Oct 2026 (as in docs/backups/2026-10-08-local-archive.json)"},
               "scope": "Full data root except model caches and environments; same-machine separate physical disk; "
                        "not offsite or continuous."}
    (dest / "receipt.json").write_text(json.dumps(receipt, indent=1))
    (ROOT / "docs" / "backups").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "backups" / f"{stamp}-{args.label}.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in ("status", "file_count", "verified_files", "bytes",
                                               "changed_during_copy_count", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
