"""Merge the Part A spectra of the frozen Huginn test (amendments 1 and 1a of docs/huginn-causal-test-design.md) into
the one folder that the frozen items analysis reads. For each question of the panel, the qNNNN folder of the
earliest-launched run whose archive holds a complete spectrum (qNNNN/fixed-spectrum.json) is copied byte for byte
from that run's hash-verified results.tgz. A question folder without fixed-spectrum.json (cut off by a time limit)
is never used. Launch order is the order of the run folders' names (UTC start stamps of the orchestrator).

Only archive member names and file presence are read; no spectrum value, residual or read-out is opened. Writes
manifest.json (question -> run, pod, archive SHA-256, file SHA-256s; every other complete copy is listed as unused,
and questions with no complete copy as missing) and refuses an existing output folder.

Usage:
  python tools/huginn_partA_merge.py --runs 20261013T151945Z 20261013T175506Z ... \
      --out llm-soft-loop/pod64/huginn/partA-spectra-all
"""

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path

DATA = Path("/data2/neural-shrimp-data")
PATTERN = re.compile(r"^explore/llm-soft-loop/pod64/huginn/partA-spectra(?:-2)?/q(\d{4})/([^/]+)$")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def scan_run(run_dir):
    """Verified archive of one run: {question: {file name: member}} for question folders in the archive."""
    result = json.loads((run_dir / "result.json").read_text())
    archive = run_dir / "results.tgz"
    digest = sha256_file(archive)
    if not result.get("hash_verified") or digest != result.get("local_sha256"):
        raise SystemExit(f"{run_dir.name}: archive not verified (hash_verified {result.get('hash_verified')}, "
                         f"local {digest[:12]} vs recorded {str(result.get('local_sha256'))[:12]})")
    questions = {}
    with tarfile.open(archive) as tar:
        for m in tar.getmembers():
            hit = PATTERN.match(m.name)
            if hit and m.isfile():
                questions.setdefault(int(hit.group(1)), {})[hit.group(2)] = m.name
    return {"run": run_dir.name, "pod": result.get("pod_id"), "archive": str(archive), "archive_sha256": digest,
            "questions": questions}


def merge(soft64, runs, out, panel):
    if out.exists():
        raise SystemExit(f"{out} exists; the merged folder is never overwritten")
    scanned = [scan_run(soft64 / r) for r in sorted(runs)]          # launch order = stamp order
    manifest = {"rule": "earliest-launched run with a complete spectrum (fixed-spectrum.json); incomplete folders "
                        "never used; amendments 1 and 1a of docs/huginn-causal-test-design.md",
                "runs": [{k: s[k] for k in ("run", "pod", "archive_sha256")} for s in scanned],
                "questions": {}, "missing": []}
    chosen = {}
    for q in panel:
        complete = [s for s in scanned if "fixed-spectrum.json" in s["questions"].get(q, {})]
        if not complete:
            manifest["missing"].append(q)
            continue
        chosen[q] = complete[0]
        manifest["questions"][f"q{q:04d}"] = {"run": complete[0]["run"], "pod": complete[0]["pod"],
                                              "unused_complete_copies": [s["run"] for s in complete[1:]]}
    out.mkdir(parents=True)
    for q, s in chosen.items():
        files = {}
        with tarfile.open(s["archive"]) as tar:
            for name, member in sorted(s["questions"][q].items()):
                data = tar.extractfile(member).read()
                (out / f"q{q:04d}").mkdir(exist_ok=True)
                (out / f"q{q:04d}" / name).write_bytes(data)
                files[name] = sha256_bytes(data)
        manifest["questions"][f"q{q:04d}"]["files"] = files
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="run folder names under --soft64")
    parser.add_argument("--soft64", default=str(DATA / "runpod" / "soft64"))
    parser.add_argument("--out", required=True, help="explore-relative output folder")
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=64)
    args = parser.parse_args()
    out = Path(args.data) / "explore" / args.out
    m = merge(Path(args.soft64), args.runs, out, range(args.start, args.start + args.count))
    print(json.dumps({"out": str(out), "questions": len(m["questions"]), "missing": m["missing"],
                      "duplicates": {q: v["unused_complete_copies"] for q, v in m["questions"].items()
                                     if v["unused_complete_copies"]}}))


if __name__ == "__main__":
    main()
