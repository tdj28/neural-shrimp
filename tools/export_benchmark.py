"""Export a Hugging Face dataset split set at a pinned revision to JSON lines, with a hash manifest.

Rows are written one per line as json.dumps(row) (ASCII-escaped, default separators), the format of the GSM8K export
used by tools/huginn_relax_eval.py. The manifest records the repository, the full commit hash, the source parquet
files with their SHA-256, and the SHA-256 and row count of each output file.

Usage:
  python tools/export_benchmark.py --repo openai/gsm8k --revision 740312ad --config main --splits train test \
      --out explore/gsm8k-check
  python tools/export_benchmark.py --repo allenai/ai2_arc --revision main --config ARC-Challenge \
      --splits train validation test --out explore/arc-challenge
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve(api, repo, revision):
    """Full commit hash for a branch name, a full hash or a unique hash prefix; refuses an ambiguous or unknown one."""
    if len(revision) == 40:
        return revision
    commits = api.list_repo_commits(repo, repo_type="dataset")
    hits = [c.commit_id for c in commits if c.commit_id.startswith(revision)]
    if len(hits) == 1:
        return hits[0]
    if hits:
        raise SystemExit(f"revision prefix {revision} is ambiguous in {repo}: {hits}")
    info = api.dataset_info(repo, revision=revision)          # a branch or tag name
    return info.sha


def main():
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    import explore_llm_attractors as ela

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--config", required=True, help="the folder of the parquet files in the repository")
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--out", required=True, help="data-relative output folder")
    args = parser.parse_args()
    api = HfApi()
    commit = resolve(api, args.repo, args.revision)
    files = api.list_repo_files(args.repo, repo_type="dataset", revision=commit)
    out = ela.DATA / args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"repo": args.repo, "revision": commit, "config": args.config, "splits": {}}
    for split in args.splits:
        sources = sorted(f for f in files if f.startswith(f"{args.config}/") and f.endswith(".parquet")
                         and Path(f).name.startswith(f"{split}-"))
        if not sources:
            raise SystemExit(f"no parquet file for split {split} under {args.config}/ at {commit}")
        rows, record = [], []
        for name in sources:
            local = hf_hub_download(args.repo, name, repo_type="dataset", revision=commit, cache_dir=ela.CACHE)
            rows += pq.read_table(local).to_pylist()
            record.append({"file": name, "sha256": sha256(local)})
        dest = out / f"{split}.jsonl"
        with open(dest, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        manifest["splits"][split] = {"sources": record, "file": dest.name, "rows": len(rows), "sha256": sha256(dest)}
        print(json.dumps({"split": split, "rows": len(rows), "sha256": manifest["splits"][split]["sha256"]}))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
