"""Source-frozen, bounded, value-free Task41 public Git-content inventory.

Run only after Root checkpoint. Never reads working-tree payloads, env-family
blobs, forbidden project trees, Git config/remotes, or untracked/ignored files.
Findings contain locations/categories only, never matches, snippets or values.
This is heuristic candidate screening, not proof that credentials are absent.
"""
import collections
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import resource
import subprocess
import time

REPO = Path("/data2/neural-shrimp")
OUT = REPO / "codex/release-audit-task41/public-findings.json"
TAG = "1314528118ac1300c2cffd34fb60cfbfc67d2c82"
HEAD = "21272a7830d74fade1f6592af2a7b8189370ffad"
CLEANUP = "e359312e339c8f2fda59f280e6d7f92d88c0552d"
MAX_BLOB = 8 * 1024**2
MAX_TOTAL = 1024**3
MAX_METADATA = 64 * 1024**2
MAX_FINDINGS = 100000
MAX_SECONDS = 560
TEXT_SUFFIXES = {
    ".py", ".sh", ".md", ".txt", ".tex", ".bib", ".bbl", ".json", ".jsonl",
    ".yaml", ".yml", ".toml", ".cfg", ".ini", ".csv", ".tsv", ".html", ".css",
    ".js", ".ts", ".rst", ".log", ".xml", ".svg", ".lock", ".cff", ".r",
}
BINARY_SUFFIXES = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".zip", ".gz", ".tar",
    ".npz", ".npy", ".pt", ".pth", ".bin", ".safetensors", ".woff", ".ttf",
}
THIRD_SUFFIXES = {".pdf", ".csv", ".tsv", ".parquet", ".safetensors", ".bin", ".pt", ".pth"}


def forbidden(path):
    parts = PurePosixPath(path).parts
    return any(".env" in p.lower() for p in parts) or any(
        p.lower() in ("lexical-prompt-study",) for p in parts)


def path_category(path):
    if forbidden(path):
        return "prohibited_payload_path"
    suffix = PurePosixPath(path).suffix.lower()
    if suffix in THIRD_SUFFIXES:
        return "third_party_provenance_review_candidate"
    if re.search(r"(?i)(?:^|/)(?:datasets?|vendor|third.party|papers?|literature)(?:/|$)", path):
        return "third_party_provenance_review_candidate"
    if re.search(r"(?i)(?:^|/)(?:id_rsa|id_ed25519|credentials|secrets?)(?:[./_-]|$)|\.(?:pem|key|p12|pfx)$", path):
        return "credential_filename_candidate"
    return None


PATTERNS = [
    ("credential_vendor_token_shape", re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|"
        r"hf_[A-Za-z0-9]{20,}|sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}|"
        r"AKIA[A-Z0-9]{16}|ASIA[A-Z0-9]{16}|AIza[A-Za-z0-9_-]{30,}|"
        r"xox[baprs]-[A-Za-z0-9-]{15,}|pypi-[A-Za-z0-9_-]{40,}|"
        r"npm_[A-Za-z0-9]{20,}|SG\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,})\b")),
    ("credential_private_key_marker", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----")),
    ("credential_jwt_shape", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{12,}\b")),
    ("credential_url_userinfo_candidate", re.compile(r"(?i)\b(?:https?|postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^/\s:@]{1,80}:[^/\s@]{4,160}@")),
    ("credential_literal_assignment_candidate", re.compile(
        r"""(?i)(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret[_-]?(?:key|token)?|password|passwd|bearer|runpod[_-]?key|hf[_-]?token)\s*["']?\s*[:=]\s*["'][A-Za-z0-9_+/=.-]{16,}["']""")),
    ("credential_bearer_literal_candidate", re.compile(r"(?i)\b(?:authorization|bearer)\s*[:=]?\s+[A-Za-z0-9_.+/=-]{20,}")),
    ("contact_email_candidate", re.compile(r"\b[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("off_limits_project_mention", re.compile(r"(?i)lexical[-_/ ]prompt[-_/ ]study")),
]
MONEY = re.compile(r"(?:\\?\$|USD\s*|US\s*dollars?\s*)\s*\d[\d,.]*(?:\s*[kKmM])?", re.I)
MONEY_CONTEXT = re.compile(r"\b(?:cost|costs|budget|paid|spend|spent|spending|balance|balances|price|pricing|rate|rates|ceiling|dollars?|funding|rent|income)\b", re.I)
PERSON = re.compile(r"\b(?:author|user|timothy|tim|his|her|my|i|he|she)\b", re.I)
PERSONAL = re.compile(r"\b(?:family|wife|husband|spouse|children|retir\w*|career|employment|unemploy\w*|salary|income|mortgage|pension|debt|savings|illness|health|medical|hospital|diagnosis|afford|financial|finances)\b", re.I)
PRIVATE_CONTACT = re.compile(r"\b(?:phone|mobile|home address|personal email|private email|contact me)\b", re.I)
PHONE = re.compile(r"(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4}\b")


def line_categories(line):
    found = {name for name, pattern in PATTERNS if pattern.search(line)}
    if MONEY.search(line):
        found.add("monetary_context_candidate" if MONEY_CONTEXT.search(line)
                  else "dollar_numeric_candidate_may_be_math")
    if PERSON.search(line) and PERSONAL.search(line):
        found.add("personal_context_candidate")
    if PRIVATE_CONTACT.search(line) and PHONE.search(line):
        found.add("private_phone_candidate")
    if re.search(r"(?i)SPDX-License-Identifier|copyright|licensed under|creative commons|apache.?2|MIT License|redistribut", line):
        found.add("license_provenance_marker")
    return sorted(found)


def main():
    if OUT.exists() or OUT.is_symlink():
        raise ValueError("Fresh audit output required")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("Offline environment required")
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise ValueError("nice10 required")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise ValueError("Single-thread environment required")
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (300,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2,) * 2)
    started = time.monotonic()
    report = dict(schema="task41-redacted-git-candidate-audit-v1", complete=False,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  targets=dict(arxiv_v1=TAG, head=HEAD, cleanup=CLEANUP),
                  findings=[], path_inventory=[], skipped=[], cleanup_changes=[],
                  ci=[], counts={}, failures=[], credential_validity_tested=False,
                  content_snippets_emitted=False, env_payloads_read=0,
                  limits=dict(blob_bytes=MAX_BLOB, aggregate_bytes=MAX_TOTAL,
                              metadata_bytes=MAX_METADATA, wall_seconds=MAX_SECONDS))
    git_env = dict(os.environ, GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
    payload_bytes = 0
    cache = {}
    batch = None
    stage = "metadata"

    def git(*args, input_bytes=None):
        left = MAX_SECONDS - (time.monotonic() - started)
        if left <= 0:
            raise TimeoutError("Audit wall bound")
        result = subprocess.run(["git", "-C", str(REPO), *args], input=input_bytes,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                env=git_env, timeout=left, check=False)
        if result.returncode or len(result.stdout) > MAX_METADATA:
            raise RuntimeError("Git command failed or metadata bound exceeded")
        return result.stdout

    def read_blob(oid, size, denied):
        if oid in denied or not re.fullmatch("[0-9a-f]{40}", oid):
            raise ValueError("Prohibited or malformed blob request")
        if not 0 <= size <= MAX_BLOB:
            raise ValueError("Blob request exceeds bound")
        batch.stdin.write((oid + "\n").encode())
        batch.stdin.flush()
        fields = batch.stdout.readline(256).split()
        if fields != [oid.encode(), b"blob", str(size).encode()]:
            raise ValueError("Batch blob identity differs")
        data = batch.stdout.read(size)
        if len(data) != size or batch.stdout.read(1) != b"\n":
            raise ValueError("Truncated batch blob")
        return data

    def tree(commit):
        rows = {}
        for item in git("ls-tree", "-r", "-l", "-z", commit).split(b"\0"):
            if not item:
                continue
            meta, raw_path = item.split(b"\t", 1)
            mode, kind, oid, size = meta.split()
            path = raw_path.decode("utf-8", "replace")
            rows[path] = dict(mode=mode.decode(), kind=kind.decode(),
                              blob=oid.decode(), bytes=int(size) if size != b"-" else 0)
        return rows

    try:
        for value in (TAG, HEAD, CLEANUP):
            if git("rev-parse", value + "^{commit}").decode().strip() != value:
                raise ValueError("Frozen commit identity differs")
        trees = {"arxiv_v1": tree(TAG), "head": tree(HEAD)}
        before = git("rev-parse", CLEANUP + "^").decode().strip()
        old_cleanup, new_cleanup = tree(before), tree(CLEANUP)
        # Resolve every historical env-family object ID before reading any blobs.
        # Raw diff metadata only: env payloads are never requested.
        env_meta = git("log", "--all", "--format=", "--raw", "--no-abbrev",
                       "--no-renames", "--root", "--", "*.env*", ":(glob)**/*.env*",
                       ":(glob)**/*.env*/**", "lexical-prompt-study",
                       ":(glob)**/lexical-prompt-study/**")
        denied = set()
        for line in env_meta.splitlines():
            if line.startswith(b":"):
                fields = line.split()
                for field in fields[2:4]:
                    if re.fullmatch(b"[0-9a-f]{40}", field) and field != b"0" * 40:
                        denied.add(field.decode())
        for rows in (*trees.values(), old_cleanup, new_cleanup):
            denied.update(row["blob"] for path, row in rows.items() if forbidden(path))

        versions = collections.defaultdict(dict)
        for label, rows in trees.items():
            for path, row in rows.items():
                if row["kind"] == "blob":
                    versions[row["blob"]][path] = label if path not in versions[row["blob"]] else "both_trees"
                category = path_category(path)
                if category or row["mode"] == "120000":
                    report["path_inventory"].append(dict(
                        path=path, category=category or "tracked_symlink_not_followed",
                        commit=TAG if label == "arxiv_v1" else HEAD,
                        blob=row["blob"], bytes=row["bytes"], at_head=path in trees["head"]))
        # History paths are names reported by Git, never directories to traverse.
        historical = {}
        for item in git("rev-list", "--objects", TAG, HEAD).splitlines():
            oid, _, raw_path = item.partition(b" ")
            if not re.fullmatch(b"[0-9a-f]{40}", oid):
                raise ValueError("Unexpected object identifier")
            if raw_path:
                historical[oid.decode()] = raw_path.decode("utf-8", "replace")
        for oid, path in historical.items():
            if forbidden(path):
                denied.add(oid)
        ordered = list(versions)
        ordered.extend(oid for oid in historical if oid not in versions)
        checked = git("cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)",
                      input_bytes="".join(oid + "\n" for oid in ordered).encode())
        metadata = {}
        for line in checked.splitlines():
            oid, kind, size = line.split()
            metadata[oid.decode()] = (kind.decode(), int(size))
        stage = "blob_screen"
        batch = subprocess.Popen(["git", "-C", str(REPO), "cat-file", "--batch"],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, env=git_env)
        scanned_blobs = 0
        for oid in ordered:
            kind, size = metadata[oid]
            if kind != "blob":
                continue
            locations = versions.get(oid) or {historical[oid]: "history"}
            paths = list(locations)
            suffixes = {PurePosixPath(path).suffix.lower() for path in paths}
            reason = None
            if oid in denied:
                reason = "prohibited_env_or_project_payload"
            elif size > MAX_BLOB:
                reason = "over_per_blob_bound"
            elif suffixes <= BINARY_SUFFIXES:
                reason = "binary_inventory_only"
            elif payload_bytes + size > MAX_TOTAL:
                reason = "aggregate_payload_bound"
            elif time.monotonic() - started > MAX_SECONDS - 20:
                reason = "wall_bound_before_next_blob"
            if reason:
                report["skipped"].append(dict(blob=oid, path=paths[0], bytes=size, category=reason))
                continue
            data = read_blob(oid, size, denied)
            if len(data) != size:
                raise ValueError("Git payload byte count differs")
            payload_bytes += size
            if b"\0" in data[:8192]:
                report["skipped"].append(dict(blob=oid, path=paths[0], bytes=size,
                                              category="binary_probe_only"))
                continue
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = data.decode("utf-8", "replace")
            rows = []
            for number, line in enumerate(text.splitlines(), 1):
                cats = line_categories(line)
                if cats:
                    rows.append((number, cats))
            cache[oid] = rows
            scanned_blobs += 1
            for path, where in locations.items():
                head_oid = trees["head"].get(path, {}).get("blob")
                for number, categories in rows:
                    if len(report["findings"]) >= MAX_FINDINGS:
                        raise ValueError("Candidate row bound")
                    for category in categories:
                        report["findings"].append(dict(
                            path=path, line=number, category=category, blob=oid,
                            tree=where, at_head=head_oid == oid,
                            head_path_exists=head_oid is not None,
                            status="candidate_not_confirmed",
                            recommendation="Review location without copying content"))
            if any(path.startswith(".github/workflows/") for path in paths):
                # Record security-relevant configuration shape, never expressions/values.
                rules = {
                    "manual_dispatch": r"\bworkflow_dispatch\b",
                    "pull_request_target": r"\bpull_request_target\b",
                    "pull_request": r"\bpull_request\b",
                    "secret_reference": r"\bsecrets\s*(?:\.|\[)",
                    "write_permission": r"\b(?:write-all|write)\b",
                    "credential_persistence_disabled": r"persist-credentials\s*:\s*false",
                    "environment_dump": r"\b(?:printenv|env\s*$|set\s+-x)\b",
                    "untrusted_checkout": r"(?:pull_request\.head|github\.event\..*head)",
                }
                for number, line in enumerate(text.splitlines(), 1):
                    for category, pattern in rules.items():
                        if re.search(pattern, line):
                            report["ci"].append(dict(path=paths[0], line=number,
                                                      category=category, blob=oid,
                                                      at_head=any(trees["head"].get(p, {}).get("blob") == oid for p in paths)))
        batch.stdin.close()
        if batch.wait(timeout=5):
            raise RuntimeError("Batch reader exit failure")
        batch.stdout.close()
        batch = None
        stage = "commit_message_screen"
        messages = git("log", "--format=%H%x00%B%x00", TAG, HEAD).split(b"\0")
        message_count = 0
        for i in range(0, len(messages) - 1, 2):
            commit = messages[i].strip().decode()
            if not re.fullmatch("[0-9a-f]{40}", commit):
                raise ValueError("Commit message record boundary")
            message_count += 1
            for number, line in enumerate(messages[i + 1].decode("utf-8", "replace").splitlines(), 1):
                for category in line_categories(line):
                    report["findings"].append(dict(path="<commit-message>", line=number,
                        category=category, commit=commit, blob=None, tree="history",
                        at_head=False, head_path_exists=False, status="candidate_not_confirmed",
                        recommendation="Review historical message without copying content"))
        stage = "cleanup_comparison"
        # Cleanup comparison is metadata plus category counts; changed text never escapes.
        changed = [p for p in sorted(set(old_cleanup) | set(new_cleanup))
                   if old_cleanup.get(p, {}).get("blob") != new_cleanup.get(p, {}).get("blob")]
        for path in changed:
            old = old_cleanup.get(path, {})
            new = new_cleanup.get(path, {})
            counts = {}
            for label, row in (("before", old), ("after", new)):
                oid = row.get("blob")
                counts[label] = dict(collections.Counter(
                    c for _, cats in cache.get(oid, []) for c in cats))
            report["cleanup_changes"].append(dict(path=path, commit=CLEANUP,
                before_blob=old.get("blob"), after_blob=new.get("blob"),
                at_head=trees["head"].get(path, {}).get("blob") == new.get("blob"),
                category_counts=counts,
                payloads_scanned=all(r.get("blob") in cache for r in (old, new))))
        report["counts"] = dict(
            tree_entries={k:len(v) for k,v in trees.items()},
            history_named_objects=len(historical), prohibited_blob_ids=len(denied),
            commit_messages_scanned=message_count,
            unique_text_blobs_scanned=scanned_blobs, payload_bytes_read=payload_bytes,
            candidate_rows=len(report["findings"]), skipped_blobs=len(report["skipped"]),
            category_rows=dict(collections.Counter(r["category"] for r in report["findings"])))
        report["complete"] = True
    except Exception as exc:
        # Do not serialize exception messages or tracebacks: library messages can contain input.
        report["failures"].append({"category": type(exc).__name__,
                                   "stage": stage,
                                   "recommendation": "Review scanner stage without printing payload"})
    finally:
        if batch is not None:
            if batch.poll() is None:
                batch.kill()
            batch.wait(timeout=5)
            for pipe in (batch.stdin, batch.stdout):
                if pipe is not None and not pipe.closed:
                    pipe.close()
        report["elapsed_seconds"] = time.monotonic() - started
        with OUT.open("x") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(json.dumps({"complete":report["complete"],"counts":report["counts"],
                          "failure_categories":[x["category"] for x in report["failures"]]}))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
