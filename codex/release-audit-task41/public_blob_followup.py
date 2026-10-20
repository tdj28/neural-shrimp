"""Task41 follow-up: frozen target trees and cleanup, never a history rescan.

Run once only after Root checkpoint. All payloads come from bounded Git blob
reads. Env-family/prohibited-path object identities are denied before payload
reads, including historical aliases. No matched values or snippets are emitted.
Detailed candidates go to scratch; the Git receipt groups their locations.
Heuristic coverage is explicitly not a credential-absence or license clearance.
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
OUT = REPO / "codex/release-audit-task41/public-findings-followup002.json"
RAW = Path("/data2/neural-shrimp-data/scratch/audit-codex-1020b/public-candidates-followup002.jsonl")
TAG = "1314528118ac1300c2cffd34fb60cfbfc67d2c82"
HEAD = "21272a7830d74fade1f6592af2a7b8189370ffad"
CLEANUP = "e359312e339c8f2fda59f280e6d7f92d88c0552d"
MAX_BLOB = 8 * 1024**2
MAX_TOTAL = 1024**3
MAX_METADATA = 64 * 1024**2
MAX_ROWS = 20000
MAX_SECONDS = 560
BINARY_SUFFIXES = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".zip", ".gz", ".tar",
    ".npz", ".npy", ".pt", ".pth", ".bin", ".safetensors", ".woff", ".ttf",
}
PROVENANCE_SUFFIXES = {
    ".pdf", ".csv", ".tsv", ".parquet", ".safetensors", ".bin", ".pt", ".pth",
}
CODE_SUFFIXES = {".py", ".sh", ".js", ".ts", ".css", ".svg"}
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
MONEY_CONTEXT = re.compile(r"\b(?:costs?|budget|paid|spend|spent|spending|balances?|prices?|pricing|ceiling|dollars?|funding|rent|income)\b", re.I)
CURRENCY_KEY = re.compile(
    r'''(?ix)["']?(?:[a-z][a-z0-9_./-]*(?:usd|dollars?)[a-z0-9_./-]*|
    usd|dollars?|cost|price|hourly_price|hourly_rate|cost_per_hour)["']?\s*[:=]\s*
    ["']?-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?''')
PERSON = re.compile(r"\b(?:author|user|timothy|tim|his|her|my|i|he|she)\b", re.I)
PERSONAL = re.compile(r"\b(?:family|wife|husband|spouse|children|retir\w*|career|employment|unemploy\w*|salary|income|mortgage|pension|debt|savings|illness|health|medical|hospital|diagnosis|afford|financial|finances)\b", re.I)
PRIVATE_CONTACT = re.compile(r"\b(?:phone|mobile|home address|personal email|private email|contact me)\b", re.I)
PHONE = re.compile(r"(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4}\b")
LICENSE = re.compile(r"(?i)SPDX-License-Identifier|copyright|licensed under|creative commons|apache.?2|MIT License|redistribut")
MATH_CLOSE = re.compile(r"^\$[^$\n]{1,220}\$")


def forbidden(path):
    return any(".env" in p.lower() or p.lower() == "lexical-prompt-study"
               for p in PurePosixPath(path).parts)


def categories(line, suffix):
    found = {name for name, pattern in PATTERNS if pattern.search(line)}
    counters = collections.Counter()
    money = list(MONEY.finditer(line))
    if money:
        # Mathematical notation is aggregated, not discarded silently. An
        # escaped dollar or explicit currency word is never classified as math.
        math = all(
            m.group().startswith("$") and not (m.start() and line[m.start()-1] == "\\")
            and (suffix in {".tex", ".bib", ".bbl"} or MATH_CLOSE.match(line[m.start():]))
            for m in money)
        if math:
            counters["mathematical_currency_shape_lines"] += 1
        else:
            found.add("monetary_context_candidate" if MONEY_CONTEXT.search(line)
                      else "currency_shape_candidate")
    if CURRENCY_KEY.search(line):
        found.add("numeric_currency_or_cost_key_candidate")
    if PERSON.search(line) and PERSONAL.search(line):
        if suffix in CODE_SUFFIXES:
            counters["code_personal_keyword_cooccurrence_lines"] += 1
        else:
            found.add("personal_context_candidate")
    if PRIVATE_CONTACT.search(line) and PHONE.search(line):
        found.add("private_phone_candidate")
    if LICENSE.search(line):
        counters["license_marker_lines"] += 1
    return sorted(found), counters


def main():
    if any(p.exists() or p.is_symlink() for p in (OUT, RAW)):
        raise ValueError("Exclusive follow-up allocation required")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise ValueError("Offline nice10 execution required")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise ValueError("Single-thread environment required")
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (300,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024**2,) * 2)
    started = time.monotonic()
    report = dict(
        schema="task41-frozen-tree-followup-v2", complete=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        targets=dict(arxiv_v1=TAG, frozen_head=HEAD, cleanup=CLEANUP),
        scope="All eligible blobs in the two frozen target trees; old/new versions of changed cleanup paths. No general historical payload or commit-message rescan.",
        limits=dict(blob_bytes=MAX_BLOB, aggregate_bytes=MAX_TOTAL,
                    metadata_bytes=MAX_METADATA, candidate_rows=MAX_ROWS,
                    wall_seconds=MAX_SECONDS, cpu_seconds=300,
                    address_space_bytes=2 * 1024**3),
        findings=[], cleanup_changes=[], path_inventory=[], ci=[],
        coverage={}, failures=[], content_snippets_emitted=False,
        env_payloads_read=0, credential_validity_tested=False,
        absence_or_license_clearance_claimed=False)
    git_env = dict(os.environ, GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
    batch = raw = None
    raw_hash = hashlib.sha256()
    raw_bytes = raw_rows = payload_bytes = 0
    stage = "metadata"
    trees = {}
    versions = collections.defaultdict(dict)
    metadata = {}
    dispositions = {}
    screened = {}
    groups = {}
    skipped = collections.Counter()
    aggregate = collections.Counter()
    denied = set()

    def git(*args):
        left = MAX_SECONDS - (time.monotonic() - started)
        if left <= 0:
            raise TimeoutError("Wall bound")
        result = subprocess.run(["git", "-C", str(REPO), *args],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                env=git_env, timeout=left, check=False)
        if result.returncode or len(result.stdout) > MAX_METADATA:
            raise RuntimeError("Git metadata command failed or bound exceeded")
        return result.stdout

    def tree(commit):
        rows = {}
        for item in git("ls-tree", "-r", "-l", "-z", commit).split(b"\0"):
            if not item:
                continue
            meta, path = item.split(b"\t", 1)
            mode, kind, oid, size = meta.split()
            path = path.decode("utf-8", "strict")
            rows[path] = dict(mode=mode.decode(), kind=kind.decode(),
                             blob=oid.decode(), bytes=int(size) if size != b"-" else 0)
        return rows

    def read_blob(oid, size):
        if oid in denied or not re.fullmatch("[0-9a-f]{40}", oid) or not 0 <= size <= MAX_BLOB:
            raise ValueError("Refused blob request")
        batch.stdin.write((oid + "\n").encode())
        batch.stdin.flush()
        if batch.stdout.readline(256).split() != [oid.encode(), b"blob", str(size).encode()]:
            raise ValueError("Blob identity mismatch")
        value = batch.stdout.read(size)
        if len(value) != size or batch.stdout.read(1) != b"\n":
            raise ValueError("Blob length mismatch")
        if hashlib.sha1(b"blob " + str(size).encode() + b"\0" + value).hexdigest() != oid:
            raise ValueError("Blob object hash mismatch")
        return value

    def emit(path, oid, labels, number, category):
        nonlocal raw_rows, raw_bytes
        if raw_rows >= MAX_ROWS:
            raise ValueError("Meaningful candidate row bound")
        head = trees["head"].get(path, {})
        row = dict(path=path, line=number, category=category, blob=oid,
                   trees=sorted(labels), at_head=head.get("blob") == oid,
                   head_path_exists=bool(head), status="candidate_not_confirmed",
                   recommendation="Review location without copying matched content")
        encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
        raw.write(encoded)
        raw_hash.update(encoded)
        raw_bytes += len(encoded)
        raw_rows += 1
        key = (path, oid, category)
        group = groups.setdefault(key, dict(
            path=path, blob=oid, category=category, trees=sorted(labels),
            at_head=row["at_head"], head_path_exists=bool(head),
            count=0, lines=[], status="candidate_not_confirmed",
            recommendation="Review exact locations in the value-free scratch evidence"))
        group["count"] += 1
        if len(group["lines"]) < 12:
            group["lines"].append(number)

    try:
        for value in (TAG, HEAD, CLEANUP):
            if git("rev-parse", value + "^{commit}").decode().strip() != value:
                raise ValueError("Frozen commit identity mismatch")
        trees = {"arxiv_v1": tree(TAG), "head": tree(HEAD)}
        before = git("rev-parse", CLEANUP + "^").decode().strip()
        old, new = tree(before), tree(CLEANUP)
        changed = [p for p in sorted(set(old) | set(new))
                   if old.get(p, {}).get("blob") != new.get(p, {}).get("blob")]
        report["cleanup_parent"] = before
        # Path-only Git metadata is read first. No env-family payload is requested.
        env_meta = git("log", "--all", "--format=", "--raw", "--no-abbrev",
                       "--no-renames", "--root", "--", ":(icase)*.env*",
                       ":(icase,glob)**/*.env*", ":(icase,glob)**/*.env*/**",
                       ":(icase)lexical-prompt-study",
                       ":(icase,glob)**/lexical-prompt-study/**")
        for line in env_meta.splitlines():
            if line.startswith(b":"):
                for field in line.split()[2:4]:
                    if re.fullmatch(b"[0-9a-f]{40}", field) and field != b"0" * 40:
                        denied.add(field.decode())
        for rows in (*trees.values(), old, new):
            denied.update(r["blob"] for p, r in rows.items() if forbidden(p))

        all_rows = [
            ("arxiv_v1", trees["arxiv_v1"]), ("head", trees["head"]),
            ("cleanup_before", {p:old[p] for p in changed if p in old}),
            ("cleanup_after", {p:new[p] for p in changed if p in new})]
        for label, rows in all_rows:
            for path, row in rows.items():
                if row["kind"] != "blob":
                    continue
                oid = row["blob"]
                versions[oid].setdefault(path, set()).add(label)
                if oid in metadata and metadata[oid] != row["bytes"]:
                    raise ValueError("Metadata size inconsistency")
                metadata[oid] = row["bytes"]
                suffix = PurePosixPath(path).suffix.lower()
                if label in trees and (
                    suffix in PROVENANCE_SUFFIXES or row["mode"] == "120000"
                    or forbidden(path) or re.search(r"(?i)(?:^|/)(?:datasets?|vendor|third.party|literature)(?:/|$)", path)
                    or re.search(r"(?i)(?:^|/)(?:id_rsa|id_ed25519|credentials|secrets?)(?:[./_-]|$)|\.(?:pem|key|p12|pfx)$", path)):
                    report["path_inventory"].append(dict(
                        path=path, blob=oid, bytes=row["bytes"], tree=label,
                        category="prohibited_payload_path" if forbidden(path)
                        else "symlink_not_followed" if row["mode"] == "120000"
                        else "provenance_or_sensitive_filename_review_candidate",
                        at_head=trees["head"].get(path, {}).get("blob") == oid))
        RAW.parent.mkdir(parents=True, exist_ok=True)
        raw = RAW.open("xb")
        stage = "target_and_cleanup_blob_screen"
        batch = subprocess.Popen(["git", "-C", str(REPO), "cat-file", "--batch"],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, env=git_env)
        for oid, locations in versions.items():
            size = metadata[oid]
            suffixes = {PurePosixPath(p).suffix.lower() for p in locations}
            reason = ("prohibited_env_or_project_payload" if oid in denied
                      else "over_per_blob_bound" if size > MAX_BLOB
                      else "binary_inventory_only" if suffixes <= BINARY_SUFFIXES
                      else "aggregate_payload_bound" if payload_bytes + size > MAX_TOTAL
                      else "wall_bound_before_next_blob" if time.monotonic() - started > MAX_SECONDS - 20
                      else None)
            if reason:
                dispositions[oid] = reason
                skipped[reason] += 1
                continue
            value = read_blob(oid, size)
            payload_bytes += size
            if b"\0" in value[:8192]:
                dispositions[oid] = "binary_probe_only"
                skipped["binary_probe_only"] += 1
                continue
            text = value.decode("utf-8", "replace")
            # Per-path suffix handling prevents a TeX alias from suppressing a
            # candidate in a non-TeX alias of identical bytes.
            by_suffix = {}
            for suffix in suffixes:
                rows, counters = [], collections.Counter()
                for number, line in enumerate(text.splitlines(), 1):
                    cats, counts = categories(line, suffix)
                    counters.update(counts)
                    rows.extend((number, cat) for cat in cats)
                by_suffix[suffix] = (rows, counters)
            screened[oid] = {}
            for path, labels in locations.items():
                rows, counters = by_suffix[PurePosixPath(path).suffix.lower()]
                screened[oid][path] = dict(collections.Counter(cat for _, cat in rows))
                for category, count in counters.items():
                    aggregate[category] += count
                for number, category in rows:
                    emit(path, oid, labels, number, category)
                if path.startswith(".github/workflows/"):
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
                                report["ci"].append(dict(path=path, line=number,
                                    category=category, blob=oid, trees=sorted(labels),
                                    at_head=trees["head"].get(path, {}).get("blob") == oid))
            dispositions[oid] = "text_screen_complete"
        batch.stdin.close()
        if batch.wait(timeout=5):
            raise RuntimeError("Batch reader exit failure")
        batch.stdout.close()
        batch = None
        stage = "cleanup_comparison"
        for path in changed:
            a, b = old.get(path, {}), new.get(path, {})
            report["cleanup_changes"].append(dict(
                path=path, commit=CLEANUP, before_blob=a.get("blob"), after_blob=b.get("blob"),
                at_head=trees["head"].get(path, {}).get("blob") == b.get("blob"),
                before_counts=screened.get(a.get("blob"), {}).get(path),
                after_counts=screened.get(b.get("blob"), {}).get(path),
                before_disposition=dispositions.get(a.get("blob"), "not_a_blob"),
                after_disposition=dispositions.get(b.get("blob"), "not_a_blob")))
        report["complete"] = not any(dispositions.get(oid) in (
            "aggregate_payload_bound", "wall_bound_before_next_blob") for oid in versions)
    except Exception as exc:
        report["failures"].append(dict(category=type(exc).__name__, stage=stage))
    finally:
        if batch is not None:
            if batch.poll() is None:
                batch.kill()
            batch.wait(timeout=5)
            for pipe in (batch.stdin, batch.stdout):
                if pipe is not None and not pipe.closed:
                    pipe.close()
        if raw is not None:
            raw.flush()
            os.fsync(raw.fileno())
            raw.close()
        report["raw_candidates"] = dict(path=str(RAW), bytes=raw_bytes,
            sha256=raw_hash.hexdigest(), rows=raw_rows, exists=raw is not None)
        report["findings"] = sorted(groups.values(), key=lambda r: (
            not r["at_head"], r["category"], r["path"], r["blob"]))
        report["coverage"] = dict(
            tree_entries={k:len(v) for k,v in trees.items()},
            requested_unique_blobs=len(versions), completed_dispositions=len(dispositions),
            unique_text_blobs=sum(v == "text_screen_complete" for v in dispositions.values()),
            payload_bytes_read=payload_bytes, prohibited_blob_id_count=len(denied),
            skipped=dict(skipped), aggregate_line_counts=dict(aggregate),
            candidate_rows=raw_rows, grouped_candidates=len(groups),
            historical_payload_scope="Only changed cleanup-path versions; no general history or commit-message payloads.",
            target_tree_status={
                label:dict(collections.Counter(dispositions.get(r["blob"], "not_reached")
                    for r in rows.values() if r["kind"] == "blob"))
                for label, rows in trees.items()})
        # Persist exact coverage without putting every history/blob row in Git.
        report["skipped_target_locations"] = [
            dict(path=p, blob=oid, bytes=metadata[oid], trees=sorted(labels),
                 category=dispositions.get(oid, "not_reached"))
            for oid, locations in versions.items()
            if dispositions.get(oid) != "text_screen_complete"
            for p, labels in locations.items()]
        report["elapsed_seconds"] = time.monotonic() - started
        with OUT.open("x") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(json.dumps(dict(complete=report["complete"], coverage=report["coverage"],
            failure_categories=[r["category"] for r in report["failures"]])))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
