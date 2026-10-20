"""Close only the seven oversized-text screening gaps in Task41.

Preparation only until Root checkpoints and authorizes one bounded invocation.
No tree/history rescan, model calls, credential validation, or matched values.
The frozen heuristic categories are reused through an exact, hash-bound AST
selection; neither the original module nor its main() is executed.
"""
import ast
import collections
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import resource
import signal
import subprocess
import time

REPO = Path("/data2/neural-shrimp")
HERE = REPO / "codex/release-audit-task41"
OUT = HERE / "public-large-text-gap-evidence.json"
MANIFEST = HERE / "public-findings.json"
MANIFEST_PIN = (138425, "277813a8b2911a825ca50a1febd654370f3fb9b57d497d032a2cf32be7ebdabd")
CATEGORY_SOURCE = HERE / "public_blob_followup.py"
CATEGORY_PIN = (20635, "7ebd63c331f1f76a888a1b627b8228dc2fb6c998756b34c2cc75f3f810de1819")
TREES = {
    "arxiv_v1": "1314528118ac1300c2cffd34fb60cfbfc67d2c82",
    "head": "21272a7830d74fade1f6592af2a7b8189370ffad",
}
TARGETS = (
    ("evidence/EXP-004/attempt-001/selected.json", "fffd3664e38ae8557f26a610329f1ac963e12a7c", 16572495),
    ("evidence/EXP-006/attempt-001/records.jsonl", "0d40130c270b990acc7626746b1f79ff4f2efa5b", 23484808),
    ("evidence/EXP-007/attempt-001/records.jsonl", "116f03ac03496710e7330973156ed5ba0f710e83", 24143057),
    ("evidence/EXP-008/attempt-001/development-features.json", "3d7bf8cbae97e51c04fcba5e407543e35196d559", 10206330),
    ("evidence/EXP-008/attempt-001/development-targets.jsonl", "aa5cd6da44dbdf8a9ffb22f9725d750c5e642170", 15237444),
    ("evidence/EXP-008/attempt-001/test-features.json", "30aab2f426ada53d7c48962b1e9ec437181697a9", 9824036),
    ("evidence/EXP-008/attempt-001/test-targets.jsonl", "6485f1fd1160e87d7a19bc83aa88a0d7b06b961c", 15140600),
)
MAX_BLOB = 32 * 1024**2
MAX_TOTAL = 128 * 1024**2
MAX_ROWS = 20000
WALL_SECONDS = 170  # External envelope: 180 wall / 120 CPU / 2 GiB.
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def fail(message):
    raise ValueError(message)


def load_pinned(path, pin):
    if path.is_symlink() or path.resolve() != path:
        fail("Metadata path must be canonical and not a link")
    with path.open("rb") as handle:
        value = handle.read(pin[0] + 1)
    if len(value) != pin[0] or hashlib.sha256(value).hexdigest() != pin[1]:
        fail("Metadata source pin mismatch")
    return value


def load_categories(source):
    names = {
        "CODE_SUFFIXES", "PATTERNS", "MONEY", "MONEY_CONTEXT", "CURRENCY_KEY",
        "PERSON", "PERSONAL", "PRIVATE_CONTACT", "PHONE", "LICENSE", "MATH_CLOSE",
    }
    selected = []
    seen = set()
    for node in ast.parse(source, filename=str(CATEGORY_SOURCE)).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                if target.id in seen:
                    fail("Duplicate category definition")
                seen.add(target.id)
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in {"categories", "forbidden"}:
            if node.name in seen:
                fail("Duplicate category function")
            seen.add(node.name)
            selected.append(node)
    if seen != names | {"categories", "forbidden"}:
        fail("Incomplete frozen category selection")
    namespace = dict(re=re, collections=collections, PurePosixPath=PurePosixPath)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(CATEGORY_SOURCE), "exec"), namespace)
    return namespace["categories"], namespace["forbidden"]


def timeout_handler(_signal, _frame):
    raise TimeoutError("Screening wall bound reached")


def main():
    # Pure admission failures do not touch an occupied allocation.
    if OUT.exists() or OUT.is_symlink():
        fail("Fresh exclusive output required")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        fail("Offline CPU-only environment required")
    if os.getpriority(os.PRIO_PROCESS, 0) < 10 or any(os.environ.get(k) != "1" for k in THREAD_KEYS):
        fail("Nice10 single-thread environment required")
    for kind, desired in (
        (resource.RLIMIT_AS, 2 * 1024**3),
        (resource.RLIMIT_CPU, 120),
        (resource.RLIMIT_FSIZE, 16 * 1024**2),
    ):
        _, hard = resource.getrlimit(kind)
        limit = desired if hard == resource.RLIM_INFINITY else min(desired, hard)
        resource.setrlimit(kind, (limit, limit))
    started = time.monotonic()
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(WALL_SECONDS)
    report = dict(
        schema="task41-seven-large-text-gaps-v1", complete=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="Only seven exact oversized JSON/JSONL Git blobs from the pinned prior receipt.",
        trees=TREES, inputs=[], candidates=[], failures=[],
        limits=dict(blob_bytes=MAX_BLOB, aggregate_bytes=MAX_TOTAL,
                    candidate_rows=MAX_ROWS, cpu_seconds=120,
                    wall_seconds=WALL_SECONDS, external_wall_seconds=180,
                    address_space_bytes=2 * 1024**3),
        content_snippets_emitted=False, matched_values_emitted=False,
        env_payloads_read=0, model_calls=0, credential_validity_tested=False,
        absence_or_license_clearance_claimed=False,
        limitations=[
            "Heuristic text categories only; a candidate is not a confirmed disclosure.",
            "No new general tree/history or prohibited-object alias scan; exact targets come from the prior pinned receipt.",
            "The prior receipt reported zero prohibited blob identities.",
            "PDFs, other binary files, image text, and third-party ownership are outside this pass.",
            "UTF-8 text is classified, not a semantic JSON interpretation or scientific recomputation.",
        ])
    git_env = dict(os.environ, GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
    stage = "pinned_metadata"
    batch = None
    payload_bytes = 0
    # Reserve the result before any payload. A failure never overwrites it.
    with OUT.open("x", encoding="utf-8") as output:
        try:
            manifest_bytes = load_pinned(MANIFEST, MANIFEST_PIN)
            category_bytes = load_pinned(CATEGORY_SOURCE, CATEGORY_PIN)
            manifest = json.loads(manifest_bytes)
            categories, forbidden = load_categories(category_bytes)
            report["metadata_sources"] = [
                dict(path=str(path), bytes=pin[0], sha256=pin[1])
                for path, pin in ((MANIFEST, MANIFEST_PIN), (CATEGORY_SOURCE, CATEGORY_PIN))
            ]
            expected_rows = [
                dict(path=p, blob=o, bytes=n, trees=["arxiv_v1", "head"], category="over_per_blob_bound")
                for p, o, n in TARGETS
            ]
            actual_rows = [
                row for row in manifest["skipped_target_locations"]
                if row["category"] == "over_per_blob_bound" and not row["path"].endswith(".pdf")
            ]
            if actual_rows != expected_rows or manifest["coverage"]["prohibited_blob_id_count"] != 0:
                fail("Exact seven-target prior coverage mismatch")
            if manifest["targets"]["arxiv_v1"] != TREES["arxiv_v1"] or manifest["targets"]["frozen_head"] != TREES["head"]:
                fail("Prior target-tree mismatch")
            if len({o for _, o, _ in TARGETS}) != 7 or sum(n for _, _, n in TARGETS) > MAX_TOTAL:
                fail("Target inventory or aggregate bound")
            if any(forbidden(p) or PurePosixPath(p).suffix not in {".json", ".jsonl"}
                   or not re.fullmatch("[0-9a-f]{40}", o) or not 0 < n <= MAX_BLOB for p, o, n in TARGETS):
                fail("Target path, identity or size refusal")
            stage = "exact_path_metadata"
            wanted = {p: (o, n) for p, o, n in TARGETS}
            for label, commit in TREES.items():
                # Pathspecs are the seven literal files, never a broad tree walk.
                proc = subprocess.run(
                    ["git", "-C", str(REPO), "ls-tree", "-r", "-l", "-z", commit, "--", *wanted],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=git_env,
                    timeout=10, check=False)
                if proc.returncode or len(proc.stdout) > 16384:
                    fail("Bounded Git path metadata failed")
                observed = {}
                for item in proc.stdout.split(b"\0"):
                    if item:
                        meta, path = item.split(b"\t", 1)
                        mode, kind, oid, size = meta.split()
                        if mode != b"100644" or kind != b"blob":
                            fail("Target must be ordinary tracked blob")
                        decoded = path.decode("utf-8", "strict")
                        if decoded in observed:
                            fail("Duplicate target metadata")
                        observed[decoded] = (oid.decode(), int(size))
                if observed != wanted:
                    fail("Exact frozen path inventory mismatch")
            stage = "seven_blob_screen"
            batch = subprocess.Popen(
                ["git", "-C", str(REPO), "cat-file", "--batch"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                env=git_env)
            for path, oid, size in TARGETS:
                batch.stdin.write((oid + "\n").encode("ascii"))
                batch.stdin.flush()
                if batch.stdout.readline(256).split() != [oid.encode(), b"blob", str(size).encode()]:
                    fail("Git payload header mismatch")
                payload = batch.stdout.read(size)
                if len(payload) != size or batch.stdout.read(1) != b"\n":
                    fail("Git payload length mismatch")
                payload_bytes += size
                sha1 = hashlib.sha1(b"blob " + str(size).encode() + b"\0")
                sha1.update(payload)
                if sha1.hexdigest() != oid:
                    fail("Git object hash mismatch")
                entry = dict(path=path, blob=oid, bytes=size,
                             sha256=hashlib.sha256(payload).hexdigest(),
                             trees=["arxiv_v1", "head"], at_frozen_head=True,
                             complete=False, line_count=0, category_line_counts={},
                             auxiliary_line_counts={})
                report["inputs"].append(entry)
                if b"\0" in payload:
                    fail("Unexpected NUL in text target")
                # Strict decoding refuses replacement-character loss. Iterate
                # without materializing a second list of every source line.
                text = payload.decode("utf-8", "strict")
                del payload
                counts, auxiliary = collections.Counter(), collections.Counter()
                for number, line in enumerate(io.StringIO(text), 1):
                    cats, counters = categories(line.rstrip("\r\n"), PurePosixPath(path).suffix)
                    auxiliary.update(counters)
                    for category in cats:
                        if len(report["candidates"]) >= MAX_ROWS:
                            fail("Candidate location bound")
                        report["candidates"].append(dict(
                            path=path, blob=oid, line=number, category=category,
                            at_frozen_head=True, status="candidate_not_confirmed"))
                        counts[category] += 1
                    entry["line_count"] = number
                entry.update(complete=True, category_line_counts=dict(counts),
                             auxiliary_line_counts=dict(auxiliary))
                del text
            batch.stdin.close()
            if batch.wait(timeout=5):
                fail("Git blob reader failed")
            batch.stdout.close()
            batch = None
            report["complete"] = len(report["inputs"]) == 7 and all(r["complete"] for r in report["inputs"])
        except Exception as exc:
            # Exception messages and tracebacks can contain payload text.
            report["failures"].append(dict(stage=stage, category=type(exc).__name__))
        finally:
            signal.alarm(0)
            if batch is not None:
                if batch.poll() is None:
                    batch.kill()
                batch.wait(timeout=5)
                for pipe in (batch.stdin, batch.stdout):
                    if pipe is not None and not pipe.closed:
                        pipe.close()
            report["payload_bytes_read"] = payload_bytes
            report["category_line_counts"] = dict(collections.Counter(r["category"] for r in report["candidates"]))
            report["elapsed_seconds"] = time.monotonic() - started
            json.dump(report, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
    print(json.dumps(dict(complete=report["complete"], blobs=len(report["inputs"]),
                         payload_bytes=payload_bytes, candidates=len(report["candidates"]),
                         failure_categories=[r["category"] for r in report["failures"]])))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
