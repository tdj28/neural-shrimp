"""Value-free classification of exactly 27 previously inventoried PDFs.

Source-only preparation until Root checkpoints and admits one invocation.
No rendering, OCR, embedded-file extraction, scientific replay or title/author
text output. Original PDF bytes are retained only in an exclusive scratch
allocation; decoded metadata/text stay in memory.
"""
import ast
import collections
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import resource
import selectors
import signal
import subprocess
import time

REPO = Path("/data2/neural-shrimp")
HERE = REPO / "codex/release-audit-task41"
OUT = HERE / "public-pdf-inventory-evidence.json"
SCRATCH = Path("/data2/neural-shrimp-data/scratch/audit-codex-1020b/pdf-inventory-001")
MANIFEST = HERE / "public-findings.json"
MANIFEST_PIN = (138425, "277813a8b2911a825ca50a1febd654370f3fb9b57d497d032a2cf32be7ebdabd")
CATEGORY_SOURCE = HERE / "public_blob_followup.py"
CATEGORY_PIN = (20635, "7ebd63c331f1f76a888a1b627b8228dc2fb6c998756b34c2cc75f3f810de1819")
TOOLS = {
    "pdfinfo": ("/usr/bin/pdfinfo", 76368, "19ae9ad02f055bbeb1f13f5ad4cb56c2ff130312bff967703840fe425e0ebc0a"),
    "pdftotext": ("/usr/bin/pdftotext", 64104, "aadbaa027e404c401760e3b8855de7848b18dd442d54c585ba39d461a01c19b2"),
}
TREES = {
    "arxiv_v1": "1314528118ac1300c2cffd34fb60cfbfc67d2c82",
    "head": "21272a7830d74fade1f6592af2a7b8189370ffad",
}
TARGETS = (
    ("paper/figures/analytic-training-cycle.pdf", "9f30353685a93529f0c4f4fc3a12ac677913deff", 32608),
    ("paper/figures/conceptual-readout.pdf", "1b0959c78c604846b8cf72dc54443b700772f8a5", 21815),
    ("paper/figures/conceptual-roadmap.pdf", "4590ee201977de6ea4414df83a74b92fe06b4bb3", 30425),
    ("paper/figures/conceptual-spaces.pdf", "5027b1ac3e9e112ebb9436b48f37c3abd4eb276e", 36398),
    ("paper/figures/empirical-henon-map.pdf", "1d491399464ae0798a6b5f33575dac21cf360743", 39803),
    ("paper/figures/empirical-henon-orbits.pdf", "eaa112977da51654c31fc6786e21d8ea21b4c674", 55241),
    ("paper/figures/empirical-neural-map.pdf", "5a33723309a6d24b46200dafea7c3a425b7ed2c0", 55347),
    ("paper/figures/empirical-neural-tail-diagnostics.pdf", "3bbe2e3a68cbf2a02729e7a669291f1f37c467f0", 50104),
    ("paper/figures/empirical-utility-census.pdf", "53474fbe532eec5d25453b5335f3065e22df2146", 25813),
    ("paper/figures/empirical-utility-trajectories.pdf", "d87b1e06ce88058ce0f6e350c57c659cf45356cc", 54800),
    ("paper/figures/neuralshrimp-aligned-map.pdf", "b3766aba91bff7fbea4c44c22cb4f02c5c59224b", 149459),
    ("paper/figures/neuralshrimp-controls.pdf", "6f25ac5207e9e90474c773555a60c44491f8a7e1", 265238),
    ("paper/figures/neuralshrimp-drift-anchors.pdf", "d66e8d07bb0e81158a631e7362744f0113ccc683", 33624),
    ("paper/figures/neuralshrimp-drift-continuation.pdf", "82d3d2ec8f5f54d8437e32bfdaff2f0fad608811", 45426),
    ("paper/figures/neuralshrimp-drift-utility.pdf", "2f6bdff105d6e9036d6dbbd5b5f3089fbc6eeeaa", 48015),
    ("paper/figures/neuralshrimp-drift-waveforms.pdf", "533787d00e1e90594d9e3e14eee81a894c529cdf", 136308),
    ("paper/figures/neuralshrimp-holdout-anchors.pdf", "05cbb71e6c30e33d1caaefc29ca72326de30e99c", 44289),
    ("paper/figures/neuralshrimp-holdout-utility.pdf", "4b0209c9618d5d6515f031e113372aa54a5a5ae3", 47348),
    ("paper/figures/neuralshrimp-holdout-waveforms.pdf", "9ed50215c01089850a4fd99d913ac9440abbfd57", 171984),
    ("paper/figures/neuralshrimp-information-map.pdf", "64d50d3d85b75f182a11db1ee2e103b9503057cf", 164394),
    ("paper/figures/neuralshrimp-information-utility.pdf", "f3d3b97c6dfb394fa4d6817aa145367abe803a81", 36833),
    ("paper/figures/neuralshrimp-information-waveforms.pdf", "07fec28e17e2a04044bd0a68ad9555a7e8f6aa20", 78010),
    ("paper/figures/neuralshrimp-map.pdf", "b6cabb678d3c45b515228b7021945c9bd6c37b79", 49108),
    ("paper/figures/neuralshrimp-transect.pdf", "22afc176966be396c6a60e175a0edf98946eb421", 217960),
    ("paper/figures/neuralshrimp-waveforms.pdf", "026fde99bd18c8b69a17c9dac1d8c20139370466", 170679),
    ("paper/llm/main.pdf", "25cdabfce28ef465e0d999405dee9c8247ca078e", 26773810),
    ("paper/neural-shrimp-working-draft.pdf", "d70c2d2d376ef42742fd655397f816b3c9fd2e3e", 2013795),
)
MAX_BLOB = MAX_PAYLOAD = 32 * 1024**2
MAX_TEXT = 2 * 1024**2
MAX_ALL_TEXT = 16 * 1024**2
MAX_METADATA = 64 * 1024
MAX_CANDIDATES = 10000
WALL = 170
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
META_FIELDS = (
    "Title", "Author", "Subject", "Keywords", "Creator", "Producer",
    "CreationDate", "ModDate",
)


def pinned(path, size, digest):
    path = Path(path)
    if path.is_symlink() or path.resolve() != path:
        raise ValueError("Noncanonical pinned input")
    with path.open("rb") as handle:
        value = handle.read(size + 1)
    if len(value) != size or hashlib.sha256(value).hexdigest() != digest:
        raise ValueError("Pinned bytes mismatch")
    return value


def category_definitions(source):
    names = {
        "CODE_SUFFIXES", "PATTERNS", "MONEY", "MONEY_CONTEXT", "CURRENCY_KEY",
        "PERSON", "PERSONAL", "PRIVATE_CONTACT", "PHONE", "LICENSE", "MATH_CLOSE",
    }
    selected, seen = [], set()
    for node in ast.parse(source, filename=str(CATEGORY_SOURCE)).body:
        name = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in names:
                name = node.targets[0].id
        elif isinstance(node, ast.FunctionDef) and node.name in {"categories", "forbidden"}:
            name = node.name
        if name is not None:
            if name in seen:
                raise ValueError("Duplicate category definition")
            selected.append(node)
            seen.add(name)
    if seen != names | {"categories", "forbidden"}:
        raise ValueError("Incomplete category definition set")
    namespace = dict(re=re, collections=collections, PurePosixPath=PurePosixPath)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(CATEGORY_SOURCE), "exec"), namespace)
    return namespace


def alarm(_signal, _frame):
    raise TimeoutError("Global wall bound")


def capture(argv, cap, env, overall_deadline):
    """Bound actual stdout allocation; never print child diagnostics."""
    deadline = min(overall_deadline, time.monotonic() + 20)
    chunks, size = [], 0
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, env=env)
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    try:
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("Per-tool deadline")
            if not selector.select(left):
                raise TimeoutError("Per-tool deadline")
            value = os.read(proc.stdout.fileno(), min(65536, cap - size + 1))
            if not value:
                break
            size += len(value)
            if size > cap:
                raise ValueError("Tool stdout bound")
            chunks.append(value)
        if proc.wait(timeout=max(0.001, deadline - time.monotonic())):
            raise RuntimeError("Tool exited nonzero")
        return b"".join(chunks)
    finally:
        selector.close()
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)
        proc.stdout.close()


def main():
    if any(p.exists() or p.is_symlink() for p in (OUT, SCRATCH)):
        raise ValueError("Fresh exclusive PDF allocation required")
    if HERE.resolve() != HERE or SCRATCH.parent.resolve() != SCRATCH.parent:
        raise ValueError("Canonical existing output parents required")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Offline CPU environment required")
    if os.getpriority(os.PRIO_PROCESS, 0) < 10 or any(os.environ.get(k) != "1" for k in THREAD_KEYS):
        raise ValueError("Nice10 single-thread environment required")
    for kind, desired in ((resource.RLIMIT_AS, 2 * 1024**3), (resource.RLIMIT_CPU, 120),
                          (resource.RLIMIT_FSIZE, 32 * 1024**2)):
        _, hard = resource.getrlimit(kind)
        limit = desired if hard == resource.RLIM_INFINITY else min(desired, hard)
        resource.setrlimit(kind, (limit, limit))
    started = time.monotonic()
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(WALL)
    env = dict(os.environ, LC_ALL="C.UTF-8", GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
    deadline = started + WALL
    report = dict(
        schema="task41-exact-pdf-inventory-v1", complete=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        trees=TREES, scratch=str(SCRATCH), pdfs=[], candidates=[], failures=[],
        limits=dict(blob_bytes=MAX_BLOB, aggregate_git_payload_bytes=MAX_PAYLOAD,
                    text_bytes_per_pdf=MAX_TEXT, aggregate_text_bytes=MAX_ALL_TEXT,
                    metadata_bytes_per_pdf=MAX_METADATA, tool_wall_seconds=20,
                    inner_wall_seconds=WALL, external_wall_seconds=180, cpu_seconds=120,
                    address_space_bytes=2 * 1024**3, candidate_rows=MAX_CANDIDATES),
        snippets_emitted=False, titles_or_authors_emitted=False, values_emitted=False,
        credential_validity_tested=False, ownership_or_license_clearance_claimed=False,
        scientific_replay=False, env_payloads_read=0,
        limitations=[
            "Only the 27 fixed PDFs from the prior inventory; no new tree or historical-content scan.",
            "Path-based project manuscript/figure groups are provenance hints, not ownership or license proof.",
            "Poppler metadata/text extraction is not OCR, visual review or an embedded-content/JavaScript audit.",
            "Empty extracted pages and text/metadata parse failures remain explicit coverage limits.",
            "Heuristic candidates do not establish disclosures; absence of candidates is not clearance.",
            "First-page counts are a subset of full-text counts, not independent additional findings.",
            "Only closed dollar-delimited shapes without monetary-context keywords retain mathematical bookkeeping suppression.",
        ])
    stage, payload_bytes, text_bytes, batch = "allocation", 0, 0, None
    # Existing scratch or result is never replaced, resumed, or cleaned up.
    SCRATCH.mkdir(mode=0o700)
    with OUT.open("x", encoding="utf-8") as output:
        try:
            stage = "metadata_and_tool_pins"
            manifest = json.loads(pinned(MANIFEST, *MANIFEST_PIN))
            definitions = category_definitions(pinned(CATEGORY_SOURCE, *CATEGORY_PIN))
            for _, (path, size, digest) in TOOLS.items():
                pinned(path, size, digest)
            report["metadata_sources"] = [
                dict(path=str(p), bytes=pin[0], sha256=pin[1])
                for p, pin in ((MANIFEST, MANIFEST_PIN), (CATEGORY_SOURCE, CATEGORY_PIN))
            ]
            report["tools"] = {
                name: dict(path=p, bytes=n, sha256=h) for name, (p, n, h) in TOOLS.items()
            }
            wanted = {p: (o, n) for p, o, n in TARGETS}
            pdf_rows = [r for r in manifest["path_inventory"] if r["path"].endswith(".pdf")]
            if len(wanted) != 27 or len({o for _, o, _ in TARGETS}) != 27 or len(pdf_rows) != 54:
                raise ValueError("Exact PDF inventory count")
            for label in TREES:
                selected = [r for r in pdf_rows if r["tree"] == label]
                if len(selected) != 27 or {r["path"]: (r["blob"], r["bytes"]) for r in selected} != wanted:
                    raise ValueError("Prior exact PDF inventory mismatch")
                if not all(r["at_head"] is True for r in selected):
                    raise ValueError("Prior frozen-head identity mismatch")
            if manifest["targets"]["arxiv_v1"] != TREES["arxiv_v1"] or manifest["targets"]["frozen_head"] != TREES["head"]:
                raise ValueError("Prior tree identity mismatch")
            if manifest["coverage"]["prohibited_blob_id_count"] != 0:
                raise ValueError("Prior prohibited-object scope changed")
            if sum(n for _, _, n in TARGETS) > MAX_PAYLOAD or any(
                definitions["forbidden"](p) or not p.endswith(".pdf") or not 0 < n <= MAX_BLOB
                or not re.fullmatch("[0-9a-f]{40}", o) for p, o, n in TARGETS
            ):
                raise ValueError("PDF path, size or identity refusal")
            stage = "exact_path_metadata"
            for commit in TREES.values():
                value = capture(["git", "-C", str(REPO), "ls-tree", "-r", "-l", "-z",
                                 commit, "--", *wanted], MAX_METADATA, env, deadline)
                observed = {}
                for row in value.split(b"\0"):
                    if row:
                        meta, path = row.split(b"\t", 1)
                        mode, kind, oid, size = meta.split()
                        key = path.decode("utf-8", "strict")
                        if mode != b"100644" or kind != b"blob" or key in observed:
                            raise ValueError("PDF must be a unique ordinary tracked blob")
                        observed[key] = (oid.decode(), int(size))
                if observed != wanted:
                    raise ValueError("Frozen exact PDF path mismatch")
            batch = subprocess.Popen(["git", "-C", str(REPO), "cat-file", "--batch"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, env=env)

            def classify(entry, text, scope, page=None):
                counts, auxiliary = collections.Counter(), collections.Counter()
                for line_number, line in enumerate(text.splitlines(), 1):
                    cats, extra = definitions["categories"](line, ".pdf")
                    # The old classifier can suppress a closed dollar-shaped
                    # expression. Explicit monetary context defeats that
                    # suppression here; all such bookkeeping remains visible.
                    if extra.get("mathematical_currency_shape_lines") and definitions["MONEY_CONTEXT"].search(line):
                        extra.pop("mathematical_currency_shape_lines")
                        cats = sorted(set(cats) | {"monetary_context_candidate"})
                    auxiliary.update(extra)
                    for category in cats:
                        if len(report["candidates"]) >= MAX_CANDIDATES:
                            raise ValueError("Candidate location bound")
                        report["candidates"].append(dict(
                            path=entry["path"], blob=entry["blob"], scope=scope,
                            page=page, line=line_number, category=category,
                            at_frozen_head=True, status="candidate_not_confirmed"))
                        counts[category] += 1
                return dict(counts), dict(auxiliary)

            for index, (path, oid, size) in enumerate(TARGETS):
                stage = "pdf_blob"
                entry = dict(path=path, blob=oid, bytes=size, complete=False,
                             trees=["arxiv_v1", "head"], at_frozen_head=True,
                             group=("project_figure_path" if path.startswith("paper/figures/")
                                    else "project_llm_report_path" if path == "paper/llm/main.pdf"
                                    else "project_working_draft_path"),
                             ownership_verified=False)
                report["pdfs"].append(entry)
                disk = SCRATCH / f"item-{index:02d}.pdf"
                batch.stdin.write((oid + "\n").encode())
                batch.stdin.flush()
                if batch.stdout.readline(256).split() != [oid.encode(), b"blob", str(size).encode()]:
                    raise ValueError("PDF blob header mismatch")
                sha1 = hashlib.sha1(b"blob " + str(size).encode() + b"\0")
                sha256 = hashlib.sha256()
                left = size
                with disk.open("xb") as handle:
                    while left:
                        block = batch.stdout.read(min(1024**2, left))
                        if not block:
                            raise ValueError("PDF blob short read")
                        handle.write(block)
                        sha1.update(block)
                        sha256.update(block)
                        left -= len(block)
                        payload_bytes += len(block)
                    handle.flush()
                    os.fsync(handle.fileno())
                if batch.stdout.read(1) != b"\n" or sha1.hexdigest() != oid:
                    raise ValueError("PDF blob identity mismatch")
                entry.update(sha256=sha256.hexdigest(), scratch_file=disk.name)
                stage = "pdf_metadata"
                info_bytes = capture([TOOLS["pdfinfo"][0], "-enc", "UTF-8", str(disk)],
                                     MAX_METADATA, env, deadline)
                info = info_bytes.decode("utf-8", "strict")
                entry["metadata_extraction"] = dict(
                    bytes=len(info_bytes), sha256=hashlib.sha256(info_bytes).hexdigest())
                fields = {}
                pages = []
                for line in info.splitlines():
                    key, separator, value = line.partition(":")
                    if separator and key in META_FIELDS:
                        if key in fields:
                            raise ValueError("Duplicate standard metadata field")
                        encoded = value.strip().encode("utf-8")
                        fields[key] = dict(present=bool(encoded), bytes=len(encoded),
                                           sha256=hashlib.sha256(encoded).hexdigest())
                    elif separator and key == "Pages":
                        if not re.fullmatch(r"[1-9][0-9]*", value.strip()):
                            raise ValueError("Invalid page count")
                        pages.append(int(value))
                if len(pages) != 1:
                    raise ValueError("Missing or duplicate page count")
                entry["pages"] = pages[0]
                entry["metadata_fields"] = {k: fields.get(k, dict(present=False, bytes=0)) for k in META_FIELDS}
                counts, extra = classify(entry, info, "metadata")
                entry["metadata_category_counts"] = counts
                entry["metadata_auxiliary_counts"] = extra
                stage = "pdf_text"
                remaining = MAX_ALL_TEXT - text_bytes
                text_data = capture(
                    [TOOLS["pdftotext"][0], "-layout", "-enc", "UTF-8", str(disk), "-"],
                    min(MAX_TEXT, remaining), env, deadline)
                text_bytes += len(text_data)
                text = text_data.decode("utf-8", "strict")
                entry["text_extraction"] = dict(
                    bytes=len(text_data), sha256=hashlib.sha256(text_data).hexdigest())
                segments = text.split("\f")
                if segments and not segments[-1].strip():
                    segments.pop()
                if len(segments) != entry["pages"]:
                    raise ValueError("Extracted page separator count mismatch")
                entry["empty_extracted_pages"] = [i for i, page in enumerate(segments, 1) if not page.strip()]
                entry["first_page"] = dict(
                    bytes=len(segments[0].encode()), sha256=hashlib.sha256(segments[0].encode()).hexdigest())
                counts, extra = classify(entry, segments[0], "first_page", 1)
                entry["first_page_category_counts"] = counts
                entry["first_page_auxiliary_counts"] = extra
                all_counts, all_extra = collections.Counter(), collections.Counter()
                for page_number, page in enumerate(segments, 1):
                    counts, extra = classify(entry, page, "full_text", page_number)
                    all_counts.update(counts)
                    all_extra.update(extra)
                entry["full_text_category_counts"] = dict(all_counts)
                entry["full_text_auxiliary_counts"] = dict(all_extra)
                entry["complete"] = True
            batch.stdin.close()
            if batch.wait(timeout=5):
                raise RuntimeError("Git blob reader exit failure")
            batch.stdout.close()
            batch = None
            report["complete"] = len(report["pdfs"]) == 27 and all(r["complete"] for r in report["pdfs"])
        except Exception as exc:
            report["failures"].append(dict(stage=stage, category=type(exc).__name__,
                                          pdf_index=len(report["pdfs"]) - 1))
        finally:
            signal.alarm(0)
            if batch is not None:
                if batch.poll() is None:
                    batch.kill()
                batch.wait(timeout=5)
                for pipe in (batch.stdin, batch.stdout):
                    if pipe is not None and not pipe.closed:
                        pipe.close()
            report["git_payload_bytes"] = payload_bytes
            report["extracted_text_bytes"] = text_bytes
            report["category_counts_by_scope"] = {
                scope: dict(collections.Counter(r["category"] for r in report["candidates"] if r["scope"] == scope))
                for scope in ("metadata", "first_page", "full_text")
            }
            report["elapsed_seconds"] = time.monotonic() - started
            json.dump(report, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
    print(json.dumps(dict(complete=report["complete"], pdfs=len(report["pdfs"]),
                         candidates=len(report["candidates"]), git_payload_bytes=payload_bytes,
                         extracted_text_bytes=text_bytes,
                         failure_categories=[r["category"] for r in report["failures"]])))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
