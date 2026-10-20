"""Single-pass archive/member hashes, direct release closure, text-risk screening.

Only the declared Zenodo directory is read. Archives are never extracted.
Potential secret/personal-data matches are reported as filenames/categories,
never matching contents. No complete-secret-free guarantee is asserted.
Large JSON is screened as text but not parsed into a giant in-memory object.
"""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import time

ROOT = Path("/data2/neural-shrimp")
OUT = ROOT / "codex/arxiv-gonogo2-task40"
PACKAGE = Path("/data2/neural-shrimp-data/zenodo/neural-shrimp-proofs-v1")
SCRATCH = Path("/data2/neural-shrimp-data/scratch/audit-codex-1020")
RELEASES = ["code/certificates/llm-p8-certificate-final-001.json",
            "code/codex/coexistence-task33/pair-certificate-release.json"]
RELEASE_PINS = {
    RELEASES[0]: ("539ead0d344f0eb3e0b91f55999da1c6cd76f801d62e274b4537241018949f97", 10471, 15),
    RELEASES[1]: ("eb1e2238f2607408ddcbb31383c6a7e4c1c25f0803f5d9a5af90769a196fcf97", 8219, 19),
}
CODE_ARCHIVE = "code-and-release-records-fcf2d462.tar.gz"
CODE_PREFIX = "neural-shrimp-fcf2d462/"
CHUNK = 1024**2
PATTERNS = {
    "private_key_header": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "github_token_shape": re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "service_key_shape": re.compile(rb"\b(?:sk-[A-Za-z0-9_-]{24,}|(?:AKIA|ASIA)[A-Z0-9]{16}|hf_[A-Za-z0-9]{24,}|rpa_[A-Za-z0-9_-]{24,})\b"),
    "credential_assignment_candidate": re.compile(rb"""(?i)(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*["'][A-Za-z0-9_+/-]{20,}["']"""),
    "email_address": re.compile(rb"[A-Za-z0-9._%+-]{1,80}@[A-Za-z0-9.-]{1,100}\.[A-Za-z]{2,12}\b"),
    "private_host_path": re.compile(rb"/(?:home|Users)/[^/\s\"']{1,80}/"),
}
TEXT_EXT = {".json", ".jsonl", ".py", ".md", ".txt", ".log", ".toml", ".yaml", ".yml",
            ".sh", ".csv", ".tsv", ".ini", ".cfg", ".tex", ".bib"}
NEEDLES = {
    "private_key_header": (b"-----BEGIN ",),
    "github_token_shape": (b"gh", b"github_pat_"),
    "service_key_shape": (b"sk-", b"AKIA", b"ASIA", b"hf_", b"rpa_"),
    "email_address": (b"@",),
    "private_host_path": (b"/home/", b"/Users/"),
}


class HashedReader:
    def __init__(self, stream):
        self.stream = stream
        self.hash = hashlib.sha256()
        self.bytes = 0

    def read(self, n=-1):
        data = self.stream.read(n)
        self.hash.update(data)
        self.bytes += len(data)
        return data


def scan_chunk(data, found):
    for category, expression in PATTERNS.items():
        if category in found:
            continue
        if category in NEEDLES and not any(x in data for x in NEEDLES[category]):
            continue
        if category == "credential_assignment_candidate":
            lower = data.lower()
            if not any(x in lower for x in (b"key", b"token", b"password", b"secret")):
                continue
        if expression.search(data):
            found.add(category)


def refs(obj, trail="", issues=None):
    found = []
    if isinstance(obj, dict):
        if "path" in obj:
            if isinstance(obj["path"], str) and re.fullmatch(r"[a-f0-9]{64}", str(obj.get("sha256", ""))):
                found.append({"path": obj["path"], "sha256": obj["sha256"], "field": trail,
                              "bytes": obj.get("bytes")})
            elif issues is not None:
                issues.append({"field": trail, "reason": "path without accepted sha256 schema"})
        for key, value in obj.items():
            found.extend(refs(value, trail + "/" + str(key), issues))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            found.extend(refs(value, trail + "/" + str(i), issues))
    return found


def canonical_ref(path):
    for original, mapped in (
        ("/data2/neural-shrimp-data/codex/llm-p8-certificate/", "p8/"),
        ("/data2/neural-shrimp-data/codex/coexistence/", "coexistence/"),
    ):
        if path.startswith(original):
            return mapped + path[len(original):]
    for prefix in (
        "/data2/neural-shrimp/",
        "/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp/",
    ):
        if path.startswith(prefix):
            return "code/" + path[len(prefix):]
    if path.startswith(("codex/", "certificates/")):
        return "code/" + path
    return None


def safe_open(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError("Nonregular package input")
    return os.fdopen(fd, "rb")


def strict_json(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    # Historical JSON may contain nonstandard numerical NaNs. Record identity
    # without silently dropping the entire record; this is not numeric admission.
    return json.loads(raw, object_pairs_hook=pairs)


def main():
    start = time.monotonic()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    summary_path = OUT / "package-evidence.json"
    manifest_path = SCRATCH / "package-member-manifest.json"
    if summary_path.exists() or manifest_path.exists():
        raise ValueError("Immutable output already exists")
    if PACKAGE.resolve() != PACKAGE:
        raise ValueError("Package directory contains symlink component")
    entries = list(PACKAGE.iterdir())
    if any(not stat.S_ISREG(p.lstat().st_mode) for p in entries):
        raise ValueError("Package contains linked or nonregular top-level entries")
    with safe_open(PACKAGE / "SHA256SUMS.txt") as handle:
        sums_bytes = handle.read()
    declared = {}
    for line in sums_bytes.decode().splitlines():
        digest, name = line.split(None, 1)
        name = name.lstrip("*")
        if name in ("", ".", "..") or "/" in name or name in declared or not re.fullmatch("[a-f0-9]{64}", digest):
            raise ValueError("Invalid/duplicate checksum filename")
        declared[name] = digest
    actual_names = {p.name for p in entries}
    required_archives = ({CODE_ARCHIVE}
                         | {f"p8-evidence-part{i:02d}-of-08.tar" for i in range(1, 9)}
                         | {f"coexistence-evidence-part{i:02d}-of-07.tar" for i in range(1, 8)})
    required_files = (required_archives | {"README.md", "steering-vectors-l12.json",
                                          "steering-vectors-l12.npz"}
                      | {n[:-4] + ".listing.txt" for n in required_archives if n.endswith(".tar")})
    if not required_files <= set(declared) or not required_files <= actual_names:
        raise ValueError("Required archive part, listing, README or steering asset missing")
    members = {}
    all_refs = defaultdict(list)
    release_objects = {}
    checksum_rows, risk_rows, unsafe, duplicates, nonregular = [], [], [], [], []
    json_errors, reference_schema_issues = [], {}
    parsed_json_count = 0
    large_json_count = 0
    text_bytes = 0
    for name, expected in declared.items():
        path = PACKAGE / name
        before = path.lstat()
        with safe_open(path) as raw:
            opened = os.fstat(raw.fileno())
            if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                raise ValueError("Package input changed before open")
            stream = HashedReader(raw)
            if name.endswith((".tar", ".tar.gz")):
                if name == CODE_ARCHIVE:
                    family = "code"
                elif re.fullmatch(r"p8-evidence-part0[1-8]-of-08\.tar", name):
                    family = "p8"
                elif re.fullmatch(r"coexistence-evidence-part0[1-7]-of-07\.tar", name):
                    family = "coexistence"
                else:
                    raise ValueError("Unexpected archive family/name")
                with tarfile.open(fileobj=stream, mode="r|gz" if name.endswith(".gz") else "r|") as archive:
                    for item in archive:
                        pure = PurePosixPath(item.name)
                        if pure.is_absolute() or ".." in pure.parts or str(pure) != item.name.rstrip("/"):
                            unsafe.append({"archive": name, "member": item.name})
                            raise ValueError("Unsafe/noncanonical archive member")
                        if item.isdir():
                            continue
                        if not item.isfile():
                            nonregular.append({"archive": name, "member": item.name, "type": repr(item.type)})
                            continue
                        if family == "code" and not item.name.startswith(CODE_PREFIX):
                            raise ValueError("Unexpected code archive root")
                        rel = item.name[len(CODE_PREFIX):] if family == "code" else item.name
                        key = family + "/" + rel
                        is_text = pure.suffix.lower() in TEXT_EXT or pure.name.startswith(("LICENSE", "README"))
                        if pure.name == ".env" or pure.name in ("id_rsa", "id_ed25519"):
                            unsafe.append({"archive": name, "member": item.name, "reason": "sensitive filename"})
                        digest = hashlib.sha256()
                        size = 0
                        risks = set()
                        keep_json = pure.suffix.lower() == ".json" and item.size <= 16*1024**2
                        pieces = [] if keep_json else None
                        tail = b""
                        handle = archive.extractfile(item)
                        while True:
                            chunk = handle.read(CHUNK)
                            if not chunk:
                                break
                            digest.update(chunk)
                            size += len(chunk)
                            if is_text:
                                scan_chunk(tail + chunk, risks)
                                tail = chunk[-512:]
                                text_bytes += len(chunk)
                            if keep_json:
                                pieces.append(chunk)
                        if size != item.size:
                            raise ValueError(f"Tar member length mismatch: {key}")
                        row = {"archive": name, "member": item.name, "bytes": size, "sha256": digest.hexdigest()}
                        if key in members:
                            duplicates.append({"key": key, "same_hash": members[key]["sha256"] == row["sha256"]})
                        members[key] = row
                        if risks:
                            risk_rows.append({"file": key, "categories": sorted(risks)})
                        if keep_json:
                            try:
                                obj = strict_json(b"".join(pieces))
                            except (ValueError, UnicodeError) as exc:
                                json_errors.append({"file": key, "error_type": type(exc).__name__})
                                obj = None
                            if obj is not None:
                                parsed_json_count += 1
                                issues = []
                                all_refs[key] = refs(obj, issues=issues)
                                if issues:
                                    reference_schema_issues[key] = issues
                                if key in RELEASES:
                                    release_objects[key] = obj
                        elif pure.suffix.lower() == ".json":
                            large_json_count += 1
            else:
                risks = set()
                tail = b""
                while True:
                    chunk = stream.read(CHUNK)
                    if not chunk:
                        break
                    if path.suffix.lower() in TEXT_EXT or path.name.startswith(("LICENSE", "README")):
                        scan_chunk(tail + chunk, risks)
                        tail = chunk[-512:]
                        text_bytes += len(chunk)
                if risks:
                    risk_rows.append({"file": name, "categories": sorted(risks)})
            # Include tar padding/footer or unread compressed trailer in outer SHA.
            while stream.read(CHUNK):
                pass
            got = stream.hash.hexdigest()
            row = {"name": name, "bytes": stream.bytes, "expected": expected, "sha256": got, "match": got == expected}
        after = path.lstat()
        row["unchanged_during_read"] = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        checksum_rows.append(row)
        print(json.dumps({"archive_or_file_finished": name, "match": row["match"],
                          "member_count_so_far": len(members)}), flush=True)
    closure = {}
    for release in RELEASES:
        if release not in release_objects:
            raise ValueError(f"Missing release: {release}")
        expected_hash, expected_size, expected_refs = RELEASE_PINS[release]
        if (members[release]["sha256"], members[release]["bytes"]) != (expected_hash, expected_size):
            raise ValueError("Frozen release identity mismatch")
        if len(all_refs[release]) != expected_refs or release in reference_schema_issues:
            raise ValueError("Frozen release direct reference schema/count mismatch")
        checks = []
        direct = list(all_refs[release])
        if release == RELEASES[0]:
            pins = release_objects[release]["inherited_accepted_review_pins"]
            if len(pins) != 5:
                raise ValueError("Inherited review count mismatch")
            direct.extend({"path": "codex/certify-llm/" + name, "sha256": digest,
                           "bytes": None, "field": "/inherited_accepted_review_pins/" + name}
                          for name, digest in pins.items())
        for ref in direct:
            key = canonical_ref(ref["path"])
            actual = members.get(key)
            checks.append({**ref, "package_key": key, "present": actual is not None,
                           "hash_match": actual is not None and actual["sha256"] == ref["sha256"],
                           "size_match": actual is not None and (ref["bytes"] is None or actual["bytes"] == ref["bytes"])})
        closure[release] = {"release_sha256": members[release]["sha256"],
                            "frozen_identity_matches": True, "path_sha_references": expected_refs,
                            "inherited_review_references": len(direct)-expected_refs,
                            "references": checks}
    # Transitive path+sha references are classified separately: the original
    # records also pin external runtime/assets and historical source versions.
    reachable, queue = set(RELEASES), list(RELEASES)
    transitive_missing, transitive_mismatch, external = [], [], []
    matched = 0
    while queue:
        owner = queue.pop()
        for ref in all_refs.get(owner, []):
            key = canonical_ref(ref["path"])
            item = {"from": owner, **ref, "package_key": key}
            if key is None:
                external.append(item)
            elif key not in members:
                transitive_missing.append(item)
            elif members[key]["sha256"] != ref["sha256"]:
                transitive_mismatch.append(item)
            else:
                matched += 1
                if key not in reachable:
                    reachable.add(key)
                    queue.append(key)
    manifest = {"members": members, "text_risk_candidates": risk_rows,
                "transitive_missing": transitive_missing, "transitive_mismatch": transitive_mismatch,
                "external_references": external, "json_parse_errors": json_errors,
                "reference_schema_issues": reference_schema_issues}
    with manifest_path.open("x") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    result = {
        "sha256sums_sha256": hashlib.sha256(sums_bytes).hexdigest(), "checksums": checksum_rows,
        "unlisted_files": sorted(actual_names - set(declared) - {"SHA256SUMS.txt"}),
        "missing_files": sorted(set(declared) - actual_names), "member_count": len(members),
        "member_bytes": sum(r["bytes"] for r in members.values()), "direct_release_closure": closure,
        "unsafe_members": unsafe, "nonregular_members": nonregular, "duplicates": duplicates,
        "text_bytes_screened": text_bytes, "risk_candidate_files": len(risk_rows),
        "risk_category_counts": {k: sum(k in r["categories"] for r in risk_rows) for k in PATTERNS},
        "parsed_small_json_count": parsed_json_count, "large_json_not_structurally_parsed": large_json_count,
        "json_parse_error_count": len(json_errors),
        "reference_schema_issue_files": len(reference_schema_issues),
        "transitive": {"reachable_small_json_or_files": len(reachable), "matched_reference_edges": matched,
                       "missing_edges": len(transitive_missing), "mismatched_edges": len(transitive_mismatch),
                       "external_edges": len(external)},
        "manifest": {"path": str(manifest_path), "bytes": manifest_path.stat().st_size,
                     "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()},
        "seconds": time.monotonic() - start,
        "limits": ["Text regex screening cannot establish absence of every possible secret or personal datum.",
                   "Large JSON contents hashed and text-screened, but nested references not parsed.",
                   "Identity checks are not new mathematical certificate verification.",
                   "Five named P8 inherited reviews checked directly; domain/map hashes are semantic bindings, not file digests.",
                   "README semantic accuracy and risk-candidate interpretation require separate review."],
    }
    result["integrity_and_direct_closure_passed"] = (
        all(r["match"] and r["unchanged_during_read"] for r in checksum_rows)
        and not result["unlisted_files"] and not result["missing_files"]
        and not unsafe and not nonregular and not duplicates
        and all(r["hash_match"] and r["size_match"] for v in closure.values() for r in v["references"]))
    result["publication_clearance"] = False
    with summary_path.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({"output": str(summary_path), "seconds": result["seconds"],
                      "checksum_failures": sum(not r["match"] for r in checksum_rows),
                      "direct_reference_failures": sum(not r["hash_match"] for v in closure.values() for r in v["references"])}))
    return 0 if result["integrity_and_direct_closure_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
