"""Offline fail-closed regeneration of C1 certificates bound by a Git receipt.

One command: python tools/verify_certificates.py certificates/c1-receipt.json
The small receipt binds immutable inputs, output and executable source hashes;
raw files stay in the external data directory. No network access is used.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from neural_shrimp.rigorous_orbits import canonical_bytes, check_orbit, sha256


def verify(receipt_path):
    receipt = json.loads(Path(receipt_path).read_bytes())
    if receipt.get("schema") != "neural-shrimp-c1-receipt-v1":
        raise ValueError("unknown receipt schema")
    raw = Path(receipt["external_root"])
    required = {"inputs.json", "certificate.json", "runtime-start.json", "runtime-end.json"}
    if set(receipt["files"]) != required:
        raise ValueError("incomplete or unexpected file inventory")
    for name, record in receipt["files"].items():
        content = (raw / name).read_bytes()
        if len(content) != record["bytes"] or sha256(content) != record["sha256"]:
            raise ValueError(f"artifact hash/size mismatch: {name}")
    expected_sources = {"src/neural_shrimp/rigorous_orbits.py", "tools/certify_c1.py", "tools/verify_certificates.py"}
    if set(receipt["source_sha256"]) != expected_sources:
        raise ValueError("incomplete checker source inventory")
    for name, digest in receipt["source_sha256"].items():
        if not (ROOT / name).resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("invalid source path")
        if sha256((ROOT / name).read_bytes()) != digest:
            raise ValueError(f"checker source hash mismatch: {name}")
    certificate = json.loads((raw / "certificate.json").read_bytes())
    inputs = json.loads((raw / "inputs.json").read_bytes())
    if certificate.get("schema") != "neural-shrimp-c1-certificate-v1":
        raise ValueError("unknown certificate schema")
    if certificate["inputs_sha256"] != sha256(canonical_bytes(inputs)):
        raise ValueError("input binding mismatch")
    if certificate["runtime"]["source_sha256"] != receipt["source_sha256"]:
        raise ValueError("runtime source binding mismatch")
    recomputed = check_orbit(inputs, **certificate["result"]["settings"])
    if canonical_bytes(recomputed) != canonical_bytes(certificate["result"]):
        raise ValueError("regenerated enclosures or verdict differ")
    if not recomputed["passed"] or not receipt["passed"] or receipt["statement"] != recomputed["exact_statement"]:
        raise ValueError("certificate does not pass every required gate")
    return {"passed": True, "evidence_rung": 3, "statement": recomputed["exact_statement"],
            "research_status": recomputed["research_status"], "certificate_sha256": sha256((raw / "certificate.json").read_bytes())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.receipt), indent=2))
    except Exception as error:
        print(json.dumps({"passed": False, "error": type(error).__name__ + ": " + str(error)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
