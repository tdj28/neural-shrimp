"""Verify the gd-route manuscript against its saved evidence.

Checks, in order, and exits non-zero on the first category with a failure:

1. every quoted number re-derives from its sources (build_numbers.py --verify);
2. every figure regenerates byte-identically from its inputs (each figure script
   with --verify), and every figure receipt is listed in provenance.json;
3. every macro used in main.tex is generated, and every included figure exists;
4. every repository path the text or captions cite exists in the committed tree
   (git cat-file -e HEAD:<path>);
5. no em dashes in the manuscript sources (house style).

Run from anywhere: python paper/gd-route/scripts/verify.py
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parents[1]
PYTHON = sys.executable


def run(args, cwd):
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    return result.returncode, (result.stdout + result.stderr).strip()


def main():
    failures = []
    code, out = run([PYTHON, str(PAPER / "scripts/build_numbers.py"), "--verify"], PAPER)
    print(out.splitlines()[-1] if out else "numbers: no output")
    if code:
        failures.append("numbers")
    scripts = sorted((PAPER / "figures").glob("f[0-9]*_*.py"))
    for script in scripts:
        code, out = run([PYTHON, script.name, "--verify"], PAPER / "figures")
        print(out.splitlines()[-1] if out else f"{script.name}: no output")
        if code:
            failures.append(script.name)
    checks = [
        ["tools/analyze_tiny_lm.py", "--receipt", "docs/explore-tiny-lm-receipt.json"],
        ["tools/analyze_scalar_mechanism.py", "--receipt", "docs/explore-minimal-model-scalar-receipt.json"],
        ["tools/check_warmup_intervention.py", str(Path(os.environ.get("NEURAL_SHRIMP_DATA", "/data2/neural-shrimp-data")) / "explore/warmup-intervention/attempt-002")],
        ["tools/verify_certificates.py", "certificates/c1-row35-receipt.json", "certificates/c1-row121-receipt.json", "certificates/s1r-period2-receipt.json", "certificates/s2-period18-receipt.json"],
        ["tools/certify_scalar_ns.py", "verify", "certificates/scalar-ns-receipt.json"],
    ]
    for args in checks:
        code, out = run([PYTHON, *args], ROOT)
        print(f"{args[0]}: {'passed' if code == 0 else 'FAILED'}")
        if code:
            print(out)
            failures.append(args[0])
    # The unchanged fold driver writes only to an exclusive, persistent scratch
    # bundle. Keep failed as well as successful evidence, including process output.
    fold_receipt = json.loads((ROOT / "certificates/s2-period18-fold-receipt.json").read_text())
    fold_scratch = Path(tempfile.mkdtemp(prefix="neural-shrimp-fold-replay-"))
    print(f"fold replay artifacts retained at: {fold_scratch}", flush=True)
    fold_args = ["timeout", "--signal=TERM", "--kill-after=5s", "135s", "nice", "-n", "10",
                 PYTHON, "tools/certify_s2_period18_fold.py", "--execute", "--mode", "replay",
                 "--bundle", str(Path(fold_receipt["external_root"]) / "run"),
                 "--manifest-sha256", fold_receipt["files"]["run/manifest.json"]["sha256"],
                 "--out", str(fold_scratch / "run")]
    fold_env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                    MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    (fold_scratch / "command.json").write_text(json.dumps(fold_args, indent=2) + "\n")
    with (fold_scratch / "process.log").open("w") as log:
        fold_code = subprocess.run(fold_args, cwd=ROOT, env=fold_env, stdout=log,
                                   stderr=subprocess.STDOUT).returncode
    (fold_scratch / "exit-code.txt").write_text(str(fold_code) + "\n")
    print(f"local fold replay: {'passed' if fold_code == 0 else 'FAILED'}; retained {fold_scratch}")
    if fold_code:
        failures.append("local fold replay")
    # The flip has a distinct exact two-rate family and retains its own replay.
    flip_receipt = json.loads((ROOT / "certificates/s2-period18-flip-receipt.json").read_text())
    flip_scratch = Path(tempfile.mkdtemp(prefix="neural-shrimp-flip-replay-"))
    print(f"flip replay artifacts retained at: {flip_scratch}", flush=True)
    flip_args = ["timeout", "--signal=TERM", "--kill-after=5s", "135s", "nice", "-n", "10",
                 PYTHON, "tools/certify_s2_period18_flip.py", "--execute", "--mode", "replay",
                 "--bundle", str(Path(flip_receipt["external_root"]) / "run"),
                 "--manifest-sha256", flip_receipt["run_manifest_sha256"],
                 "--out", str(flip_scratch / "run")]
    (flip_scratch / "command.json").write_text(json.dumps(flip_args, indent=2) + "\n")
    with (flip_scratch / "process.log").open("w") as log:
        flip_code = subprocess.run(flip_args, cwd=ROOT, env=fold_env, stdout=log,
                                   stderr=subprocess.STDOUT).returncode
    (flip_scratch / "exit-code.txt").write_text(str(flip_code) + "\n")
    print(f"local flip replay: {'passed' if flip_code == 0 else 'FAILED'}; retained {flip_scratch}")
    if flip_code:
        failures.append("local flip replay")
    # Distinct exact exponential-family flip and fixed-dyadic primitive36 map.
    for kind, receipt_name in (("flip", "s2-ray51-flip-receipt.json"),
                               ("period36", "s2-period36-receipt.json")):
        receipt = json.loads((ROOT / "certificates" / receipt_name).read_text())
        scratch = Path(tempfile.mkdtemp(prefix=f"neural-shrimp-ray51-{kind}-replay-"))
        args = ["timeout", "--signal=TERM", "--kill-after=5s", "75s", "nice", "-n", "10",
                PYTHON, "tools/certify_s2_ray51.py", "--execute", "--kind", kind, "--mode", "replay",
                "--bundle", str(Path(receipt["target"]["root"]) / "run"),
                "--manifest-sha256", receipt["target"]["run_manifest"]["sha256"],
                "--out", str(scratch / "run")]
        (scratch / "command.json").write_text(json.dumps(args, indent=2) + "\n")
        print(f"ray51 {kind} replay artifacts retained at: {scratch}", flush=True)
        with (scratch / "process.log").open("w") as log:
            code = subprocess.run(args, cwd=ROOT, env=fold_env, stdout=log,
                                  stderr=subprocess.STDOUT).returncode
        (scratch / "exit-code.txt").write_text(str(code) + "\n")
        print(f"ray51 {kind} replay: {'passed' if code == 0 else 'FAILED'}; retained {scratch}")
        if code:
            failures.append(f"ray51 {kind} replay")
    # Replay the reviewed joined-root theorem and its complete closed stability
    # cover once. The unchanged reviewer owns its135s+5s child timeout/cleanup.
    joined_review = json.loads((ROOT / "docs/reviews/s2-period18-joined-branch-independent-review.json").read_text())
    if joined_review["reviewer_execution_commit"] != "9bdeafd338e987c072168d96c41c9180c8e29f19":
        raise ValueError("unexpected joined-branch reviewer source")
    for name, expected in joined_review["reviewer_source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"joined-branch frozen source changed: {name}")
    slices = joined_review["sliced_stability"]
    if (slices["execution_commit"] != "71bcfa97a565d40ca209a68a90128f329ba96966"
            or slices["run_manifest"] != "62dc7bb7eb51510645042cd3eb7ed00907cd37b276374dc5750020508d6a36cb"):
        raise ValueError("unexpected joined-branch slice pins")
    joined_scratch = Path(tempfile.mkdtemp(prefix="neural-shrimp-joined-branch-replay-"))
    joined_args = ["nice", "-n", "10", PYTHON, "tools/reviews/review_joined_period18_branch.py",
                   "--out", str(joined_scratch / "attempt"), "--sliced", slices["raw"],
                   "--sliced-manifest", slices["run_manifest"],
                   "--sliced-commit", slices["execution_commit"]]
    (joined_scratch / "command.json").write_text(json.dumps(joined_args, indent=2) + "\n")
    print(f"joined-branch replay artifacts retained at: {joined_scratch}", flush=True)
    with (joined_scratch / "process.log").open("w") as log:
        joined_code = subprocess.run(joined_args, cwd=ROOT, env=fold_env,
                                     stdout=log, stderr=subprocess.STDOUT).returncode
    (joined_scratch / "exit-code.txt").write_text(str(joined_code) + "\n")
    print(f"joined-branch replay: {'passed' if joined_code == 0 else 'FAILED'}; retained {joined_scratch}")
    if joined_code:
        failures.append("joined-branch replay")
    index = json.loads((PAPER / "provenance.json").read_text())
    listed = {Path(r["path"]).name for r in index["figure_receipts"]}
    for script in scripts:
        if f"{script.stem}.provenance.json" not in listed:
            failures.append(f"{script.stem} receipt missing from provenance.json")
    tex = (PAPER / "main.tex").read_text()
    generated = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (PAPER / "generated-numbers.tex").read_text()))
    defined_here = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", tex))
    used = set(re.findall(r"\\([nxpt][A-Z][A-Za-z]*)\b", tex)) - defined_here
    missing = sorted(used - generated)
    if missing:
        failures.append(f"macros used but not generated: {missing}")
    unused = sorted(generated - used)
    if unused:
        print(f"note: generated but unused macros: {', '.join(unused)}")
    for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        if not (PAPER / "figures" / name).exists():
            failures.append(f"missing figure {name}")
    cited = set(re.findall(r"\\provenance\{([^}]+)\}", tex))
    cited |= set(re.findall(r"\\path\{([^}]+)\}", tex))
    cited |= set(re.findall(r"\\texttt\{((?:paper|docs|tools|src|experiments)/[^}]+)\}", tex))
    for path in sorted(cited):
        path = path.replace("\\_", "_")
        candidates = [path, f"paper/gd-route/{path}"]
        if not any(run(["git", "cat-file", "-e", f"HEAD:{c}"], ROOT)[0] == 0 for c in candidates):
            failures.append(f"cited path not in the committed tree: {path}")
    for source in (PAPER / "main.tex", PAPER / "references.bib", PAPER / "generated-numbers.tex"):
        text = source.read_text()
        if "\u2014" in text or re.search(r"(?<!-)---(?!-)", text):
            failures.append(f"em dash in {source.name}")
    if failures:
        print("VERIFY FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print(f"verified: {len(scripts)} figures, numbers manifest, macros, figure files and cited paths")


if __name__ == "__main__":
    main()
