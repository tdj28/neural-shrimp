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

import json
import os
import re
import subprocess
import sys
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
    scripts = sorted((PAPER / "figures").glob("f[0-9]_*.py"))
    for script in scripts:
        code, out = run([PYTHON, script.name, "--verify"], PAPER / "figures")
        print(out.splitlines()[-1] if out else f"{script.name}: no output")
        if code:
            failures.append(script.name)
    checks = [
        ["tools/analyze_tiny_lm.py", "--receipt", "docs/explore-tiny-lm-receipt.json"],
        ["tools/analyze_scalar_mechanism.py", "--receipt", "docs/explore-minimal-model-scalar-receipt.json"],
        ["tools/check_warmup_intervention.py", str(Path(os.environ.get("NEURAL_SHRIMP_DATA", "/data2/neural-shrimp-data")) / "explore/warmup-intervention/attempt-002")],
        ["tools/verify_certificates.py", "certificates/c1-row35-receipt.json", "certificates/c1-row121-receipt.json"],
        ["tools/certify_scalar_ns.py", "verify", "certificates/scalar-ns-receipt.json"],
    ]
    for args in checks:
        code, out = run([PYTHON, *args], ROOT)
        print(f"{args[0]}: {'passed' if code == 0 else 'FAILED'}")
        if code:
            print(out)
            failures.append(args[0])
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
