"""Read-only Task40 bundle build/content/layout preparation; run only after checkpoint.

Never runs the producer's bundle script or writes in the article checkout.
All child commands are serial, inherit the Root runner's one-thread/nice10
limits, and receive at most 300 CPU seconds, 4 GiB AS and a shared 600s deadline.
The output is audit evidence, not a publication or a numerical-science verdict.
"""
import difflib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import resource
import subprocess
import tarfile
import time
import traceback

ARTICLE = Path("/data2/chaos-journal-neural-shrimp")
COMMIT = "bb42a82189da91c37f50804c077832df9e2237ae"
BUNDLE = Path("/data2/neural-shrimp-data/arxiv/arxiv-source-bb42a82.tar.gz")
OUT = Path("/data2/neural-shrimp-data/scratch/audit-codex-1020/bundle-001")
REPORT = Path("/data2/neural-shrimp/codex/arxiv-gonogo2-task40/bundle-evidence.json")
MAIN_SHA = "b0ecb1d8b74d19bb0b1a6b87c402846679f12b50346d135f9a0beecf45d562f1"
BIB_SHA = "d71ac43d72519f400d1dafa19db0b3de12d1358bbaa0d50c806902eb0b2b838e"
MAX_BYTES = 64 * 1024**2


def sha(data):
    return hashlib.sha256(data).hexdigest()


def comment_free(text):
    """Independent lexical comparison; escaped percent signs remain literal."""
    lines = []
    for line in text.splitlines():
        cut = None
        for index, char in enumerate(line):
            if char != "%":
                continue
            backslashes = 0
            prior = index - 1
            while prior >= 0 and line[prior] == "\\":
                backslashes += 1
                prior -= 1
            if backslashes % 2 == 0:
                cut = index
                break
        if cut is not None:
            if not line[:cut].strip():
                continue
            line = line[:cut + 1]
        else:
            line = line.rstrip()
        lines.append(line)
    return "\n".join(lines) + "\n"


def main():
    if REPORT.exists() or REPORT.is_symlink() or OUT.exists() or OUT.is_symlink():
        raise ValueError("Fresh output/report allocation required; preserve earlier attempts")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("Root must set HF_HUB_OFFLINE=1")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise ValueError("Root must set every declared thread count to one")
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise ValueError("Root must start the worker under nice10")
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + 600
    evidence = {
        "article_commit": COMMIT, "script_sha256": sha(Path(__file__).read_bytes()),
        "scratch": str(OUT), "commands": [], "files": [], "failures": [],
        "publication_authorized": False, "science_revalidated": False,
        "visual_inspection_completed_by_script": False,
        "limits": {"one_serial_child": True, "cpu_seconds_per_child": 300,
                   "address_space_bytes": 4 * 1024**3, "shared_wall_seconds": 600},
    }
    env = dict(os.environ)
    for key in ("TEXINPUTS", "BIBINPUTS", "BSTINPUTS", "TEXMFHOME", "TEXMFCNF"):
        env.pop(key, None)
    env.update(TEXMFHOME=str(OUT / "no-texmfhome"), openin_any="p", openout_any="p")

    def child_limits():
        resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (300,) * 2)
        resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024**2,) * 2)

    def run(argv, cwd=OUT):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Shared 600s audit deadline exhausted")
        index = len(evidence["commands"]) + 1
        stdout = OUT / f"command-{index:03d}.stdout"
        stderr = OUT / f"command-{index:03d}.stderr"
        record = {"argv": argv, "cwd": str(cwd)}
        evidence["commands"].append(record)
        before = time.monotonic()
        try:
            with stdout.open("xb") as of, stderr.open("xb") as ef:
                result = subprocess.run(argv, cwd=cwd, env=env, stdout=of, stderr=ef,
                                        timeout=remaining, preexec_fn=child_limits)
            record["exit_code"] = result.returncode
            if result.returncode:
                raise RuntimeError(f"Child {index} failed: {argv[0]} exit {result.returncode}")
        except subprocess.TimeoutExpired:
            record["timeout"] = True
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - before
            for name, path in (("stdout", stdout), ("stderr", stderr)):
                if path.exists():
                    record[name] = {"path": str(path), "bytes": path.stat().st_size,
                                    "sha256": sha(path.read_bytes())}
        return stdout.read_bytes()

    def blob(name):
        size = int(run(["git", "-C", str(ARTICLE), "cat-file", "-s", f"{COMMIT}:{name}"]))
        if not 0 <= size <= MAX_BYTES:
            raise ValueError("Unexpected Git blob size")
        data = run(["git", "-C", str(ARTICLE), "show", f"{COMMIT}:{name}"])
        if len(data) != size:
            raise ValueError("Git blob size changed")
        return data

    try:
        if BUNDLE.stat().st_size > MAX_BYTES:
            raise ValueError("Archive larger than bounded input ceiling")
        archive = BUNDLE.read_bytes()
        evidence["bundle"] = {"path": str(BUNDLE), "bytes": len(archive), "sha256": sha(archive)}
        stage, canonical = OUT / "bundle", OUT / "canonical"
        stage.mkdir()
        canonical.mkdir()
        members, total = {}, 0
        # Manual regular-file extraction; no links, devices, duplicate or escaping names.
        import io
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tf:
            for member in tf:
                path = PurePosixPath(member.name)
                if (len(members) >= 128 or not member.isfile() or path.is_absolute()
                        or ".." in path.parts or str(path) != member.name
                        or member.name in members or not 0 <= member.size <= MAX_BYTES):
                    raise ValueError(f"Unsafe/unexpected archive member: {member.name}")
                total += member.size
                if total > MAX_BYTES:
                    raise ValueError("Archive expanded-byte ceiling exceeded")
                payload = tf.extractfile(member).read(MAX_BYTES + 1)
                if len(payload) != member.size:
                    raise ValueError("Archive member size mismatch")
                members[member.name] = payload
                dest = stage / member.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(payload)
        if not {"main.tex", "main.bbl", "00README.json"} <= members.keys():
            raise ValueError("Required source-bundle files absent")
        readme = json.loads(members["00README.json"])
        if readme != {"process": {"compiler": "pdflatex"},
                      "sources": [{"filename": "main.tex", "usage": "toplevel"}]}:
            raise ValueError("Unexpected arXiv compiler/top-level declaration")
        source_texts = {}
        for name, payload in sorted(members.items()):
            row = {"name": name, "bundle_bytes": len(payload), "bundle_sha256": sha(payload)}
            if name not in ("00README.json", "main.bbl"):
                original = blob(name)
                dest = canonical / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(original)
                row["article_sha256"] = sha(original)
                row["comparison"] = "comment-stripped TeX" if name.endswith(".tex") else "exact bytes"
                expected = comment_free(original.decode()).encode() if name.endswith(".tex") else original
                row["matches"] = payload == expected
                if not row["matches"]:
                    raise ValueError(f"Bundle differs from article: {name}")
                if name.endswith(".tex"):
                    source_texts[name] = original.decode()
            evidence["files"].append(row)
        if sha((canonical / "main.tex").read_bytes()) != MAIN_SHA:
            raise ValueError("Frozen main.tex SHA differs")
        bibliography = blob("references.bib")
        if sha(bibliography) != BIB_SHA:
            raise ValueError("Frozen bibliography SHA differs")
        (canonical / "references.bib").write_bytes(bibliography)
        frozen_pdf = OUT / "frozen-main.pdf"
        frozen_pdf.write_bytes(blob("main.pdf"))
        body = "\n".join(comment_free(text) for text in source_texts.values())
        labels = re.findall(r"\\label\{([^}]+)\}", body)
        refs = re.findall(r"\\(?:ref|eqref|pageref|autoref)\{([^}]+)\}", body)
        evidence["source_xrefs"] = {
            "duplicate_labels": sorted({key for key in labels if labels.count(key) > 1}),
            "missing_labels": sorted(set(refs) - set(labels)),
            "figure_environments": len(re.findall(r"\\begin\{figure\*?\}", body)),
            "table_environments": len(re.findall(r"\\begin\{table\*?\}", body)),
            "captionof_figures": len(re.findall(r"\\captionof\{figure\}", body)),
        }
        # Canonical build separately regenerates bibliography from exact Git bytes.
        for label, cwd, sequence in (
            ("canonical", canonical, ["tex", "bib", "tex", "tex"]),
            ("bundle", stage, ["tex", "tex", "tex"]),
        ):
            for step in sequence:
                if step == "bib":
                    run(["/usr/bin/bibtex", "main"], cwd)
                else:
                    run(["/usr/bin/pdflatex", "-no-shell-escape", "-interaction=nonstopmode",
                         "-halt-on-error", "-recorder", "main.tex"], cwd)
            log = (cwd / "main.log").read_text(errors="replace")
            evidence[label + "_log"] = {
                "problems": [line for line in log.splitlines() if re.search(
                    r"undefined|not found|Rerun to get|Missing|No file main\.bbl|Emergency stop", line)],
                "overfull": [line for line in log.splitlines() if line.startswith("Overfull")],
            }
        evidence["bbl_exact_equal"] = (canonical / "main.bbl").read_bytes() == members["main.bbl"]
        pdfs = {"frozen": frozen_pdf, "canonical": canonical / "main.pdf", "bundle": stage / "main.pdf"}
        texts, pixels = {}, {}
        for label, pdf in pdfs.items():
            metadata = run(["/usr/bin/pdfinfo", str(pdf)]).decode(errors="replace")
            page_count = int(re.search(r"^Pages:\s+(\d+)", metadata, re.M).group(1))
            if page_count > 40:
                raise ValueError("Unexpected PDF page count above render ceiling")
            text = run(["/usr/bin/pdftotext", "-layout", str(pdf), "-"]).decode()
            (OUT / (label + "-layout.txt")).write_text(text)
            texts[label] = text
            render_dir = OUT / (label + "-pages")
            render_dir.mkdir()
            run(["/usr/bin/pdftoppm", "-r", "90", "-png", str(pdf), str(render_dir / "page")])
            rendered = sorted(render_dir.glob("page-*.png"))
            if len(rendered) != page_count:
                raise ValueError("Incomplete rendered page inventory")
            pixels[label] = [sha(p.read_bytes()) for p in rendered]
            evidence[label + "_pdf"] = {
                "path": str(pdf), "bytes": pdf.stat().st_size, "sha256": sha(pdf.read_bytes()),
                "pages": page_count, "metadata": metadata, "page_png_sha256": pixels[label],
                "layout_sha256": sha(text.encode()),
            }
        evidence["comparisons"] = {}
        for other in ("frozen", "canonical"):
            evidence["comparisons"][other + "_versus_bundle"] = {
                "pdf_bytes_equal": pdfs[other].read_bytes() == pdfs["bundle"].read_bytes(),
                "layout_text_exact_equal": texts[other] == texts["bundle"],
                "normalized_text_equal": " ".join(texts[other].split()) == " ".join(texts["bundle"].split()),
                "rendered_pages_equal": pixels[other] == pixels["bundle"],
                "different_rendered_page_numbers": [i + 1 for i, (a, b) in
                                                   enumerate(zip(pixels[other], pixels["bundle"])) if a != b],
            }
            if texts[other] != texts["bundle"]:
                (OUT / (other + "-versus-bundle.diff")).write_text("".join(difflib.unified_diff(
                    texts[other].splitlines(True), texts["bundle"].splitlines(True))))
        aux = (stage / "main.aux").read_text()
        evidence["numbered_labels"] = [
            {"label": m[0], "number": m[1], "page": int(m[2])}
            for m in re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]+)\}\{(\d+)\}", aux)
            if m[0].startswith(("fig:", "tab:"))
        ]
        pages = texts["bundle"].split("\f")
        evidence["figure_mention_candidates"] = {
            str(i): [{"page": p + 1, "line": line.strip()}
                     for p, text in enumerate(pages) for line in text.splitlines()
                     if re.search(rf"\b(?:Fig(?:ure)?s?\.?)\s+{i}(?!\d)", line)]
            for i in range(1, 28)
        }
        evidence["manual_review_required"] = [
            "Inspect all34 rendered bundle pages, figure/caption alignment and table legibility.",
            "Resolve first-mention candidates, including plural references and line-wrap cases, against source.",
            "PDF byte inequality alone is not content inequality; inspect metadata/text/pixels separately.",
        ]
        evidence["completed"] = True
    except Exception as exc:
        evidence["completed"] = False
        evidence["failures"].append({"type": type(exc).__name__, "message": str(exc),
                                     "traceback": traceback.format_exc()})
    finally:
        evidence["elapsed_seconds"] = time.monotonic() - started
        with REPORT.open("x") as handle:
            json.dump(evidence, handle, indent=2)
            handle.write("\n")
        print(json.dumps({"completed": evidence.get("completed"), "report": str(REPORT),
                          "elapsed_seconds": evidence["elapsed_seconds"],
                          "failures": evidence["failures"]}))
    return 0 if evidence.get("completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
