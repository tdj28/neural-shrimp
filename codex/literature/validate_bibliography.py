"""Validate Task 3 bibliography syntax/citation coverage; not factual validation."""
from pathlib import Path
import datetime
import hashlib
import json
import re
import shutil
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2]
bib = root / 'paper/llm-references.bib'
draft = root / 'docs/related-work-draft.md'
text = bib.read_text()
keys = re.findall(r'@\w+\s*\{\s*([^,]+),', text)
assert len(keys) == len(set(keys)), 'Duplicate BibTeX keys'
references = set(re.findall(r'`([a-z]+\d{4}[a-z]+)`', draft.read_text()))
assert not (references - set(keys)), sorted(references - set(keys))
parts = [root / f'codex/literature/{name}.bib' for name in ('inference', 'cascades', 'behavior')]
for part in parts:
    assert part.read_text() in text, f'Supplement stale relative to {part.name}'
with tempfile.TemporaryDirectory(prefix='neural-shrimp-literature-') as directory:
    temp = Path(directory)
    shutil.copyfile(bib, temp / 'references.bib')
    (temp / 'check.aux').write_text('\\relax\n\\citation{*}\n\\bibstyle{plainnat}\n\\bibdata{references}\n')
    result = subprocess.run(['bibtex', 'check'], cwd=temp, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Warning--' not in result.stdout, result.stdout
    bbl = (temp / 'check.bbl').read_text()
    assert len(re.findall(r'\\bibitem', bbl)) == len(keys)
    # Compile the generated bibliography in a disposable harness. No manuscript edits.
    (temp / 'check.tex').write_text(
        '\\documentclass{article}\n\\usepackage[T1]{fontenc}\n\\usepackage[utf8]{inputenc}\n'
        '\\usepackage{natbib}\n\\usepackage{url}\n\\begin{document}\n'
        '\\input{check.bbl}\n\\end{document}\n')
    latex = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'check.tex'],
                           cwd=temp, text=True, capture_output=True, timeout=60)
    assert latex.returncode == 0, latex.stdout[-6000:] + latex.stderr
    assert (temp / 'check.pdf').is_file()

old = (root / 'paper/llm/references.bib').read_text()
existing = set(re.findall(r'@\w+\s*\{\s*([^,]+),', old))
files = [draft, bib, *parts, *(root / f'codex/literature/{name}.md' for name in ('inference', 'cascades', 'behavior'))]
receipt = {
    'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'source_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
    'checks': {'unique_keys': True, 'draft_key_coverage': True, 'source_assembly_current': True,
               'bibtex_exit_status': result.returncode, 'bibtex_warnings': False,
               'bibliography_tex_exit_status': latex.returncode},
    'entry_count': len(keys),
    'keys_already_in_manuscript_bibliography': sorted(set(keys) & existing),
    'scope': 'Syntax, rendering compilation and key coverage only; factual/source review is recorded in the audit notes.',
    'sha256': {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
}
(root / 'codex/literature/validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({k: v for k, v in receipt.items() if k != 'sha256'}, indent=2))
