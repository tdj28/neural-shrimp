"""Small read-only Task37 source, packaging and scalar anatomy checks; no model."""
import collections
import decimal
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile

ARTICLE = '/data2/chaos-journal-neural-shrimp'
RESEARCH = '/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp'
OUT = Path('/tmp/task37-root')
def show(repo, ref, path):
    return subprocess.check_output(['git', 'show', f'{ref}:{path}'], cwd=repo)
def sha(b): return hashlib.sha256(b).hexdigest()
src = show(ARTICLE, '18490e2', 'main.tex').decode()
bib = show(ARTICLE, '18490e2', 'references.bib').decode()
pdf = show(ARTICLE, '18490e2', 'main.pdf')
abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', src, re.S)[1].strip()
proposal = Path('/data2/neural-shrimp-data/arxiv/chaos-v1/arxiv-abstract-proposed.txt').read_text()
keys = re.findall(r'@\w+\s*\{\s*([^,]+),', bib)
cites = set(k.strip() for group in re.findall(r'\\cite\w*\s*(?:\[[^\]]*\]\s*)*\{([^}]+)\}',src) for k in group.split(','))
labels = re.findall(r'\\label\{([^}]+)\}',src)
refs = re.findall(r'\\(?:ref|eqref|pageref|autoref)\{([^}]+)\}',src)
info = subprocess.check_output(['pdfinfo', str(OUT/'article-18490e2.pdf')]).decode()
text = (OUT/'article-18490e2.txt').read_text()
manifest_path = Path('/data2/neural-shrimp-data/arxiv/chaos-v1/manifest.json')
manifest = json.loads(manifest_path.read_text())
archive = Path(manifest['archive'])
with tarfile.open(archive) as tar:
    member = next(m for m in tar.getmembers() if m.name in ('main.tex','./main.tex'))
    archive_tex = tar.extractfile(member).read()
report = {
 'schema':'task37-submission-readonly-checks-v1',
 'article_commit':subprocess.check_output(['git','rev-parse','18490e2'],cwd=ARTICLE,text=True).strip(),
 'research_commit':subprocess.check_output(['git','rev-parse','35eabc6c'],cwd=RESEARCH,text=True).strip(),
 'source_sha256':sha(src.encode()), 'bibliography_sha256':sha(bib.encode()), 'pdf_sha256':sha(pdf),
 'undefined_citations':sorted(cites-set(keys)), 'uncited_bib_entries':sorted(set(keys)-cites),
 'duplicate_bib_keys':[k for k,v in collections.Counter(keys).items() if v>1],
 'undefined_labels':sorted(set(refs)-set(labels)),
 'duplicate_labels':[k for k,v in collections.Counter(labels).items() if v>1],
 'figure_environments':len(re.findall(r'\\begin\{figure\*?\}',src)),
 'table_environments':len(re.findall(r'\\begin\{table\*?\}',src)),
 'pdf_pages':int(re.search(r'^Pages:\s+(\d+)',info,re.M)[1]),
 'pdf_numbered_figures':re.findall(r'FIG\.\s+(\d+)\.',text),
 'pdf_numbered_tables':re.findall(r'TABLE\s+([IVX]+)\.',text),
 'abstract_characters_stripped':len(proposal.strip()),
 'abstract_whitespace_normalized_equal':' '.join(abstract.split())==' '.join(proposal.split()),
 'archive': {'sha256':sha(archive.read_bytes()), 'manifest_sha256':sha(manifest_path.read_bytes()),
  'source_commit':manifest['source_commit'],'test_pages':manifest['test_build']['pages'],
  'main_tex_sha256':sha(archive_tex),
  'manifest_main_tex_match':sha(archive_tex)==manifest['files']['main.tex']['upload_sha256'],
  'manifest_archive_match':sha(archive.read_bytes())==manifest['archive_sha256']},
 'abstract_body': {'passages':[50,135,749], 'changed_text':[51,52,570,571,573,574],
  'model_range':[44,119,121,495,498,499], 'readout_scope':[47,131,421,424]},
}
decimal.getcontext().prec=60
D=decimal.Decimal
u,v=D('-.26'),D('-1.1')
def a(x,t): return x*x-t/(4*x)
def b(x): return 1/(16*x*x)-x
def da(x,t): return 2*x+t/(4*x*x)
def db(x): return -1/(8*x*x*x)-1
for it in range(30):
    f,g=a(u,D(1))-a(v,D(-1)),b(u)-b(v)
    if max(abs(f),abs(g))<D('1e-45'): break
    aa,bb,cc,dd=da(u,D(1)),-da(v,D(-1)),db(u),-db(v)
    det=aa*dd-bb*cc
    u,v=u-(dd*f-bb*g)/det,v-(-cc*f+aa*g)/det
else: raise RuntimeError('tiny scalar Newton did not converge')
exact_a=[a(u,D(1)),b(u)]
stored=json.loads(show(RESEARCH,'35eabc6c','docs/oc/shrimp-anatomy-models.json'))['gallas']
gallas=[D('1.01905888'),D('1.16529086')]
report['anatomy']={'scalar_iterations':it, 'residual':str(max(abs(f),abs(g))),
 'A_scalar_solution':[str(x) for x in exact_a], 'B_scalar_solution':[str(x) for x in exact_a[::-1]],
 'fold_state':str(u),'flip_state':str(v),'stored':stored,
 'stored_A_matches_Gallas_at_six_significant_digits':all(format(float(x),'.6g')==format(float(y),'.6g') for x,y in zip(stored['A'],gallas)),
 'stored_A_does_not_match_at_six_decimal_places':any(format(float(x),'.6f')!=format(float(y),'.6f') for x,y in zip(stored['A'],gallas)),
 'R_exact':['5/4','5/4'], 'R_flip_states':['(-1-sqrt(2))/2','(-1+sqrt(2))/2']}
availability={}
for path in ['codex/certify-llm','certificates/llm-p8-certificate-final-001.json','codex/coexistence-task33']:
 availability[path]=subprocess.check_output(['git','ls-tree','35eabc6c','--',path],cwd=RESEARCH,text=True).strip()
report['availability_paths']=availability
report['limits']='Source/metadata and tiny scalar checks only; no full proof re-audit, no transformer/model call. Native proof evidence pinned under external data paths is not thereby included in Git. No public access claim was verified.'
(OUT/'submission-checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'pdf_pages':report['pdf_pages'],'archive_pages':report['archive']['test_pages'], 'anatomy':report['anatomy']},indent=2))
