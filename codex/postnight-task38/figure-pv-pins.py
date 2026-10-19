#!/usr/bin/env python3
"""Task38 figure/PV addendum: hashes and JSON excerpts only; no arrays or target code."""
import hashlib,json,os,resource,signal,subprocess
from pathlib import Path
os.nice(10); resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3)); resource.setrlimit(resource.RLIMIT_CPU,(30,30)); signal.alarm(45)
O=Path('/tmp/task38-geometry'); R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp'); A=Path('/data2/chaos-journal-neural-shrimp'); T=Path('/tmp/task38-root')
pins=[]
def pin(p):
 b=Path(p).read_bytes(); h=hashlib.sha256(b).hexdigest(); pins.append({'path':str(p),'bytes':len(b),'sha256':h}); return h
names=['fig-repro','fig-fold','fig-real-shrimp','fig-p6-shrimp','fig-p6-kite']
cmp=[]
for name in names:
 for repo,rel,rev in [(A,'figures/'+name+'.png','b8413881e7ce97a9e58adcf1bf7d78a8cebb0514'),(R,'paper/llm/figures/'+name+'.png','651a50059a145e2bb0526f9675bcc41e9cb895a8')]:
  h=pin(repo/rel); b=subprocess.check_output(['git','-C',str(repo),'show',rev+':'+rel],stderr=subprocess.PIPE)
  cmp.append({'repository':str(repo),'commit':rev,'path':rel,'git_blob_sha256':hashlib.sha256(b).hexdigest(),'current_matches_commit':h==hashlib.sha256(b).hexdigest()})
for p in [T/'article-assigned-main.tex',T/'article-p6-only-main.tex',T/'report-assigned-main.tex',T/'blog-index.md',T/'snapshots.json',T/'pv_steering_audit.py',T/'pv-steering-result-001.json',R/'tools/plot_repro.py',R/'tools/plot_p6_kite.py',R/'tools/plot_real_shrimp.py',R/'paper/llm/figures/fig-fold.json',R/'paper/llm/figures/fig-p6-shrimp.json',R/'paper/llm/figures/fig-p6-kite.json',R/'docs/fa/fa-100-summary-closeup.json',O/'geometry-review.md',O/'geometry-manifest.json']:
 pin(p)
q=json.loads((T/'pv-steering-result-001.json').read_text())
starts=[{k:v for k,v in r.items() if k in ['condition','chaotic_count','chaos_without_detected_period_count','periodic_and_positive_exponent_count','chaos_binary_agreement','period_label_agreement','chaos_jaccard']} for r in q['rows'] if r['condition'].startswith('s')]
for r in starts:
 r['threshold_mask_agreement_count_from_saved_fraction']=round(16384*r['chaos_binary_agreement'])
 r['period_label_agreement_count_from_saved_fraction']=round(16384*r['period_label_agreement'])
out={'scope':'Only byte hashing, static source/JSON review, and arithmetic on previously saved summary fractions. No new numerical audit or model/array execution.', 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'pins':pins,'figure_snapshot_comparisons':cmp,'pv_saved_starts':starts,'pv_positive_with_detected_period':{r['condition']:r['periodic_and_positive_exponent_count'] for r in q['rows']},'absent_adjacent_records':[str(p) for p in [R/'paper/llm/figures/fig-repro.json',R/'paper/llm/figures/fig-real-shrimp.json'] if not p.exists()]}
dest=O/'figure-pv-manifest-001.json'; assert not dest.exists(); dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'path':str(dest),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'all_images_match_assigned_commits':all(c['current_matches_commit'] for c in cmp),'starts':starts}))
