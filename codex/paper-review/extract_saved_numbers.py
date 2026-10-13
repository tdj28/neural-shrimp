"""Task13 read-only saved-JSON transcription; no producer imports, fitting, or model calls."""
import hashlib, json, pathlib
ROOT=pathlib.Path(__file__).resolve().parents[2]
DATA=pathlib.Path('/data2/neural-shrimp-data')
FILES=[
 'explore/llm-soft-loop/pod64/map-stripes-256/scan.json',
 'explore/llm-soft-loop/pod64/cascade/crescent-t-0.14798/cascade.json',
 'explore/llm-soft-loop/pod64/cascade/crescent-t-0.14798-v2/cascade.json',
 'explore/llm-soft-loop/feigenbaum-crescent-t-0.14798/cascade.json',
 'explore/llm-soft-loop/pod64/superstable/crescent-t-0.14798/superstable.json',
 'explore/llm-soft-loop/pod64/superstable/crescent-deep2/superstable.json',
 'explore/llm-soft-loop/pod64/edges/stripes-t-0.115/edges.json',
 'explore/llm-soft-loop/passage-p2-t-0.115/passage.json',
 'explore/llm-msm/p1-committor-analysis/committor.json',
 'explore/llm-msm/p1-controller-l10/controller.json']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def summary(p,j):
 if p.name=='cascade.json':
  rows=[]
  for r in j['flips']:
   f=r['flip']; rows.append(dict(period=r['period'],parameter=f['lambda'],critical_multiplier=f['critical_multiplier'],critical_distance_minus_one=abs(f['critical_multiplier'][0]+1),newton_residual=f['newton_residuals'][-1],ritz_residual=f['ritz_residual'],next_modulus=f['next_modulus'],computed_ritz_count=len(f['multipliers'])))
  return dict(rows=rows,ratios=j['ratios'])
 if p.name=='superstable.json':
  rows=[]
  for r in j['bands']:
   if 'error' in r: rows.append(r);continue
   f=r['record']; vals=f['multipliers']; mod=sorted([(a*a+b*b)**.5 for a,b in vals],reverse=True)
   rows.append({k:r[k] for k in ['period','detected_period','superstable','how','clean_bracket','mu']}|dict(record_parameter=f['lambda'],record_newton_residual=f['newton_residuals'][-1],record_computed_moduli=mod,computed_ritz_count=len(vals)))
  return dict(rows=rows,ratios=j['ratios'])
 if p.name=='edges.json':
  return {'rows':[dict(period=r['period'],detected_period=r.get('detected_period'),edges=[{k:v for k,v in e.items() if k!='record'}|({'record_gate':{k:e['record'][k] for k in ['critical_multiplier','ritz_residual','next_modulus','newton_residuals']}} if 'record'in e else {}) for e in r['edges']]) for r in j['windows']]}
 if p.name=='committor.json':
  return {k:j[k] for k in ['states_with_rollouts','k','h','coverage','calibration','calibration_gate','primary_admissible','rescore','committor_auc','committor_probe']}
 return j
out={'schema':'task13-saved-number-transcription-v1','date_utc':'2026-10-13','source_commit':'1491f37972e706426d3343ce5b69fa91ab89da01','scope':'Saved JSON values and deterministic arithmetic only; producer status fields are reported, not independently rerun. No Huginn held-out outputs read. No complete raw archive authentication claimed.', 'manuscript_sha256':sha(ROOT/'paper/llm/main.tex'),'bib_sha256':sha(ROOT/'paper/llm/references.bib'),'files':[]}
for f in FILES:
 p=DATA/f; raw=p.read_bytes();j=json.loads(raw)
 out['files'].append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),summary=summary(p,j)))
values=[.06597169426728654,.06608014054384855,.06611413278446639,.06612258989268897,.06612447851007745,.06612488685381053]
out['recomputed_superstable_ratios']=[(values[i]-values[i-1])/(values[i+1]-values[i]) for i in range(1,len(values)-1)]
path=ROOT/'codex/paper-review/saved-number-transcription.json'
path.write_text(json.dumps(out,indent=2)+'\n')
print(path,sha(path))
