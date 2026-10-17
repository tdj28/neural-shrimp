"""Focused manufactured-only checks for the disclosed sign-token correction."""
import argparse
import hashlib
import importlib.util
import json
from decimal import Decimal
from pathlib import Path

P=Path(__file__).resolve().parent/'check.py'
spec=importlib.util.spec_from_file_location('number_sign_recovery',P)
check=importlib.util.module_from_spec(spec);spec.loader.exec_module(check)

CASES=[
 ('prose_identifiers',r'period-8 OLMo-2 Llama-3 step_4-5', ['8','2','3','4','5']),
 ('tex_and_plain_ranges',r'7--8B, 3-5, 11---18; -5', ['7','8','3','5','11','18','-5']),
 ('inline_math_negatives',r'$-8$ and $x-2$ and $5--2$ and $\alpha-0.02$', ['-8','-2','5','-2','-0.02']),
 ('display_delimiters',r'\(-8\) and \[-3.2\] and $$-4$$; period-7', ['-8','-3.2','-4','7']),
 ('math_environment',r'\begin{align*}x-2\end{align*} period-3', ['-2','3']),
 ('escaped_dollar',r'price\$ text period-8, followed by $-8$', ['8','-8']),
 ('scientific_regression',r'$-3.2\times10^{-4}$, $10^{-10}$, 11\,040, 2.5e-3, −.25', ['-.00032','1e-10','11040','.0025','-.25']),
 ('line_column_and_raw', 'Heading\nperiod-8 $-8$', ['8','-8'])
]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
 out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
 rows=[]
 for name,text,expected in CASES:
  tokens=list(check.numeric_tokens(text));got=[v for a,b,v,q,raw in tokens]
  row={'name':name,'source':text,'expected':expected,'actual':[str(v) for v in got],
       'tokens':[{'a':a,'b':b,'raw':raw,'literal':text[a:b],'quantum':str(q)} for a,b,v,q,raw in tokens]}
  row['passed']=got==list(map(Decimal,expected)) and all(raw==text[a:b] for a,b,v,q,raw in tokens)
  if name=='line_column_and_raw':
   extracted,_=check.extract(text,'manufactured.tex')
   row['locations']=[(r['line'],r['column'],r['raw']) for r in extracted]
   row['passed']=row['passed'] and row['locations']==[(2,8,'8'),(2,11,'-8')]
  rows.append(row)
 report={'scope':'Focused manufactured sign parser controls only. No target reads or full profile repetition.',
         'source_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'groups':rows,
         'passed':sum(r['passed'] for r in rows),'failed':sum(not r['passed'] for r in rows)}
 (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
 if report['failed']:raise SystemExit(1)
if __name__=='__main__':main()
