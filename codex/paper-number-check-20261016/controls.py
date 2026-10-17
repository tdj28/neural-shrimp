#!/usr/bin/env python3
"""Manufactured parser, matching, provenance, and lossless-candidate checks."""
import gzip, hashlib, importlib.util, json, resource, time, traceback
from decimal import Decimal
from pathlib import Path
_spec=importlib.util.spec_from_file_location('number_candidate_own_check',Path(__file__).with_name('check.py'))
check=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)

def run(out):
    groups=[]
    def require(name,ok):
        if not ok: raise AssertionError(name)
        groups.append(name)
    s=r'Values $-3.2\times10^{-4}$, $10^{-10}$, 11\,040, 2.5e-3, and −.25.'
    vals=[d for a,b,d,q,r in check.numeric_tokens(s)]
    require('scientific powers thin thousands unicode minus', vals==list(map(Decimal,['-.00032','1e-10','11040','.0025','-.25'])))
    require('scientific written precision quantum',[q for a,b,d,q,r in check.numeric_tokens('2.50e-3')]==[Decimal('.00001')])
    require('range dash not unary minus',[d for a,b,d,q,r in check.numeric_tokens('3-5; -5')]==list(map(Decimal,['3','5','-5'])))
    s=r'''\documentclass[11pt]{article}
\begin{document}
\section{Results}
In 2026, 2000 bootstrap draws gave 14.5 and
19.5\,\% with $[-0.17,0.44]$ and $2\times10^{-4}$.
Figure 12 and \cite{smith2026x1} and \ref{fig:32} are references.
\includegraphics[width=0.5\linewidth]{x32.png}
% 999 must not appear
'''
    rows,skip=check.extract(s,'paper/llm/main.tex')
    values=[Decimal(x['value']) for x in rows]
    require('refs dates render settings comments excluded', all(x not in values for x in map(Decimal,['11','2026','12','32','.5','999'])))
    require('2000 draws retained',Decimal(2000) in values)
    require('shared percentage suffix across line',all(x['percentage'] for x in rows if x['value'] in ('14.5','19.5')))
    require('outward interval directions',[x['bound_direction'] for x in rows if x['value'] in ('-0.17','0.44')]==['lower','upper'])
    require('line section preserved',all(x['line']==4 and x['section']=='Results' for x in rows if x['value']=='2000'))
    cap=r'\begin{tikzpicture}\draw (1,2)--(3,4);\end{tikzpicture}\caption{N=7; \cite{x2048}.}'
    cr,_=check.extract(cap,'fig.tex','Figures',True)
    require('caption excludes drawing geometry',[x['value'] for x in cr]==['7'])
    td=Path(out)/'functional'; td.mkdir(parents=False,exist_ok=False)
    if True:
        root=Path(td)/'tree'; out=Path(td)/'out'
        (root/'paper/llm/figures').mkdir(parents=True); (root/'docs').mkdir()
        (root/'paper/llm/main.tex').write_text(r'''\begin{document}
\section{Test}
The accuracy is 0.44; rate 50\,\%. It is at most 0.44. Unique 9.875; unknown 555.2.
We used 2000 draws. \input{figures/cap}
''')
        (root/'paper/llm/figures/cap.tex').write_text(r'\caption{The value is 7.125.}')
        (root/'docs/one.json').write_text('{"accuracy":0.435,"exclusive":0.445,"upper":0.431,"rate":0.5,"args":{"draws":2000},"truth":true,"nonfinite":NaN,"numeric_string":"7.125","quoted":"text 555.2","api_key":123456,"cost_usd":0.44}')
        (root/'docs/two.json').write_text('{"accuracy":0.44,"count":50}')
        (root/'docs/bad.json').write_text('{broken')
        (root/'docs/research-log.md').write_text('The unique diagnostic is 9.875.\n')
        files=[]
        for p in sorted(root.rglob('*')):
            if p.is_file():
                b=p.read_bytes(); files.append({'repo_path':str(p.relative_to(root)),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)})
        inv=Path(td)/'inventory.json'; inv.write_text(json.dumps({'commit':'manufactured','files':files}))
        summary=check.check(root,out,inv,True)
        rows=json.loads((out/'numbers.json').read_text())['numbers']
        groups_obj=json.loads(gzip.decompress((out/'candidate-groups.json.gz').read_bytes()))
        idx=json.loads(gzip.decompress((out/'receipt-index.json.gz').read_bytes()))
        def refs(row):
            vis={v for g in row['candidate_groups'] for v in groups_obj[g]['value_ids']}
            return {tuple(idx['refs'][r]) for v in vis for r in idx['values'][v]['refs']}
        x=next(r for r in rows if r['value']=='0.44')
        require('half-open precision interval',('docs/one.json','/accuracy','data') in refs(x) and ('docs/one.json','/exclusive','data') not in refs(x))
        x=next(r for r in rows if r['value']=='50')
        require('percent fraction and native candidates',('docs/one.json','/rate','data') in refs(x) and ('docs/two.json','/count','data') in refs(x))
        x=next(r for r in rows if r['value']=='0.44' and r['bound_direction']=='upper')
        require('outward upper cell distinct from nearest',x['status']=='bound' and ('docs/one.json','/upper','data') in refs(x))
        require('log-only and unmatched distinction',next(r for r in rows if r['value']=='9.875')['status']=='log_only' and next(r for r in rows if r['value']=='555.2')['status']=='unmatched')
        require('args design tag', 'design_parameter_candidate' in next(r for r in rows if r['value']=='2000')['tags'])
        require('include caption resolved',summary['input_files'][0]['status']=='caption_arguments_only' and any(r['value']=='7.125' for r in rows))
        require('invalid JSON and nonfinite explicit',summary['invalid_json_files']==1 and sum(x.get('nonfinite_leaves',0) for x in json.loads((out/'coverage.json').read_text())['json_files'])==1)
        require('booleans and named protected fields excluded',not any(x[1] in ('/truth','/api_key','/cost_usd') for x in idx['refs']) and summary['named_secret_or_monetary_exclusions']==2)
        require('all candidates retained without preview truncation',all(r['candidate_count']==len(refs(r)) for r in rows))
        (root/'docs/two.json').write_text('{"changed":1}')
        refused=False
        try: check.check(root,Path(td)/'changed',inv,True)
        except ValueError as e: refused='Changed input' in str(e)
        require('changed input refusal',refused)
    return {'passed':len(groups),'groups':groups}

def profile(out,files=1361):
    """Representative 29MB manufactured receipts; no target source/data."""
    td=Path(out)/'profile'; td.mkdir(parents=False,exist_ok=False)
    if True:
        root=Path(td)/'tree'; (root/'docs').mkdir(parents=True); (root/'paper/llm').mkdir(parents=True)
        text='\\begin{document}\n\\section{Manufactured}\n'+''.join(f'Cell {i%100} has accuracy {i/137:.4f} and at most {i/123:.3f}.\n' for i in range(2200))
        (root/'paper/llm/main.tex').write_text(text)
        entries=[]
        for i in range(files):
            # Unique keys and values exercise index size; overlapping IDs exercise ambiguity.
            obj={'args':{'steps':2000},'cells':[{'accuracy':(i*181+j)/19997,'id':j%64,'residual':(j%17)*1e-5} for j in range(240)]}
            (root/f'docs/{i:04}.json').write_text(json.dumps(obj,separators=(',',':')))
        for p in root.rglob('*'):
            if p.is_file():
                b=p.read_bytes(); entries.append({'repo_path':str(p.relative_to(root)),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)})
        inv=Path(td)/'inventory.json'; inv.write_text(json.dumps({'commit':'manufactured-profile','files':entries}))
        t=time.perf_counter(); summary=check.check(root,Path(td)/'out',inv,False)
        return {'elapsed_seconds':time.perf_counter()-t,'input_bytes':sum(x['bytes'] for x in entries),'output_bytes':sum(p.stat().st_size for p in (Path(td)/'out').iterdir()),'summary':summary}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--profile',action='store_true'); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    a.out.mkdir(parents=False,exist_ok=False)
    t=time.perf_counter()
    try:
        result={'controls':run(a.out)}
        if a.profile: result['profile']=profile(a.out)
        result['elapsed_seconds']=time.perf_counter()-t
        result['rss_KiB_before_final_report']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['source_sha256']={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ('check.py','controls.py')}
        (a.out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    except BaseException as e:
        (a.out/'failure.json').write_text(json.dumps({'error':repr(e),'traceback':traceback.format_exc(),'elapsed_seconds':time.perf_counter()-t},indent=2)+'\n')
        raise
