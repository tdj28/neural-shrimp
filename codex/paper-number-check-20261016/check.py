#!/usr/bin/env python3
"""Lossless numeric candidate finder; a numeric match is NOT claim verification."""
from __future__ import annotations
import argparse, bisect, collections, gzip, hashlib, json, re, subprocess, time
from decimal import Decimal, localcontext
from pathlib import Path

VERSION = 'paper-number-candidates-v1'
SPACE = r'(?:\s|\\[,!;:]|~)*'
MANT = r'(?:\d{1,3}(?:(?:\\[,!;:]|[\u2009\u202f,])\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)'
SIGN = r'[-+\u2212]?'
NUM = re.compile(r'(?<![\d.])'+SIGN+MANT+r'(?:[eE][+-]?\d+|'+SPACE+r'\\(?:times|cdot)'+SPACE+r'10\s*\^\s*\{?\s*[+-]?\d+\s*\}?)?|(?<![\d.])'+SIGN+r'10\s*\^\s*\{?\s*[+-]?\d+\s*\}?')
POWER = re.compile(r'(?<![\d.])'+SIGN+r'10\s*\^\s*\{?\s*[+-]?\d+\s*\}?')
SIMPLE = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z')
STOP = set('the and for that this with from into were was are has have had not its their these those than then all one two three four five can may per our out both only such when where which whose each also latex text frac log times cdot begin end alpha beta sigma theta left right mathrm textbf textit label ref appendix section figure table percent'.split())
DESIGN = {'args','arguments','run_args','config','configuration','settings','parameters','hyperparameters'}
PRIVATE = {'password','secret','api_key','access_token','private_key','cost_usd','price_usd','balance_usd','dollars'}

def sha(b): return hashlib.sha256(b).hexdigest()
def dump(path, obj, compressed=False):
    b=json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(',',':')).encode()
    Path(path).write_bytes(gzip.compress(b, mtime=0) if compressed else b+b'\n')
def words(s):
    s=re.sub(r'([a-z])([A-Z])',r'\1 \2',s)
    return {w.rstrip('s') for w in re.findall(r'[a-zA-Z]{3,}',s.lower()) if w not in STOP}
def blank(s): return ''.join('\n' if c=='\n' else ' ' for c in s)
def brace_end(s,p):
    if p>=len(s) or s[p]!='{': return None
    depth=1; i=p+1
    while i<len(s):
        if s[i]=='\\': i+=2; continue
        if s[i]=='{': depth+=1
        if s[i]=='}':
            depth-=1
            if not depth: return i+1
        i+=1
    return None

def clean_tex(original, captions_only=False):
    """Length-preserving masks keep exact original line/column locations."""
    s=re.sub(r'(?<!\\)%[^\n]*',lambda m:blank(m[0]),original)
    dispositions=[]
    if captions_only:
        keep=[]
        for m in re.finditer(r'\\caption(?:of\{[^}]+\})?\*?(?:\[[^\]]*\])?\s*\{',s):
            a=m.end()-1; b=brace_end(s,a)
            if b: keep.append((a+1,b-1))
        chars=list(blank(s))
        for a,b in keep: chars[a:b]=s[a:b]
        s=''.join(chars)
    else:
        p=s.find(r'\begin{document}')
        if p>=0: s=blank(s[:p])+s[p:]
        for m in list(re.finditer(r'\\begin\{(tikzpicture|axis|pgfpicture)\}.*?\\end\{\1\}',s,re.S)):
            s=s[:m.start()]+blank(m[0])+s[m.end():]
    # Rendering controls and reference IDs must never become scientific numbers.
    pattern=r'\\(?:cite\w*|label|[a-z]*ref|ref|url|path|includegraphics|input|include|bibliography|bibliographystyle|date|vspace|hspace|setlength|addtolength)\*?(?:\[[^\]]*\])*\s*\{'
    spans=[]
    for m in re.finditer(pattern,s):
        b=brace_end(s,m.end()-1)
        if b: spans.append((m.start(),b,'reference_or_render_command'))
    for m in re.finditer(r'\\href\s*\{',s):
        b=brace_end(s,m.end()-1)
        if b: spans.append((m.start(),b,'hyperlink_target'))
    for a,b,reason in spans:
        dispositions.append({'reason':reason,'line':original.count('\n',0,a)+1,'raw':original[a:b]})
        s=s[:a]+blank(s[a:b])+s[b:]
    # Skip explicitly dated years, not the number of 2000 bootstrap draws.
    pats=[(r'\b(?:19|20)\d\d-\d\d-\d\d(?:T[\d:.]+Z)?\b','date'),
          (r'\b(?:\d{1,2}\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:\d{1,2},?\s+)?(?:19|20)\d\d\b','date'),
          (r'\b(?:in|since|year)\s+((?:19|20)\d\d)\b','year'),
          (r'\b(?:Figures?|Tables?|Sections?|Appendices|Appendix)\s*[~ ]\d+(?:\.\d+)*(?:[a-z])?\b','numbered_reference')]
    for pat,reason in pats:
        for m in list(re.finditer(pat,s,re.I)):
            a,b=(m.span(1) if reason=='year' else m.span())
            dispositions.append({'reason':reason,'line':original.count('\n',0,a)+1,'raw':original[a:b]})
            s=s[:a]+blank(s[a:b])+s[b:]
    return s,dispositions

def math_spans(text):
    """Literal TeX math delimiters; macros are deliberately not expanded."""
    spans=[]; opened=None; skip_until=-1
    token=re.compile(r'\$|\\[()\[\]]|\\(?:begin|end)\{(?:equation|align|alignat|gather|multline|math|displaymath)\*?\}')
    closes={r'\(':r'\)',r'\[':r'\]'}
    for m in token.finditer(text):
        a=m.start(); t=m[0]
        if a<skip_until: continue
        # A row break \\[2pt] is not display math. An even run of preceding
        # backslashes leaves the delimiter active; an odd run escapes it.
        k=a-1
        while k>=0 and text[k]=='\\': k-=1
        if (a-k-1)%2: continue
        if t=='$':
            if opened is None:
                end='$$' if text.startswith('$$',a) else '$'
                opened=(a,end); skip_until=a+len(end)
            elif opened[1]=='$':
                # In $a$$b$, close the first inline span, then open the next.
                spans.append((opened[0],a+1)); opened=None
            elif opened[1]=='$$' and text.startswith('$$',a):
                spans.append((opened[0],a+2)); opened=None; skip_until=a+2
        elif opened is not None:
            if t==opened[1]: spans.append((opened[0],m.end())); opened=None
        elif t in closes: opened=(a,closes[t])
        elif t.startswith(r'\begin{'): opened=(a,t.replace(r'\begin{',r'\end{',1))
    if opened is not None: spans.append((opened[0],len(text)))
    return spans

def numeric_tokens(text):
    # Power-of-ten alternatives must be considered before their leading 10.
    pattern=re.compile(POWER.pattern+'|'+NUM.pattern)
    spans=math_spans(text); starts=[a for a,b in spans]
    for m in pattern.finditer(text):
        raw=m[0]; a=m.start(); b=m.end()
        # Prose hyphens (period-8, Llama-3, 7--8B) are separators.
        # Preserve unary negatives and doubled subtraction inside literal math.
        si=bisect.bisect_right(starts,a)-1
        in_math=si>=0 and a<spans[si][1]
        if raw.startswith('-') and a:
            previous=text[a-1]
            prose_separator=not in_math and (previous.isalpha() or previous in '_-')
            numeric_range=previous.isdigit() or previous=='}'
            if prose_separator or numeric_range:
                raw=raw[1:]; a+=1
        t=re.sub(r'\\[,!;:]|[\u2009\u202f,~\s]','',raw).replace('\u2212','-')
        t=re.sub(r'\\(?:times|cdot)10\^\{?([+-]?\d+)\}?',r'e\1',t)
        p=re.fullmatch(r'([+-]?)10\^\{?([+-]?\d+)\}?',t)
        if p: t=p[1]+'1e'+p[2]
        try:
            d=Decimal(t)
            if not d.is_finite(): continue
            quantum=Decimal(1).scaleb(d.as_tuple().exponent)
        except Exception: continue
        yield a,b,d,quantum,raw

def section_events(s):
    events=[(0,'Unsectioned')]
    for m in re.finditer(r'\\(part|chapter|section|subsection|subsubsection)\*?(?:\[[^\]]*\])?\s*\{',s):
        b=brace_end(s,m.end()-1)
        if b: events.append((m.start(),re.sub(r'\s+',' ',s[m.end():b-1])))
    for m in re.finditer(r'\\begin\{abstract\}',s): events.append((m.start(),'Abstract'))
    return sorted(events)
def sentence(s,a,b):
    # Decimal points are not sentence boundaries; retain TeX syntax verbatim.
    left=max(0,a-700); right=min(len(s),b+700)
    breaks=list(re.finditer(r'(?<!\d)[.!?](?=\s)|(?<=\d)[.!?](?=\s+[A-Z\\])|\n\s*\n',s[left:right]))
    lo=left; hi=right
    for m in breaks:
        x,y=left+m.start(),left+m.end()
        if y<=a: lo=y
        elif x>=b: hi=y; break
    return re.sub(r'\s+',' ',s[lo:hi]).strip()
def percent_at(s,a,b):
    tail=s[b:b+100]
    m=re.search(r'\\%|\bper\s*cent\b|\bpercent\b',tail,re.I)
    if not m: return False
    between=tail[:m.start()]
    between=re.sub(r'\\[,!;:]|\b(?:and|to|or|through)\b','',between)
    return not re.search(r'[A-Za-z;!?]|\n\s*\n|(?<!\d)\.(?!\d)',between)
def bound_at(s,a,b):
    before=s[max(0,a-95):a].lower(); after=s[b:b+65].lower()
    if re.search(r'(?:at most|no more than|within|up to|not exceed\w*|never exceed\w*|upper bound|\\leq?|<=)\s*[${~ ]*$',before): return 'upper'
    if re.search(r'(?:at least|no less than|lower bound|\\geq?|>=)\s*[${~ ]*$',before): return 'lower'
    # Numeric interval endpoints are candidates for outward rounding, not proof.
    if re.search(r'\[\s*[$ ]*$',before) and re.match(r'\s*,',after): return 'lower'
    if re.search(r',\s*[$ ]*$',before) and re.match(r'\s*[$ ]*\]',after): return 'upper'
    return None

def extract(s,file,inherited='Unsectioned',captions_only=False):
    clean,skipped=clean_tex(s,captions_only)
    events=section_events(clean); positions=[x for x,_ in events]
    lines=[-1]+[m.start() for m in re.finditer('\n',s)]
    rows=[]
    for a,b,d,q,raw in numeric_tokens(clean):
        sec=events[bisect.bisect_right(positions,a)-1][1]
        if sec=='Unsectioned': sec=inherited
        tags=[]
        if a and (s[a-1].isalpha() or s[a-1]=='_') or b<len(s) and s[b].isalpha(): tags.append('identifier_or_unit_adjacent')
        rows.append({'file':file,'line':bisect.bisect_right(lines,a),'column':a-lines[bisect.bisect_right(lines,a)-1],
                     'offset':a,'raw':s[a:b],'value':str(d),'quantum':str(q),'section':sec,
                     'sentence':sentence(clean,a,b),'percentage':percent_at(clean,a,b),'bound_direction':bound_at(clean,a,b),'tags':tags})
    return rows,skipped

def inventory(root,manifest=None):
    if manifest:
        obj=json.loads(Path(manifest).read_text())
        files={x['repo_path']:x for x in obj['files']}
        return obj['commit'],files
    env={'GIT_OPTIONAL_LOCKS':'0'}
    import os
    env={**os.environ,**env}
    head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],env=env,text=True).strip()
    entries=subprocess.check_output(['git','-C',str(root),'ls-tree','-r','-z','--full-tree',head,'--','docs','paper/llm'],env=env).decode().split('\0')
    tracked={}
    for entry in entries:
        if not entry: continue
        metadata,path=entry.split('\t',1); mode,kind,blob=metadata.split()
        if kind!='blob': continue
        tracked[path]=blob
    selected=[p for p in tracked if p.endswith('.json') or p.endswith('.tex') or p=='docs/research-log.md']
    files={}
    for p in sorted(selected):
        f=root/p
        if f.is_symlink(): raise ValueError('Refuse symlink '+p)
        b=f.read_bytes()
        blob_bytes=('blob '+str(len(b))+'\0').encode()+b
        git_hash=hashlib.sha1(blob_bytes).hexdigest() if len(tracked[p])==40 else hashlib.sha256(blob_bytes).hexdigest()
        if git_hash!=tracked[p]: raise ValueError('Uncommitted input bytes; commit or use an explicitly hash-bound snapshot: '+p)
        files[p]={'repo_path':p,'sha256':sha(b),'bytes':len(b),'git_blob':tracked[p]}
    return head,files

def check(root,out,manifest=None,include_log=False):
    started=time.perf_counter(); root=Path(root).resolve(); out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    commit,files=inventory(root,manifest); used={}
    def read(p):
        if p not in files: raise ValueError('Input not in tracked/frozen inventory: '+p)
        f=(root/p)
        if f.is_symlink() or not f.resolve().is_relative_to(root): raise ValueError('Escaping input '+p)
        b=f.read_bytes(); rec=files[p]
        if sha(b)!=rec['sha256'] or len(b)!=rec['bytes']: raise ValueError('Changed input '+p)
        used[p]={'sha256':sha(b),'bytes':len(b)}
        return b.decode('utf-8')
    main='paper/llm/main.tex'; text=read(main)
    numbers,skipped=extract(text,main)
    main_events=section_events(clean_tex(text)[0]); mp=[x for x,_ in main_events]
    input_rows=[]; seen={main}
    def inputs(p,s,inherited):
        clean,_=clean_tex(s)
        # clean_tex masks input calls, so scan comments-stripped source instead.
        stripped=re.sub(r'(?<!\\)%[^\n]*',lambda m:blank(m[0]),s)
        for m in re.finditer(r'\\(?:input|include)\s*\{([^}]+)\}',stripped):
            rel=m[1]; options=[str(Path(p).parent/rel),str(Path(main).parent/rel)]
            options=[x if x.endswith('.tex') else x+'.tex' for x in options]
            child=next((x for x in options if x in files),None)
            if not child: input_rows.append({'parent':p,'literal':rel,'status':'unresolved'}); continue
            if child in seen: input_rows.append({'parent':p,'resolved':child,'status':'already_indexed'}); continue
            seen.add(child); ct=read(child)
            sec=main_events[bisect.bisect_right(mp,m.start())-1][1] if p==main else inherited
            rows,skip=extract(ct,child,sec,True); numbers.extend(rows)
            skipped.extend(dict(x,file=child) for x in skip)
            input_rows.append({'parent':p,'resolved':child,'status':'caption_arguments_only','numbers':len(rows)})
            inputs(child,ct,sec)
    inputs(main,text,'Unsectioned')
    values=[]; vmap={}; refs=[]; ref_v=[]; ref_terms=[]; term_cache={}; file_dispositions=[]; policy_exclusions=[]
    def add(d,p,pointer,kind):
        if not d.is_finite(): return False
        if d not in vmap: vmap[d]=len(values); values.append({'value':str(d),'refs':[]})
        vi=vmap[d]; ri=len(refs); refs.append([p,pointer,kind]); ref_v.append(vi); values[vi]['refs'].append(ri)
        key=re.sub(r'\d+','',p+' '+pointer)
        if key not in term_cache: term_cache[key]=words(key)
        ref_terms.append(term_cache[key])
        return True
    def leaves(obj,path=(),design=False):
        if isinstance(obj,dict):
            for k,v in obj.items():
                if k.lower() in PRIVATE:
                    policy_exclusions.append({'file':p,'pointer':'/'+'/'.join(path+(k,)),'reason':'named_secret_or_monetary_field','values_not_copied':True})
                    continue
                yield from leaves(v,path+(k,),design or k.lower() in DESIGN)
        elif isinstance(obj,list):
            for i,v in enumerate(obj): yield from leaves(v,path+(str(i),),design)
        elif isinstance(obj,Decimal): yield path,obj,design
        elif isinstance(obj,str) and SIMPLE.fullmatch(obj.strip()) and len(obj.strip())<1000:
            # Numeric strings are indexed explicitly; nonnumeric strings are never mined.
            yield path,Decimal(obj.strip()),design
    selected=sorted(p for p in files if p.endswith('.json') and (p.startswith('docs/') or p.startswith('paper/llm/figures/')))
    for p in selected:
        s=read(p); count=nonfinite=0
        try: obj=json.loads(s,parse_float=Decimal,parse_int=Decimal,parse_constant=Decimal)
        except (ValueError,TypeError) as e:
            file_dispositions.append({'file':p,'status':'invalid_json','error':str(e)}); continue
        for path,d,design in leaves(obj):
            pointer='/'+'/'.join(k.replace('~','~0').replace('/','~1') for k in path)
            if add(d,p,pointer,'design_parameter' if design else 'data'): count+=1
            else: nonfinite+=1
        file_dispositions.append({'file':p,'status':'indexed','numeric_leaves':count,'nonfinite_leaves':nonfinite})
    if include_log and 'docs/research-log.md' in files:
        p='docs/research-log.md'; s=read(p); newlines=[-1]+[m.start() for m in re.finditer('\n',s)]
        for a,b,d,q,raw in numeric_tokens(s):
            add(d,p,'line:'+str(bisect.bisect_right(newlines,a))+';offset:'+str(a),'log_only')
    sorted_values=sorted(vmap); sorted_ids=[vmap[x] for x in sorted_values]
    groups=[]; group_map={}; group_sets={}
    def group(d,q,mode,scale):
        with localcontext() as ctx:
            ctx.prec=max(80,len(d.as_tuple().digits)+30)
            if mode=='nearest': lo,hi,lc,hc=d-q/2,d+q/2,True,False
            elif mode=='upper': lo,hi,lc,hc=d-q,d,False,True
            else: lo,hi,lc,hc=d,d+q,True,False
            lo/=scale; hi/=scale
        key=(lo,hi,lc,hc,mode,scale)
        if key in group_map: return group_map[key]
        a=bisect.bisect_left(sorted_values,lo) if lc else bisect.bisect_right(sorted_values,lo)
        b=bisect.bisect_right(sorted_values,hi) if hc else bisect.bisect_left(sorted_values,hi)
        ids=sorted_ids[a:b]; gid=len(groups); counts=collections.Counter()
        for vi in ids:
            for ri in values[vi]['refs']: counts[refs[ri][2]]+=1
        groups.append({'mode':mode,'receipt_multiplier':str(scale),'interval':[str(lo),str(hi)],'closed':[lc,hc],
                       'value_ids':ids,'reference_counts':dict(counts)})
        group_sets[gid]=set(ids); group_map[key]=gid
        return gid
    for i,n in enumerate(numbers):
        d,q=Decimal(n['value']),Decimal(n['quantum']); modes=['nearest']
        if n['bound_direction']: modes.append(n['bound_direction'])
        gids=[group(d,q,m,scale) for scale in ([Decimal(1),Decimal(100)] if n['percentage'] else [Decimal(1)]) for m in modes]
        viable=set().union(*(group_sets[g] for g in gids)); kinds=collections.Counter()
        # Count references once, even when native/percent or bound/nearest overlap.
        for vi in viable:
            for ri in values[vi]['refs']: kinds[refs[ri][2]]+=1
        data=kinds['data']+kinds['design_parameter']; nearest_data=any(sum(groups[g]['reference_counts'].get(k,0) for k in ('data','design_parameter')) for g in gids if groups[g]['mode']=='nearest')
        bound_data=any(sum(groups[g]['reference_counts'].get(k,0) for k in ('data','design_parameter')) for g in gids if groups[g]['mode']!='nearest')
        status='bound' if bound_data else 'data_match' if nearest_data else 'log_only' if kinds['log_only'] else 'unmatched'
        n.update(id=i,status=status,candidate_groups=gids,candidate_count=sum(kinds.values()),candidate_counts=dict(kinds),ambiguous=data>1)
        if kinds['design_parameter']: n['tags'].append('design_parameter_candidate')
        import heapq
        context_terms=words(n['sentence']+' '+n['section'])
        ranked=((len(ref_terms[ri]&context_terms),-ri) for vi in viable for ri in values[vi]['refs'])
        top=heapq.nlargest(5,ranked)
        n['preview']=[{'reference_id':-neg,'key_overlap':score,'file':refs[-neg][0],'pointer':refs[-neg][1],'kind':refs[-neg][2],'value':values[ref_v[-neg]]['value']} for score,neg in top]
    # Recheck every used byte, including the paper, before publishing the report.
    for p,rec in used.items():
        if sha((root/p).read_bytes())!=rec['sha256']: raise ValueError('Input changed during scan '+p)
    counts=collections.Counter(n['status'] for n in numbers)
    summary={'schema':VERSION,'source_commit':commit,'main_sha256':used[main]['sha256'],'runtime_seconds':time.perf_counter()-started,
             'numbers':len(numbers),'status_counts':dict(counts),'ambiguous_data_numbers':sum(n['ambiguous'] for n in numbers),
             'design_parameter_candidates':sum('design_parameter_candidate' in n['tags'] for n in numbers),
             'receipt_files':len(selected),'numeric_references':len(refs),'unique_values':len(values),'candidate_groups':len(groups),
             'input_files':input_rows,'log_indexed':include_log,'invalid_json_files':sum(x['status']=='invalid_json' for x in file_dispositions),
             'named_secret_or_monetary_exclusions':len(policy_exclusions),
             'limitations':['Numeric coincidence and lexical ranking do not authenticate the claimed cohort, statistic, extrema, source or units.',
               'Bounds search one outward rounding quantum and do not establish a global extremum or sufficient inequality.',
               'Literal TeX input captions only; no macro expansion, OCR, generated figure labels or number words. Exclusions and unresolved inputs are reported.',
               'All numeric leaves from committed JSON are candidates, including unrelated experiments and receipt metadata; design tags come from args/config paths.',
               'Every candidate is retained through candidate_groups -> value_ids -> refs. Preview is only the five highest lexical scores.']}
    dump(out/'numbers.json',{'summary':summary,'numbers':numbers})
    dump(out/'candidate-groups.json.gz',groups,True)
    dump(out/'receipt-index.json.gz',{'values':values,'refs':refs,'reference_format':['file','JSON_pointer_or_log_line_offset','kind']},True)
    dump(out/'coverage.json',{'inputs':used,'json_files':file_dispositions,'excluded_tex':skipped,'policy_exclusions':policy_exclusions})
    md=['# Paper number candidates','', '**A numerical match is a search lead, not verification of a scientific claim.**','',
        f"Frozen source `{commit}`; main.tex SHA256 `{summary['main_sha256']}`.",'',
        f"{len(numbers)} numeric tokens; statuses: "+', '.join(f'{k} {v}' for k,v in sorted(counts.items()))+'.',
        f"{summary['ambiguous_data_numbers']} ambiguous data candidates; {summary['design_parameter_candidates']} have an args/config design candidate.",'',
        'All candidates are losslessly indexed in numbers.json, candidate-groups.json.gz and receipt-index.json.gz. Each number lists group IDs; group value IDs index values, whose refs index the file/pointer records. No match list is truncated.','',
        'A figure input contributes caption arguments only. Main prose, equations and table values remain. Citation/reference keys, rendering controls, explicit dates and contextually identified years are excluded; 2000 bootstrap draws are retained.','',
        '## Numbers needing context review','']
    sections=collections.defaultdict(list)
    for n in numbers:
        if n['status'] in ('unmatched','log_only','bound') or n['ambiguous']: sections[n['section']].append(n)
    for sec,rows in sections.items():
        md += ['### '+sec,'']
        for n in rows:
            raw=n['raw'].replace('`','\\`'); ctx=n['sentence'].replace('\n',' ')
            preview='; '.join(f"{x['file']} {x['pointer']} (overlap {x['key_overlap']})" for x in n['preview'][:2])
            md.append(f"- **#{n['id']} `{raw}`** — {n['file']}:{n['line']}, {n['status']}; {n['candidate_count']} candidates"+(' [ambiguous]' if n['ambiguous'] else '')+(' [design parameter candidate]' if 'design_parameter_candidate' in n['tags'] else '')+'. '+ctx+(f" Preview: {preview}." if preview else ''))
        md.append('')
    md+=['## Limitations','']+['- '+x for x in summary['limitations']]
    (out/'report.md').write_text('\n'.join(md)+'\n')
    dump(out/'run-summary.json',summary)
    outputs={p.name:{'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in sorted(out.iterdir()) if p.is_file()}
    dump(out/'manifest.json',{'schema':VERSION,'source_commit':commit,'main_sha256':used[main]['sha256'],'code_sha256':sha(Path(__file__).read_bytes()),'outputs':outputs})
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inventory',type=Path); p.add_argument('--include-log',action='store_true')
    a=p.parse_args(); print(json.dumps(check(a.root,a.output,a.inventory,a.include_log),indent=2))
