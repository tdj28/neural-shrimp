"""PM2F one-shot independent saved-row audit. Prepared cards refuse all work."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import time

MODELS = ['r1qwen7b','r1llama8b','qwq32b']
EXPECTED_SOURCE = '21d300a6c3a6f73a5f098f8a3c8387f7b668a71b'


def load_module(name):
    p=Path(__file__).with_name(name+'.py')
    spec=importlib.util.spec_from_file_location('own_pm2f_'+name,p)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def pins_from_metadata(meta):
    out={}
    def add(p):
        k=p['path']
        if k in out and (out[k]['sha256'],out[k]['bytes']) != (p['sha256'],p['bytes']):
            raise ValueError('conflicting metadata pin: '+k)
        out[k]={'path':k,'sha256':p['sha256'],'bytes':p['bytes']}
    for m in meta['models']:
        add(m['records']);add(m['run'])
    for h in meta['history']:
        add(h['log_pin'])
        for f in h['files']:add(f)
    for f in meta['sources']:add(f)
    add(meta['publication'])
    return list(out.values())


def authenticate(pins, ceiling):
    if sum(p['bytes'] for p in pins)>ceiling:raise ValueError('declared input byte ceiling')
    for p in pins:
        f=Path(p['path'])
        if not f.is_file() or f.stat().st_size!=p['bytes'] or digest(f)!=p['sha256']:
            raise ValueError('input pin mismatch: '+str(f))


def save(out,name,value,core):
    p=out/name
    with p.open('x') as f:json.dump(core.json_safe(value),f,indent=2,allow_nan=False);f.write('\n')
    return {'path':name,'bytes':p.stat().st_size,'sha256':digest(p)}


def perform(card,out):
    start=time.monotonic()
    expected={'core.py','controls.py','run.py'}
    if set(card['source_pins'])!=expected:raise ValueError('exact source inventory required')
    for name,sha in card['source_pins'].items():
        if digest(Path(__file__).with_name(name))!=sha:raise ValueError('own source mismatch: '+name)
    core=load_module('core');ctl=load_module('controls')
    controls=ctl.execute();measurement=ctl.manufactured_full_shape()
    written=[save(out,'controls.json',{'controls':controls,'manufactured':measurement},core)]
    if not controls['passed'] or controls['count']!=10:raise ValueError('manufactured controls failed')
    mp=card['metadata'];authenticate([mp],card['input_byte_limit'])
    meta=json.loads(Path(mp['path']).read_text())
    pins=[mp]+pins_from_metadata(meta)
    authenticate(pins,card['input_byte_limit'])
    if [m['id'] for m in meta['models']]!=MODELS:raise ValueError('exact model order mismatch')
    if any(s['commit']!=EXPECTED_SOURCE for s in meta['sources']):raise ValueError('original source association')
    if len(meta['history'])!=11:raise ValueError('original attempt coverage')
    history=meta['history']
    for arm,model in zip('ABC',MODELS):
        success=[h for h in history if h['arm']==arm and h.get('runner_returncode')==0]
        if len(success)!=1 or success[0]['commit']!=EXPECTED_SOURCE:raise ValueError('successful original attempt association')
        h=success[0];m=next(m for m in meta['models'] if m['id']==model)
        for kind in ('records','run'):
            matches=[p for p in h['selected_archive_members'] if p['local_path']==m[kind]['path']]
            if len(matches)!=1 or matches[0]['sha256']!=m[kind]['sha256'] or matches[0]['bytes']!=m[kind]['bytes'] or not matches[0]['local_matches']:
                raise ValueError('original archive member association')
    own={};projections={'endpoint':{},'fitted_slope':{}}
    for m in meta['models']:
        run=json.loads(Path(m['run']['path']).read_text())
        if run!=m['metadata']:raise ValueError('run metadata changed')
        a=run['args'];mid=m['id']
        fixed={'model':mid,'twins':2,'eps0':1e-10,'think_budget':1536,'answer_budget':256,
               'temperature':.6,'top_n':15,'top_p':.95,'offset':256,'seed':20261016,'device':'cuda',
               'limit':32 if mid=='qwq32b' else 128,'group':4 if mid=='qwq32b' else 8}
        if any(a[k]!=v for k,v in fixed.items()):raise ValueError('fixed physical/run arguments mismatch')
        rows=[]
        with open(m['records']['path']) as f:
            for line in f:
                if not line.strip():raise ValueError('blank record line')
                rows.append(json.loads(line))
        if len(rows)!=m['row_count'] or [r['index'] for r in rows]!=m['indices']:
            raise ValueError('row count/order/index identity mismatch')
        result=core.reduce_rows(rows,a)
        result['record_pin']=m['records'];result['run_pin']=m['run']
        result['source_group_count']=run['groups']
        result['source_group_count_matches_row_count']=len(rows)==run['groups']*a['group']
        own[mid]=result
        for definition in projections:
            projections[definition][mid]=core.declared_publication_projection(rows,a,definition)
    pub=json.loads(Path(meta['publication']['path']).read_text())
    if set(pub)!= {'run','source','eps0','models'} or set(pub['models'])!=set(MODELS):
        raise ValueError('publication schema mismatch')
    comparisons={d:core.compare_leaves(projections[d],pub['models']) for d in projections}
    metadata_leaves=[{'path':'$.run','published':pub['run'],'status':'inherited_label_not_independently_reconstructed'},
                     {'path':'$.source','published':pub['source'],'status':'inherited_label_not_independently_reconstructed'},
                     {'path':'$.eps0','published':pub['eps0'],'own':1e-10,'status':'exact' if pub['eps0']==1e-10 else 'different'}]
    written.append(save(out,'models.json',own,core))
    written.append(save(out,'comparisons.json',{'projections':projections,'all_comparisons':comparisons,
                    'publication_metadata':metadata_leaves,'original_publication':pub,
                    'semantics':'Both declared views retained; no view chosen by agreement. Missing historical inline summary implementation.'},core))
    authenticate(pins,card['input_byte_limit'])
    report={'computational_completed':True,'scientific_claims_accepted':False,'models':MODELS,
            'input_pins':pins,'source_pins':card['source_pins'],'card_sha256':card['_sha256'],
            'original_source':EXPECTED_SOURCE,'original_attempts':history,
            'controls':controls,'manufactured':measurement,
            'comparison_counts':{d:dict((s,sum(x['status']==s for x in comparisons[d])) for s in sorted(set(x['status'] for x in comparisons[d]))) for d in comparisons},
            'publication_metadata':metadata_leaves,
            'model_summaries':{k:{a:b for a,b in v.items() if a!='rows'} for k,v in own.items()},
            'historical_summary_invocation':meta['historical_summary_invocation'],
            'elapsed_through_posthash_before_report':time.monotonic()-start,
            'rss_KiB_observed_before_report':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'limitations':['No retained native distance series: fit/window/scalar measurements inherited',
                           'Nominal eps0 comparison, not observed initial-distance comparison',
                           'No independently reconstructed text/tokenizer extraction or loaded-weight attestation',
                           'Non-renormalized finite separation; no asymptotic chaos/causal exclusivity inference']}
    written.append(save(out,'report.json',report,core))
    save(out,'manifest.json',{'files':written},core)
    if sum(p.stat().st_size for p in out.iterdir() if p.is_file())>card['output_byte_limit']:
        raise ValueError('final output byte ceiling')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--card',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();card=json.loads(Path(a.card).read_text())
    if card.get('status')!='reviewed':raise SystemExit('prepared/unreviewed card refuses execution')
    if card.get('task')!='PM2F-independent-saved-audit':raise SystemExit('wrong task')
    out=Path(a.out)
    if str(out)!=card['output'] or not out.is_absolute():raise SystemExit('fixed absolute output mismatch')
    if out.exists():raise SystemExit('output already exists; untouched')
    card['_sha256']=digest(a.card)
    out.mkdir(parents=False,exist_ok=False)
    try:perform(card,out)
    except BaseException as e:
        with (out/'failure.json').open('x') as f:
            json.dump({'computational_completed':False,'error':repr(e)},f);f.write('\n')
        raise


if __name__=='__main__':main()
