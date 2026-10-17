"""Strict, bounded E5 saved-data adapter. No model or producer imports."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import time
import numpy as np

HERE=Path(__file__).resolve().parent

def adjacent(name):
    spec=importlib.util.spec_from_file_location('e5_own_'+name,HERE/(name+'.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

A=adjacent('arithmetic')


def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def verify(pin):
    p=Path(pin['path'])
    if not p.is_file() or p.stat().st_size!=pin['bytes'] or digest(p)!=pin['sha256']:
        raise ValueError('pin mismatch: '+str(p))


def readj(p):return json.loads(Path(p).read_text())

def write(p,x):
    with open(p,'x') as f:json.dump(A.clean(x),f,sort_keys=True,separators=(',',':'),allow_nan=False)


def all_pins(meta):
    pins=[f for r in meta['runs'] for f in r['files']]+meta['sources']+meta['logs']+[meta[k] for k in ('publication','choices','choices_at_preheldout')]
    by={}
    for p in pins:
        key=p['path']; short={k:p[k] for k in ('path','bytes','sha256')}
        if key in by and by[key]!=short:raise ValueError('conflicting input pin')
        by[key]=short
    return list(by.values())


def load_npz(file):
    with np.load(file['path'],allow_pickle=False) as z:
        expected={m['member'][:-4]:m for m in file['members']}
        if set(z.files)!=set(expected):raise ValueError('NPZ key inventory')
        data={k:z[k] for k in z.files}
        for k,a in data.items():
            if a.shape!=tuple(expected[k]['shape']) or a.dtype!=np.dtype(expected[k]['dtype']):
                raise ValueError('NPZ header mismatch: '+file['path']+'/'+k)
    return data


def file_named(run, name):
    a=[f for f in run['files'] if Path(f['path']).name==name]
    if len(a)!=1:raise ValueError('missing/duplicate named input '+run['id']+'/'+name)
    return a[0]


def settings(run):return readj(file_named(run,'settings.json')['path'])


def load_loops(run, evidence, comparisons):
    cfg=settings(run); pids=cfg['prompts'];grid=cfg['log_t']; found={};batches=[]
    if cfg['transient']!=1000 or cfg['measured']!=1000 or cfg['eps']!=[1e-6,1e-8] or cfg['qmax']!=64 or cfg['tol']!=1e-8 or cfg['lambda_min']!=.001:raise ValueError('unplanned loop settings')
    records=sorted([f for f in run['files'] if Path(f['path']).name.startswith('loop-b') and f['path'].endswith('.json')],key=lambda f:f['path'])
    if len(records)!=len(pids)//8:raise ValueError('loop batch coverage')
    for batch,jfile in enumerate(records):
        jp=readj(jfile['path']);group=pids[batch*8:(batch+1)*8];z=load_npz(file_named(run,Path(jfile['path']).with_suffix('.npz').name))
        if jp['batch']!=batch or jp['prompts']!=group or z['batch_prompts'].tolist()!=group:raise ValueError('batch/group association')
        expected=[(p,t) for p in group for t in range(24)]
        if list(zip(z['cell_prompt'].tolist(),z['cell_tindex'].tolist()))!=expected:raise ValueError('NPZ native cell order')
        if [(c['prompt'],c['t_index']) for c in jp['cells']]!=expected or len(jp['cells'])!=192:raise ValueError('JSON cell order')
        if [e['prompt'] for e in jp['first_step_entropy']]!=group:raise ValueError('entropy prompt order')
        ev=A.loop_evidence(z,cfg);ev.update(run=run['id'],batch=batch,json_path=jfile['path'])
        comparisons.extend(A.comparison(ev['gate_failures'],jp['gate_failures'],f"/{run['id']}/{batch}/gates"))
        comparisons.extend(A.comparison(ev['virtual_retirement_mismatches'],jp['virtual_retirement_mismatches'],f"/{run['id']}/{batch}/virtual_retirement"))
        owncells=[]
        for j,(p,t) in enumerate(expected):
            row={k:(int(z[k][j]) if k in ('period','q_transient','q_end','retire_step','lead_period') else bool(z[k][j])) for k in ('period','q_transient','q_end','retire_step','lead_period','twins','supplementary','retired')}
            row.update(prompt=p,t_index=t,log10_T=grid[t],label=str(z['label'][j]),**{'class':str(z['raw_class'][j])},lyapunov=[float(v) if np.isfinite(v) else None for v in z['lyapunov'][:,j]],entropy_mean=float(z['entropy_mean'][j]) if np.isfinite(z['entropy_mean'][j]) else None)
            owncells.append(row)
        comparisons.extend(A.comparison(owncells,jp['cells'],f"/{run['id']}/{batch}/cells"))
        ev['entropy']=[]
        for ix,p in enumerate(group):
            if p in found:raise ValueError('duplicate loop prompt')
            logits=z['first_step_logits'][ix];mask=np.ones(len(logits),bool);mask[list(A.MASKED)]=False
            er=A.entropy_root(logits); em=A.entropy_root(logits[mask]);curve=[A.entropy(logits,10.**v) for v in grid];curvem=[A.entropy(logits[mask],10.**v) for v in grid]
            ownent=dict(prompt=p,entropy_by_T=curve,entropy_by_T_masked=curvem,T_H=er['T'],T_H_masked=em['T'],entropy_at_T1=A.entropy(logits,1.),max_probability_T1=float(1/np.exp(logits-logits.max()).sum()))
            comparisons.extend(A.comparison(ownent,jp['first_step_entropy'][ix],f"/{run['id']}/{batch}/entropy/{p}"))
            comparisons.extend(A.comparison({'T_H':z['T_H'][ix],'T_H_masked':z['T_H_masked'][ix]},{k:jp['first_step_entropy'][ix][k] for k in ('T_H','T_H_masked')},f"/{run['id']}/{batch}/entropy_array/{p}"))
            ev['entropy'].append(dict(prompt=p,full=er,masked=em,own_record=ownent,saved_record=jp['first_step_entropy'][ix]))
            # Endpoint analysis uses the saved H scalar, independently checked above from the saved logits.
            # This isolates entropy solver rounding from the original fixed rank-statistic contract.
            cs=owncells[ix*24:(ix+1)*24]
            found[p]=dict(labels=[c['label'] for c in cs],period=[c['period'] for c in cs],lyapunov=[c['lyapunov'] for c in cs],entropy=jp['first_step_entropy'][ix],middle_entropy=A.entropy(logits,10**-.25),gate_failures=jp['gate_failures'],virtual_retirement_mismatches=jp['virtual_retirement_mismatches'],vocab=len(logits))
        evidence['loops'].append(ev);batches.append(z)
    if set(found)!=set(pids):raise ValueError('loop prompt coverage')
    return cfg,found,batches


def load_samples(run, vocab, evidence, comparisons):
    cfg=settings(run);found={};names=sorted(Path(f['path']).name for f in run['files'] if f['path'].endswith('.npz'))
    if names!=sorted(f'samples-p{p:02d}.npz' for p in cfg['prompts']):raise ValueError('sample file inventory')
    for p in cfg['prompts']:
        z=load_npz(file_named(run,f'samples-p{p:02d}.npz'))
        z['saved_gate_failures']=json.loads(str(z['gate_failures']))
        ev=A.sample_evidence(z,(24,cfg['samples'],256),vocab);ev.update(run=run['id'],prompt=p,seconds=float(z['seconds'][0]))
        comparisons.extend(A.comparison(ev['gate_failures'],z['saved_gate_failures'],f"/{run['id']}/{p}/sample_gates"))
        evidence['samples'].append(ev);found[p]=z
    return cfg,found


def selected_projection_comparison(own,saved,comparisons,path):
    # Metadata are associated against authenticated source/CLI below, never reported as new numerical evidence.
    for k in ('git','choices_file'):
        if k in saved:comparisons.append(dict(path=path+'/'+k,status='inherited_execution_metadata',saved=saved[k]))
    comparisons.extend(A.comparison(own,{k:v for k,v in saved.items() if k not in ('git','choices_file')},path))


def execute(card,out):
    start=time.perf_counter();meta=readj(card['metadata']['path']);pins=all_pins(meta)
    if len(pins)!=card['input_file_count'] or sum(p['bytes'] for p in pins)!=card['input_bytes']:raise ValueError('input accounting')
    if sum(p['bytes'] for p in pins)>card['max_input_bytes']:raise ValueError('input byte cap')
    for p in pins:verify(p)
    C=adjacent('controls');controls=C.checks();write(out/'controls.json',controls)
    if controls['failed']:raise ValueError('manufactured control failure')
    runs={r['id']:r for r in meta['runs']}
    if len(runs)!=10:raise ValueError('run inventory')
    cfg0=settings(runs['dev-loop-1']);cfg1=settings(runs['heldout-loop-1'])
    manifest=next(f['manifest_metadata'] for f in runs['dev-loop-1']['files'] if 'manifest_metadata' in f)
    names=manifest['prompt_lists'];grid=cfg0['log_t']
    if grid!=cfg1['log_t'] or len(grid)!=24 or cfg0['prompts']!=names['dev'] or cfg1['prompts']!=names['heldout']:raise ValueError('fixed cohort/grid')
    if names['dev']!=[8*s+o for s in (0,1) for o in range(8) if (s+o)%2==0] or names['heldout']!=[8*s+o for s in range(2,8) for o in range(8) if (s+o)%2==0]:raise ValueError('cohort rule')
    choices=readj(meta['choices_at_preheldout']['path'])
    if any(choices[k]!=v for k,v in dict(rep_window=96,rep_qmax=48,h_target=2.,retire=True,samples=12).items()):raise ValueError('frozen-choice contract')
    comparisons=[];evidence=dict(loops=[],samples=[],scope='saved arrays and scalar histories; no model/native trajectory regeneration')
    devcfg,dl,db=load_loops(runs['dev-loop-1'],evidence,comparisons)
    hc,hl,hb=load_loops(runs['heldout-loop-1'],evidence,comparisons)
    if devcfg['retire'] or not hc['retire']:raise ValueError('frozen retirement setting')
    dc,ds=load_samples(runs['dev-sample-1'],151936,evidence,comparisons)
    sc,ss=load_samples(runs['heldout-sample-1'],151936,evidence,comparisons)
    bc,bs=load_samples(runs['dev-sample-buffered-1'],151936,evidence,comparisons)
    if dc['samples']!=16 or sc['samples']!=12 or sc.get('sampler_impl')!='buffered' or bc['prompts']!=[0]:raise ValueError('sample allocation')
    if any(c['log_t']!=grid or c['tokens']!=256 for c in (dc,sc,bc)):raise ValueError('sample grid/length')
    if dc['prompts']!=names['dev'] or sc['prompts']!=names['heldout']:raise ValueError('sample cohort')
    for p in names['heldout']:
        if ss[p]['mean_entropy'].shape!=(24,12):raise ValueError('entropy shape')
    plain,buf=ds[0],bs[0]
    evidence['buffered_validation']=dict(tokens_bitwise_identical=bool(np.array_equal(plain['tokens'],buf['tokens'])),differing_samples=int((plain['tokens']!=buf['tokens']).any(-1).sum()),entropy_max_abs_diff=float(np.abs(plain['mean_entropy']-buf['mean_entropy']).max()),seconds=[float(plain['seconds'][0]),float(buf['seconds'][0])])
    bpub=readj(file_named(runs['dev-sample-buffered-1'],'compare-dev-sample-1.json')['path'])
    comparisons.extend(A.comparison(evidence['buffered_validation'],{k:v for k,v in bpub['files'][0].items() if k!='file'},'/buffered_validation'))
    dev_events=[r for r in evidence['loops'] if r['run']=='dev-loop-1'];sample_events=[r for r in evidence['samples'] if r['run']=='dev-sample-1']
    choice=A.development_choice([ds[p] for p in names['dev']],db[0],grid,all(e['source_gate_pass'] for e in dev_events+sample_events),sum(len(e['virtual_retirement_mismatches']) for e in dev_events),float(db[0]['seconds'][0]),[float(ds[p]['seconds'][0]) for p in names['dev']],[dl[p]['middle_entropy'] for p in names['dev']])
    evidence['development_recomputed_choices']=choice
    comparisons.extend(A.comparison(choice,{k:choices[k] for k in choice},'/frozen_choices'))
    evidence['development_unique_virtual_mismatch_count']=sum(len(e['virtual_retirement_mismatches']) for e in dev_events)
    results={};draws={};sample_witness=[]
    # Main fixed choices and the already-recorded original-card-rule sensitivity, both predetermined.
    for label,qmax,pubid in [('heldout',48,'analysis-heldout-1'),('heldout_cardrule',32,'analysis-heldout-cardrule-1')]:
        rows=[]
        for p in names['heldout']:
            r,w=A.prompt_summary(p,names,grid,hl[p],ss[p],hl[p]['entropy'],96,qmax);rows.append(r);sample_witness.append(dict(analysis=label,prompt=p,**w))
        res,w=A.aggregate(rows,grid,10000);A.definitions(res)
        res.update(split='heldout',loop_runs=['heldout-loop-1'],sample_runs=['heldout-sample-1'],rule=dict(window=96,qmax=qmax,h_target=2.),provisional=False)
        results[label]=res;draws[label]=w
        pubfile=runs[pubid]['files'][0];saved=readj(pubfile['path'])
        selected_projection_comparison(res,saved,comparisons,'/'+label)
        if label=='heldout':
            publication=readj(meta['publication']['path'])
            selected_projection_comparison(res,publication,comparisons,'/publication')
    evidence['sample_witness']=sample_witness
    # Precision records contain only scalar summaries; ratios are recomputable, native derivatives are not.
    prec=readj(runs['precision-1']['files'][0]['path']);precrows=[]
    for r in prec['cells']:
        v=r['successive_relative_change'];rr=[float(v[i]/v[i+1]) if v[i+1]>0 else None for i in range(len(v)-1)]
        own=dict(min_successive_change=min(v),ratio_per_decade=rr)
        comparisons.extend(A.comparison(own,{k:r[k] for k in own},'/precision/'+str(r['log10_T'])))
        precrows.append(dict(saved=r,recomputed_scalar_summary=own,native_derivative_recomputation_available=False))
    evidence['precision']=dict(saved_prompt=prec['prompt'],h=prec['h'],cells=precrows,scope='source/recorded scalar check, not native precision validation')
    evidence['inherited_metadata']=dict(choice_chronology=meta['choice_commit_metadata'],choice_ancestor_of_versions=meta['choice_ancestor_of_versions'],runs=[dict(id=r['id'],invocations=[f['manifest_metadata'] for f in r['files'] if 'manifest_metadata' in f],progress=[{k:f[k] for k in ('path','rows','event_sequence') if k in f} for f in r['files'] if f['path'].endswith('jsonl')]) for r in meta['runs']],choices=choices)
    write(out/'analysis.json',results);write(out/'evidence.json',evidence);write(out/'draws.json',draws);write(out/'comparisons.json',comparisons)
    elapsed=time.perf_counter()-start;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for p in pins:verify(p)
    verify(card['metadata'])
    outputs=[dict(path=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(out.iterdir())]
    report=dict(schema='e5-independent-saved-audit-v1',computational_status='complete',source_sha256=card['source_sha256'],input_pins=pins,metadata_pin=card['metadata'],outputs=outputs,controls=controls,summary={k:{f:v[f] for f in ('n_prompts','primary','secondary_partial','secondary_matched','descriptive','censoring','first_step_entropy_range','virtual_retirement_mismatches')} for k,v in results.items()},loop_cells=sum(len(e['rows']) for e in evidence['loops']),sample_prompts=len(evidence['samples']),comparison_counts={s:sum(r['status']==s for r in comparisons) for s in sorted(set(r['status'] for r in comparisons))},source_gate_failures=[{'run':e['run'],'batch':e['batch'],'failures':e['gate_failures']} for e in evidence['loops'] if e['gate_failures']],sample_gate_failures=[{'run':e['run'],'prompt':e['prompt'],'failures':e['gate_failures']} for e in evidence['samples'] if e['gate_failures']],runtime=dict(seconds_before_final_rehash_report=elapsed,observed_peak_kib_before_final_rehash_report=rss,scope='includes data decoding, controls, arithmetic and five output serializations; excludes final rehash/report; supervisor owns whole-worker wall'),scientific_limitations=['finite fixed prompt cohort, not randomized causal intervention','partial rank correlation is not a causal adjustment or evidence of a shared cause','source-recorded finite-twin native periods and retired/supplementary histories are not regenerated','entropy inversion is independent float64 bisection; raw saved H used for original rank statistics after separately retaining all root discrepancies','no numerical primary-success cutoff was frozen; all intervals, censoring and undefined draws retained','sampling is float32 growing context with two EOS tokens masked; loop is float64 fixed soft slot','source-reading false bootstrap-regression claim was withdrawn: both exact current/original sources use paired indices'])
    write(out/'report.json',report)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--card',required=True);ap.add_argument('--out',required=True);args=ap.parse_args();card=readj(args.card)
    if card.get('status')!='reviewed':raise ValueError('prepared/unadmitted card')
    if set(card['source_sha256'])!={'run','arithmetic','controls'}:raise ValueError('source inventory')
    for name,sha in card['source_sha256'].items():
        if digest(HERE/(name+'.py'))!=sha:raise ValueError('source hash: '+name)
    verify(card['metadata']);out=Path(args.out)
    if str(out)!=card['output']:raise ValueError('output contract')
    out.mkdir(parents=True,exist_ok=False)
    try:execute(card,out)
    except BaseException as e:
        write(out/'failure.json',dict(type=type(e).__name__,message=str(e)));raise

if __name__=='__main__':main()
