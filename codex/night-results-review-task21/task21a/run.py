"""Bounded saved-scalar adapter; original producer code is never imported or executed."""
import argparse, hashlib, json, math, shlex, sys, time
from decimal import Decimal
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arithmetic as A


def decode(data):
    return json.loads(data, parse_float=Decimal, parse_constant=lambda s: (_ for _ in ()).throw(ValueError('nonfinite '+s)))


def encode(value):
    return json.dumps(value, indent=2, default=lambda x: str(x) if isinstance(x, Decimal) else A.decimal_string(x), allow_nan=False) + '\n'


def read_pin(pin, seen):
    p = Path(pin['path']); b = p.read_bytes()
    A.require(len(b) == pin['bytes'] and hashlib.sha256(b).hexdigest() == pin['sha256'], 'input pin mismatch: '+str(p))
    seen[str(p)] = pin
    return b


def command_options(command):
    tokens = shlex.split(command); out = {}
    for i, t in enumerate(tokens):
        if t.startswith('--'):
            values = []
            for value in tokens[i+1:]:
                if value.startswith('--'): break
                values.append(value)
            out[t[2:].replace('-', '_')] = values if t == '--points' else values[0]
    return out


def same(a, b):
    if isinstance(a, list) or isinstance(b, list):
        return isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    try: return A.f64(a) == A.f64(b)
    except (ValueError, TypeError, OverflowError): return a == b


def compare_rows(values, saved):
    A.require(len(values) == len(saved), 'saved scalar-list length mismatch')
    return [dict(index=i, independent_binary64=v, saved=A.scalar_record(s), saved_minus_independent=A.f64(s)-v) for i,(v,s) in enumerate(zip(values,saved))]


def literal_ratios(values, saved):
    A.require(len(saved) == max(0,len(values)-2), 'saved ratio-list length mismatch')
    return [A.ratio_record(values[i:i+3], saved[i]) for i in range(len(saved))]


def cascade(name, doc, options):
    flips = doc['flips']; args = doc['args']; p0 = int(options.get('period', args['period']))
    A.require(len(flips) <= int(options.get('flips',args['flips'])), 'more flips than requested')
    A.require([r['period'] for r in flips] == [p0*2**i for i in range(len(flips))], 'source period sequence mismatch')
    metadata = {k: same(args.get(k),v) for k,v in options.items() if k in args}
    A.require(all(metadata.values()), 'saved arguments disagree with original command')
    pars = [r['flip']['lambda'] for r in flips]
    A.require(len(doc['flip_parameters']) == len(flips), 'flip parameter count')
    levels = []; evaluated = []
    for i,r in enumerate(flips):
        p = int(r['period']); check = A.check_evaluation(r['flip'],p,args['tol'], '1e-6', True)
        check['saved_parameter_array_matches'] = same(r['flip']['lambda'],doc['flip_parameters'][i])
        check['half_period_guard'] = 'not checked for first orbit' if i == 0 else 'source guard; split distance not saved'
        levels.append(dict(attempt=name,kind='flip',period=p,parameter=r['flip']['lambda'],
                           disposition='accepted_flip' if check['saved_flip_gates_pass'] else 'saved accepted row fails an independently checked scalar gate',checks=check))
        evaluated.append([A.check_evaluation(e,p,args['tol']) for e in r['evaluations']])
    return dict(levels=levels, metadata_matches=metadata, requested_start_period=p0,
        requested_flips=int(options.get('flips',args['flips'])), actual_saved_flip_count=len(flips),
        original_completed_run_is_separate=True, evaluation_checks=evaluated,
        interval_comparisons=compare_rows([A.f64(pars[i+1])-A.f64(pars[i]) for i in range(len(pars)-1)],doc['intervals']),
        literal_ratio_comparisons=literal_ratios(pars,doc['ratios']), consecutive_level_ratios=A.ratio_sequence(levels), original=doc)


def superstable(name,doc,options):
    bands=doc['bands']; specs=options['points']
    A.require(len(bands)==len(specs), 'superstable band count vs command')
    levels=[]; checks=[]
    for r,spec in zip(bands,specs):
        ps,ts,aa,bb=spec.split(':'); p=int(ps)
        A.require(r['period']==p and same(r['log10_temperature'],ts) and same(r['bracket'],[aa,bb]),'superstable command association')
        row=dict(period=p,detected_period=r['detected_period'],detected_period_matches=r['detected_period']==p)
        level=dict(attempt=name,kind='superstable',period=p,parameter=None,disposition='original error or missing root')
        if 'superstable' in r:
            pair=r['clean_bracket']; mus=r['mu']; A.require(len(pair)==len(mus)==2,'clean endpoint pair')
            root=A.secant(*pair,*mus); saved=A.exact(r['superstable']); record=r['record']; lead=record['multipliers'][0]
            point=A.f64(r['superstable']); af,bf,pf,mf=map(A.f64,(*pair,*mus))
            literal=(af*mf-bf*pf)/(mf-pf)
            row.update(positive_endpoint_record=A.check_evaluation(record,p,doc['args']['tol']),
                secant_exact_decimal=A.rational_record(root), saved_minus_exact_decimal=A.rational_record(saved-root),
                secant_binary64=literal,saved_minus_secant_binary64=point-literal,
                signed_endpoints=A.f64(mus[0])>0>A.f64(mus[1]),
                positive_endpoint_parameter_matches=same(record['lambda'],pair[0]),
                positive_endpoint_leading_real_matches=same(lead[0],mus[0]),
                positive_endpoint_leading_imaginary_small=abs(A.f64(lead[1]))<=1e-8,
                saved_point_inside_clean_bracket=min(map(A.exact,pair))<=saved<=max(map(A.exact,pair)),
                clean_bracket=A.interval_record(*sorted(map(A.exact,pair)),places=18),
                root_was_directly_evaluated=False, evaluation_checks=[])
            vals=[complex(A.f64(z[0]),A.f64(z[1])) for z in record['multipliers']]
            row['positive_endpoint_clean_from_saved_pairs']=len(vals)<2 or abs(vals[0])>10*abs(vals[1])
            for ev in r['evaluations']:
                # Only the real part is logged. For clean rows the source enforces |imag|<=1e-8;
                # at an unclean row the imaginary part is unavailable, so do not pretend |Re mu| is its modulus.
                realclean=ev['next_modulus'] is None or abs(A.f64(ev['mu']))>10*A.f64(ev['next_modulus'])
                row['evaluation_checks'].append(dict(original=ev, real_part_dominance=realclean,
                    claimed_clean=ev['clean'], disposition='saved clean real-leading claim; original source guards imaginary part' if ev['clean'] else 'unclean retained; missing imaginary part prevents full modulus reconstruction'))
            gates=[row[k] for k in ('detected_period_matches','signed_endpoints','positive_endpoint_parameter_matches','positive_endpoint_leading_real_matches','positive_endpoint_leading_imaginary_small','saved_point_inside_clean_bracket','positive_endpoint_clean_from_saved_pairs')]
            gates.append(row['positive_endpoint_record']['newton_converged'])
            level.update(parameter=r['superstable'],clean_bracket=pair,
                disposition='superstable_point_estimate' if all(gates) else 'saved superstable row fails an independently checked scalar gate')
        else: row['error']=r.get('error','missing root without recorded error')
        levels.append(level); checks.append(row)
    successful=[r['superstable'] for r in bands if 'superstable' in r]
    A.require(same(successful,doc['superstable']),'producer successful-band list association')
    return dict(levels=levels,band_checks=checks,original=doc,
        literal_success_only_ratio_comparisons=literal_ratios(successful,doc['ratios']),
        original_success_only_ratio_array_skips_failed_bands=len(successful)!=len(bands),consecutive_level_ratios=A.ratio_sequence(levels))


def progress_checks(rows,tol):
    out=[]
    for i,r in enumerate(rows):
        if 'lambda' in r and 'g' in r:
            mu=A.f64(r['mu']); g=mu+1.; p=int(r['period'])
            out.append(dict(index=i,period=p,parameter=A.scalar_record(r['lambda']),g_reconstructed=g,
                logged_g_minus_reconstructed=A.f64(r['g'])-g,
                newton_converged=A.f64(r['newton'])<max(A.f64(tol),1e-13*p),
                ritz_small=A.f64(r['ritz_residual'])<=1e-6,
                next_modulus_below_one=r['next_modulus'] is not None and A.f64(r['next_modulus'])<1,
                imaginary_part='not logged; cannot independently check this evaluation realness'))
    return out


def selected_levels(attempts,selectors):
    out=[]
    for name,p in selectors:
        rows=[r for r in attempts[name].get('analysis',{}).get('levels',[]) if r['period']==p]
        A.require(len(rows)<=1,'duplicate selected period')
        out.append(rows[0] if rows else dict(attempt=name,kind='flip',period=p,parameter=None,disposition='missing selected period'))
    return out


def run(card):
    start=time.monotonic();seen={}; manifest=decode(read_pin(card['input_manifest'],seen)); authority=decode(read_pin(manifest['authority'],seen))
    A.require(len(manifest['files'])==16 and manifest['selected_file_bytes']==164161,'exact scientific inventory')
    read_pin(card['local_c_original_card'],seen)
    original={r['name']:r for r in authority['original_runs']}; A.require(set(original)==set(card['original_runs']), 'exact original run set')
    for p in manifest['source_files'].values(): read_pin(p,seen)
    files={}
    for r in manifest['files']:
        A.require(r['id'] not in files and r['selected_for_saved_review'] is True,'unique selected input')
        raw=read_pin(r['file'],seen)
        files[r['id']]=[decode(line) for line in raw.splitlines() if line.strip()] if r['file']['path'].endswith('.jsonl') else decode(raw)
    attempts={}
    for name in card['attempt_order']:
        options={}; provenance={}
        if name in original:
            source=original[name]; A.require(source['recorded_termination_verified'] is True and source['hash_verified'] is True,'original local receipts lack termination/hash authority')
            A.require(source['commit'] in manifest['source_files'],'original source not captured')
            options=command_options(source['upload']['jobset_args'])
            stdout=read_pin(source['stdout'],seen).decode('utf-8')
            refusal=[line for line in stdout.splitlines() if any(word in line for word in ('SearchError','Collapsed','refusing','not a clean flip','no flip of period','did not converge','returned to the period','attractor at the start'))]
            provenance=dict(original_source_commit=source['commit'],original_job_command=source['upload']['jobset_args'],
                original_inner_status_lines=source['status_lines'],runner_returncode=source['runner_returncode'],
                original_recorded_termination_verified=True,refusal_lines=refusal,stdout_pin=source['stdout'],original_receipt_pin=source['receipt'])
        else:
            A.require(name=='local-FB156c' and authority['local_completed_c']['terminal_done_true'] is True,'local completion basis')
            provenance=dict(local_completion_authority=authority['local_completed_c'],original_source_commit=None,
                original_exit_status=None,original_process_closure_independently_observed=False)
        attempt=dict(provenance=provenance,original_options=options)
        rows=files.get(name+'/progress.jsonl',[])
        if rows:
            attempt.update(original_progress=rows,progress_scalar_checks=progress_checks(rows,'1e-12'))
            if name=='local-FB156c': A.require(rows[-1].get('done') is True,'local terminal done required')
        if name+'/cascade.json' in files:
            doc=files[name+'/cascade.json']
            if name=='local-FB156c':
                A.require(same(doc['args']['log_temperature'],'-0.156') and doc['args']['free']=='alpha','local scalar metadata scope')
                options={'period':str(card['local_c_start_period']),'flips':str(card['local_c_requested_flips'])}
            attempt['analysis']=cascade(name,doc,options)
            summaries=[r for r in rows if 'flip' in r]
            A.require(len(summaries)==len(doc['flips']),'progress flip count vs saved table')
            associations=[]
            for i,(event,row) in enumerate(zip(summaries,doc['flips'])):
                A.require(event['flip']==i+1 and event['period']==row['period'],'progress flip index/period')
                associations.append(dict(index=i,parameter_matches=same(event['lambda'],row['flip']['lambda']),
                    mu_matches=same(event['mu'],row['flip']['critical_multiplier'][0]),
                    next_modulus_matches=same(event['next_modulus'],row['flip']['next_modulus']),
                    ratios=literal_ratios([q['flip']['lambda'] for q in doc['flips'][:i+1]],event['ratios'])))
            attempt['progress_flip_associations']=associations
            if rows and rows[-1].get('done') is True:
                attempt['terminal_ratio_comparisons']=literal_ratios([q['flip']['lambda'] for q in doc['flips']],rows[-1]['ratios'])
        elif name+'/superstable.json' in files: attempt['analysis']=superstable(name,files[name+'/superstable.json'],options)
        else: attempt['analysis']=dict(disposition='no saved accepted root table; original refusal retained',levels=[])
        attempts[name]=attempt
    seq=selected_levels(attempts,card['flip_156_sequence']); ss=selected_levels(attempts,card['superstable_156_sequence'])
    flip_ratios=A.ratio_sequence(seq); super_ratios=A.ratio_sequence(ss)
    first156=A.ratio(*[x['parameter'] for x in seq[:3]]) if len(seq)>=3 and all(x.get('parameter') is not None and x['disposition']=='accepted_flip' for x in seq[:3]) else None
    base152=attempts['FB152b']['analysis']['levels']; first152=A.ratio(*[x['parameter'] for x in base152[:3]]) if len(base152)>=3 and all(x['disposition']=='accepted_flip' for x in base152[:3]) else None
    e=[r for r in files['FBD/progress.jsonl'] if 'lambda' in r and 'g' in r]
    A.require(len(e)==card['e_expected_evaluations'] and all(r['period']==384 for r in e),'exact e evaluation set')
    fits={'logged_decimal_g':A.ols_root(e),'represented_binary64_g':A.ols_root([{k:A.f64(v) if k in ('lambda','g') else v for k,v in r.items()} for r in e]),
        'decimal_mu_plus_one':A.ols_root([dict(r,g=A.exact(r['mu'])+1) for r in e])}
    root=A.exact(fits['logged_decimal_g']['root']['numerator'])/A.exact(fits['logged_decimal_g']['root']['denominator'])
    prior=selected_levels(attempts,card['e_ratio_prior_levels']); e_ratio=None
    if all(r.get('parameter') is not None and r['disposition']=='accepted_flip' for r in prior):
        pars=[r['parameter'] for r in prior]; spans=[min(A.exact(r['lambda']) for r in e),max(A.exact(r['lambda']) for r in e)]
        e_ratio=dict(point=A.ratio_record(pars+[root]),logged_parameter_span_sensitivity=A.coupled_ratio_envelope([[x,x] for x in pars]+[spans]),
            explicit_one_picomagnitude_sensitivity=A.coupled_ratio_envelope([[x,x] for x in pars]+[[root-A.exact('1e-12'),root+A.exact('1e-12')]]),original_locator_accepted=False)
    overlaps=[]
    for left,right,p in card['overlap_comparisons']:
        lr=selected_levels(attempts,[(left,p),(right,p)])
        overlaps.append(dict(attempts=[left,right],period=p,right_minus_left=None if any(x.get('parameter') is None for x in lr) else A.rational_record(A.exact(lr[1]['parameter'])-A.exact(lr[0]['parameter']))))
    for pin in list(seen.values()): read_pin(pin,{})
    return dict(schema='task21a-saved-arithmetic-v1', computational_status='complete',
        attempt_count=len(attempts),attempts=attempts,scientific_inputs=manifest['files'],excluded_pending=manifest['excluded'],
        input_manifest=card['input_manifest'],authority=manifest['authority'],all_read_pins=list(seen.values()),
        scientific_file_count=16,scientific_input_bytes=164161,refusal_metadata_bytes=sum(original[n]['stdout']['bytes'] for n in original),
        flip_156_sequence=seq,flip_156_ratios=flip_ratios,overlap_comparisons=overlaps,
        superstable_156_sequence=ss,superstable_156_ratios=super_ratios,pf1=A.pf1(first156,first152),
        pf2_minus_156=A.pf2(ss,'2026-10-15T06:05:00Z',192),pf2_minus_152=dict(disposition='cannot check: no admitted superstable sequence'),
        e_interpolation=dict(all_points=e,fit_variants=fits,ratio_diagnostics=e_ratio,original_refusal_unchanged=True,
            selection='All twelve admitted period384 evaluation rows; no plateau or outlier selection.'),
        limitations=['No original model/eigensolver/Newton/continuation rerun. Saved partial Ritz diagnostics are not full-spectrum proof.',
            'g_accept tests best single residual; scatter-based refusal is not implemented in reviewed source.',
            'At -0.152 no later flip found or a half-period collapse does not exclude a narrower cascade.',
            'Local156c has completed saved records but no independently attested historical loaded-source or process-exit identity.',
            'Superstable clean-bracket sensitivity and e interpolation error diagnostics are not rigorous exact-model enclosures.'],
        wall_seconds=time.monotonic()-start)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--card',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    card=decode(Path(args.card).read_bytes());A.require(card['status']=='reviewed','card not reviewed')
    for name, expected in card['source_sha256'].items():
        A.require(Path(name).name==name,'adjacent source basename')
        A.require(hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==expected,'executed source pin')
    out=Path(args.out);out.mkdir(exist_ok=False,parents=True)
    report=run(card);payload=encode(report).encode();A.require(len(payload)<=16*1024*1024,'output budget')
    (out/'report.json').write_bytes(payload)
    (out/'manifest.json').write_text(encode(dict(files=[dict(path='report.json',bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())])))

if __name__=='__main__':main()
