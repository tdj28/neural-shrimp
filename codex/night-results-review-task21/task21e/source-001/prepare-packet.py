from pathlib import Path
import ast, datetime, hashlib, json, re, subprocess, tarfile, time

R = Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp')
D = Path('/tmp/task21e-source-001')
P27 = Path('/data2/neural-shrimp-data/codex/paper-review-task27/source-001')
DATA = Path('/data2/neural-shrimp-data')
D.mkdir(exist_ok=False)
start = time.perf_counter()
H = lambda b: hashlib.sha256(b).hexdigest()
bytes_read = 0

def read(p):
    global bytes_read
    b = Path(p).read_bytes(); bytes_read += len(b); return b

def pin(p):
    b = read(p); return {'path': str(p), 'bytes': len(b), 'sha256': H(b)}

def put(name, b):
    p = D / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b)
    return {'path': str(p), 'bytes': len(b), 'sha256': H(b)}

def dump(name, obj):
    return put(name, (json.dumps(obj, indent=2) + '\n').encode())

def git(c, p):
    return subprocess.check_output(['git', 'show', c + ':' + p], cwd=R)

current = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=R).decode().strip()
ids = ['CL1','CL2','CL3','CL4','CL1b','CL2b','CL3b','CL4b','CL5','CL6','CL6b']
attempts, rows, curves, commits = [], [], {}, []
p27_manifest = json.loads(read(P27 / 'compact-evidence-manifest.json'))
p27_rows = {Path(x['original_path']).parent.name: x for x in p27_manifest['local_saved_files'] if x.get('lane') == 'cusps'}

def keys_only(q):
    if isinstance(q, dict): return {k: keys_only(v) for k,v in q.items()}
    if isinstance(q, list): return {'type':'list','length':len(q),'element_schema':keys_only(q[0]) if q else None}
    return type(q).__name__

for ident in ids:
    op = DATA / 'runpod' / (ident + '-orchestrator.log'); ob = read(op); events = []
    for line in ob.splitlines():
        try: events.append(json.loads(line))
        except json.JSONDecodeError: pass
    initial = next(e for e in events if 'jobset_args' in e)
    c = initial['commit']; commits.append(c)
    init = {k:initial[k] for k in ['commit','jobset','jobset_args','code_bytes','data_bytes','code_sha256','data_sha256','extra_data'] if k in initial}
    init_pin = dump('authority/'+ident+'/initial-metadata.json', init)
    a = {'id':ident,'recorded_source_commit':c,'initial':init_pin,'command_specs':initial['jobset_args'].split(),
         'orchestrator':{'path':str(op),'bytes':len(ob),'sha256':H(ob)},
         'historical_limit':'Recorded local producer source/upload/retrieval/termination evidence; no reconstructed historical process births or remotely loaded model/source attestation.'}
    terminals = [e for e in events if 'terminated' in e]
    a['termination'] = dump('authority/'+ident+'/termination.json', terminals[-1]) if terminals else None
    outs = [e for e in events if 'out' in e]
    if not outs:
        a.update({'disposition':'initial attempt without terminal/retrieval event in this preserved orchestrator log', 'retrieved_archive':None,
                  'limit':'The frozen 01:40 producer log attributes the initial CL restart to a NumPy bool JSON serialization bug. This local log alone supplies no tool exit or independent termination evidence.'})
        attempts.append(a); continue
    original_root = Path(outs[-1]['out']); rp = original_root / 'result.json'; rb = read(rp); result = json.loads(rb)
    assert result['commit'] == c
    safe = {k:v for k,v in result.items() if not any(word in k.lower() for word in ['cost','rate','price'])}
    safe['original_record'] = {'path':str(rp),'bytes':len(rb),'sha256':H(rb),'financial_fields_omitted':True}
    a['retrieval'] = dump('authority/'+ident+'/retrieval.json', safe)
    ap = original_root / 'results.tgz'
    if not ap.exists():
        a.update({'disposition':'recorded termination; no retrieved archive or runner exit in original result', 'retrieved_archive':None})
        attempts.append(a); continue
    ab = read(ap); assert result['hash_verified'] and H(ab) == result['local_sha256'] == result['remote_sha256']
    sb = read(original_root / 'runner-stdout.txt')
    status = [l.decode() for l in sb.splitlines() if re.search(rb' start cl-| end cl-| runner done', l)]
    a['tool_status'] = dump('authority/'+ident+'/tool-status.json', {'original_path':str(original_root/'runner-stdout.txt'),'bytes':len(sb),'sha256':H(sb),'exact_status_lines':status,'runner_returncode':result['runner_returncode']})
    a['disposition'] = 'retrieved archive and per-tool status 0, runner status 0, producer termination recorded'
    a['retrieved_archive'] = {'path':str(ap),'bytes':len(ab),'sha256':H(ab)}
    a['selected_members'] = []
    with tarfile.open(ap) as tf:
        member_map = {m.name:m for m in tf.getmembers() if m.isfile()}
        a['small_native_member_inventory'] = [{'name':m.name,'bytes':m.size} for m in member_map.values() if m.name.endswith(('/locate.json','/state.npy','/progress.jsonl'))]
        for spec in a['command_specs']:
            test, curve, left = spec.split(':'); left = int(left)
            rid = Path(curve).name + '-' + test + '-' + str(left)
            member = 'explore/llm-soft-loop/pod64/codim2/' + rid + '/locate.json'
            b = tf.extractfile(member_map[member]).read()
            original = DATA / member; assert read(original) == b
            old = p27_rows[rid]; assert old['sha256'] == H(b) and read(P27 / old['snapshot_path']) == b
            file = put('inputs/'+rid+'/locate.json', b)
            q = json.loads(b)
            assert q['curve'] == curve and q['left'] == left and q['test'] == test
            assert any((' end cl-'+rid+' status 0') in s for s in status)
            row = {'id':rid,'test':test,'curve_id':curve,'left':left,'right':left+1,'attempt':ident,'source_commit':c,
                   'file':file,'original_path':str(original),'archive_member':member,'archive_sha256':H(ab),
                   'task27_snapshot':{'path':str(P27/old['snapshot_path']),'sha256':old['sha256'],'bytes':old['bytes']},
                   'declared_args':q['args'],'declared_period':q['period'],'declared_kind':q['kind'],
                   'schema':keys_only(q),'selection':'producer-declared cusp reversal' if test=='cusp' else 'separately retained non-cusp candidate disposition'}
            if test == 'cusp':
                cp = DATA / 'explore' / curve / 'curve.json'; cb = read(cp); cq = json.loads(cb)
                if curve not in curves:
                    cf = put('curves/'+curve.replace('/','__')+'.json', cb)
                    curves[curve] = {'id':curve,'file':cf,'original_path':str(cp),'declared_args':cq['args'],
                                     'declared_kind':cq['kind'],'declared_period':cq['period'],'point_count':len(cq['points']),
                                     'point_schema':keys_only(cq['points'][0]),'selected_left_indices':[],
                                     'association_limit':'Current local original curve JSON is pinned and its path is named in the original uploaded extra_data/command. Native upload archive contents and historical remote loaded bytes are not reconstructed.'}
                curves[curve]['selected_left_indices'].append(left)
                row['curve_metadata_scale'] = cq['args'].get('alpha_scale',10.0)
                row['curve_file'] = curves[curve]['file']
            rows.append(row); a['selected_members'].append({'id':rid,'member':member,**file})
    attempts.append(a)

assert len(rows)==15 and sum(r['test']=='cusp' for r in rows)==9 and len(curves)==7
source_rows=[]
paths=['tools/cusp_locate.py','tests/test_cusp_locate.py','tools/bifurcation_curves.py','tools/explore_llm_soft_loop.py','tools/explore_llm_attractors.py']
seen={}
for c in dict.fromkeys(commits):
    for p in paths:
        b=git(c,p); sha=H(b)
        if (p,sha) not in seen: seen[(p,sha)]=put('source/'+c[:8]+'/'+p,b)
        source_rows.append({'source_commit':c,'repository_path':p,**seen[(p,sha)]})
    p='tools/runpod_soft_runner.sh';b=git(c,p);ls=b.splitlines(keepends=True)
    i=next(i for i,l in enumerate(ls) if re.match(rb'  codim2\)',l));j=next(j for j in range(i+1,len(ls)) if re.match(rb'  [A-Za-z0-9_]+\)',ls[j]))
    helper=next(i for i,l in enumerate(ls) if l.startswith(b'run()'));hend=next(j for j in range(helper+1,len(ls)) if ls[j].strip()==b'}')+1
    for lo,hi in [(helper,hend),(i,j)]:
        e=b''.join(ls[lo:hi]);x=put('source/'+c[:8]+f'/runner.lines{lo+1}-{hi}.sh',e)
        source_rows.append({'source_commit':c,'repository_path':p,'full_source_sha256':H(b),'full_source_bytes':len(b),'start_line':lo+1,'end_line':hi,**x})
diff=subprocess.check_output(['git','diff',commits[0],attempts[4]['recorded_source_commit'],'--','tools/cusp_locate.py','tests/test_cusp_locate.py'],cwd=R)
put('source/initial-serialization-fix.diff',diff)

claims=[]
log=git('66ffa102c56f6d5b22f116a1bb194f503c75562e','docs/research-log.md');ls=log.splitlines(keepends=True)
for stamp in ['01:19','01:40','02:28','02:30','02:32','02:50','03:17']:
    i=next(i for i,l in enumerate(ls) if l.startswith(('## 2026-10-15 '+stamp+' UTC:').encode()))
    j=next((j for j in range(i+1,len(ls)) if ls[j].startswith(b'## ')),len(ls))
    x=put('claims/'+stamp.replace(':','')+'.md',b''.join(ls[i:j]));claims.append({'source_commit':'66ffa102c56f6d5b22f116a1bb194f503c75562e','repository_path':'docs/research-log.md','full_source_sha256':H(log),'start_line':i+1,'end_line':j,**x})
paper=git(current,'paper/llm/main.tex');ls=paper.splitlines(keepends=True)
for phrase in [b'All nine reversals',b'All nine tangent']:
    i=next(i for i,l in enumerate(ls) if phrase in l);lo=max(0,i-6);hi=min(len(ls),i+12)
    x=put('claims/paper-lines'+str(lo+1)+'-'+str(hi)+'.tex',b''.join(ls[lo:hi]));claims.append({'source_commit':current,'repository_path':'paper/llm/main.tex','full_source_sha256':H(paper),'start_line':lo+1,'end_line':hi,**x})
comms=git(current,'comms/claude-to-codex.md');put('claims/assignment-and-queue.md',comms[comms.index(b'## 2026-10-15T06:27Z'):])
source_map=dump('source-map.json',{'schema':'task21e-source-map-v1','capture_commit':current,'source_files':source_rows,'claims':claims,
    'task27_locator_equal_all_completed':all(git(a['recorded_source_commit'],'tools/cusp_locate.py')==read(P27/'tools/cusp_locate.py') for a in attempts if a.get('retrieved_archive')),
    'scope':'Exact original Git source exports, not imports or execution; common source bytes stored once and all original commit associations retained.'})
authority=dump('original-authority.json',{'schema':'task21e-original-authority-v1','attempts':attempts,'source_map':source_map,
    'task27_compact_evidence_manifest':pin(P27/'compact-evidence-manifest.json'),
    'coverage':'Nine originally commanded cusp localizations across seven named source curves, plus six separately retained crossing/reversal dispositions. The broader continued-curve universe is producer-declared; the future scalar check may compare all saved point flags and tangent-dot candidate predicates within these seven pinned curves only.',
    'scope':'Small local saved JSON/source/record/archive-member authentication only. No state.npy, model weights, derivatives or remote processes decoded or queried. No original producer process births reconstructed.',
    'preparation_exposure':'Task27 and original outcome-bearing logs were previously read by the team. This packet preparation copied records and inspected schemas/args/identities only; no independent cusp arithmetic was executed.'})
inputs=dump('inputs.json',{'schema':'task21e-saved-inputs-v1','prepared_unlaunchable':True,'source_commit':current,'authority':authority,'source_map':source_map,
    'records':rows,'curves':list(curves.values()),'selected_cusp_count':9,'separate_other_count':6,'curve_count':7,
    'locate_bytes':sum(r['file']['bytes'] for r in rows),'curve_bytes':sum(c['file']['bytes'] for c in curves.values()),
    'numerical_scope':'Own finite saved-scalar consistency/gate reconstruction and within-seven-curve candidate association only. Native derivative/state/multiplier calculations are inherited or explicitly cannot-check; no model rerun.'})
files=[{'path':str(p.relative_to(D)),'bytes':p.stat().st_size,'sha256':H(p.read_bytes())} for p in sorted(D.rglob('*')) if p.is_file()]
manifest=dump('manifest.json',{'schema':'task21e-source-input-packet-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':current,
    'inputs':inputs,'authority':authority,'source_map':source_map,'files':files,'packet_bytes':sum(f['bytes'] for f in files),
    'metadata_bytes_read':bytes_read,'metadata_elapsed_seconds':time.perf_counter()-start,
    'measurement_scope':'Finite source/hash/archive-member/metadata preparation, not target arithmetic runtime; Git source reads and final manifest hashing not included in byte counter.',
    'target_numerical_execution':False,'native_arrays_decoded':False})
print(json.dumps({'manifest':manifest,'inputs':inputs,'authority':authority,'source_map':source_map,'records':len(rows),'curves':len(curves),'packet_bytes':sum(f['bytes'] for f in files)},indent=2))
