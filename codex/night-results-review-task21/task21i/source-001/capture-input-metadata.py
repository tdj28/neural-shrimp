from pathlib import Path
import json,hashlib,tarfile,io,re,subprocess,time,datetime,difflib
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');D=Path('/tmp/task21i-source-001');DATA=Path('/data2/neural-shrimp-data');m=json.loads((D/'source-and-archive-index.json').read_text());H=lambda b:hashlib.sha256(b).hexdigest();start=time.perf_counter();readbytes=0;files=[];sources=m['source_files'].copy()
def put(n,b):p=D/n;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_bytes(b);return dict(path=str(p),bytes=len(b),sha256=H(b))
def dump(n,q):return put(n,(json.dumps(q,indent=2)+'\n').encode())
def schema(q):
 if isinstance(q,dict):return {k:schema(v)for k,v in q.items()}
 if isinstance(q,list):return {'type':'list','length':len(q),'first':schema(q[0])if q else None}
 return type(q).__name__
def git(c,p):return subprocess.check_output(['git','-C',str(R),'show',c+':'+p],timeout=20)
# Historical analysis implementation is explicitly distinct from the BZ producer.
for c,repo in [('99ee6416','tools/behavior_zones.py')]:
 full=subprocess.check_output(['git','-C',str(R),'rev-parse',c]).decode().strip();b=git(full,repo);sources.append({'source_commit':full,'repository_path':repo,'role':'later_analysis_source_not_original_generator',**put('source/'+full[:8]+'/'+repo,b)})
 old=(D/'source/9113b64e/tools/behavior_zones.py').read_text();put('source/bz-analysis-addition.diff',''.join(difflib.unified_diff(old.splitlines(True),b.decode().splitlines(True),fromfile='original-BZ-generator',tofile='later-BZ-analyzer')).encode())
for c,paths in [('9113b64e73f6b4f9981ef45ac31fe5b6f2f02280',['tools/explore_llm_phase_map.py','tools/explore_llm_attractors.py']),('6a7c7ac1c1cbef76a93bc798d712bb2d5d965970',['tools/hidden_chaos.py','tools/explore_llm_soft_loop.py','tools/explore_llm_phase_map.py','tools/explore_llm_attractors.py','tools/behavior_zones.py'])]:
 for repo in paths:sources.append({'source_commit':c,'repository_path':repo,'role':'historical_called_dependency',**put('source/'+c[:8]+'/'+repo,git(c,repo))})
for a in m['selected_archives']:
 b=Path(a['archive']['path']).read_bytes();readbytes+=len(b);assert H(b)==a['archive']['sha256']
 with tarfile.open(fileobj=io.BytesIO(b))as tf:
  for member in a['members']:
   name=member['member']
   if not re.fullmatch(r'explore/llm-soft-loop/pod64/(bz/pilot-[012]/cell-\d{3}\.json|bp/(q05|q7)-(gen/generations\.json|label-[0-5]/labels\.jsonl))',name):continue
   raw=tf.extractfile(name).read();assert len(raw)==member['bytes'];readbytes+=len(raw);original=DATA/name;original_bytes=original.read_bytes();readbytes+=len(original_bytes);assert original_bytes==raw
   ref=put('inputs/'+name.split('pod64/',1)[1],raw);record={'attempt':a['id'],'source_commit':a['source_commit'],'archive_member':name,'original_path':str(original),**ref}
   if name.endswith('labels.jsonl'):
    rows=[json.loads(l)for l in raw.splitlines()];record['kind']='bp-label';record['metadata']={'rows':len(rows),'identifiers_in_file_order':[{'prompt':x['prompt'],'task':x['task'],'temperatures':[c['log10_temperature']for c in x['cells']]}for x in rows],'first_record_schema':schema(rows[0])if rows else None,'scientific_outcomes_interpreted':False}
   else:
    q=json.loads(raw)
    if '/bz/'in name:
     record['kind']='bz-cell';record['metadata']={'cell':q['cell'],'task_order':list(q['groups']),'groups':{k:{'rows':len(v['rows']),'row_key_order':list(v['rows'][0])if v['rows']else [],'item_sample_identities':[[x['item'],x['sample']]for x in v['rows']]}for k,v in q['groups'].items()},'first_group_schema':schema(next(iter(q['groups'].values())))if q['groups']else None,'scientific_outcomes_interpreted':False}
    else:
     record['kind']='bp-generation';record['metadata']={'greedy_prompt_order':list(q['greedy']),'sampled_temperature_order':list(q['sampled']),'sampled':{t:{'prompt_order':list(ps),'sample_counts':{k:len(v)for k,v in ps.items()},'first_sample_schema':schema(next(iter(ps.values()))[0])if ps else None}for t,ps in q['sampled'].items()},'greedy_first_schema':schema(next(iter(q['greedy'].values()))),'scientific_outcomes_interpreted':False}
   files.append(record)
assert len(files)==100 and sum(f['kind']=='bz-cell'for f in files)==90
published=[]
for name in ['bz-analysis.json','bp-q05.json','bp-q7.json']:
 p=DATA/'scratch'/name;b=p.read_bytes();readbytes+=len(b);published.append({'original_path':str(p),'association':'Named scratch summary in original research-log entry; exact original analysis command/process/source attestation not recorded.',**put('published/'+name,b),'schema':schema(json.loads(b))})
# Source-defined order disclosed separately from unknown historical CLI invocation.
analysis_order={'bz':{'runs':['llm-soft-loop/pod64/bz/pilot-0','llm-soft-loop/pod64/bz/pilot-1','llm-soft-loop/pod64/bz/pilot-2'],'within_run':'filename-sorted','duplicate_policy':'Original analyzer last assignment by cell ID; future own adapter must retain duplicates and refuse silent overwrite. Current original archive scopes are disjoint by declared shard.','tasks':['arithmetic','nq','mbpp'],'metrics':['d_acc','d_entropy','d_length'],'rng':'one default_rng(20261015) for all tasks/metrics; pairs sorted by pair ID','historical_cli_order_recorded':False},'bp':{'model_order':['q05','q7'],'label_runs':{'q05':['q05-label-0','q05-label-1'],'q7':['q7-label-0','q7-label-1','q7-label-2','q7-label-3','q7-label-4','q7-label-5']},'temperature_order':[0.25,0.32,0.4],'subset_order':['all','greedy_correct'],'rng':'analyze initializes one default_rng(20261015) per invocation/model; no reset across temperatures/subsets. Labels insertion follows supplied runs then file lines.','historical_cli_order_recorded':False},'scope':'Explicit canonical reproduction ordering from source and original command shards, not a fabricated original analyze invocation. Original permutation p-values may depend on missing invocation order; retain exact differences.','ad_hoc_adjustment':'Exact inline code/invocation not found in scoped original source/current tools/top-level scratch metadata; source log gives only acc~chaotic+generation entropy+task+greedy and rounded results. Cannot claim exact historical adjusted-estimator replay.'}
order=dump('analysis-order-and-limitations.json',analysis_order)
# Bind source log cards/claims verbatim; these are disclosed historical summaries, not independent outcomes.
commit=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD']).decode().strip();log=git(commit,'docs/research-log.md');ls=log.splitlines(True);claims=[]
for stamp in ['04:49','04:59','05:03','05:53','08:25','08:45','08:54','08:58','08:59','09:11','09:44']:
 for i,l in enumerate(ls):
  if l.startswith(('## 2026-10-15 '+stamp+' UTC:').encode()):
   j=next((j for j in range(i+1,len(ls))if ls[j].startswith(b'## ')),len(ls));claims.append({'source_commit':commit,'repository_path':'docs/research-log.md','full_source_sha256':H(log),'start_line':i+1,'end_line':j,**put('claims/'+stamp.replace(':','')+'.md',b''.join(ls[i:j]))})
for repo in ['docs/bpc-test.md','tools/bpc_analyze.py','tools/bpcs_analyze.py','tools/bpc_posthoc.py']:
 sources.append({'source_commit':commit,'repository_path':repo,'role':'exposed_BPC_BPCS_source_context_not_numeric_scope',**put('source/context/'+repo,git(commit,repo))})
sm=dump('source-map.json',{'schema':'task21i-source-map-v1','source_files':sources,'claims':claims,'analysis_order':order,'scope':'Exact historical original generator/dependencies and separately identified later/current analysis/context. No model/native precision requalification; no historical analysis invocation fabricated.'})
au=dump('original-authority.json',{'schema':'task21i-original-authority-v1','attempt_index':{'path':str(D/'original-attempt-index.json'),'bytes':(D/'original-attempt-index.json').stat().st_size,'sha256':H((D/'original-attempt-index.json').read_bytes())},'archive_index':{'path':str(D/'source-and-archive-index.json'),'bytes':(D/'source-and-archive-index.json').stat().st_size,'sha256':H((D/'source-and-archive-index.json').read_bytes())},'source_map':sm,'scope':'All21 original attempts retained.13 complete archives authenticated against recorded retrieval SHA and selected local member bytes; six initial q7 label failures have no archive and two later b attempts lack retrieval/termination. No remote queries, no invented historical process/source/model loaded-byte attestation.'})
inp=dump('inputs.json',{'schema':'task21i-saved-inputs-v1','status':'metadata_preparation_no_target_run_admission','files':files,'published':published,'authority':au,'source_map':sm,'analysis_order':order,'target_files':len(files),'target_bytes':sum(f['bytes']for f in files),'metadata_only_scope':'Only source definitions, JSON schema, source cell/task/prompt/sample identities/order/counts, original commands/receipts and bytes inspected. No accuracy/entropy/LE/period/class outcomes summarized or used for decisions.'})
rows=[{'path':p.relative_to(D).as_posix(),'bytes':p.stat().st_size,'sha256':H(p.read_bytes())}for p in sorted(D.rglob('*'))if p.is_file()];man=dump('manifest.json',{'schema':'task21i-source-input-packet-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'inputs':inp,'source_map':sm,'authority':au,'files':rows,'packet_bytes':sum(r['bytes']for r in rows),'selected_io_bytes':readbytes,'seconds':time.perf_counter()-start,'scope':'Metadata-only capture; no model/generated-code execution or statistical calculation. Original archive reads and selected member/local equality are included in IO; Git source reads excluded.'})
print(json.dumps({'manifest':man,'inputs':inp,'source_map':sm,'authority':au,'counts':{kind:sum(f['kind']==kind for f in files)for kind in ['bz-cell','bp-label','bp-generation']},'target_bytes':sum(f['bytes']for f in files),'selected_io_bytes':readbytes,'seconds':time.perf_counter()-start},indent=2))
