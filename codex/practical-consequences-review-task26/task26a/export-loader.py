from pathlib import Path
import ast,subprocess,json,hashlib,datetime
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');O=Path('/tmp/task26a-loader-supplement-001');O.mkdir();ref='a9c411ff1bf992a20f595039000c55be15c6d12c';rows=[]
def put(name,b,meta):
 p=O/name;p.write_bytes(b);rows.append(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),**meta))
for f,c,m in [('tools/explore_llm_soft_loop.py','SoftLoop','__init__'),('tools/explore_llm_phase_map.py','SteeredModel','__init__'),('tools/explore_llm_attractors.py','Engine','__init__')]:
 b=subprocess.check_output(['git','show',ref+':'+f],cwd=R);s=b.decode();tree=ast.parse(s);cl=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==c);method=next(x for x in cl.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==m)
 start=method.lineno;end=method.end_lineno;data=''.join(s.splitlines(keepends=True)[start-1:end]).encode();put(c+'-initializer.py',data,{'original_path':f,'original_commit':ref,'original_file_sha256':hashlib.sha256(b).hexdigest(),'first_line':start,'last_line':end,'exact_source_excerpt':True})
note='''# Task 26(a) source-only loader supplement

These exact original initializer excerpts bind the call chain repro_task.load -> SoftLoop -> SteeredModel -> Engine at the original REPRO-T source. Full original precision-helper source remains in the prior source packet. The loader defines state precision and the embedding table separately from the later bfloat16 cast; this source review is not a new model execution or loaded-weight attestation.

No separate saved inline bootstrap/sign-flip analysis script or exact producer RNG/quantile/p-value implementation was located in the tracked source-name inventory. The frozen design supplies seed 0 and 10,000 draws; the independently declared convention is prospective and is not an exact producer-script replay. No outcomes were used to calibrate that convention.

The source-bound 07:05 log says the CPU smoke test used two development problems and one held-out problem at four soft steps, but does not identify that held-out row. Its index is unavailable in this compact source record; freshness should retain this limit. No smoke outcome files were opened, and no target data were decoded.
'''
put('scope.md',note.encode(),{'source_only':True})
manifest={'schema':'task26a-loader-source-supplement-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_commit':ref,'files':rows,'original_source_packet_manifest_sha256':'008c566befb4826479dafaa1671cae8a8f74d48290e5196cb1055dd922a10640','no_target_decoding':True}
p=O/'manifest.json';p.write_text(json.dumps(manifest,indent=2)+'\n');print(str(p),hashlib.sha256(p.read_bytes()).hexdigest());print(json.dumps(rows,indent=2))
