from pathlib import Path
import ast,subprocess,json,hashlib,datetime
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');O=Path('/tmp/task26b-loop-supplement-001');O.mkdir();ref='dbd125a70dd1b1a1f54a12c1ff2fde275b4c7a49';rows=[]
for f,c,m in [('tools/explore_llm_soft_loop.py','SoftLoop',None),('tools/explore_llm_phase_map.py','SteeredModel','__init__'),('tools/explore_llm_attractors.py','Engine','__init__'),('tools/explore_llm_attractors.py',None,'keep_float64_norms')]:
 b=subprocess.check_output(['git','show',ref+':'+f],cwd=R);s=b.decode();tree=ast.parse(s)
 if c:
  n=next(x for x in tree.body if isinstance(x,ast.ClassDef)and x.name==c)
  if m:n=next(x for x in n.body if isinstance(x,ast.FunctionDef)and x.name==m)
 else:n=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name==m)
 name=(c or m)+('-initializer'if c and m else'')+'.py';data=''.join(s.splitlines(keepends=True)[n.lineno-1:n.end_lineno]).encode();p=O/name;p.write_bytes(data);rows.append(dict(path=str(p),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),original_path=f,original_commit=ref,original_file_sha256=hashlib.sha256(b).hexdigest(),first_line=n.lineno,last_line=n.end_lineno,exact_source_excerpt=True))
p=O/'manifest.json';p.write_text(json.dumps({'schema':'task26b-loop-source-supplement-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_commit':ref,'files':rows,'source_only':True,'no_target_reads_or_execution':True},indent=2)+'\n');print(str(p),hashlib.sha256(p.read_bytes()).hexdigest())
