import importlib.util,json,hashlib,tempfile,struct,ast
from pathlib import Path
from unittest.mock import patch
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');p=R/'codex/coexistence-task33/audit_thin_pilot_002.py'
spec=importlib.util.spec_from_file_location('audit002',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rows=[]
def ok(name,fn):fn();rows.append({'control':name,'passed':True})
def refuse(name,fn):
 try:fn()
 except (ValueError,FileNotFoundError,RuntimeError):rows.append({'control':name,'passed':True});return
 raise AssertionError(name+' did not refuse')
with tempfile.TemporaryDirectory(prefix='thin-audit002-')as td:
 d=Path(td);canonical=d/'canonical';mid=d/'mid';snapshot=d/'snapshot'
 header=json.dumps({'model.embed_tokens.weight':{'dtype':'BF16','shape':[2,2],'data_offsets':[0,8]}}).encode()
 canonical.write_bytes(struct.pack('<Q',len(header))+header+struct.pack('<4H',0,0,0x3f80,0xbf80));mid.symlink_to('canonical');snapshot.symlink_to('mid')
 sha=hashlib.sha256(canonical.read_bytes()).hexdigest()
 with patch.multiple(m,ASSET=snapshot,ASSET_RESOLVED=canonical,ASSET_LINKS=((snapshot,'mid'),(mid,'canonical')),ASSET_SHA=sha):
  def valid():
   destination,record=m.checkpoint_authority(snapshot);assert destination==canonical and len(record['links'])==2
   values,authority=m.read_bf16_row(destination,sha,(2,2),1);assert values==[1,-1] and authority['asset']==m.pin(canonical)
  ok('literal two-link chain and exact BF16 row accepted',valid)
  refuse('different requested snapshot refused',lambda:m.checkpoint_authority(mid))
  with patch.object(m,'ASSET_LINKS',((snapshot,'mid'),)):
   refuse('one-link metadata refused',lambda:m.checkpoint_authority(snapshot))
  with patch.object(m,'ASSET_LINKS',((snapshot,'wrong'),(mid,'canonical'))):
   refuse('wrong link target refused',lambda:m.checkpoint_authority(snapshot))
  with patch.object(m,'ASSET_RESOLVED',d/'other'):
   refuse('different canonical destination refused',lambda:m.checkpoint_authority(snapshot))
  mid.unlink();mid.symlink_to('missing')
  refuse('changed/dangling cache link refused',lambda:m.checkpoint_authority(snapshot))
  mid.unlink();mid.write_bytes(b'not a link')
  refuse('regular replacement of cache link refused',lambda:m.checkpoint_authority(snapshot))
  mid.unlink();mid.symlink_to('canonical')
  refuse('generic evidence symlink remains refused',lambda:m.pin(snapshot))
  refuse('wrong canonical content hash refused',lambda:m.read_bf16_row(canonical,'0'*64,(2,2),1))
  alias=d/'alias';alias.symlink_to('canonical')
  with patch.object(m,'ASSET_RESOLVED',alias):
   refuse('canonical symlink destination refused',lambda:m.checkpoint_authority(snapshot))
  assert m.RESULT.name=='result.json' and m.RESULT.parent.name=='thin-pilot-001-audit-002';rows.append({'control':'fresh audit002 allocation differs from preserved001','passed':True})
# Changed orchestration only; original arithmetic and authority functions exactly match.
original=ast.parse((p.parent/'audit_thin_pilot.py').read_text());new=ast.parse(p.read_text())
f=lambda tree:{x.name:ast.dump(x,include_attributes=False)for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))}
a,b=f(original),f(new);changed=[name for name in a if a[name]!=b[name]]
assert changed==['argument_admission','run'],changed
rows.append({'control':'only declared allocation/run functions differ; all numeric/authority helpers unchanged','passed':True})
result={'source':{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()},'controls':rows,'count':len(rows),'target_evidence_reads':False,'model_calls':0,'scope':'New exact-cache-chain adapter controls only; prior suites not repeated.'}
Path('/tmp/task33-audit002-link-controls.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':len(rows),'changed_original_functions':changed,'source_sha256':result['source']['sha256']}))
