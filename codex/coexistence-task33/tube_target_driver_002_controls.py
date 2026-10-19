"""New audit002/checkpoint-join controls only. All target I/O mocked."""
import ast,contextlib,copy,hashlib,importlib.util,json
from fractions import Fraction as F
from pathlib import Path
from unittest.mock import patch
P=Path('/tmp/tube_target_driver_002.py')
expected='2c8e6ae4cb86b5d06b4e6a534d2e6f8a518d19f55fcab92b9b5d0b25210befc9'
assert hashlib.sha256(P.read_bytes()).hexdigest()==expected
spec=importlib.util.spec_from_file_location('tube_driver_002_tested',P)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
# Reuse only fixture-builder definitions, never the original66 control actions.
old=Path('/tmp/tube_target_driver_controls.py').read_text();tree=ast.parse(old)
parts=[ast.get_source_segment(old,n)for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))and n.name in ('native','Fixture')]
fixture='\n\n'.join(parts)
fixture=fixture.replace("'b'*40","m.AUDIT_SOURCE")
fixture=fixture.replace("m.D/'thin-audit-fixture-card.json'","m.AUDIT_CARD").replace("m.D/'thin-audit-fixture-plan.json'","m.AUDIT_PLAN")
fixture=fixture.replace('ac,None,argv','ac,m.AUDIT_CARD_SHA,argv')
fixture=fixture.replace('audit_thin_pilot.py','audit_thin_pilot_002.py')
fixture=fixture.replace("Path('/synthetic/model.safetensors')","m.CHECKPOINT_RESOLVED")
fixture=fixture.replace("tensor='model.embed_tokens.weight',", "snapshot_authority=m.checkpoint_definition(),tensor='model.embed_tokens.weight',")
fixture=fixture.replace("path=str(path);p=dict(","path=str(path);sha=m.AUDIT_PLAN_SHA if path==str(m.AUDIT_PLAN) else sha;p=dict(")
exec(compile(fixture,'manufactured-fixture-definitions-only','exec'),globals())
rows=[]
def checked(name,fn):fn();rows.append(dict(name=name,passed=True))
def refused(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('expected audit002 delta refusal absent')
def positive_candidate():
    f=Fixture()
    with f.mocked():_,_,_,selection=m.candidate(f.card)
    assert selection['audit_result']['path']==str(m.AUDIT_RESULT)
    assert selection['embedding']['snapshot_authority']==m.checkpoint_definition()
    assert selection['embedding']['asset']['path']==str(m.CHECKPOINT_RESOLVED)
checked('actual candidate admits exact audit002 source/card/plan/entry/result with canonical row authority',positive_candidate)
def candidate_bad(name,change):
    def run():
        f=Fixture();change(f)
        with f.mocked():refused(lambda:m.candidate(f.card))
    checked(name,run)
candidate_bad('audit001 source cannot substitute for explicit original002 commit',lambda f:f.audit.__setitem__('source_commit','b'*40))
candidate_bad('audit001 card path cannot substitute',lambda f:f.audit['card'].__setitem__('path',str(m.D/'thin-pilot-001-audit-card.json')))
candidate_bad('changed audit002 card hash refused',lambda f:f.audit['card'].__setitem__('sha256','0'*64))
candidate_bad('audit001 plan path cannot substitute',lambda f:f.audit['plan'].__setitem__('path',str(m.D/'thin-pilot-001-audit-plan.json')))
candidate_bad('changed audit002 plan hash refused',lambda f:f.audit['plan'].__setitem__('sha256','0'*64))
def old_argv(f):
    command=list(f.audit['argv']);command[3]=str(m.D/'audit_thin_pilot.py')
    f.audit['argv']=command;f.rc(f.audit)['argv']=command;f.ad(f.audit)['argv']=command
    f.plan(f.audit)['lanes']['audit']['entry']=command[3]
candidate_bad('synchronized original001 executable argv still refused by002 candidate',old_argv)
candidate_bad('audit002 report requires snapshot authority field',lambda f:f.report_doc()['initial_state']['embedding'].__delitem__('snapshot_authority'))
candidate_bad('audited original snapshot alias cannot masquerade as canonical asset pin',lambda f:f.report_doc()['initial_state']['embedding']['asset'].__setitem__('path',str(m.CHECKPOINT_SNAPSHOT)))
candidate_bad('audited declared link target cannot change',lambda f:f.report_doc()['initial_state']['embedding']['snapshot_authority']['links'][1].__setitem__('link_target','other'))
candidate_bad('audited canonical destination cannot change',lambda f:f.report_doc()['initial_state']['embedding']['snapshot_authority'].__setitem__('resolved_path','/tmp/other'))
candidate_bad('audited expected full content hash cannot change',lambda f:f.report_doc()['initial_state']['embedding']['snapshot_authority'].__setitem__('expected_content_sha256','0'*64))

def binding_fixture():
    asset=dict(path=str(m.CHECKPOINT_RESOLVED),bytes=100,sha256=m.ASSETS['model.safetensors'])
    embedding=dict(asset=asset,snapshot_authority=m.checkpoint_definition())
    return {str(m.CHECKPOINT_RESOLVED):copy.deepcopy(asset)},embedding

def binding_case(change=None,bad=False,change_after_hash=False):
    inputs,embedding=binding_fixture()
    if change:change(inputs,embedding)
    responses=[(m.CHECKPOINT_RESOLVED,m.checkpoint_definition()),(m.CHECKPOINT_RESOLVED,m.checkpoint_definition())]
    if change_after_hash:responses[1][1]['links'][0]['link_target']='changed after hash'
    calls=[]
    def verified(record):calls.append(copy.deepcopy(record));return copy.deepcopy(record)
    with patch.object(m,'checkpoint_authority',side_effect=responses)as authority,patch.object(m,'verify',side_effect=verified), \
         patch.object(Path,'open',side_effect=AssertionError('target filesystem access forbidden')):
        call=lambda:m.checkpoint_binding(inputs,embedding,m.CHECKPOINT_SNAPSHOT)
        if bad:refused(call)
        else:
            result=call();assert result==dict(asset=embedding['asset'],snapshot_authority=m.checkpoint_definition())
            assert calls==[embedding['asset']]
            assert authority.call_args_list==[((m.CHECKPOINT_SNAPSHOT,),),((m.CHECKPOINT_SNAPSHOT,),)]
checked('new loader join verifies full canonical content and rechecks declared chain',binding_case)
checked('new loader join refuses absent canonical preflight input',lambda:binding_case(lambda i,e:i.clear(),True))
checked('new loader join refuses differing row asset byte count',lambda:binding_case(lambda i,e:e['asset'].__setitem__('bytes',101),True))
checked('new loader join refuses undeclared audit chain',lambda:binding_case(lambda i,e:e.__setitem__('snapshot_authority',{}),True))
checked('new loader join refuses alias path in audited asset',lambda:binding_case(lambda i,e:e['asset'].__setitem__('path',str(m.CHECKPOINT_SNAPSHOT)),True))
def wrong_content(i,e):
    e['asset']['sha256']='0'*64;i[str(m.CHECKPOINT_RESOLVED)]['sha256']='0'*64
checked('new loader join refuses synchronized wrong full content hash',lambda:binding_case(wrong_content,True))
checked('new loader join catches chain change during full content hashing',lambda:binding_case(bad=True,change_after_hash=True))

# Source equality checks do not repeat runtime operator/receiver/audit controls.
R=m.R
original=(R/'codex/coexistence-task33/tube_target_driver.py').read_text()
new=P.read_text()
def definition(text,name):
    n=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name==name)
    return ast.dump(n,include_attributes=False)
for name in ('pin','verify','source_pin','original_execution','current_admission','bounds','vector','prefix_authority'):
    checked('original '+name+' semantics preserved verbatim in AST',lambda name=name:(_ for _ in ()).throw(AssertionError(name))if definition(original,name)!=definition(new,name)else None)
checked('explicit installed002 filename retained',lambda:None if m.SELF==m.D/'tube_target_driver_002.py'else(_ for _ in ()).throw(AssertionError('wrong SELF')))
# The auditor's exact chain constants are read from source AST only; no import/read of raw.
aud=(R/'codex/coexistence-task33/audit_thin_pilot_002.py').read_text()
assignments={}
for node in ast.parse(aud).body:
    if isinstance(node,ast.Assign)and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)and node.targets[0].id in ('ASSET','ASSET_SHA','ASSET_RESOLVED','ASSET_LINKS'):
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'auditor-literal-assignments','exec'),{'Path':Path,**assignments},assignments)
checked('new driver exactly reproduces independently reviewed auditor002 snapshot chain literals',lambda:None if (m.CHECKPOINT_SNAPSHOT,m.CHECKPOINT_RESOLVED,m.CHECKPOINT_LINKS,m.ASSETS['model.safetensors'])==(assignments['ASSET'],assignments['ASSET_RESOLVED'],assignments['ASSET_LINKS'],assignments['ASSET_SHA'])else(_ for _ in ()).throw(AssertionError('chain literal mismatch')))
# Check that both pre-evaluation and post-evaluation bindings remain in main.
main=next(n for n in ast.parse(new).body if isinstance(n,ast.FunctionDef)and n.name=='main')
calls=[n for n in ast.walk(main)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='checkpoint_binding']
checked('main binds canonical checkpoint before loading and again after evaluation',lambda:None if len(calls)==2 else(_ for _ in ()).throw(AssertionError('missing pre/post binding')))

# Configuration has its own newly disclosed one-link mapping; no generic exception.
def config_case(change=None,bad=False,path=None,target=None,resolved=None,destination_link=False,mutate_after=False):
    asset=dict(path=str(m.CONFIG_RESOLVED),bytes=659,sha256=m.ASSETS['config.json'])
    inputs={str(m.CONFIG_RESOLVED):asset}
    if change:change(inputs)
    hashes=[];link_reads=[]
    def symlink(p):return p==m.CONFIG_SNAPSHOT or (destination_link and p==m.CONFIG_RESOLVED)
    def readlink(p):
        assert p==m.CONFIG_SNAPSHOT;link_reads.append(p)
        if mutate_after and len(link_reads)>1:return Path('changed-after-hash')
        return Path(target or m.CONFIG_LINK_TARGET)
    def resolve(p,strict=False):
        assert strict and p in (m.CONFIG_SNAPSHOT,m.CONFIG_RESOLVED)
        return resolved or m.CONFIG_RESOLVED
    def verify(p):hashes.append(copy.deepcopy(p));return copy.deepcopy(p)
    with patch.object(Path,'is_symlink',autospec=True,side_effect=symlink), \
         patch.object(Path,'readlink',autospec=True,side_effect=readlink), \
         patch.object(Path,'resolve',autospec=True,side_effect=resolve), \
         patch.object(Path,'is_file',autospec=True,side_effect=lambda p:p==m.CONFIG_RESOLVED), \
         patch.object(Path,'open',side_effect=AssertionError('target filesystem access forbidden')), \
         patch.object(m,'verify',side_effect=verify):
        call=lambda:m.config_binding(inputs,path or m.CONFIG_SNAPSHOT)
        if bad:refused(call)
        else:
            result=call();assert result['asset']==asset and len(hashes)==1 and len(link_reads)==2
            assert result['snapshot_authority']==dict(snapshot_path=str(m.CONFIG_SNAPSHOT),
                links=[dict(path=str(m.CONFIG_SNAPSHOT),link_target=m.CONFIG_LINK_TARGET)],
                resolved_path=str(m.CONFIG_RESOLVED),expected_bytes=659,expected_content_sha256=m.ASSETS['config.json'])
checked('explicit one-link config mapping hashes canonical659-byte content and rechecks mapping',config_case)
checked('other configuration alias refused',lambda:config_case(path=Path('/tmp/config.json'),bad=True))
checked('alternate config link target refused',lambda:config_case(target='other',bad=True))
checked('other config canonical resolution refused',lambda:config_case(resolved=Path('/tmp/other-config.json'),bad=True))
checked('config canonical destination cannot itself be a symlink',lambda:config_case(destination_link=True,bad=True))
checked('config canonical input missing refused',lambda:config_case(change=lambda i:i.clear(),bad=True))
checked('config full file byte count bound659 required',lambda:config_case(change=lambda i:i[str(m.CONFIG_RESOLVED)].__setitem__('bytes',660),bad=True))
checked('config exact original full content hash required',lambda:config_case(change=lambda i:i[str(m.CONFIG_RESOLVED)].__setitem__('sha256','0'*64),bad=True))
checked('config mapping mutation after content hashing refused',lambda:config_case(mutate_after=True,bad=True))
config_calls=[n for n in ast.walk(main)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='config_binding']
checked('main binds canonical config before loading and again after evaluation',lambda:None if len(config_calls)==2 else(_ for _ in ()).throw(AssertionError('missing pre/post config binding')))

print(json.dumps(dict(schema='codex-task33-tube-driver002-delta-controls-v1',source_sha256=expected,count=len(rows),passed=True,controls=rows,
 scope='Only new002 source/path/checkpoint joins and unchanged-source assertions. All evidence synthetic and I/O mocked; prior66+26 controls not executed. No actual target/raw/model/process/Git actions; runtime admission pending.'),indent=2))
