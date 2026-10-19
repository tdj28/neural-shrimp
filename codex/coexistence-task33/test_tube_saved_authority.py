"""New mocked integration controls; no original raw/model/asset/backend reads.

Only the four frozen standard-library helpers are loaded. Scientific helpers are
mocked at the orchestration seam; their old controls are not repeated. New cache
and allocation checks use exclusively fresh tiny temporary files.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location('tube_saved_authority',Path(__file__).with_name('tube_saved_authority.py'))
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


def pin(path,payload=b'{}'):
    return dict(path=str(path),bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())


def arguments(out=None):
    return argparse.Namespace(card=str(M.CARD),card_sha256='a'*64,out=str(M.OUT if out is None else out))


def upstream_fixture(execution):
    card_pin=pin(M.D/'synthetic-card.json'); plan_pin=pin(M.D/'synthetic-plan.json')
    root=str(M.B/'synthetic-upstream'); command=[M.PYTHON,'-I','-B',str(M.D/'synthetic.py')]
    authority=dict(source_commit='1'*40,card=card_pin,plan=plan_pin,receipt=pin(root+'-supervision/receipt.json'),
                   admission=pin(root+'-supervision/admission.json'),lane='synthetic',output=root,argv=command)
    input_pin=pin('/tmp/synthetic-input.json'); output_pin=pin(root+'/result.json')
    limits=dict(as_bytes=100,cpu_seconds=10,wall_seconds=20,output_bytes=100)
    card=dict(inputs=[input_pin]); lane=dict(entry=command[-1],arguments=[],output=root,supervision=root+'-supervision',limits=limits,preflight_pins=[input_pin])
    plan=dict(status='reviewed_for_execution',repository=str(M.ROOT),raw_root=str(M.B),python=M.PYTHON,cpu_threads=1,
              lanes={'synthetic':lane},source_files=[card_pin])
    receipt=dict(source_commit='1'*40,lane='synthetic',argv=command,returncode=0,failure=None,worker_reaped=True,
                 worker_identity_absent=True,worker_group_session_absent=True,output_files=[output_pin],actual_output_bytes=2)
    admission=dict(source_commit='1'*40,lane='synthetic',argv=command,plan=plan_pin,limits=limits,
                   threads={key:'1' for key in execution.THREADS},source_blobs=[dict(relative_path=str(Path(card_pin['path']).relative_to(M.ROOT)),bytes=2,sha256=card_pin['sha256'])])
    values={authority[key]['path']:value for key,value in zip(('card','plan','receipt','admission'),(card,plan,receipt,admission))}
    io=mock.Mock(); io.e=execution; io.read.side_effect=lambda p,decode=False:copy.deepcopy(values[p['path']])
    io.inventory.return_value={output_pin['path']:output_pin}
    return io,authority,values


def integration_fixture():
    """Deliberately mocked authenticated stages; this is no numerical fixture."""
    events=[]; expected=dict(source_commit=M.ORIGINAL_SOURCE,source_files=[])
    for key in ('card','plan','receipt','admission','worker_startup','host_closure'): expected[key]=pin('/tmp/synthetic-'+key+'.json')
    expected['card']['path']=str(M.D/'tube-target-001-card.json')
    command=[M.PYTHON,'-I','-B',str(M.D/'tube_target_driver_002.py'),'--card',expected['card']['path'],
             '--card-sha256',expected['card']['sha256'],'--out',str(M.TARGET)]
    identity=dict(pid=9,starttime=99,pgid=9,sid=9)
    receipt=dict(source_commit=M.ORIGINAL_SOURCE,lane='tube',argv=command,returncode=0,failure=None,
                 worker_reaped=True,worker_identity_absent=True,worker_group_session_absent=True,output_files=[])
    admission=dict(source_commit=M.ORIGINAL_SOURCE,lane='tube',argv=command,supervisor=identity,source_blobs=[])
    card=dict(audit_execution={'synthetic':'audit'},thin_execution={'synthetic':'thin'},prefix_execution={'synthetic':'prefix'})
    plan=dict(lanes={'tube':{'preflight_pins':[]}}); closure=dict(preflight_pins_verified=[],source_blobs_verified=[])
    audit_card=dict(original_expected=pin('/tmp/synthetic-expected.json'),original_closure=pin('/tmp/synthetic-closure.json'))
    values={audit_card['original_expected']['path']:expected,audit_card['original_closure']['path']:closure,
            expected['card']['path']:card,expected['plan']['path']:plan,expected['receipt']['path']:receipt,
            expected['admission']['path']:admission,expected['worker_startup']['path']:{'identity':identity},expected['host_closure']['path']:{}}
    io=mock.Mock(); io.pins={}; io.git_checks=[]
    io.read.side_effect=lambda p,decode=False:copy.deepcopy(values[p['path']])
    io.e.pins.side_effect=lambda rows:{row['path']:row for row in rows}
    io.inventory.return_value={}; io.verify_pins.side_effect=lambda rows:events.append('preflight')
    files=mock.Mock(); files.read_flat_output_inventory.return_value=dict(records={'admission.json':{},'status.json':{}},pins={})
    execution=mock.Mock(); execution.pins.side_effect=io.e.pins.side_effect; execution.identity.side_effect=lambda v:v
    execution.check_target_execution.side_effect=lambda *a:(events.append('target') or {'reconstructed_binding':{}})
    assets_result={'assets':{'model.safetensors':{'asset':pin('/tmp/synthetic-model'),'snapshot_authority':{'synthetic':True}}}}
    io.assets.side_effect=lambda:(events.append('assets') or copy.deepcopy(assets_result))
    assets=mock.Mock(); assets.check_target_assets.side_effect=lambda:(events.append('assets-final') or copy.deepcopy(assets_result))
    scalar=mock.Mock(); scalar.check_bundle.side_effect=lambda *a:(events.append('scalar') or {'L_candidate':{'numerator':3,'denominator':2},'scope':'old helper scope'})
    io.recheck.side_effect=lambda:events.append('recheck')
    upstream=dict(initial={},step={},report={},upstream_pins={},embedding=copy.deepcopy(assets_result['assets']['model.safetensors']))
    patches=dict(source_join=lambda *a:events.append('source'),host_join=lambda *a:events.append('host'),
                 join_upstream=lambda *a:(events.append('upstream') or upstream),join_prefix=lambda *a:(events.append('prefix') or {}))
    return io,(files,execution,assets,scalar),audit_card,values,patches,events


def admission_fixture(execution):
    card_pin=dict(pin(M.CARD),sha256='a'*64); plan_pin=pin(M.PLAN)
    host_pin=pin(M.D/'tube-target-001-host-closure.json')
    card=dict(schema='codex-task33-tube-saved-audit-card-v1',status='reviewed_for_saved_audit',repository=str(M.ROOT),output=str(M.OUT),
              original_expected=copy.deepcopy(M.EXPECTED_PIN),original_closure=copy.deepcopy(M.CLOSURE_PIN),helper_sha256=copy.deepcopy(M.HELPERS),
              supervision_plan=str(M.PLAN),supervision_admission=str(M.SUP/'admission.json'),auditor_sha256='a'*64,
              limits=copy.deepcopy(M.LIMITS),cpu_ids=[0],**M.FLAGS)
    sources=[card_pin,dict(pin(M.SELF),sha256='a'*64),
             dict(pin(M.ROOT/'codex/certify-chaos/supervise.py'),sha256=execution.SOURCE_HASHES['codex/certify-chaos/supervise.py']),
             *[dict(pin(M.D/name),sha256=digest) for name,digest in M.HELPERS.items()],
             copy.deepcopy(M.EXPECTED_PIN),copy.deepcopy(M.CLOSURE_PIN),host_pin]
    argv=[M.PYTHON,'-I','-B',str(M.SELF),'--card',str(M.CARD),'--card-sha256',card_pin['sha256'],'--out',str(M.OUT)]
    plan=dict(status='reviewed_for_execution',repository=str(M.ROOT),raw_root=str(M.B),python=M.PYTHON,cpu_threads=1,cpu_ids=[0],startup_ram_gib=24,
              source_files=sources,lanes={'audit':dict(entry=str(M.SELF),arguments=argv[4:],output=str(M.OUT),supervision=str(M.SUP),
              limits=copy.deepcopy(M.LIMITS),preflight_pins=[copy.deepcopy(M.EXPECTED_PIN),copy.deepcopy(M.CLOSURE_PIN)])})
    admission=dict(source_commit='1'*40,plan=plan_pin,lane='audit',argv=argv,limits=copy.deepcopy(M.LIMITS),
                   threads={key:'1' for key in execution.THREADS},capacity=dict(available_ram_bytes=24*2**30,disk_free_bytes=16*2**30),
                   source_blobs=[dict(relative_path=str(Path(row['path']).relative_to(M.ROOT)),bytes=row['bytes'],sha256=row['sha256']) for row in sources])
    io=mock.Mock();io.e=execution;io.current.return_value=(pin(M.SUP/'admission.json'),admission)
    def read(record,decode=False):
        if record['path']==str(M.PLAN):return plan
        if record['path']==M.EXPECTED_PIN['path']:return {'host_closure':host_pin}
        return b'synthetic pinned bytes'
    io.read.side_effect=read
    return io,card_pin,card,plan,admission


def prefix_fixture(execution,assets_helper):
    defs=assets_helper._target_definitions(); assets={}
    for name,value in defs.items():
        authority=None
        if value['links']:
            authority=dict(snapshot_path=value['snapshot_path'],links=copy.deepcopy(value['links']),resolved_path=value['asset']['path'],expected_content_sha256=value['asset']['sha256'])
            if name=='config.json':authority['expected_bytes']=value['asset']['bytes']
        assets[name]=dict(asset=copy.deepcopy(value['asset']),snapshot_authority=authority)
    prefix=execution.PREFIX
    authority=dict(source_commit=prefix['prefix_source_commit'],lane='prefix',output=prefix['prefix_output'],
        receipt=dict(pin(prefix['prefix_receipt']),sha256=prefix['prefix_receipt_sha256']),
        admission=dict(pin(prefix['prefix_admission_path']),sha256=prefix['prefix_admission_sha256']),
        card=dict(pin(M.D/'prefix-card.json'),sha256='6c2ad90ef0332733f8f249833ec9d2adcd0799b1ec6c4dc415202df0b78e851c'),
        plan=pin(M.D/'prefix-plan.json'),argv=[M.PYTHON,'-I','-B',str(M.D/'prefix_setup.py'),'--card',str(M.D/'prefix-card.json'),'--out',prefix['prefix_output']])
    map_value=dict(schema='codex-qwen-exact-soft-map-v1',dimension=896,vocabulary=151936,layers=24,
        pinned_assets={name:value['asset']['sha256'] for name,value in assets.items()},prefix_ids=list(M.PREFIX_IDS),steering_layer=12,
        alpha_ratio=list((.06468).as_integer_ratio()),log10_temperature_ratio=list((-.148).as_integer_ratio()),
        temperature='exp(log(10)*exact_dyadic_log10T)',rotary='real base^(-2j/head_dim), real sin/cos',
        state='one appended full-dimensional soft embedding after the fixed prefix',readout='full-vocabulary expected embedding with the checkpoint embedding matrix',
        checkpoint_constants='stored binary constants interpreted as exact dyadics')
    backend=dict(precision=2048,threads=1,python_flint='synthetic-python-flint',flint='synthetic-flint')
    card=dict(prefix_execution=authority,map=map_value,map_sha256=execution.canonical_hash(map_value),backend=backend,backend_sha256=execution.canonical_hash(backend),
              regularity_review=pin(M.D/'tube-target-001-regularity-review.md'),input_files=[copy.deepcopy(M.CACHE_PIN),copy.deepcopy(M.PREFIX_INPUT_PIN),
                  copy.deepcopy(authority['receipt']),copy.deepcopy(authority['admission']),*[copy.deepcopy(value['asset']) for value in assets.values()]])
    value=dict(prefix=copy.deepcopy(prefix),prefix_files={'inputs.json':M.PREFIX_INPUT_PIN['sha256'],'prefix-cache.json.gz':M.CACHE_PIN['sha256']},
        original_prefix_execution=copy.deepcopy(authority),checkpoint=copy.deepcopy(assets['model.safetensors']),config=copy.deepcopy(assets['config.json']),
        stored_precision=2048,receiver_precision=2048,original_radii_retained=True,map=copy.deepcopy(map_value),map_sha256=card['map_sha256'],
        backend=copy.deepcopy(backend),backend_sha256=card['backend_sha256'],regularity_contract=execution.REGULARITY_CONTRACT,
        regularity_review=copy.deepcopy(card['regularity_review']),**execution.SAVED_FLAGS)
    meta=dict(source_commit=authority['source_commit'],pinned_assets=copy.deepcopy(map_value['pinned_assets']),prefix_ids=list(M.PREFIX_IDS),precision=2048,steering_layer=12,
              versions={'python_flint':backend['python_flint'],'flint':backend['flint']},
              **{key:copy.deepcopy(map_value[key]) for key in ('alpha_ratio','log10_temperature_ratio','temperature','rotary')})
    report=dict(completed=True,precision=2048,layers=list(range(24)))
    upstream=dict(card=dict(alpha=.06468,log10T=-.148,precision=2048),files={row['path']:row for row in
        [copy.deepcopy(M.CACHE_PIN),copy.deepcopy(M.PREFIX_INPUT_PIN),pin(M.B/'prefix-001/layers.jsonl'),pin(M.B/'prefix-001/report.json')]},
        sources={str(M.D/'prefix_setup.py'):dict(sha256='72a95c042b4d5e229264b07886733b23c385f37b0dd95009dfa067935877fbc3'),
                 str(M.ROOT/'codex/certify-llm/ball_qwen.py'):dict(sha256=prefix['prefix_map_source_sha256'])})
    io=mock.Mock();io.e=execution;io.cache_observation={'synthetic_streamed_cache':True}
    io.read.side_effect=lambda record,decode=False:meta if record['path']==M.PREFIX_INPUT_PIN['path'] else report
    return io,card,{'prefix-binding.json':value},{'assets':assets},upstream,meta,report


class AuthorityControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files,cls.execution,cls.assets,cls.scalar=M.load_helpers()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='task33-authority-synthetic-'); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_01_pure_argument_refusal_precedes_all_io(self):
        for field,value in (('card','/tmp/other'),('out','/tmp/other'),('card_sha256','A'*64)):
            args=arguments(); setattr(args,field,value)
            with mock.patch.object(M,'_allocate',side_effect=AssertionError('must not allocate')):
                with self.assertRaises(ValueError): M.execute(args)

    def test_02_fresh_directory_refuses_existing_directory_file_or_link(self):
        out=self.root/'audit'
        with mock.patch.object(M,'OUT',out):
            out.mkdir()
            with self.assertRaises(FileExistsError): M._allocate()
            out.rmdir(); out.write_bytes(b'old')
            with self.assertRaises(FileExistsError): M._allocate()
            out.unlink(); out.symlink_to(self.root/'missing')
            with self.assertRaises(FileExistsError): M._allocate()

    def test_03_helper_load_failure_is_preserved_after_allocation(self):
        out=self.root/'audit'
        with mock.patch.object(M,'OUT',out),mock.patch.object(M,'load_helpers',side_effect=ValueError('synthetic helper pin refusal')):
            self.assertEqual(M.execute(arguments(out)),1)
        saved=json.loads((out/'failure.json').read_text())
        self.assertEqual(saved['message'],'synthetic helper pin refusal'); self.assertNotIn('L_candidate',saved)
        self.assertFalse((out/'result.json').exists())

    def _execute_mocked(self,error=None):
        out=self.root/'audit'; card_pin=pin(M.CARD); card_pin['sha256']='a'*64
        result=dict(status='synthetic_consistency',L_candidate={'numerator':1,'denominator':2},**M.FLAGS)
        with mock.patch.object(M,'OUT',out),mock.patch.object(M,'load_helpers',return_value=(self.files,self.execution,self.assets,self.scalar)),\
             mock.patch.object(M.Evidence,'current',return_value=(card_pin,{})),mock.patch.object(M,'audit_admission',return_value={}),\
             mock.patch.object(M,'integrate',side_effect=error,return_value=result):
            code=M.execute(arguments(out))
        return code,out

    def test_04_mocked_complete_report_keeps_all_admission_false(self):
        code,out=self._execute_mocked(); self.assertEqual(code,0)
        saved=json.loads((out/'result.json').read_text())
        for key in M.FLAGS:self.assertIs(saved[key],False)
        self.assertEqual(saved['L_candidate'],{'numerator':1,'denominator':2})
        self.assertFalse((out/'failure.json').exists())

    def test_05_closed_original_failure_preserved_without_L(self):
        code,out=self._execute_mocked(M.ClosedOriginalFailure({'returncode':1,'failure':'synthetic'}))
        self.assertEqual(code,1); saved=json.loads((out/'failure.json').read_text())
        self.assertEqual(saved['status'],'closed_original_failure_no_L'); self.assertNotIn('L_candidate',saved)
        self.assertEqual(saved['original_failure']['returncode'],1)

    def test_06_held_card_refusal_never_calls_integration(self):
        out=self.root/'audit'; card_pin=pin(M.CARD); card_pin['sha256']='a'*64
        with mock.patch.object(M,'OUT',out),mock.patch.object(M,'load_helpers',return_value=(self.files,self.execution,self.assets,self.scalar)),\
             mock.patch.object(M.Evidence,'current',return_value=(card_pin,{})),mock.patch.object(M,'audit_admission',side_effect=ValueError('held')),\
             mock.patch.object(M,'integrate') as integrate:
            self.assertEqual(M.execute(arguments(out)),1); integrate.assert_not_called()
        self.assertEqual(json.loads((out/'failure.json').read_text())['message'],'held')

    def test_07_output_replacement_refuses_alias_write(self):
        out=self.root/'audit'
        with mock.patch.object(M,'OUT',out):
            fd=M._allocate()
            try:
                out.rename(self.root/'old'); out.mkdir()
                with self.assertRaises(ValueError): M._save(fd,'result.json',{})
            finally:os.close(fd)
        self.assertFalse((out/'result.json').exists()); self.assertFalse((self.root/'old/result.json').exists())

    def test_08_same_pinned_bytes_supply_strict_decoder(self):
        payload=b'{"value":7}'; record=pin('/tmp/synthetic-record.json',payload)
        io=M.Evidence(self.files,self.execution,self.assets)
        with mock.patch.object(self.files,'read_pinned_bytes',return_value=payload) as read:
            self.assertEqual(io.read(record,True),{'value':7}); self.assertEqual(io.read(record),payload); read.assert_called_once_with(record)
        changed=dict(record,sha256='0'*64)
        with self.assertRaises(ValueError):io.read(changed,True)

    def test_09_ambiguous_json_is_not_decoded_via_loose_reader(self):
        payload=b'{"value":1,"value":2}'; record=pin('/tmp/synthetic-record.json',payload)
        io=M.Evidence(self.files,self.execution,self.assets)
        with mock.patch.object(self.files,'read_pinned_bytes',return_value=payload):
            with self.assertRaises(ValueError):io.read(record,True)

    def test_10_original_execution_covers_sources_outputs_and_every_preflight(self):
        io,authority,values=upstream_fixture(self.execution)
        result=M.original_execution(io,authority)
        self.assertEqual(result['card'],values[authority['card']['path']])
        self.assertEqual(io.sources.call_count,2)
        io.verify_pins.assert_called_once_with(values[authority['plan']['path']]['lanes']['synthetic']['preflight_pins'])
        io.inventory.assert_called_once_with(authority['output'],values[authority['receipt']['path']]['output_files'])

    def test_11_failed_or_boolean_upstream_exit_refused(self):
        for code in (1,True,None):
            io,authority,values=upstream_fixture(self.execution);values[authority['receipt']['path']]['returncode']=code
            with self.assertRaises(ValueError):M.original_execution(io,authority)
            io.inventory.assert_not_called()

    def test_12_upstream_source_command_and_input_joins_refuse(self):
        for mode in ('command','source','preflight','source_blob','bytes'):
            io,authority,values=upstream_fixture(self.execution)
            if mode=='command':values[authority['receipt']['path']]['argv']=[]
            elif mode=='source':values[authority['admission']['path']]['source_commit']='2'*40
            elif mode=='preflight':values[authority['card']['path']]['inputs'][0]['sha256']='0'*64
            elif mode=='source_blob':values[authority['admission']['path']]['source_blobs'][0]['relative_path']='codex/./synthetic'
            else:values[authority['receipt']['path']]['actual_output_bytes']=True
            with self.subTest(mode=mode),self.assertRaises(ValueError):M.original_execution(io,authority)

    def test_13_git_uses_explicit_original_source_and_exact_bytes(self):
        payload=b'synthetic source'; record=pin(M.D/'synthetic.py',payload); io=M.Evidence(self.files,self.execution,self.assets)
        with mock.patch.object(M.subprocess,'check_output',side_effect=[str(len(payload)).encode(),payload]) as command:
            self.assertEqual(io.blob('1'*40,record),payload)
        spec='1'*40+':codex/coexistence-task33/synthetic.py'
        self.assertEqual(command.call_args_list[0].args[0],['git','cat-file','-s',spec])
        self.assertEqual(command.call_args_list[1].args[0],['git','cat-file','blob',spec])
        self.assertNotIn('HEAD',str(command.call_args_list))

    def test_14_wrong_git_blob_size_or_bytes_refuses(self):
        payload=b'xx'; record=pin(M.D/'synthetic.py',payload); io=M.Evidence(self.files,self.execution,self.assets)
        for answers in ([b'3'],[b'2',b'zz']):
            with mock.patch.object(M.subprocess,'check_output',side_effect=answers),self.assertRaises(ValueError):io.blob('1'*40,record)

    def test_15_input_router_has_only_literal_streaming_exceptions(self):
        io=M.Evidence(self.files,self.execution,self.assets); ordinary=pin('/tmp/ordinary.json')
        asset=self.assets._target_definitions()['model.safetensors']['asset']
        with mock.patch.object(io,'stream_cache') as cache,mock.patch.object(io,'assets') as assets,mock.patch.object(io,'read') as read:
            io.verify_pins([M.CACHE_PIN,asset,ordinary]);cache.assert_called_once();assets.assert_called_once();read.assert_called_once_with(ordinary)
        wrong=dict(M.CACHE_PIN,sha256='0'*64)
        with self.assertRaises(ValueError):io.verify_pins([wrong])
        with self.assertRaises(ValueError):io.verify_pins([dict(ordinary,bytes=68579136)])

    def test_16_tiny_cache_streamed_without_read_all_or_decode(self):
        payload=b'original-radius-encoding'*19; path=self.root/'cache.gz';path.write_bytes(payload)
        io=M.Evidence(self.files,self.execution,self.assets)
        with mock.patch.object(M,'CACHE_PIN',pin(path,payload)),mock.patch.object(self.files,'read_pinned_bytes',side_effect=AssertionError('no read-all')),\
             mock.patch.object(self.files,'strict_json_bytes',side_effect=AssertionError('no decode')):
            first=io.stream_cache();self.assertEqual(io.stream_cache(),first)
        self.assertEqual(first['pin']['bytes'],len(payload))

    def test_17_cache_rejects_leaf_and_parent_aliases(self):
        folder=self.root/'cache';folder.mkdir();path=folder/'cache.gz';path.write_bytes(b'x');record=pin(path,b'x')
        io=M.Evidence(self.files,self.execution,self.assets)
        with mock.patch.object(M,'CACHE_PIN',record):
            path.rename(folder/'other');path.symlink_to(folder/'other')
            with self.assertRaises((ValueError,OSError)):io.stream_cache()
            path.unlink();(folder/'other').rename(path);folder.rename(self.root/'moved');folder.symlink_to(self.root/'moved',target_is_directory=True)
            with self.assertRaises((ValueError,OSError)):io.stream_cache()

    def test_18_cache_growth_during_hash_refuses(self):
        payload=b'x'*50;path=self.root/'cache.gz';path.write_bytes(payload);io=M.Evidence(self.files,self.execution,self.assets)
        original=self.assets._hash_fd
        def grow(fd,length):
            value=original(fd,length)
            with path.open('ab') as stream:stream.write(b'x')
            return value
        with mock.patch.object(M,'CACHE_PIN',pin(path,payload)),mock.patch.object(self.assets,'_hash_fd',side_effect=grow),self.assertRaises(ValueError):io.stream_cache()

    def test_19_final_whole_inventory_recheck_catches_change(self):
        io=M.Evidence(self.files,self.execution,self.assets);io.inventories=[('/tmp/synthetic',[],('before',{}))]
        with mock.patch.object(io,'snapshot',return_value=('after',{})),self.assertRaises(ValueError):io.recheck()

    def test_20_integrated_order_and_result_scope(self):
        io,modules,card,values,patches,events=integration_fixture()
        with mock.patch.multiple(M,**patches):result=M.integrate(io,modules,card)
        self.assertEqual(events,['source','host','target','preflight','upstream','assets','prefix','scalar','recheck','assets-final'])
        self.assertEqual(result['L_candidate'],{'numerator':3,'denominator':2})
        for key in M.FLAGS:self.assertIs(result[key],False)
        self.assertNotIn('scope',result['scalar_check'])

    def test_21_each_authority_failure_prevents_scalar_call(self):
        for stage in ('source_join','host_join','join_upstream','join_prefix'):
            io,modules,card,values,patches,events=integration_fixture()
            patches[stage]=mock.Mock(side_effect=ValueError('synthetic '+stage))
            with mock.patch.multiple(M,**patches),self.assertRaises(ValueError):M.integrate(io,modules,card)
            modules[3].check_bundle.assert_not_called()
        io,modules,card,values,patches,events=integration_fixture();modules[1].check_target_execution.side_effect=ValueError('target')
        with mock.patch.multiple(M,**patches),self.assertRaises(ValueError):M.integrate(io,modules,card)
        modules[3].check_bundle.assert_not_called()

    def test_22_closed_failed_original_cannot_use_success_bundle(self):
        io,modules,card,values,patches,events=integration_fixture()
        expected=values[card['original_expected']['path']];values[expected['receipt']['path']]['returncode']=1
        with mock.patch.multiple(M,**patches),self.assertRaises(M.ClosedOriginalFailure):M.integrate(io,modules,card)
        self.assertEqual(events,['source','host']);modules[0].read_flat_output_inventory.assert_not_called();modules[3].check_bundle.assert_not_called()

    def test_23_failed_original_host_refusal_is_not_claimed_closed(self):
        io,modules,card,values,patches,events=integration_fixture();expected=values[card['original_expected']['path']]
        values[expected['receipt']['path']]['returncode']=1;patches['host_join']=mock.Mock(side_effect=ValueError('host mismatch'))
        with mock.patch.multiple(M,**patches):
            with self.assertRaises(ValueError) as raised:M.integrate(io,modules,card)
        self.assertNotIsInstance(raised.exception,M.ClosedOriginalFailure);modules[3].check_bundle.assert_not_called()

    def test_24_post_scalar_mutation_still_prevents_completed_report(self):
        io,modules,card,values,patches,events=integration_fixture();io.recheck.side_effect=ValueError('late raw mutation')
        with mock.patch.multiple(M,**patches),self.assertRaises(ValueError):M.integrate(io,modules,card)
        modules[3].check_bundle.assert_called_once();modules[2].check_target_assets.assert_not_called()
        io,modules,card,values,patches,events=integration_fixture();modules[2].check_target_assets.side_effect=lambda:{'changed':True}
        with mock.patch.multiple(M,**patches),self.assertRaises(ValueError):M.integrate(io,modules,card)

    def test_25_output_is_exclusive_and_unknown_names_refuse(self):
        out=self.root/'audit'
        with mock.patch.object(M,'OUT',out):
            fd=M._allocate()
            try:
                M._save(fd,'result.json',{'synthetic':True})
                with self.assertRaises(FileExistsError):M._save(fd,'result.json',{})
                with self.assertRaises(ValueError):M._save(fd,'other.json',{})
            finally:os.close(fd)

    def _real_admission(self,io,card_pin,card,uid=(1000,1000,1000),bad_rlimit=False):
        environment={key:'1' for key in self.execution.THREADS};environment['HF_HUB_OFFLINE']='1'
        rlimits={M.resource.RLIMIT_AS:M.LIMITS['as_bytes'],M.resource.RLIMIT_CPU:M.LIMITS['cpu_seconds'],M.resource.RLIMIT_FSIZE:M.LIMITS['output_bytes']}
        with mock.patch.dict(M.os.environ,environment),mock.patch.object(M.os,'getpriority',return_value=10),\
             mock.patch.object(M.os,'sched_getaffinity',return_value={0}),mock.patch.object(M.os,'getresuid',return_value=uid),\
             mock.patch.object(M.resource,'getrlimit',side_effect=lambda kind:(1,1) if bad_rlimit else (rlimits[kind],)*2):
            return M.audit_admission(io,card_pin,card)

    def test_26_real_audit_admission_accepts_complete_synthetic_metadata(self):
        io,card_pin,card,plan,admission=admission_fixture(self.execution)
        result=self._real_admission(io,card_pin,card)
        self.assertEqual(result['card'],card_pin);self.assertEqual(result['source_commit'],'1'*40)
        io.verify_pins.assert_called_once_with(plan['lanes']['audit']['preflight_pins'])

    def test_27_real_audit_admission_refuses_held_sources_capacity_and_runtime(self):
        for mode in ('held','supervisor','helper','ram','uid','threads','rlimit','bool_cpu'):
            io,card_pin,card,plan,admission=admission_fixture(self.execution);uid=(1000,1000,1000)
            if mode=='held':card['status']='held'
            elif mode in ('supervisor','helper'):
                path=str(M.ROOT/'codex/certify-chaos/supervise.py') if mode=='supervisor' else str(M.D/'tube_saved_scalar.py')
                plan['source_files']=[row for row in plan['source_files'] if row['path']!=path]
                admission['source_blobs']=[row for row in admission['source_blobs'] if str(M.ROOT/row['relative_path'])!=path]
            elif mode=='ram':admission['capacity']['available_ram_bytes']=24*2**30-1
            elif mode=='uid':uid=(1000,0,1000)
            elif mode=='threads':admission['threads'][self.execution.THREADS[0]]='2'
            elif mode=='bool_cpu':plan['cpu_ids']=[False]
            with self.subTest(mode=mode),self.assertRaises(ValueError):self._real_admission(io,card_pin,card,uid,bad_rlimit=mode=='rlimit')

    def test_28_real_prefix_join_accepts_complete_synthetic_metadata(self):
        io,card,records,assets,upstream,meta,report=prefix_fixture(self.execution,self.assets)
        with mock.patch.object(M,'original_execution',return_value=upstream):result=M.join_prefix(io,card,records,assets)
        self.assertEqual(result['execution'],card['prefix_execution']);self.assertEqual(result['cache'],io.cache_observation)

    def test_29_real_prefix_join_refuses_changed_binding_metadata(self):
        for mode in ('checkpoint','cache','radii','precision','backend','review','prefix_source','metadata_precision','layers'):
            io,card,records,assets,upstream,meta,report=prefix_fixture(self.execution,self.assets);value=records['prefix-binding.json']
            if mode=='checkpoint':value['checkpoint']['asset']['sha256']='0'*64
            elif mode=='cache':value['prefix_files']['prefix-cache.json.gz']='0'*64
            elif mode=='radii':value['original_radii_retained']=False
            elif mode=='precision':value['receiver_precision']=1024
            elif mode=='backend':value['backend_sha256']='0'*64
            elif mode=='review':value['regularity_review']['sha256']='0'*64
            elif mode=='prefix_source':card['prefix_execution']['source_commit']='1'*40
            elif mode=='metadata_precision':meta['precision']=1024
            else:report['layers'].pop()
            with self.subTest(mode=mode),mock.patch.object(M,'original_execution',return_value=upstream),self.assertRaises(ValueError):M.join_prefix(io,card,records,assets)


if __name__=='__main__':unittest.main()
