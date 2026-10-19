"""New stdlib-only actual002 adapter controls; no old suite or numerical replay.

All original metadata below is manufactured. Source files are read only for the
new module and immutable-code comparison. No original evidence, /proc, Git,
NumPy, model or backend is accessed; loader numerical interfaces are mocked.
Fresh temporary tiny files exercise new metadata/inventory/output boundaries.
"""
import ast
import copy
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
SOURCE = HERE/'audit_phase_p6_002.py'
OLD = HERE/'audit_phase_p6.py'
spec = importlib.util.spec_from_file_location('phase002_subject', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    """Small independent dictionaries with the exact changed structural schema."""
    cell = dict(name=m.CELL, period=6, alpha=.065693, log10T=-.14798, seed='p3',
                alpha_ratio=[147927485160425,2251799813685248],
                log10T_ratio=[-333221336429143,2251799813685248],
                states=str(m.B/'settle-001/cascade-p6-states.npy'),
                settle_report=str(m.B/'settle-001/cascade-p6-report.json'),
                seed_path='/data2/neural-shrimp-data/explore/sheets/seeds/p3.npy')
    inputs = [dict(path=cell[key], bytes=1, sha256='a'*64) for key in ('states','settle_report')]
    card = dict(repository=str(m.R), cells=[cell], inputs=copy.deepcopy(inputs))
    argv = ['/data2/neural-shrimp/.venv/bin/python','-I','-B',str(m.D/'phase_preconditioner.py'),
            '--card',str(m.CARD),'--cell',m.CELL,'--out',str(m.RAW)]
    sources = [dict(path=str(m.R/name), bytes=1, sha256=sha) for name,sha in m.SCIENCE.items()]
    sources.append(copy.deepcopy(m.PINS['resource_card']))
    plan = dict(status='reviewed_for_execution',repository=str(m.R),raw_root=str(m.B),
                python=argv[0],cpu_threads=1,cpu_ids=[0],source_files=sources,
                lanes={'phase':dict(entry=argv[3],arguments=argv[4:],output=str(m.RAW),supervision=str(m.SUP),
                                    limits=copy.deepcopy(m.ORIGINAL_LIMITS),preflight_pins=copy.deepcopy(inputs))})
    def actor(identity, command):
        return dict(**identity,nice=10,threads=1,uid_line='Uid:\t1000\t1000\t1000\t1000',cmdline=' '.join(command)+' ')
    supervisor = actor(m.SUPERVISOR,['synthetic-original-supervisor'])
    worker = actor(m.WORKER,argv)
    outputs = [dict(path=path,bytes=1,sha256='b'*64) for path in sorted(m.expected_outputs())]
    receipt = dict(source_commit=m.SOURCE,argv=argv.copy(),lane='phase',returncode=0,failure=None,
                   worker_reaped=True,worker_identity_absent=True,worker_group_session_absent=True,
                   worker_pid=m.WORKER['pid'],worker_observed_after_exec=copy.deepcopy(worker),
                   supervisor=copy.deepcopy(supervisor),output_files=outputs,actual_output_bytes=14,
                   external_elapsed_seconds=1.)
    blobs = [dict(relative_path=str(Path(row['path']).relative_to(m.R)),bytes=row['bytes'],sha256=row['sha256'])
             for row in sources]
    admission = dict(source_commit=m.SOURCE,argv=argv.copy(),lane='phase',supervisor=copy.deepcopy(supervisor),
                     plan=copy.deepcopy(m.PINS['plan']),limits=copy.deepcopy(m.ORIGINAL_LIMITS),
                     threads={key:'1' for key in m.THREADS},source_blobs=copy.deepcopy(blobs))
    startup = dict(identity=copy.deepcopy(worker),planned_argv=argv.copy(),affinity=[0],
                   address_space=[m.ORIGINAL_LIMITS['as_bytes']]*2,cpu_seconds=[5080]*2,
                   file_size=[m.ORIGINAL_LIMITS['output_bytes']]*2)
    startup['identity']['cmdline']='synthetic preexec command; planned argv binds subsequent exec'
    root = dict(source_commit=m.SOURCE,admission=copy.deepcopy(m.PINS['admission']),
                worker_startup=copy.deepcopy(m.PINS['worker_startup']),
                root_supervisor_postexec=copy.deepcopy(m.SUPERVISOR),root_worker_postexec=copy.deepcopy(m.WORKER))
    recovery = dict(status='reviewed_for_execution',attempt=m.ATTEMPT,scientific_card=copy.deepcopy(m.PINS['scientific_card']),
                    cell=copy.deepcopy(cell),new_output=str(m.RAW),new_supervision=str(m.SUP),
                    limits=copy.deepcopy(m.ORIGINAL_LIMITS))
    host = dict(checked_utc='2026-10-19T18:15:54.306291+00:00',
                identities=[dict(pid=value['pid'],expected_starttime=value['starttime'],present=False)
                            for value in (m.SUPERVISOR,m.WORKER)],group_session_members=[],scan_permission_errors=[])
    completion = dict(schema='codex-task34-phase002-operational-completion-v1',
                      status='closed_exit0_pending_independent_saved_audit',source_commit=m.SOURCE,
                      failure=None,session_exit_code=0,original_returncode=0,saved_numerical_consistency_admitted=False,
                      **{key:False for key in m.FALSE_FLAGS})
    for key in ('receipt','admission','scientific_card','resource_card','plan','worker_startup','root_startup'):
        completion[key]=copy.deepcopy(m.PINS[key])
    completion.update(source_blobs_checked=copy.deepcopy(blobs),
                      original_preflight_pins_checked=copy.deepcopy(inputs),
                      original_card_input_pins_checked=copy.deepcopy(inputs),outputs=copy.deepcopy(outputs),
                      complete_output_inventory_checked=True,all14_output_bytes_authenticated=True,output_bytes=14,
                      original_host_closure=dict(pin=copy.deepcopy(m.PINS['host_closure']),observation=copy.deepcopy(host),
                                                group_ids_checked=[787986,788101],session_ids_checked=[787986,788101]))
    records = dict(scientific_card=card,plan=plan,receipt=receipt,admission=admission,worker_startup=startup,
                   root_startup=root,resource_card=recovery,completion=completion,host_closure=host)
    return records


def args():
    return SimpleNamespace(attempt=m.ATTEMPT,source=m.SOURCE,cell=m.CELL,
        receipt_sha256=m.PINS['receipt']['sha256'],admission_sha256=m.PINS['admission']['sha256'],output=str(m.RESULT))


def join(records):
    receipt,admission,plan,card = [records[key] for key in ('receipt','admission','plan','scientific_card')]
    cell,files=m.allocation(receipt,admission,plan,card,m.PINS['plan'])
    return m.original_execution(records,receipt,admission,plan,card,files)


class Controls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='phase002-new-controls-')
        self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name)

    def test_01_numerical_checker_and_old_source_unchanged(self):
        data=OLD.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(),'61c8a466235eedbfcc9454d5965e8e999a46a88a79552c3ab7e799f7f2aaa92e')
        def function(text):
            return ast.get_source_segment(text,next(n for n in ast.parse(text).body
                if isinstance(n,ast.FunctionDef) and n.name=='check_saved_arrays'))
        self.assertEqual(function(data.decode()),function(SOURCE.read_text()))
        self.assertEqual(hashlib.sha256(function(SOURCE.read_text()).encode()).hexdigest(),
                         '0252e55786b6b1f6644fcc23ac630516beef203ace719e023317df7a1210f929')
        self.assertEqual(m.SOURCE,'9028465dfb8e7a9001b34a091cc61067b464d80e')
        self.assertEqual(m.PERIOD,6);self.assertEqual(m.DIMENSION,896)

    def test_02_exact002_cli_and_mixed_allocation_refusals(self):
        valid=['--attempt',m.ATTEMPT,'--source',m.SOURCE,'--cell',m.CELL,
               '--receipt-sha256',m.PINS['receipt']['sha256'],'--admission-sha256',m.PINS['admission']['sha256'],
               '--output',str(m.RESULT)]
        self.assertEqual(m.arguments(valid).output,str(m.RESULT))
        for index,value in ((1,'phase-cascade-p6-001'),(3,'36366f7fb9b81e5f90d61aee1d09ae5730dff666'),
                            (5,'band-p6'),(7,'a'*64),(9,'b'*64),(11,str(m.RAW/'result.json'))):
            with self.subTest(index=index):
                changed=valid.copy();changed[index]=value
                with self.assertRaises(ValueError):m.arguments(changed)

    def test_03_real002_pure_join_accepts_distinct_supervisor_group(self):
        value=join(fixture())
        self.assertEqual(value['supervisor']['pid'],787987)
        self.assertEqual(value['group_ids'],[787986,788101])

    def test_04_mixed_source_card_plan_receipt_or_closure_refused(self):
        changes=[lambda r:r['receipt'].update(source_commit='0'*40),
                 lambda r:r['admission']['plan'].update(sha256='0'*64),
                 lambda r:r['resource_card']['scientific_card'].update(sha256='0'*64),
                 lambda r:r['completion']['receipt'].update(sha256='0'*64),
                 lambda r:r['completion']['original_host_closure']['pin'].update(sha256='0'*64),
                 lambda r:r['resource_card'].update(attempt='phase-cascade-p6-001'),
                 lambda r:r['receipt'].update(returncode=1),
                 lambda r:r['completion'].update(session_exit_code=True)]
        for index,change in enumerate(changes):
            with self.subTest(index=index):
                records=fixture();change(records)
                with self.assertRaises(ValueError):join(records)

    def test_05_startup_birth_affinity_limits_uid_and_command_refused(self):
        changes=[lambda r:r['worker_startup']['identity'].update(starttime=1),
                 lambda r:r['admission']['supervisor'].update(pgid=787987),
                 lambda r:r['worker_startup'].update(affinity=[False]),
                 lambda r:r['worker_startup'].update(address_space=[1,1]),
                 lambda r:r['receipt']['worker_observed_after_exec'].update(uid_line='Uid: 0 0 0 0'),
                 lambda r:r['receipt']['worker_observed_after_exec'].update(threads=True),
                 lambda r:r['receipt']['worker_observed_after_exec'].update(cmdline='wrong target'),
                 lambda r:r['root_startup']['root_worker_postexec'].update(starttime=1)]
        for index,change in enumerate(changes):
            with self.subTest(index=index):
                records=fixture();change(records)
                with self.assertRaises(ValueError):join(records)

    def test_06_missing_source_preflight_output_and_byte_mismatch_refused(self):
        changes=[lambda r:r['admission']['source_blobs'].pop(),
                 lambda r:r['completion']['original_preflight_pins_checked'].pop(),
                 lambda r:r['plan']['lanes']['phase']['preflight_pins'].pop(),
                 lambda r:r['receipt']['output_files'].pop(),
                 lambda r:r['completion']['outputs'].pop(),
                 lambda r:r['receipt'].update(actual_output_bytes=True),
                 lambda r:r['completion'].update(output_bytes=15),
                 lambda r:r['completion'].update(certificate_passed=True)]
        for index,change in enumerate(changes):
            with self.subTest(index=index):
                records=fixture();change(records)
                with self.assertRaises(ValueError):join(records)

    def test_07_root_host_observation_and_coverage_refused(self):
        changes=[lambda r:r['host_closure'].update(checked_utc='2026-10-19T18:15:55Z'),
                 lambda r:r['completion']['original_host_closure'].update(group_ids_checked=[787987,788101]),
                 lambda r:r['host_closure']['identities'][0].update(present=True),
                 lambda r:r['host_closure'].update(group_session_members=[dict(pid=2)]),
                 lambda r:r['host_closure'].update(scan_permission_errors=['denied'])]
        for index,change in enumerate(changes):
            with self.subTest(index=index):
                records=fixture();change(records)
                # Bind the same altered observation to exercise semantic, not merely equality, refusal.
                records['completion']['original_host_closure']['observation']=copy.deepcopy(records['host_closure'])
                with self.assertRaises(ValueError):join(records)

    def test_08_current_namespace_queries_recorded_groups_not_supervisor_pid(self):
        ident,members=Mock(return_value=None),Mock(return_value=[])
        m.original_host_closed(ident,members)
        self.assertEqual([call.args[0] for call in ident.call_args_list],[787987,788101])
        self.assertEqual([call.args[0] for call in members.call_args_list],[787986,788101])
        with self.assertRaisesRegex(ValueError,'recorded group'):
            m.original_host_closed(lambda pid:None,lambda group:[1] if group==787986 else [])
        with self.assertRaisesRegex(ValueError,'identity'):
            m.original_host_closed(lambda pid:{} if pid==788101 else None,lambda group:[])

    def pinned_json(self,data):
        path=self.base/'metadata.json';path.write_bytes(data)
        return dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())

    def test_09_new_metadata_decodes_exact_pinned_bytes_and_refuses_changes(self):
        record=self.pinned_json(b'{"hello": 1}')
        self.assertEqual(m.metadata(record),{'hello':1})
        changed=copy.deepcopy(record);changed['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'bytes differ'):m.metadata(changed)
        changed=copy.deepcopy(record);changed['bytes']=True
        with self.assertRaisesRegex(ValueError,'bounded metadata'):m.metadata(changed)

    def test_10_new_metadata_duplicate_nonfinite_and_symlink_refusals(self):
        for data in (b'{"a":{"x":1,"x":2}}',b'{"x":NaN}',b'{"x":1e999}'):
            with self.subTest(data=data):
                with self.assertRaises(ValueError):m.metadata(self.pinned_json(data))
        record=self.pinned_json(b'{}');link=self.base/'alias.json';link.symlink_to(record['path'])
        record['path']=str(link)
        with self.assertRaisesRegex(ValueError,'canonical'):m.metadata(record)

    def make_inventory(self):
        raw=self.base/'original';(raw/m.CELL).mkdir(parents=True)
        (raw/'report.json').write_bytes(b'{}')
        for phase in range(6):
            (raw/m.CELL/f'phase-{phase}-matrix.npy').write_bytes(b'x')
            (raw/m.CELL/f'phase-{phase}-residual.npz').write_bytes(b'x')
        (raw/m.CELL/'preconditioner.npz').write_bytes(b'x')
        return raw,{str(path):{} for path in raw.rglob('*') if path.is_file()}

    def test_11_exact_inventory_refuses_added_directory_file_link_and_missing_phase(self):
        raw,files=self.make_inventory()
        with patch.object(m,'RAW',raw):
            m.whole_inventory(files)
            extra=raw/'unexpected';extra.mkdir()
            with self.assertRaises(ValueError):m.whole_inventory(files)
            extra.rmdir();extra=raw/m.CELL/'unexpected';extra.write_bytes(b'x')
            with self.assertRaises(ValueError):m.whole_inventory(files)
            extra.unlink();extra.symlink_to(raw/'report.json')
            with self.assertRaises(ValueError):m.whole_inventory(files)
            extra.unlink();(raw/m.CELL/'phase-5-matrix.npy').unlink()
            with self.assertRaises(ValueError):m.whole_inventory(files)

    def mocked_loader(self, final_inventory_error=False):
        records=fixture()
        # Source retrieval and allocation were separately joined above. Here the
        # source loop is empty to isolate actual loader ordering without Git.
        records['admission']['source_blobs']=[]
        records['completion']['source_blobs_checked']=[]
        cell=records['scientific_card']['cells'][0]
        files=m.inventory(records['receipt']['output_files'])
        row=dict(cell=cell,period=6,certificate_passed=False,product_order=m.PRODUCT_ORDER,
                 phases=[dict(phase=i,seconds=0.) for i in range(6)])
        history=object();arrays={key:object() for key in m.ARRAYS}
        matrices=[object() for _ in range(6)]
        residuals=[{key:object() for key in ('center','image','residual')} for _ in range(6)]
        class Archive:
            def __init__(self,values):self.values=values;self.files=list(values)
            def __getitem__(self,key):return self.values[key]
            def __enter__(self):return self
            def __exit__(self,*args):return False
        loads={cell['states']:history,str(m.RAW/m.CELL/'preconditioner.npz'):Archive(arrays)}
        for i in range(6):
            loads[str(m.RAW/m.CELL/f'phase-{i}-matrix.npy')]=matrices[i]
            loads[str(m.RAW/m.CELL/f'phase-{i}-residual.npz')]=Archive(residuals[i])
        fake_np=SimpleNamespace(__version__='synthetic',load=Mock(side_effect=lambda path,allow_pickle:
            loads[str(path)] if allow_pickle is False else self.fail('pickle enabled')))
        metadata={value['path']:records[key] for key,value in m.PINS.items()}
        metadata[str(m.RAW/'report.json')]=dict(rows=[row])
        verified=[]
        whole=Mock(side_effect=[None,ValueError('manufactured final inventory change')] if final_inventory_error else None)
        numeric=Mock(return_value=dict(phases_checked=6,dimension=896,certificate_passed=False,rigorous_inverse_proved=False))
        supervise=SimpleNamespace(__file__=str(m.R/'codex/certify-chaos/supervise.py'),ident=Mock(return_value=None),members=Mock(return_value=[]))
        by_path={value['path']:value for value in m.PINS.values()}
        def source_pin(path,digest):
            return by_path.get(str(path),dict(path=str(path),bytes=1,sha256=digest))
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules,{'numpy':fake_np,'supervise':supervise}))
            stack.enter_context(patch.object(sys,'path',list(sys.path)))
            stack.enter_context(patch.dict(os.environ,{**{key:'1' for key in m.THREADS},'HF_HUB_OFFLINE':'1','CUDA_VISIBLE_DEVICES':''},clear=True))
            stack.enter_context(patch.object(m,'metadata',side_effect=lambda pin:metadata[pin['path']]))
            stack.enter_context(patch.object(m,'pinned_source',side_effect=source_pin))
            stack.enter_context(patch.object(m,'allocation',return_value=(cell,files)))
            stack.enter_context(patch.object(m,'verify',side_effect=lambda pin:verified.append(pin['path']) or pin))
            stack.enter_context(patch.object(m,'whole_inventory',whole))
            stack.enter_context(patch.object(m,'check_saved_arrays',numeric))
            stack.enter_context(patch.object(m,'git_blob',side_effect=AssertionError('unexpected Git')))
            result=m.run(args())
        return result,verified,whole,numeric,fake_np,supervise,history,arrays,matrices,residuals

    def test_12_mocked_loader_checks_full_inventory_twice_and_retains_false_scope(self):
        result,verified,whole,numeric,np,sup,history,arrays,matrices,residuals=self.mocked_loader()
        self.assertEqual(whole.call_count,2)
        self.assertEqual(np.load.call_count,14)
        numeric.assert_called_once()
        actual=numeric.call_args.args
        self.assertIs(actual[1],history);self.assertEqual(actual[2],arrays)
        self.assertEqual(actual[3],list(zip(matrices,residuals)))
        for path in m.expected_outputs():self.assertEqual(verified.count(path),2)
        for value in m.PINS.values():self.assertEqual(verified.count(value['path']),1)
        self.assertEqual(sup.members.call_count,4)
        self.assertFalse(result['all_original_runtime_asset_preflight_independently_rechecked'])
        for key in m.FALSE_FLAGS:self.assertIs(result[key],False)

    def test_13_final_inventory_change_refuses_after_mocked_numeric_work(self):
        with self.assertRaisesRegex(ValueError,'final inventory change'):self.mocked_loader(True)

    def test_14_each_bad_environment_refuses_before_metadata_or_numpy(self):
        valid={**{key:'1' for key in m.THREADS},'HF_HUB_OFFLINE':'1','CUDA_VISIBLE_DEVICES':''}
        for key in valid:
            for value in (None,'bad'):
                with self.subTest(key=key,value=value):
                    env=valid.copy()
                    if value is None:env.pop(key)
                    else:env[key]=value
                    with patch.dict(os.environ,env,clear=True),patch.object(m,'metadata') as read:
                        with self.assertRaisesRegex(ValueError,'environment'):m.run(args())
                        read.assert_not_called()

    def test_15_fresh_failure_retained_and_existing_output_never_overwritten(self):
        result=self.base/'audit-001/result.json'
        with patch.object(m,'RESULT',result),patch.object(m,'arguments',return_value=args()), \
             patch.object(m,'run',side_effect=ValueError('manufactured refusal')) as run:
            with self.assertRaisesRegex(ValueError,'manufactured refusal'):m.main()
            failure=(result.parent/'failure.json').read_bytes()
            value=json.loads(failure)
            self.assertFalse(result.exists());self.assertEqual(value['source_commit'],m.SOURCE)
            self.assertEqual(value['original_metadata'],m.PINS)
            for key in m.FALSE_FLAGS:self.assertIs(value[key],False)
            with self.assertRaisesRegex(ValueError,'fresh absent'):m.main()
            self.assertEqual(run.call_count,1)
            self.assertEqual((result.parent/'failure.json').read_bytes(),failure)

    def test_16_mocked_success_exclusive_result_without_original_writes(self):
        result=self.base/'audit-001/result.json'
        value=dict(status='manufactured',outputs_checked=14,**{key:False for key in m.FALSE_FLAGS})
        with patch.object(m,'RESULT',result),patch.object(m,'arguments',return_value=args()),patch.object(m,'run',return_value=value):
            m.main()
        self.assertEqual(json.loads(result.read_text())['status'],'manufactured')
        self.assertFalse((result.parent/'failure.json').exists())


if __name__=='__main__':unittest.main(verbosity=2)
