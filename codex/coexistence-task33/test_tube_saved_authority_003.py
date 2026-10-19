"""Focused audit003 historical-source controls, using manufactured bytes only.

No old test imports, real Git execution, target records, model, backend or assets.
New wrapper/frozen stdlib helpers may load; Git and original metadata are mocked.
Only fresh temporary source text is touched for the current-file drift control.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location('tube_saved_authority_003', HERE/'tube_saved_authority_003.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
PAYLOAD = b'# manufactured source\nclass SoftLoop:\n    def prepare(self):\n        return 7\n'
TEXT = PAYLOAD.decode()
CLASS_SHA = hashlib.sha256(ast.get_source_segment(TEXT, ast.parse(TEXT).body[0]).encode()).hexdigest()


def pin(path, data=b'{}'):
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def join_fixture(execution):
    """Enough synthetic prior joins to reach the real later producer read."""
    thin = copy.deepcopy(M.THIN_AUTHORITY)
    audit = dict(source_commit='70b043192b775f2cc67860c4709bfc867de096eb',
        card=dict(path=str(M.D/'thin-pilot-001-audit-002-card.json'), bytes=2565,
                  sha256='538e6ea648fa1e115dc35eceb4392d0798da9bc19f4c77c91098ac227fed8dd7'),
        plan=dict(path=str(M.D/'thin-pilot-001-audit-002-plan.json'), bytes=5418,
                  sha256='d719c9366fc84380fa4983fd9c579a425ed8faeb99c33aad29c6373c3f0292a6'),
        output=str(M.B/'thin-pilot-001-audit-002'))
    audit['argv'] = [M.PYTHON,'-I','-B',str(M.D/'audit_thin_pilot_002.py'),
        '--source',thin['source_commit'],'--card',thin['card']['path'],'--card-sha256',thin['card']['sha256'],
        '--plan',thin['plan']['path'],'--plan-sha256',thin['plan']['sha256'],
        '--receipt-sha256',thin['receipt']['sha256'],'--admission-sha256',thin['admission']['sha256'],
        '--output',M.REPORT_PIN['path']]
    card = dict(audit_execution=audit, thin_execution=thin, audit_result=copy.deepcopy(M.REPORT_PIN))
    auditor = 'b2a84b5022f71cadfce91e6fc21fa4ad9953bb0a1a0b0c1d62d24d43f0c7e4d5'
    au = dict(card=dict(schema='codex-task33-thin-audit-execution-card-v1',
              auditor_sha256=auditor,original_execution=copy.deepcopy(thin),result_path=M.REPORT_PIN['path']),
              files={M.REPORT_PIN['path']:copy.deepcopy(M.REPORT_PIN)},
              sources={str(M.D/'audit_thin_pilot_002.py'):{'sha256':auditor}})
    authority = dict(source=copy.deepcopy(M.THIN_PRODUCER_PIN),
        method='SoftLoop.prepare: embedding of last prefix token, K=1',
        scope='exact stored checkpoint constants; not bitwise identity with floating execution',
        original_source_commit='019168d9c28b92ae80f7f2766592dccbfae69e2d',
        original_source_sha256='ead3b44a82c556c1a376f3e58c389cd3fb2f7329786ef0a712a3d92e7a920843',
        fixed_start_class_sha256='c590a4c0bb2ee3742333d66cfa204add121b63bcb43d6446f96e8cd047f3e50b')
    names = ['admission.json','centre-000.json','report.json']+[f'step-{i:03d}.json' for i in range(1,21)]
    files = {thin['output']+'/'+name:pin(thin['output']+'/'+name) for name in names}
    tc = dict(execution.PREFIX,alpha=.06468,log10T=-.148,precision=2048,steps=20,initial_token_id=198,
              inputs=[copy.deepcopy(M.THIN_PRODUCER_PIN)],producer_initial_state_authority=authority)
    tu = dict(card=tc,files=files,receipt=dict(output_files=list(files.values()),external_elapsed_seconds=1.0),
              admission=dict(source_blobs=[]),sources={
              str(M.D/'thin_pilot.py'):{'sha256':'a12fd306fb49392c0d4922b6c4d7aabec1dd56e0b0b835373ecf74287ff05cbc'},
              str(M.D/'thin_pseudo_orbit.py'):{'sha256':'1e9974f2bd5cddfc0ddfd6b1038c7d046df7f8abdb6f061b67fb764a6189369e'},
              str(M.ROOT/'codex/crosscheck/qwen_map.py'):{'sha256':execution.SOURCE_HASHES['codex/crosscheck/qwen_map.py']}})
    original = {k:copy.deepcopy(thin[k]) for k in ('source_commit','receipt','admission','card','plan')}
    original.update(raw_root=thin['output'],output_files=list(files.values()),source_blobs=[],
                    returncode=0,failure=None,host_original_groups_absent=True,elapsed_seconds=1.0)
    report = dict(schema='codex-task33-thin-pilot-saved-audit-v1',status='complete_success',worker_failure=None,
        worker_completion_present=True,reachability_proved=False,original_partial_evidence_preserved=True,
        steps_checked=20,original_execution=original,initial_state={'source_authority':copy.deepcopy(authority)},
        steps=[dict(step=i,component_count=896,file=files[thin['output']+f'/step-{i:03d}.json'],
                    checks={'synthetic_gate':True}) for i in range(1,21)])
    return card,au,tu,report


class HistoricalSourceControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files,cls.execution,cls.assets,_ = M.load_helpers()

    def setUp(self):
        self.io = M.Evidence(self.files,self.execution,self.assets)
        self.authority = copy.deepcopy(M.THIN_AUTHORITY)
        self.synthetic_pin = pin(M.THIN_PRODUCER_PIN['path'],PAYLOAD)

    def adapted(self):
        return mock.patch.multiple(M,THIN_PRODUCER_PIN=copy.deepcopy(self.synthetic_pin),THIN_CLASS_SHA256=CLASS_SHA)

    def git(self, payload=PAYLOAD, size=None):
        return mock.patch.object(M.subprocess,'check_output',
            side_effect=[str(len(payload) if size is None else size).encode(),payload])

    def test_01_literal_allocation_authority_and_unchanged_generic_gates(self):
        self.assertEqual(M.OUT,M.B/'tube-target-001-audit-003')
        self.assertEqual(M.CARD,M.D/'tube-target-001-audit-003-card.json')
        self.assertEqual(M.PLAN,M.D/'tube-target-001-audit-003-plan.json')
        self.assertEqual(M.SELF,M.D/'tube_saved_authority_003.py')
        self.assertEqual(M.THIN_AUTHORITY['source_commit'],'e8db350006098683ab170645bf7ee7c317ca3b56')
        self.assertEqual(M.THIN_AUTHORITY['card'],dict(path=str(M.D/'thin-pilot-001-card.json'),bytes=4859,
            sha256='33922427cb39bb72662122ae06c461acdbfc9314344e4e9f8a468d2b47164213'))
        self.assertEqual(M.THIN_AUTHORITY['plan'],dict(path=str(M.D/'thin-pilot-001-plan.json'),bytes=9344,
            sha256='bf07856a9637d1f174bcd4dfe65f66d2055091d59ac8758f6001ca0c3ef4ac90'))
        self.assertEqual(M.THIN_PRODUCER_PIN,dict(path=str(M.ROOT/'tools/explore_llm_soft_loop.py'),bytes=122549,
            sha256='28418408507409f52df9217c71255131a1de8cfe7d059a35761971738883e120'))
        self.assertEqual(M.THIN_CLASS_SHA256,'c590a4c0bb2ee3742333d66cfa204add121b63bcb43d6446f96e8cd047f3e50b')
        old=(HERE/'tube_saved_authority_002.py').read_bytes()
        self.assertEqual(hashlib.sha256(old).hexdigest(),'3c4a28f122eeaf89cb7bea64e14d79fcf0bb55022bf9770613c39a7a444e5399')
        oldtree,newtree=ast.parse(old),ast.parse((HERE/'tube_saved_authority_003.py').read_bytes())
        def methods(tree):
            cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='Evidence')
            return {node.name:ast.dump(node,include_attributes=False) for node in cls.body if isinstance(node,ast.FunctionDef)}
        oldmethods,newmethods=methods(oldtree),methods(newtree)
        for name in ('read','verify_pins','blob','sources','historical_source','assets','stream_cache','inventory','snapshot'):
            self.assertEqual(newmethods[name],oldmethods[name],name)
        oldf={n.name:ast.dump(n,include_attributes=False) for n in oldtree.body if isinstance(n,ast.FunctionDef)}
        newf={n.name:ast.dump(n,include_attributes=False) for n in newtree.body if isinstance(n,ast.FunctionDef)}
        for name in ('audit_admission','pure_arguments','_allocate','_save','_check_output_identity','host_join','join_prefix'):
            self.assertEqual(newf[name],oldf[name],name)
        self.assertTrue(all(value is False for value in M.FLAGS.values()))

    def test_02_exact_git_bytes_and_class_use_separate_cache_and_provenance(self):
        with self.adapted(),self.git() as git,mock.patch.object(self.files,'read_pinned_bytes',side_effect=AssertionError('no current read')):
            self.assertEqual(self.io.thin_producer(self.authority,self.synthetic_pin),PAYLOAD)
        spec=self.authority['source_commit']+':tools/explore_llm_soft_loop.py'
        self.assertEqual(git.call_args_list,[
            mock.call(['git','cat-file','-s',spec],cwd=M.ROOT,timeout=15),
            mock.call(['git','cat-file','blob',spec],cwd=M.ROOT,timeout=15)])
        self.assertEqual(self.io.pins,{})
        self.assertEqual(self.io.payloads,{})
        self.assertEqual(len(self.io.git_checks),1)
        self.assertEqual(len(self.io.historical_input_provenance),1)
        row=self.io.historical_input_provenance[0]
        self.assertEqual(row['source_pin'],self.synthetic_pin)
        self.assertEqual(row['original_card'],self.authority['card'])
        self.assertEqual(row['original_plan'],self.authority['plan'])
        self.assertEqual(row['class_source_segment_sha256'],CLASS_SHA)
        self.assertIs(row['current_file_authenticated'],False)

    def test_03_wrong_commit_card_plan_receipt_or_context_refuses_before_git(self):
        for mode in ('commit','card_path','card_hash','card_length','plan_path','plan_hash','plan_length',
                     'receipt','admission','argv','lane','output','extra'):
            authority=copy.deepcopy(self.authority)
            if mode=='commit':authority['source_commit']='0'*40
            elif mode.startswith(('card_','plan_')):
                part,field=mode.split('_')
                field={'hash':'sha256','length':'bytes'}.get(field,field)
                authority[part][field]=0 if field=='bytes' else ('0'*64 if field=='sha256' else '/tmp/wrong')
            elif mode in ('receipt','admission'):authority[mode]['sha256']='0'*64
            elif mode=='argv':authority['argv'].append('--extra')
            elif mode=='extra':authority['extra']=True
            else:authority[mode]='wrong'
            with self.subTest(mode=mode),self.adapted(),mock.patch.object(self.io,'sources') as sources,\
                 self.assertRaisesRegex(ValueError,'exact original thin producer authority'):
                self.io.thin_producer(authority,self.synthetic_pin)
            sources.assert_not_called()

    def test_04_wrong_path_length_hash_or_pin_type_has_no_fallback(self):
        for mode in ('path','bytes','sha256','boolean','extra'):
            changed=copy.deepcopy(self.synthetic_pin)
            if mode=='path':changed['path']=str(M.B/'thin-pilot-001/step-001.json')
            elif mode=='bytes':changed['bytes']+=1
            elif mode=='sha256':changed['sha256']='0'*64
            elif mode=='boolean':changed['bytes']=True
            else:changed['extra']='not a pin'
            with self.subTest(mode=mode),self.adapted(),mock.patch.object(self.io,'sources') as sources,\
                 self.assertRaises(ValueError):
                self.io.thin_producer(self.authority,changed)
            sources.assert_not_called()

    def test_05_git_size_and_hash_failures_are_retained_without_historical_admission(self):
        for payload,size in ((PAYLOAD,len(PAYLOAD)+1),(PAYLOAD.replace(b'7',b'8'),len(PAYLOAD))):
            io=M.Evidence(self.files,self.execution,self.assets)
            with self.subTest(size=size),self.adapted(),self.git(payload,size) as git,self.assertRaises(ValueError):
                io.thin_producer(self.authority,self.synthetic_pin)
            self.assertEqual(git.call_count,1 if size!=len(PAYLOAD) else 2)
            self.assertIsNone(io.thin_producer_bytes)
            self.assertEqual(io.historical_input_provenance,[])

    def test_06_wrong_missing_or_duplicate_class_refuses(self):
        for data,expected in ((PAYLOAD,'0'*64),(b'x=1\n',CLASS_SHA),
                              (PAYLOAD+b'\nclass SoftLoop:\n    pass\n',CLASS_SHA)):
            io=M.Evidence(self.files,self.execution,self.assets)
            local_pin=pin(M.THIN_PRODUCER_PIN['path'],data)
            with self.subTest(data=data),mock.patch.multiple(M,THIN_PRODUCER_PIN=local_pin,THIN_CLASS_SHA256=expected),\
                 self.git(data),self.assertRaisesRegex(ValueError,'historical thin producer class differs'):
                io.thin_producer(self.authority,local_pin)
            self.assertIsNone(io.thin_producer_bytes)
            self.assertEqual(io.historical_input_provenance,[])

    def test_07_preflight_and_later_read_share_one_historical_source(self):
        before,after=pin('/tmp/synthetic-before.json'),pin('/tmp/synthetic-after.json')
        with self.adapted(),self.git() as git,mock.patch.object(self.io,'verify_pins') as ordinary:
            M.upstream_preflight(self.io,self.authority,[before,self.synthetic_pin,after])
            self.assertEqual(self.io.thin_producer(self.authority,self.synthetic_pin),PAYLOAD)
        self.assertEqual(ordinary.call_args_list,[mock.call([before]),mock.call([after])])
        self.assertEqual(git.call_count,2)
        self.assertEqual(len(self.io.historical_input_provenance),1)
        self.assertEqual(self.io.pins,{})
        self.assertEqual(self.io.payloads,{})

    def test_08_preflight_refuses_wrong_context_changed_missing_or_duplicate_pin(self):
        for mode in ('context','changed','missing','duplicate'):
            authority=copy.deepcopy(self.authority);rows=[copy.deepcopy(self.synthetic_pin)]
            if mode=='context':authority['source_commit']='0'*40
            elif mode=='changed':rows[0]['bytes']+=1
            elif mode=='missing':rows=[]
            else:rows.append(copy.deepcopy(rows[0]))
            with self.subTest(mode=mode),self.adapted(),mock.patch.object(self.io,'thin_producer') as historical,\
                 mock.patch.object(self.io,'verify_pins') as ordinary,self.assertRaises(ValueError):
                M.upstream_preflight(self.io,authority,rows)
            historical.assert_not_called();ordinary.assert_not_called()

    def test_09_ordinary_raw_refusal_never_falls_back_to_git(self):
        raw=pin('/tmp/synthetic-raw.json')
        with self.adapted(),mock.patch.object(self.assets,'_target_definitions',return_value={}),\
             mock.patch.object(self.files,'read_pinned_bytes',side_effect=FileNotFoundError('synthetic raw absent')),\
             mock.patch.object(self.io,'thin_producer') as historical,mock.patch.object(self.io,'sources') as sources,\
             self.assertRaisesRegex(FileNotFoundError,'synthetic raw absent'):
            M.upstream_preflight(self.io,{'other':'upstream'},[raw])
        historical.assert_not_called();sources.assert_not_called()

    def test_10_historical_recheck_is_git_only_and_current_pin_still_rechecked(self):
        raw=pin('/tmp/synthetic-current.json')
        with self.adapted(),self.git():
            self.io.thin_producer(self.authority,self.synthetic_pin)
        self.io.pins[raw['path']]=raw
        with self.adapted(),self.git() as git,mock.patch.object(self.files,'read_pinned_bytes',return_value=b'{}') as read:
            self.io.recheck()
        read.assert_called_once_with(raw)
        self.assertEqual(git.call_count,2)

    def test_11_final_historical_git_mismatch_refuses(self):
        with self.adapted(),self.git():
            self.io.thin_producer(self.authority,self.synthetic_pin)
        with self.adapted(),self.git(PAYLOAD.replace(b'7',b'8')),self.assertRaises(ValueError):
            self.io.recheck()
        self.assertEqual(len(self.io.historical_input_provenance),1)

    def test_12_real_original_execution_routes_its_thin_preflight(self):
        authority=copy.deepcopy(self.authority)
        lane=dict(entry=authority['argv'][3],arguments=authority['argv'][4:],output=authority['output'],
                  supervision=authority['receipt']['path'].rsplit('/',1)[0],
                  limits=copy.deepcopy(M.LIMITS),preflight_pins=[copy.deepcopy(self.synthetic_pin)])
        card=dict(inputs=[copy.deepcopy(self.synthetic_pin)])
        plan=dict(status='reviewed_for_execution',repository=str(M.ROOT),raw_root=str(M.B),python=M.PYTHON,
                  cpu_threads=1,lanes={'thin':lane},source_files=[])
        receipt=dict(returncode=0,failure=None,worker_reaped=True,worker_identity_absent=True,
                     worker_group_session_absent=True,argv=authority['argv'].copy(),lane='thin',
                     source_commit=authority['source_commit'],output_files=[],actual_output_bytes=0)
        admission=dict(plan=copy.deepcopy(authority['plan']),limits=copy.deepcopy(M.LIMITS),
                       threads={key:'1' for key in self.execution.THREADS},argv=authority['argv'].copy(),
                       lane='thin',source_commit=authority['source_commit'])
        records={authority[k]['path']:v for k,v in zip(('card','plan','receipt','admission'),(card,plan,receipt,admission))}
        def sources(source,rows):
            return {'tools/explore_llm_soft_loop.py':PAYLOAD} if rows==[self.synthetic_pin] else {}
        with self.adapted(),mock.patch.object(self.io,'read',side_effect=lambda row,decode=False:copy.deepcopy(records[row['path']])) as read,\
             mock.patch.object(self.io,'sources',side_effect=sources),mock.patch.object(M,'source_join',return_value={}),\
             mock.patch.object(self.io,'inventory',return_value={}),mock.patch.object(self.io,'verify_pins') as ordinary:
            result=M.original_execution(self.io,authority)
        self.assertEqual(result['card'],card)
        ordinary.assert_not_called()
        self.assertNotIn(self.synthetic_pin['path'],[call.args[0]['path'] for call in read.call_args_list])
        self.assertEqual(len(self.io.historical_input_provenance),1)

    def test_13_real_join_upstream_reaches_historical_later_read(self):
        card,au,tu,report=join_fixture(self.execution)
        io=mock.Mock();io.e=self.execution
        initial=tu['files'][card['thin_execution']['output']+'/centre-000.json']
        step=tu['files'][card['thin_execution']['output']+'/step-001.json']
        values={M.REPORT_PIN['path']:report,initial['path']:{},step['path']:{}}
        io.read.side_effect=lambda row,decode=False:copy.deepcopy(values[row['path']])
        class ReachedHistoricalRead(Exception):pass
        io.thin_producer.side_effect=ReachedHistoricalRead('later producer read')
        scalar=types.SimpleNamespace(integer=mock.Mock(),thin_flags=mock.Mock(),STEP_CHECKS={'synthetic_gate'})
        with mock.patch.object(M,'original_execution',side_effect=[au,tu]) as original,\
             self.assertRaisesRegex(ReachedHistoricalRead,'later producer read'):
            M.join_upstream(io,card,scalar)
        io.thin_producer.assert_called_once_with(card['thin_execution'],M.THIN_PRODUCER_PIN)
        self.assertEqual(original.call_args_list,[mock.call(io,card['audit_execution']),mock.call(io,card['thin_execution'])])

    def test_14_changed_current_producer_is_irrelevant_only_to_exact_historical_adapter(self):
        with tempfile.TemporaryDirectory(prefix='tube-historical-source-control-') as directory:
            root=Path(directory);path=root/'tools/explore_llm_soft_loop.py';path.parent.mkdir();path.write_bytes(b'first current version')
            historical=pin(path,PAYLOAD)
            with mock.patch.multiple(M,ROOT=root,THIN_PRODUCER_PIN=historical,THIN_CLASS_SHA256=CLASS_SHA),self.git():
                self.assertEqual(self.io.thin_producer(self.authority,historical),PAYLOAD)
                path.write_bytes(b'changed current version with different length')
                self.assertEqual(self.io.thin_producer(self.authority,historical),PAYLOAD)
                with self.assertRaises(ValueError):self.io.read(historical)
            self.assertEqual(path.read_bytes(),b'changed current version with different length')
            self.assertNotIn(str(path),self.io.pins)

    def test_15_current_cache_cannot_be_mislabeled_as_historical(self):
        for field in ('pins','payloads'):
            io=M.Evidence(self.files,self.execution,self.assets)
            getattr(io,field)[self.synthetic_pin['path']]=copy.deepcopy(self.synthetic_pin) if field=='pins' else PAYLOAD
            with self.subTest(field=field),self.adapted(),mock.patch.object(io,'sources') as sources,\
                 self.assertRaisesRegex(ValueError,'historical producer cannot share'):
                io.thin_producer(self.authority,self.synthetic_pin)
            sources.assert_not_called()

    def test_16_later_failure_preserves_historical_provenance_and_false_flags(self):
        with self.adapted(),self.git():
            self.io.thin_producer(self.authority,self.synthetic_pin)
        cardpin=pin(M.CARD);card={}
        self.io.current=mock.Mock(return_value=(cardpin,card))
        args=argparse.Namespace(card=str(M.CARD),out=str(M.OUT),card_sha256=cardpin['sha256'])
        with mock.patch.object(M,'_allocate',return_value=987654),mock.patch.object(M,'load_helpers',return_value=(None,None,None,None)),\
             mock.patch.object(M,'Evidence',return_value=self.io),mock.patch.object(M,'audit_admission',return_value={}),\
             mock.patch.object(M,'integrate',side_effect=ValueError('synthetic later refusal')),\
             mock.patch.object(M,'_save') as save,mock.patch.object(M.os,'close') as close:
            self.assertEqual(M.execute(args),1)
        close.assert_called_once_with(987654)
        failure=save.call_args.args[2]
        self.assertEqual(save.call_args.args[1],'failure.json')
        self.assertEqual(failure['message'],'synthetic later refusal')
        self.assertEqual(failure['historical_input_provenance'],self.io.historical_input_provenance)
        self.assertEqual(failure['authenticated_file_pins'],[])
        for key in M.FLAGS:self.assertIs(failure[key],False)

    def test_17_only_literal003_cli_is_accepted_without_allocation(self):
        valid=argparse.Namespace(card=str(M.CARD),out=str(M.OUT),card_sha256='a'*64)
        old_card=str(M.D/'tube-target-001-audit-002-card.json')
        old_out=str(M.B/'tube-target-001-audit-002')
        with mock.patch.object(M,'_allocate') as allocate:
            self.assertIsNone(M.pure_arguments(valid))
            for card,out in ((old_card,old_out),(old_card,str(M.OUT)),(str(M.CARD),old_out)):
                args=argparse.Namespace(card=card,out=out,card_sha256='a'*64)
                with self.subTest(card=card,out=out),self.assertRaises(ValueError):
                    M.pure_arguments(args)
            allocate.assert_not_called()


if __name__=='__main__':
    unittest.main()
