"""Synthetic preflight controls only: fake file/Git/process records, no model imports.

Mocks never change process limits, affinity, priority, repository state or assets.
"""
import copy
from contextlib import ExitStack
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import unittest
from unittest.mock import patch

MODULE_PATH=Path('/tmp/task37-delivery/p6_kite.py')
spec=importlib.util.spec_from_file_location('task37_wrapper_controls',MODULE_PATH)
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)

class Fixture:
    def __init__(self):
        self.temp=tempfile.TemporaryDirectory(prefix='task37-wrapper-controls-',dir='/tmp')
        self.root=Path(self.temp.name)/'repo';self.raw=Path(self.temp.name)/'raw'
        self.folder=self.root/'codex/presubmit-task37';self.folder.mkdir(parents=True)
        self.entry=self.folder/'p6_kite.py';self.cardpath=self.folder/'r-pair-001-card.json';self.planpath=self.folder/'r-pair-001-plan.json'
        self.out=self.raw/'r-pair-001';self.sup=self.raw/'r-pair-001-supervision';self.sup.mkdir(parents=True)
        self.admissionpath=self.sup/'admission.json';self.startpath=self.sup/'worker-startup.json'
        self.source='a'*40;self.content={};self.now=datetime(2026,10,19,0,0,tzinfo=timezone.utc)
        self.prior='/data2/neural-shrimp-data/codex/certificate-ladder/phase-cascade-p6-001-supervision/receipt.json'
        self.prior_json={'worker_reaped':True,'worker_group_session_absent':True}
        self.sources=[self.entry,self.folder/'periodic_audit.py',self.root/'codex/crosscheck/qwen_map.py',self.root/'codex/certify-chaos/supervise.py']
        sourcepins=[self.pin(p,b'fake committed source '+str(p).encode()) for p in self.sources]
        snapshot=Path('/data2/neural-shrimp-data/models/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775')
        self.inventory=self.folder/'p6-seed-inventory.json';p6=Path('/data2/neural-shrimp-data/explore/llm-soft-loop/pod64/p6s')
        inputs=[self.inventory,Path(self.prior),snapshot/'tokenizer.json',snapshot/'model.safetensors',snapshot/'config.json',Path('/data2/neural-shrimp-data/explore/llm-phase-map/vectors-l12/vectors.npz')]
        inputs.extend(p6/n/f for n in ('p6s-ad-curve-plus','p6s-td-curve-minus') for f in ('states.npy','curve.json'))
        inputpins=[self.pin(p,b'fake input '+str(p).encode()) for p in inputs]
        self.limits={'as_bytes':18*1024**3,'cpu_seconds':15000,'wall_seconds':15300,'output_bytes':512*1024**2}
        self.card={'schema':'codex-task37-p6-r-pair-card-v1','status':'reviewed_for_execution','target':{'label':'R','alpha':.072912,'log10_temperature':-.154104,'period':6},'capacity_window':{'confirmed':True,'owner':'task37-r-pair-001','coordination_reference':'synthetic reservation','not_before_utc':self.now.isoformat(),'not_after_utc':(self.now+timedelta(hours=5)).isoformat()},'source_files':sourcepins,'input_files':inputpins,'seed_inventory':str(self.inventory),'tokenizer':str(snapshot/'tokenizer.json'),'prior_closed_receipt':self.prior}
        self.args=['--card',str(self.cardpath),'--out',str(self.out),'--admission',str(self.admissionpath)]
        self.argv=[sys.executable,'-I','-B',str(self.entry),*self.args]
        self.plan={'status':'reviewed_for_execution','repository':str(self.root),'raw_root':str(self.raw),'cpu_threads':1,'startup_ram_gib':32,'python':sys.executable,'source_files':[], 'lanes':{'r_pair':{'output':str(self.out),'supervision':str(self.sup),'entry':str(self.entry),'limits':self.limits.copy(),'arguments':self.args.copy(),'prior_receipts':[self.prior],'preflight_pins':copy.deepcopy(inputpins)}}}
        self.parent={'pid':os.getppid(),'starttime':123,'pgid':234,'sid':234}
        self.me={'pid':os.getpid(),'starttime':345,'pgid':os.getpid(),'sid':os.getpid()}
        self.admission={'source_commit':self.source,'lane':'r_pair','plan':{},'argv':self.argv.copy(),'limits':self.limits.copy(),'source_blobs':[], 'supervisor':self.parent.copy(),'threads':{k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},'capacity':{'available_ram_bytes':40*1024**3,'disk_free_bytes':20*1024**3}}
        self.startup={'identity':self.me.copy(),'planned_argv':self.argv.copy(),'affinity':[3],'address_space':[self.limits['as_bytes']]*2,'cpu_seconds':[15000]*2,'file_size':[self.limits['output_bytes']]*2}
    def pin(self,p,data):
        self.content[str(p)]=data
        return {'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    def materialize(self):
        data=json.dumps(self.card).encode();self.cardpath.write_bytes(data);cardpin=self.pin(self.cardpath,data)
        self.plan['source_files']=[*copy.deepcopy(self.card['source_files']),cardpin]
        pdata=json.dumps(self.plan).encode();self.planpath.write_bytes(pdata);self.admission['plan']=self.pin(self.planpath,pdata)
        self.admission['source_blobs']=[{'relative_path':str(Path(p['path']).relative_to(self.root)),'bytes':p['bytes'],'sha256':p['sha256']} for p in self.plan['source_files']]
    def run(self,after=None):
        self.materialize()
        if after:after(self)
        self.admissionpath.write_text(json.dumps(self.admission));self.startpath.write_text(json.dumps(self.startup))
        original=Path.read_text
        def fake_read(path,*a,**kw):
            if str(path)==self.prior:return json.dumps(self.prior_json)
            return original(path,*a,**kw)
        def git(argv,**kw):
            if argv[:2]==['git','show']:
                commit,relative=argv[2].split(':',1)
                if commit!=self.source:raise ValueError('unexpected fake source')
                return self.content[str(self.root/relative)]
            if argv[:2]==['git','rev-parse']:return (self.source+'\n').encode()
            raise AssertionError(argv)
        caps={resource.RLIMIT_AS:(18*1024**3,)*2,resource.RLIMIT_CPU:(15000,)*2,resource.RLIMIT_FSIZE:(512*1024**2,)*2}
        with ExitStack() as stack:
            for target,name,value in [(worker,'ROOT',self.root),(worker,'RAW',self.raw),(worker,'__file__',str(self.entry))]:stack.enter_context(patch.object(target,name,value))
            stack.enter_context(patch.object(worker,'verify',lambda p:None))
            stack.enter_context(patch.object(worker,'_utcnow',lambda:self.now))
            stack.enter_context(patch.object(worker,'_worker_argv',lambda:self.argv))
            stack.enter_context(patch.object(worker,'_identity',lambda pid:self.me if pid==os.getpid() else self.parent))
            stack.enter_context(patch.object(worker.subprocess,'check_output',git))
            stack.enter_context(patch.object(worker.os,'getpriority',lambda *a:10))
            stack.enter_context(patch.object(worker.os,'sched_getaffinity',lambda *a:{3}))
            stack.enter_context(patch.object(worker.resource,'getrlimit',lambda key:caps[key]))
            stack.enter_context(patch.object(Path,'read_text',fake_read))
            stack.enter_context(patch.dict(os.environ,{'HF_HUB_OFFLINE':'1',**self.admission['threads']}))
            return worker.preflight(str(self.cardpath),str(self.out),str(self.admissionpath))
    def close(self):self.temp.cleanup()

class WrapperControls(unittest.TestCase):
    def setUp(self):self.f=Fixture()
    def tearDown(self):self.f.close()
    def refused(self,change=None,after=None):
        if change:change(self.f)
        with self.assertRaises((ValueError,KeyError)):self.f.run(after)
    def test_valid_mock_contract(self):self.assertEqual(self.f.run()[0]['target']['period'],6)
    def test_held_card(self):self.refused(lambda f:f.card.update(status='held'))
    def test_capacity_missing(self):self.refused(lambda f:f.card.update(capacity_window=None))
    def test_capacity_uncoordinated(self):self.refused(lambda f:f.card['capacity_window'].update(coordination_reference=''))
    def test_capacity_too_short(self):self.refused(lambda f:f.card['capacity_window'].update(not_after_utc=(f.now+timedelta(seconds=100)).isoformat()))
    def test_capacity_expired(self):self.refused(lambda f:f.card['capacity_window'].update(not_after_utc=(f.now-timedelta(seconds=1)).isoformat()))
    def test_missing_source_pins(self):self.refused(lambda f:f.card.update(source_files=[]))
    def test_missing_input_pins(self):self.refused(lambda f:f.card.update(input_files=[]))
    def test_duplicate_input(self):self.refused(lambda f:f.card['input_files'].append(copy.deepcopy(f.card['input_files'][0])))
    def test_stale_card_commit(self):self.refused(after=lambda f:f.content.__setitem__(str(f.cardpath),b'different committed card'))
    def test_wrong_lane(self):self.refused(after=lambda f:f.admission.update(lane='other'))
    def test_wrong_argv(self):self.refused(after=lambda f:f.admission['argv'].append('--unreviewed'))
    def test_missing_admitted_source(self):self.refused(after=lambda f:f.admission['source_blobs'].pop())
    def test_worker_identity_mismatch(self):self.refused(after=lambda f:f.startup['identity'].update(starttime=999))
    def test_supervisor_identity_mismatch(self):self.refused(after=lambda f:f.admission['supervisor'].update(starttime=999))
    def test_prior_unclosed(self):self.refused(lambda f:f.prior_json.update(worker_reaped=False))
    def test_admission_capacity_low(self):self.refused(after=lambda f:f.admission['capacity'].update(available_ram_bytes=31*1024**3))
    def test_refuses_lower_plan_floor(self):self.refused(lambda f:f.plan.update(startup_ram_gib=28))
    def test_refuses_unexpected_18gib_cap(self):self.refused(lambda f:f.plan['lanes']['r_pair']['limits'].update(as_bytes=19*1024**3))
    def test_native_pin_requires_bytes_and_hash(self):
        p=Path(self.f.temp.name)/'pin.txt';p.write_bytes(b'checked')
        with self.assertRaises(ValueError):worker.verify({'path':str(p),'sha256':hashlib.sha256(b'checked').hexdigest()})
        worker.verify({'path':str(p),'bytes':7,'sha256':hashlib.sha256(b'checked').hexdigest()})
        with self.assertRaises(ValueError):worker.verify({'path':str(p),'bytes':7,'sha256':'0'*64})
    def test_science_entrypoint_defined(self):
        self.assertTrue(callable(getattr(worker,'run_science',None)))
    def test_no_scientific_import_at_module_load(self):
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('periodic_audit',sys.modules)

if __name__=='__main__':unittest.main(verbosity=2)
