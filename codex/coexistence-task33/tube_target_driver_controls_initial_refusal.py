"""Focused driver authority controls, exclusively manufactured/mocked evidence.

No model import/call, target/raw read or process/Git operation. The actual
candidate/original_execution/current_admission code is exercised; low-level
pin/source/filesystem/resource interfaces are mocks, not cryptographic tests.
"""
import contextlib,copy,hashlib,importlib.util,json
from fractions import Fraction as F
from pathlib import Path
from unittest.mock import patch

R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp')
SOURCE=R/'codex/coexistence-task33/tube_target_driver.py'
SOURCE_SHA='ba32975a90ddc65c32cd3153ece35d0e3d938ab6d920a758d42dda75d3f18a88'
data=SOURCE.read_bytes();assert hashlib.sha256(data).hexdigest()==SOURCE_SHA
spec=importlib.util.spec_from_file_location('tube_driver_tested',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rows=[]
def native(n=0,e=0,r=0,re=0):return dict(mid=[n,e],rad=[r,re])
def checked(name,func):func();rows.append(dict(name=name,passed=True))
def refused(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('expected actual driver refusal absent')

class Fixture:
    def __init__(self,zero=False):
        self.docs={};self.pins={};self.sources={};self.trees={}
        initial=dict(step=0,centre=[native()for _ in range(896)],trajectory_enclosure=False,certificate_passed=False)
        centres=[native((i%5)-2,-2)for i in range(896)]
        rad=0 if zero else 1
        step=dict(step=1,input_centre=copy.deepcopy(initial['centre']),next_centre=centres,
                  full_image=[dict(mid=c['mid'][:],rad=[rad,-4])for c in centres],
                  defects=[native(r=rad,re=-3)for _ in centres],defect_inf_upper=native(rad,-2),
                  trajectory_enclosure=False,certificate_passed=False)
        self.initial=self.add(m.THIN_RAW/'centre-000.json',initial)
        self.step=self.add(m.THIN_RAW/'step-001.json',step)
        thin_card=dict(m.PREFIX,precision=2048,steps=20,initial_token_id=198,alpha=m.ALPHA,
                       log10T=m.LOG10_TEMPERATURE,producer_initial_state_authority={'manufactured':'fixed source authority'})
        thin_argv=[m.PYTHON,'-I','-B',str(m.D/'thin_pilot.py'),'--card',str(m.THIN_CARD),'--out',str(m.THIN_RAW)]
        self.thin=self.execution('thin','a'*40,m.THIN_CARD,m.D/'thin-pilot-001-plan.json',m.THIN_RAW,
            thin_card,m.THIN_CARD_SHA,thin_argv,[self.initial,self.step],False,{
            'codex/coexistence-task33/thin_pilot.py':'a12fd306fb49392c0d4922b6c4d7aabec1dd56e0b0b835373ecf74287ff05cbc',
            'codex/coexistence-task33/thin_pseudo_orbit.py':'1e9974f2bd5cddfc0ddfd6b1038c7d046df7f8abdb6f061b67fb764a6189369e',
            'codex/crosscheck/qwen_map.py':m.CORE_SHA['codex/crosscheck/qwen_map.py']})
        delta=F(rad,4);maximum=F(rad,8)
        embedding=dict(asset=self.add(Path('/synthetic/model.safetensors'),{},m.ASSETS['model.safetensors']),
            tensor='model.embed_tokens.weight',stored_dtype='BF16',shape=[151936,896],row=198,
            row_bytes=1792,row_byte_offset=4096,row_raw_sha256='f'*64,
            exact_rational_values_sha256=m.canonical([m.qrecord(F(0))for _ in range(896)]))
        checks={k:True for k in ('exact_input_chain','full_image_endpoints','exact_image_midpoints',
            'signed_defect_contains_image_minus_next','saved_upper_contains_all_defect_endpoints')}
        row=dict(file=self.step,step=1,component_count=896,checks=checks,
            computed_defect_endpoint_max=m.qrecord(maximum),saved_defect_inf_upper_exact=m.qrecord(delta),
            next_centre_canonical_sha256=m.canonical(centres))
        rc=self.docs[self.thin['receipt']['path']];ad=self.docs[self.thin['admission']['path']]
        report=dict(schema='codex-task33-thin-pilot-saved-audit-v1',trajectory_enclosure=False,
            certificate_passed=False,reachability_proved=False,original_partial_evidence_preserved=True,
            status='closed_partial_failure',original_execution={
                **{k:self.thin[k]for k in ('source_commit','receipt','admission','card','plan')},
                'raw_root':str(m.THIN_RAW),'output_files':rc['output_files'],'source_blobs':ad['source_blobs'],
                'returncode':rc['returncode'],'failure':rc['failure'],'host_original_groups_absent':True},
            initial_state=dict(file=self.initial,component_count=896,exact_checkpoint_row_match=True,
                source_authority=thin_card['producer_initial_state_authority'],embedding=embedding),
            steps_checked=1,steps=[row],entry_candidate=dict(step=1,file=self.step,
                next_centre_canonical_sha256=m.canonical(centres),delta0=m.qrecord(delta),e1_upper=m.qrecord(delta)))
        self.report=self.add(m.AUDIT_RESULT,report)
        ac=dict(schema='codex-task33-thin-audit-execution-card-v1',auditor_sha256=m.AUDITOR_SHA,
                original_execution=copy.deepcopy(self.thin),result_path=str(m.AUDIT_RESULT))
        argv=[m.PYTHON,'-I','-B',str(m.D/'audit_thin_pilot.py'),'--source',self.thin['source_commit'],
            '--card',str(m.THIN_CARD),'--card-sha256',m.THIN_CARD_SHA,'--plan',self.thin['plan']['path'],
            '--plan-sha256',self.thin['plan']['sha256'],'--receipt-sha256',self.thin['receipt']['sha256'],
            '--admission-sha256',self.thin['admission']['sha256'],'--output',str(m.AUDIT_RESULT)]
        self.audit=self.execution('audit','b'*40,m.D/'thin-audit-fixture-card.json',m.D/'thin-audit-fixture-plan.json',
            m.AUDIT_RESULT.parent,ac,None,argv,[self.report],True,{
            'codex/coexistence-task33/audit_thin_pilot.py':m.AUDITOR_SHA})
        self.card=dict(thin_execution=self.thin,audit_execution=self.audit,audit_result=self.report)
    def add(self,path,doc,sha=None):
        path=str(path);p=dict(path=path,bytes=100,sha256=sha or hashlib.sha256(path.encode()).hexdigest())
        self.docs[path]=doc;self.pins[path]=p;return p
    def execution(self,lane,source,card_path,plan_path,out,card,cardsha,argv,outputs,success,science):
        cp=self.add(card_path,card,cardsha);pp=self.add(plan_path,{})
        limits=dict(as_bytes=4*1024**3,cpu_seconds=240,wall_seconds=300,output_bytes=64*1024**2)
        sources=[cp]
        for relative,sha in science.items():sources.append(self.add(m.R/relative,{},sha))
        for p in sources+[pp]:self.sources[source,p['path']]=copy.deepcopy(p)
        sup=out.parent/(out.name+'-supervision')
        plan=dict(status='reviewed_for_execution',repository=str(m.R),raw_root=str(m.B),python=m.PYTHON,
            cpu_threads=1,source_files=sources,lanes={lane:dict(entry=argv[3],arguments=argv[4:],
                output=str(out),supervision=str(sup),limits=limits)})
        self.docs[pp['path']]=plan
        ad=dict(source_commit=source,argv=argv,lane=lane,plan=pp,limits=limits,
            threads={k:'1'for k in m.THREADS},source_blobs=[dict(relative_path=str(Path(p['path']).relative_to(m.R)),
                bytes=p['bytes'],sha256=p['sha256'])for p in sources])
        rc=dict(source_commit=source,argv=argv,lane=lane,returncode=0 if success else 1,
                failure=None if success else 'synthetic outer failure',worker_reaped=True,
                worker_identity_absent=True,worker_group_session_absent=True,output_files=outputs)
        self.trees[out]=[Path(p['path'])for p in outputs]
        return dict(source_commit=source,card=cp,plan=pp,receipt=self.add(sup/'receipt.json',rc),
            admission=self.add(sup/'admission.json',ad),lane=lane,output=str(out),argv=argv)
    def report_doc(self):return self.docs[str(m.AUDIT_RESULT)]
    def step_doc(self):return self.docs[self.step['path']]
    def rc(self,which):return self.docs[which['receipt']['path']]
    def ad(self,which):return self.docs[which['admission']['path']]
    def plan(self,which):return self.docs[which['plan']['path']]
    def verify(self,record):
        if self.pins.get(record['path'])!=record:raise ValueError('synthetic pinned bytes differ')
        return copy.deepcopy(record)
    def source_pin(self,source,record,current=False):
        if self.sources.get((source,record['path']))!=record:raise ValueError('synthetic original source differs')
        if current:self.verify(record)
        return record
    @contextlib.contextmanager
    def mocked(self):
        with patch.object(m,'read',side_effect=lambda p:copy.deepcopy(self.docs[str(p)])), \
             patch.object(m,'pin',side_effect=lambda p:copy.deepcopy(self.pins[str(p)])), \
             patch.object(m,'verify',side_effect=self.verify),patch.object(m,'source_pin',side_effect=self.source_pin), \
             patch.object(Path,'rglob',autospec=True,side_effect=lambda p,q:self.trees[p]), \
             patch.object(Path,'is_file',autospec=True,side_effect=lambda p:str(p)in self.pins), \
             patch.object(Path,'open',side_effect=AssertionError('unmocked filesystem read forbidden')), \
             patch.object(m.subprocess,'check_output',side_effect=AssertionError('real subprocess forbidden')):
            yield

def positive(zero=False):
    f=Fixture(zero)
    with f.mocked():records,c0,radius,selection=m.candidate(f.card)
    assert len(records)==len(c0)==896 and radius==(F(1,2**2048)if zero else F(1,2))
    assert selection['incoming_error_le_radius'] is True
    assert all(selection[k] is v for k,v in m.FLAGS.items())
checked('actual joined audit plus honest partial thin yields full896 first-tube candidate',positive)
checked('exact zero delta uses the declared positive2^-2048 fallback',lambda:positive(True))

def mutation(name,change):
    def test():
        f=Fixture();change(f)
        with f.mocked():refused(lambda:m.candidate(f.card))
    checked(name,test)

mutation('audit unsuccessful receipt refused',lambda f:f.rc(f.audit).__setitem__('returncode',1))
mutation('audit null watchdog semantics enforced',lambda f:f.rc(f.audit).__setitem__('failure','timeout'))
mutation('audit source commit mismatch refused',lambda f:f.rc(f.audit).__setitem__('source_commit','c'*40))
mutation('audit unclosed group refused',lambda f:f.rc(f.audit).__setitem__('worker_group_session_absent',False))
mutation('audit result absent from successful receipt refused',lambda f:f.rc(f.audit).__setitem__('output_files',[]))
mutation('audit output directory mismatch refused',lambda f:f.audit.__setitem__('output',str(m.B/'other')))
mutation('audit invocation altered receipt argument refused',lambda f:f.rc(f.audit).__setitem__('argv',['wrong']))
mutation('audit original source inventory omission refused',lambda f:f.ad(f.audit).__setitem__('source_blobs',[]))
mutation('audit source binding refuses changed implementation',lambda f:f.sources.__setitem__((f.audit['source_commit'],str(m.D/'audit_thin_pilot.py')),{}))
mutation('audit execution card must bind exact original thin authority',lambda f:f.docs[f.audit['card']['path']]['original_execution'].__setitem__('source_commit','c'*40))
mutation('audit report cannot claim trajectory enclosure',lambda f:f.report_doc().__setitem__('trajectory_enclosure',True))
mutation('audit report cannot claim reachability',lambda f:f.report_doc().__setitem__('reachability_proved',True))
mutation('partial thin cannot be relabelled completed',lambda f:f.report_doc().__setitem__('status','complete_success'))
mutation('audit report original receipt join enforced',lambda f:f.report_doc()['original_execution'].__setitem__('receipt',{}))
mutation('audit report original host closure required',lambda f:f.report_doc()['original_execution'].__setitem__('host_original_groups_absent',False))
mutation('actual original thin source receipt mismatch refused',lambda f:f.rc(f.thin).__setitem__('source_commit','c'*40))
mutation('actual original thin output inventory whole-directory check enforced',lambda f:f.trees[m.THIN_RAW].append(Path('/synthetic/extra.json')))
mutation('original step1 file pin join enforced',lambda f:f.report_doc()['entry_candidate'].__setitem__('file',{}))
mutation('initial full896 dimension required',lambda f:f.docs[f.initial['path']]['centre'].pop())
mutation('initial independently audited exact row match required',lambda f:f.report_doc()['initial_state'].__setitem__('exact_checkpoint_row_match',False))
mutation('initial embedding exact row hash required',lambda f:f.report_doc()['initial_state']['embedding'].__setitem__('exact_rational_values_sha256','0'*64))
mutation('initial token198 required',lambda f:f.report_doc()['initial_state']['embedding'].__setitem__('row',199))
mutation('initial full checkpoint hash required',lambda f:f.report_doc()['initial_state']['embedding']['asset'].__setitem__('sha256','0'*64))
mutation('step1 centre chain original coordinate895 checked',lambda f:f.step_doc()['input_centre'].__setitem__(895,native(1)))
mutation('step1 next centre coordinate895 exact midpoint checked',lambda f:f.step_doc()['next_centre'].__setitem__(895,native(1)))
mutation('step1 complete full-image vector required',lambda f:f.step_doc()['full_image'].pop())
mutation('step1 signed defect containment coordinate895 checked',lambda f:f.step_doc()['defects'].__setitem__(895,native()))
mutation('step1 uncertain next centre refused',lambda f:f.step_doc()['next_centre'].__setitem__(0,native(r=1,re=-10)))
mutation('step1 understated native saved upper refused',lambda f:f.step_doc().__setitem__('defect_inf_upper',native(1,-5)))
mutation('step1 exact recombination disagrees with audit refused',lambda f:f.report_doc()['steps'][0].__setitem__('computed_defect_endpoint_max',m.qrecord(F(1,4))))
mutation('step1 missing explicit audit component check refused',lambda f:f.report_doc()['steps'][0]['checks'].__delitem__('full_image_endpoints'))
mutation('step1 candidate canonical centre hash join enforced',lambda f:f.report_doc()['entry_candidate'].__setitem__('next_centre_canonical_sha256','0'*64))
mutation('step1 candidate delta0 exact rational join enforced',lambda f:f.report_doc()['entry_candidate'].__setitem__('delta0',m.qrecord(F(1,2))))

def base_card():
    return dict(schema='codex-task33-one-tube-card-v1',status='reviewed_for_one_tube_execution',
        repository=str(m.R),output=str(m.B/'tube-fixture-001'),target_calls=1,precision=2048,
        directions=1,dimension=896,layers=24,vocabulary=151936,token_chunk=1024,
        prefix=copy.deepcopy(m.PREFIX),regularity_contract=m.REGULARITY_CONTRACT,**m.FLAGS)

def early_card(name,change):
    def test():
        card=base_card();change(card)
        with patch.object(m,'read',side_effect=AssertionError('early card refusal must precede all evidence reads')):
            refused(lambda:m.current_admission(card,{},m.B/'tube-fixture-001'))
    checked(name,test)
for key,value in [('target_calls',2),('precision',1024),('directions',2),('dimension',895),('layers',23),
                  ('vocabulary',151935),('token_chunk',512),('trajectory_enclosure',True),
                  ('certificate_passed',True),('reachability_proved',True),('conditional_operator_bound_only',False),
                  ('regularity_contract','centre Jacobian only'),('status','prepared')]:
    early_card('card '+key+' mismatch refuses before evidence reads',lambda c,k=key,v=value:c.__setitem__(k,v))
early_card('card wrong original prefix pin refuses before evidence reads',lambda c:c['prefix'].__setitem__('prefix_receipt_sha256','0'*64))

result=dict(schema='codex-task33-tube-driver-focused-controls-v1',source_sha256=SOURCE_SHA,
    count=len(rows),passed=True,controls=rows,
    scope='Actual installed original_execution/candidate and early current_admission branches; manufactured896-coordinate records. Pin/source/Git/filesystem interfaces mocked. No target/raw/model/process access. No old operator27/receiver9/audit33+15 controls rerun. Full runtime admission/coverage/backend/source review and target execution remain pending.')
print(json.dumps(result,indent=2))
