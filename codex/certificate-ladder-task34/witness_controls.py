"""Complete small synthetic witnesses for the Task34 declared 1/2/4/8 scope.

No target records, model, native target matrices or raw evidence are accessed.
The generic conditional gates generate 3-phase, 2-coordinate exact synthetic
witnesses. Four nonnormal examples first satisfy the norm criterion at 1,2,4,8.
"""
import ast,copy,hashlib,importlib.util,json,sys
from pathlib import Path
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp')
sys.path.insert(0,str(R/'codex/certify-llm'))
from flint import ctx,arb,arb_mat
import shooting_certificate as certificate
import shooting_gates as gates
from proof_gates import vector_record
W=Path(__file__).with_name('witness_audit.py');S=Path(__file__).with_name('saved_witness.py')
spec=importlib.util.spec_from_file_location('new_witness',W);w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
ctx.prec=128;ctx.threads=1
count=0;checks=[]
def passed(label):
 global count
 count+=1;checks.append(label);print(label+': passed')
def refusal(label,fn,message):
 try:fn()
 except (ValueError,KeyError)as error:assert message in str(error),(label,str(error))
 else:raise AssertionError(label+' unexpectedly accepted')
 passed(label)

def fixture(b):
 centers=[[arb(i*4),arb(0)]for i in range(3)];radii=[arb((1,-4))]*3
 images=centers[1:]+centers[:1]
 # This root-specific enclosure premise is manufactured solely for controls.
 # Each phase has the displayed affine derivative on disjoint phase boxes.
 m=arb_mat([[arb((1,-1)),arb(b)],[arb(0),arb((1,-1))]])
 eye=arb_mat([[arb(1),arb(0)],[arb(0),arb(1)]])
 jacobians=[m,eye,eye];A=jacobians;B=arb_mat([[arb(2),arb(4)*arb(b)],[arb(0),arb(2)]])
 contract=gates.ShootingEnclosureContract('1'*64,'2'*64,'3'*64,3,
  gates.domain_sha256(centers,radii),gates.enclosures_sha256(images,jacobians),
  gates.proposals_sha256(A,B),gates.SHOOTING_DECLARATION)
 result=certificate.check_shooting_certificate(centers,radii,images,jacobians,A,B,
  contract=contract,bound_method='factored',powers=(1,2,4,8))
 external=dict(centers=[vector_record(c)for c in centers],radii=vector_record(radii),
  images=[vector_record(c)for c in images],contract=result['contract'],
  jacobian_hashes=[gates._matrix_sha256(m)for m in jacobians])
 return result,external

examples={}
for b,first in [((1,-2),1),((1,-1),2),(1,4),(2,8)]:
 result,external=fixture(b);check=w.audit(result,**external)
 assert check['witness_arithmetic_consistent']is True and check['conditional_gates_passed']is True
 assert check['matrix_row_norm_entries_checked']==16 and check['scalar_blocks_checked']==9
 assert result['attraction']['witness_power']==first
 assert [r['power']for r in result['attraction']['powers']]==[1,2,4,8]
 examples[first]=(result,external);passed('complete synthetic witness first passing power '+str(first))

base,external=examples[8]
def mutate(change):
 result=copy.deepcopy(base);change(result);return w.audit(result,**external)
refusal('missing declared power four',lambda:mutate(lambda r:r['attraction']['powers'].pop(2)),'declared powers')
refusal('mismatched declared power four relabeled three',lambda:mutate(lambda r:r['attraction']['powers'][2].update(power=3)),'declared powers')
refusal('reordered declared powers',lambda:mutate(lambda r:r['attraction']['powers'].reverse()),'declared powers')
for idx,power in [(2,4),(3,8)]:
 refusal('understated higher-power row '+str(power),lambda idx=idx:mutate(lambda r:r['attraction']['powers'][idx]['weighted_row_upper'].__setitem__(0,{'mid':[0,0],'rad':[0,0]})),'saved matrix row norm understated')
refusal('aggregate witness wrongly identifies power four',lambda:mutate(lambda r:r['attraction'].update(witness_power=4)),'attraction witness differs')
refusal('higher power margin overstates its saved upper',lambda:mutate(lambda r:r['attraction']['powers'][3].update(margin={'mid':[1,0],'rad':[0,0]})),'attraction bound or margin understated')

# Source delta is exactly the two scope literals plus their failure description.
old=(R/'codex/certify-llm/witness_audit.py').read_text()
expected=old.replace("== [1,2], 'frozen powers one and two required'","== [1,2,4,8], 'declared powers one, two, four and eight required'").replace('zip((1,2),power_flags)','zip((1,2,4,8),power_flags)')
assert W.read_text()==expected;passed('only declared-power scope and matching aggregate changed')

# The wrapper interface is tested without importing or executing its target main.
module=ast.parse(S.read_text());node=next(n for n in module.body if isinstance(n,ast.FunctionDef)and n.name=='expected_input')
namespace={'Path':Path,'CARD':R/'codex/certificate-ladder-task34/cascade-p3-assembly-001-card.json'}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(S),'exec'),namespace)
expected_input=namespace['expected_input'];card=str(namespace['CARD'])
assert expected_input('a'*40,card)==namespace['CARD'];passed('explicit full source and exact Task34 card interface')
refusal('short source refused',lambda:expected_input('a'*8,card),'full lowercase source')
refusal('uppercase source refused',lambda:expected_input('A'*40,card),'full lowercase source')
refusal('foreign card namespace refused',lambda:expected_input('a'*40,card.replace('certificate-ladder-task34','coexistence-task33')),'exact original Task34')
compile(S.read_text(),str(S),'exec');passed('saved wrapper syntax')
print(json.dumps(dict(controls_passed=count,checks=checks,model_calls=0,target_records_read=0,
 generic_sha256=hashlib.sha256(old.encode()).hexdigest(),
 checker_sha256=hashlib.sha256(W.read_bytes()).hexdigest(),wrapper_sha256=hashlib.sha256(S.read_bytes()).hexdigest()),indent=2))
