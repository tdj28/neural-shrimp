#!/usr/bin/env python3
"""Saved-proposal weighted adapter; unchanged complete-domain interval kernel."""
import hashlib,math,subprocess,sys
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import certify_eigenpair_locus as B
BASE_EXEC='463e55775ec89d9f45c00ebf594f616a09b278f3'
FAILED=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/attempt-001-target')
FAILED_OUTER='35c8aeb88105a5836d47de7b6747b74da789bc266ab6084731b1f33688798369'
FAILED_RUN='6e21d614a08c336e3d7cde3127e0a6f8d6d4af461e261282df56df95106dbe26'
DIAG=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/attempt-002-diagnosis')
DIAG_EXEC='ca7cdeef4151115ab86f2336faa3c99ecfa47492'
DIAG_OUTER='dd5d8db9415931e319e70033c8843b295d1aaaa8bcf5ed2d4fc6a6c50d267206'
DIAG_RUN='0bc09fa9f98650659daf387d1b61d9d63b765ee83c2eb005f4d472d0029ea22c'
PROPOSAL_SHA='bb037a18244506702d068761b2b30a7e522e7c445d66676ab660092e23c8808c'
EXTRA=['tools/certify_eigenpair_locus_weighted.py','tools/launch_eigenpair_locus_weighted.py',
       'tests/test_eigenpair_locus_weighted.py','docs/plans/eigenpair-locus-weighted.md']


def exact_weights(proposal):
    """Interpret every saved float exactly; neither refit nor clip any entry."""
    radii=proposal['linear_radii_proposal']['radii']
    B.require(type(radii) is list and len(radii)==470,'complete470 proposal')
    B.require(all(type(v) is float and math.isfinite(v) and 0<v<=2.**-18 for v in radii),'fixed positive finite radius range')
    values=[Fraction.from_float(v) for v in radii]
    return [str(v*2**18) for v in values], [str(v) for v in values]


def measured_gate(outcome,runtime,launcher):
    """Prior acquisition failed science, but its finite measured cost is usable."""
    B.require(runtime['commit']==launcher['commit']==BASE_EXEC,'first target source')
    B.require(outcome['status']=='proof_gate_inconclusive' and outcome['passed'] is False,'retained failed target')
    B.require(launcher['exit_code']==2 and launcher['stop_reason'] is None and not launcher['owned_live_pids'],'scientific exit only')
    B.require(runtime['settings']=={'period':18,'sigma':1,'halfwidth_exponent':-20,'radius_exponent':-24,'newton_steps':2,'precision':256},'original chart settings')
    B.require(outcome['ticks']==109 and outcome['event_counts']=={'anchor_phase_return':18,'chart_return':1,'phase_return':90,'preconditioner_return':1,'proposal':2},'one full prior chart')
    B.require(B.valid_resource(outcome['elapsed_seconds'],120) and B.valid_resource(outcome['peak_rss_kib'],2*1024**2) and B.valid_resource(launcher['elapsed_seconds'],142),'finite prior resources')
    B.require(runtime['python']==sys.version and runtime['numpy']==B.np.__version__ and runtime['flint']==B.flint.__version__ and runtime['flint_library']==B.flint.__FLINT_VERSION__,'measured library identity')
    B.require(runtime['threads']==dict.fromkeys(B.THREADS,'1') and runtime['nice']>=10,'measured CPU1nice10')
    B.require(type(outcome['stored_bytes_before_final']) is int and 0<outcome['stored_bytes_before_final']<248*1024**2,'finite retained storage')
    # Count the old anchor twice and allow2x its entire two-chart estimate.
    seconds=4*outcome['elapsed_seconds']+10
    storage=3*outcome['stored_bytes_before_final']+8*1024**2
    B.require(seconds<120 and storage<256*1024**2,'measured amended cost gate')
    return {'seconds':seconds,'bytes':storage,'method':'4*old one-chart elapsed+10;3*old one-chart stored bytes+8MiB; heuristic only, unchanged actual caps'}


def bind_proposal(base_sources):
    bindings={'failed_target':B.inventory(FAILED,FAILED_OUTER),'failed_target_run':B.inventory(FAILED/'run',FAILED_RUN),
              'diagnosis':B.inventory(DIAG,DIAG_OUTER),'diagnosis_run':B.inventory(DIAG/'run',DIAG_RUN)}
    rt=B.read(FAILED/'run/runtime.json');out=B.read(FAILED/'run/outcome.json');launch=B.read(FAILED/'launcher-result.json')
    for name in base_sources:
        digest=hashlib.sha256(subprocess.check_output(['git','show',BASE_EXEC+':'+name],cwd=ROOT)).hexdigest()
        B.require(B.entry(ROOT/name)==B.entry(FAILED/'run/source'/name)==rt['sources'][name] and digest==rt['sources'][name]['sha256'],'unchanged target kernel/adapter '+name)
    bindings['measured_cost']=measured_gate(out,rt,launch)
    drt=B.read(DIAG/'run/runtime.json');dout=B.read(DIAG/'run/outcome.json');dl=B.read(DIAG/'launcher-result.json')
    B.require(drt['commit']==dl['commit']==DIAG_EXEC and dout['passed'] is True and dout['native_calls']==0 and dout['positive_resolvent_proposal'] is True,'saved-only proposal source/scope')
    B.require(dl['exit_code']==0 and dl['stop_reason'] is None and not dl['owned_live_pids'],'proposal completed')
    B.require(B.valid_resource(dout['elapsed_seconds'],45) and B.valid_resource(dout['peak_rss_kib'],1536*1024) and B.valid_resource(dl['elapsed_seconds'],67),'proposal finite resources')
    for name,rec in drt['sources'].items():
        digest=hashlib.sha256(subprocess.check_output(['git','show',DIAG_EXEC+':'+name],cwd=ROOT)).hexdigest()
        B.require(B.entry(DIAG/'run/source'/name)==rec and digest==rec['sha256'],'proposal archived source '+name)
    B.require(B.entry(DIAG/'run/floating-proposals.json')['sha256']==PROPOSAL_SHA,'complete fixed proposal bytes')
    weights,radii=exact_weights(B.read(DIAG/'run/floating-proposals.json'))
    bindings.update(proposal_sha256=PROPOSAL_SHA,exact_coordinate_radii=radii,weights=weights,
                    scope='Floating saved radii are proposals. Full interval Jacobian is reacquired on the changed box; all original strict proof gates remain decisive.')
    return weights,bindings


def install():
    """Only adapter configuration changes; retain every original source byte."""
    base_sources=list(B.SOURCES);original_bind=B.bind_inputs
    B.SOURCES=base_sources+EXTRA
    B.SETTINGS=dict(B.SETTINGS,radius_exponent=-18)
    def bind():
        inputs,proof,bindings=original_bind()
        weights,new=bind_proposal(base_sources)
        B.SETTINGS['weights']=weights
        bindings['weighted_amendment']=new
        return inputs,proof,bindings
    B.bind_inputs=bind


def main(out):
    install()
    return B.main(out)

if __name__=='__main__':raise SystemExit(main(Path(sys.argv[1])))
