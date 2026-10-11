"""Saved-only, complete-grid response summary; no new numerical solves."""
import hashlib,json,math,os,statistics,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/data2/neural-shrimp-data/explore/clark-response')
PINS={'attempt-001-profile':'c3ca49809f01c977c1d82d5675c01bd1508151667a10138eafda6ca992f76064','attempt-001-panel':'8d537914c959b2ec4ad348f24a050e31e3e71f2779d09671028151227c6f9896','attempt-002-profile':'f632e71bb41904be3961fac77464fd7fd517cc45eac5f3175b3cce1aaf00f73f','attempt-003-profile':'0adc6365139f36de7a2d33d33f19ba6d542bae4a86f7b7356501a4b962e27309','attempt-002-panel':'177b909a2274272197dfca42a13ee785924ea31e71cc14ed54f13133081dd364'}
def read(p):return json.loads(p.read_text())
def ent(p):b=p.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def main():
    assert os.getpriority(os.PRIO_PROCESS,0)>=10
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    out=BASE/'attempt-002-summary';out.mkdir(exist_ok=False);start=time.monotonic();records=[]
    for name,pin in PINS.items():
        root=BASE/name;assert ent(root/'manifest.json')['sha256']==pin;inv=read(root/'manifest.json')
        assert set(inv['files'])=={str(p.relative_to(root))for p in root.rglob('*')if p.is_file()}-{'manifest.json'}
        for name2,e in inv['files'].items():
            p=root/name2;assert ent(p)==e;assert not p.stat().st_mode&0o222
        launch=read(root/'launcher-result.json');assert not launch['owned_live_pids']
        for pid in launch['observed_owned_pids']:assert not Path('/proc',str(pid)).exists()
        records.append({'root':str(root),'manifest_sha256':pin,'files':len(inv['files']),'bytes':inv['total_bytes'],'worker':read(root/'run/result.json'),'execution':launch['source_commit'],'outer_seconds':launch['elapsed_seconds'],'exit_code':launch['exit_code'],'owned_pids_absent':True})
    root=BASE/'attempt-002-panel/run';rows=read(root/'summary.json');assert len(rows)==216
    grouped=[]
    for label in ('p10-row35','p18-row111'):
        for boundary in ('free','periodic','causal'):
            for rep in (4,16,64):
                for eta in (.01,.0001,.000001):
                    sel=[r for r in rows if r['label']==label and r['boundary']==boundary and r['repeats']==rep and r['eta']==eta]
                    assert len(sel)==4 and {r['shift']for r in sel}=={-.5,-.05,-.005,0}
                    errors=[abs(r['count_error'])for r in sel if r['count_error']is not None]
                    grouped.append({'label':label,'boundary':boundary,'periods':rep,'eta':eta,'count_errors_by_shift':[{k:r[k]for k in ('shift','count','floquet_count','qr_count','threshold_gap','count_error','solve_qualified','identity_assumptions_numerically_screened','floquet_qr_count_agree')}for r in sel],'max_abs_count_error':max(errors),'mean_abs_count_error':statistics.mean(errors)})
    probes=[];refs=[]
    for label in ('p10-row35','p18-row111'):
        p=read(root/(label+'-matrixfree.json'));c=read(root/(label+'-matrixfree-comparison.json'));r=read(root/(label+'-reference.json'))
        probes.append({'label':label,**{k:v for k,v in p.items()if k not in ('solution','readout_column')},**c})
        refs.append({'label':label,**{k:r[k]for k in ('multipliers_real','multipliers_imag','multiplier_angles','exponents_per_step','eigen_residuals','numerically_invertible_phases','qr_sorted_per_step','product_vs_saved_midpoint_frobenius')}})
    result={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'summary_source':ent(Path(__file__)),'attempts':records,'rows':216,'solve_qualified':sum(r['solve_qualified']for r in rows),'identity_screened_rows':sum(r['identity_assumptions_numerically_screened']for r in rows),'floquet_qr_agree_rows':sum(r['floquet_qr_count_agree']for r in rows),'maximum_augmented_residual':max(r['true_augmented_relative_residual']for r in rows),'groups':grouped,'matrixfree':probes,'references':refs,'elapsed_seconds':time.monotonic()-start,'scope':'Exploratory finite-window method diagnostic; all fixed rows summarized, no certificate, cavity closure, large-model or utility claim.'}
    data=(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();assert len(data)<1024**2
    (out/'summary.json').write_bytes(data);(out/'source.py').write_bytes(Path(__file__).read_bytes());(out/'manifest.json').write_text(json.dumps({'files':{p.name:ent(p)for p in out.iterdir()}},indent=2,sort_keys=True)+'\n')
    for p in out.iterdir():p.chmod(0o444)
    out.chmod(0o555)
    print(json.dumps({'root':str(out),'manifest':ent(out/'manifest.json'),'rows':216,'qualified':result['solve_qualified'],'reference_agree':result['floquet_qr_agree_rows'],'probes':probes}),flush=True)
if __name__=='__main__':main()
