"""Post-pilot descriptive saved-plane arithmetic; no models or decompositions."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse
from datetime import datetime,timezone
import json,platform,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from neural_shrimp.period2_branch_io import Store,digest,file_entry,load_record,identity
RAW=Path('/data2/neural-shrimp-data/explore/curvature-mechanism/attempt-001-panel')
MANIFEST='898880f0bc1755896251f39e5abf806cc89f32a1558ed102fc0c954bd6d9ff4b'
CARD='docs/plans/curvature-saved-plane-diagnostic.md'
SOURCES=('src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py',
 'src/neural_shrimp/period2_branch.py','src/neural_shrimp/period2_branch_io.py',
 'tools/explain_curvature_plane.py',CARD,'docs/plans/curvature-subspace-input-inventory.json')
STOP=False

def require(ok,message):
    if not ok:raise ValueError(message)

def request_stop(*_):
    global STOP
    STOP=True

def fro(x):return float(np.sqrt(np.sum(np.abs(x)**2)))

def diagnose(analysis):
    """Euclidean quantities are in the pilot's whitened coordinates, not raw theta."""
    M=np.asarray(analysis['target']);n=len(M);require(M.shape==(n,n) and not np.iscomplexobj(M) and np.isfinite(M).all(),'finite real square target')
    u,v=[np.asarray(row['u'],dtype=np.float64) for row in analysis['phases']]
    require(u.shape==v.shape==(n,) and np.isfinite(u).all() and np.isfinite(v).all(),'stored top vectors')
    un,vn=fro(u),fro(v);require(un>0 and vn>0,'nonzero top vectors');q0=u/un;vunit=v/vn
    cosine=float(q0@vunit);require(abs(cosine)<=1+1e-12,'normalized overlap')
    aligned=1-cosine*cosine<=1e-10;remainder=vunit-q0*(q0@vunit)
    if aligned:Q=q0[:,None]
    else:
        length=fro(remainder);require(length>0,'nonzero Gram-Schmidt remainder');Q=np.column_stack((q0,remainder/length))
    saved_pair=analysis['target_pair'];indices=np.asarray(saved_pair['indices']);values=np.asarray(analysis['target_eigensystem']['values']);vectors=np.asarray(analysis['target_eigensystem']['right'])
    require(indices.shape==(2,) and np.issubdtype(indices.dtype,np.integer) and len(set(indices.tolist()))==2 and np.all((indices>=0)&(indices<n)),'saved target pair indices')
    require(values.shape==(n,) and vectors.shape==(n,n) and np.isfinite(values).all() and np.isfinite(vectors).all(),'saved full eigensystem')
    target=values[indices];require(np.array_equal(target,np.asarray(saved_pair['values'])),'saved target-pair binding')
    right=vectors[:,indices];energies=np.sum(np.abs(right)**2,axis=0);require(np.all(energies>0),'nonzero leading right vectors')
    coefficients=Q.T@right;projection=Q@coefficients;residual=right-projection
    capture=np.sum(np.abs(coefficients)**2,axis=0)/energies
    MQ=M@Q;compression=Q.T@MQ;leakage=MQ-Q@compression
    out=dict(Q=Q,dimension=Q.shape[1],aligned_threshold_used=bool(aligned),cosine=cosine,Gram_Schmidt_remainder=remainder,
        orthogonality_residual=fro(Q.T@Q-np.eye(Q.shape[1])),target_indices=indices,target_values=target,
        target_right_vectors=right,right_energy=energies,captured_coefficients=coefficients,captured_vectors=projection,
        uncaptured_vectors=residual,capture_fractions=capture,uncaptured_energy_fractions=np.sum(np.abs(residual)**2,axis=0)/energies,
        MQ=MQ,compression=compression,leakage=leakage,leakage_frobenius=fro(leakage),
        relative_leakage_frobenius=fro(leakage)/max(1.,fro(MQ)),target_matrix_identity=identity(M),
        original_model_pair=analysis['models']['rotating']['leading']['values'],
        original_pair_error=analysis['comparisons']['rotating']['normalized'],
        original_analysis_status=analysis['status'],original_attribution=analysis['attribution']['qualified'],
        scope='Target-informed descriptive compression after a negative pilot; not prediction, attribution, full-spectrum fitting or a replacement engineering screen.')
    if Q.shape[1]==2:
        trace=float(compression[0,0]+compression[1,1]);det=float(compression[0,0]*compression[1,1]-compression[0,1]*compression[1,0]);discriminant=trace*trace-4*det
        root=np.sqrt(complex(discriminant));roots=np.array([(trace+root)/2,(trace-root)/2])
        errors=[float(np.max(np.abs(roots-target))),float(np.max(np.abs(roots-target[::-1])))];swap=int(errors[1]<errors[0]);radius=float(np.max(np.abs(values)))
        out.update(compression_roots=roots,compression_trace=trace,compression_determinant=det,compression_discriminant=discriminant,
            unordered_assignment=np.array([1,0] if swap else [0,1]),compression_pair_error=errors[swap]/max(1.,radius),
            compression_status='explicit quadratic roots; rounding and near-repeated roots unvalidated')
    else:out.update(compression_roots=np.array([compression[0,0]],dtype=complex),compression_pair_error=None,compression_status='one-dimensional aligned-limit convention; no two-mode pair comparison')
    a,b=analysis['phases'];Ea,Eb=a['E'],b['E'];Sa0,Sb0=a['S0'],b['S0']
    terms=dict(native_formation=(M-analysis['pair_product'])@right,
        first_phase_omission=Sb0@(Ea@right),second_phase_omission=Eb@(Sa0@right),
        interacting_omissions=Eb@(Ea@right))
    total=sum(terms.values());actual=(M-analysis['model_product'])@right;identity_residual=actual-total
    column_norm=lambda x:np.sqrt(np.sum(np.abs(x)**2,axis=0))
    out['residual_action']=dict(terms=terms,sum=total,actual_difference_action=actual,identity_residual=identity_residual,
        term_norms={k:column_norm(t) for k,t in terms.items()},
        term_unit_vector_norms={k:column_norm(t)/np.sqrt(energies) for k,t in terms.items()},
        sum_norms=column_norm(total),identity_residual_norms=column_norm(identity_residual),
        scope='Exact algebra evaluated in floating arithmetic on the supplied eigenvectors; cancellation/action diagnostics, not causal attribution of eigenvalue error.')
    require(all(np.isfinite(t).all() for t in list(terms.values())+[total,actual,identity_residual]),'finite residual-action terms')
    require(all(np.isfinite(value).all() for value in out.values() if isinstance(value,np.ndarray)),'finite diagnostic returns')
    return out


def sources():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean checkout')
    result={}
    for name in SOURCES:
        b=(ROOT/name).read_bytes();require(b==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed source '+name);result[name]=dict(bytes=len(b),sha256=digest(b))
    return result


def run(out):
    own=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice>=10')
    store=Store(out,1024**2);started=time.monotonic();rows=[]
    expected=[s['slot'] for s in json.loads((ROOT/'docs/plans/curvature-subspace-input-inventory.json').read_bytes())['panel']]
    result=dict(schema='curvature-saved-plane-v1',status='running',adapted_after_negative_pilot=True,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources=own,rows=rows,input_root=str(RAW),input_manifest_sha256=MANIFEST,new_model_calls=0,new_decompositions=0)
    def check():
        if STOP or time.monotonic()-started>=10:raise RuntimeError('saved arithmetic deadline or signal')
    try:
        store.json('runtime-start.json',dict(**result,pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),numpy=np.__version__,nice=os.getpriority(os.PRIO_PROCESS,0),threads={k:os.environ[k] for k in THREADS},internal_seconds=10,store_bytes=1024**2))
        require(file_entry(RAW/'manifest.json')['sha256']==MANIFEST,'exact input manifest')
        manifest=json.loads((RAW/'manifest.json').read_bytes());require(set(p.name for p in RAW.iterdir())==set(manifest['files'])|{'manifest.json'},'complete input inventory')
        for name,pin in manifest['files'].items():
            p=RAW/name;require(Path(name).name==name and not p.is_symlink() and file_entry(p)==pin,'input file identity '+name);check()
        runtime=json.loads((RAW/'runtime-start.json').read_bytes())
        for name,pin in runtime['sources'].items():
            b=subprocess.check_output(['git','show',runtime['source_commit']+':'+name],cwd=ROOT);require(dict(bytes=len(b),sha256=digest(b))==pin,'input execution source '+name)
        store.json('input-binding.json',dict(producer_execution=runtime['source_commit'],producer_sources=runtime['sources'],manifest_sha256=MANIFEST,inventory=manifest['files']))
        events=[json.loads(s) for s in (RAW/'journal.jsonl').read_text().splitlines()];records=[r for r in events if r['kind']=='analysis'];require(len(records)==6,'all six analysis records')
        seen=[]
        for event in records:
            check();kind,p=load_record(RAW,event['file']);require(kind=='analysis' and p['slot']==expected[len(seen)],'unchanged fixed slot order');seen.append(p['slot'])
            try:
                value=diagnose(p['analysis']);store.emit('plane_diagnostic',dict(slot=p['slot'],source_record=event['file'],source_record_identity=file_entry(RAW/event['file']),**value))
                rows.append(dict(slot=p['slot'],status='descriptive_complete',dimension=value['dimension'],capture_fractions=value['capture_fractions'].tolist(),leakage_frobenius=value['leakage_frobenius'],compression_pair_error=value['compression_pair_error'],original_pair_error=value['original_pair_error']))
            except Exception as error:
                row=dict(slot=p['slot'],status='failed',error=type(error).__name__+': '+str(error));rows.append(row);store.emit('slot_failure',row)
        require(seen==expected,'all fixed slots present');result['status']='descriptive_complete' if all(r['status']=='descriptive_complete' for r in rows) else 'descriptive_partial'
    except BaseException as error:
        result.update(status='failed_or_interrupted',error=type(error).__name__+': '+str(error))
        seen={r['slot'] for r in rows}
        rows.extend(dict(slot=name,status='not_completed',reason=result['error']) for name in expected if name not in seen)
    finally:
        result.update(elapsed_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat());store.finish(result)
        for p in store.root.iterdir():p.chmod(0o444)
        store.root.chmod(0o555)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args();signal.signal(signal.SIGTERM,request_stop)
    r=run(args.out);print(json.dumps(r,sort_keys=True));return 0 if r['status']=='descriptive_complete' else 2
if __name__=='__main__':raise SystemExit(main())
