"""Analyze saved S1R map without merging independently initialized NS families."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from neural_shrimp.rotation_map_audit import DEFAULTS, FIELDS, analyze

SOURCE_FILES = ['src/neural_shrimp/rotation_map_audit.py','tools/analyze_s1r_rotation_map.py']
OUTPUT_FILES = ['analysis.json','rotation-map.png','rotation-map.pdf','figure-metadata.json','runtime.json']


def sha(path):
    digest = hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): digest.update(chunk)
    return digest.hexdigest()


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')


def load(inputs):
    with np.load(inputs['map']) as data:
        m = {k:data[k] for k in ('log10_hidden','log10_output',*FIELDS)}
    meta = json.loads(Path(inputs['metadata']).read_text())
    families = {name:json.loads(Path(inputs[name]).read_text()) for name in ('start_1.10','start_0.95')}
    if meta['args']['system']!='S1R' or meta['args']['seed']!=28002:
        raise ValueError('wrong map system/seed')
    for f in families.values():
        if f['args']['system']!='S1R' or f['args']['seed']!=28002:
            raise ValueError('wrong trace system/seed')
    return m,meta,families


def render(m,families,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained',sharex=True,sharey=True)
    show=np.ma.masked_where((m['status']==0)|(m['status']==1)|(m['status']==3)|(m['status']==5)
          |((m['status']==2)&(m['period']<=2))|~np.isfinite(m['rho']),m['rho'])
    h,o=m['log10_hidden'],m['log10_output']
    # Half-cell outer edges retain the map's actual sample coordinates.
    extent=[o[0]-(o[1]-o[0])/2,o[-1]+(o[1]-o[0])/2,h[0]-(h[1]-h[0])/2,h[-1]+(h[1]-h[0])/2]
    for ax,(name,f) in zip(axs,families.items()):
        im=ax.imshow(show,origin='lower',aspect='auto',extent=extent,vmin=0,vmax=0.5,cmap='viridis',interpolation='nearest')
        rows=[r for r in f['rows'] if r['type']=='neimark_sacker']
        ax.scatter([r['log10_output'] for r in rows],[r['log10_hidden_boundary'] for r in rows],
                   s=22,facecolors='none',edgecolors='red',linewidths=.8,label='independent NS samples')
        ax.set_title(f"Start hidden log-rate {f['args']['start']:.2f}; no branch joining")
        ax.set_xlabel('log10 output learning rate'); ax.legend(fontsize=8,loc='upper right')
    axs[0].set_ylabel('log10 hidden learning rate')
    fig.colorbar(im,ax=axs,label='Unsigned projected turns per two GD steps',shrink=.85)
    fig.suptitle('S1R seed 28002; exploratory map and separate start families',fontsize=11)
    fig.savefig(out/'rotation-map.png',dpi=170)
    fig.savefig(out/'rotation-map.pdf',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)
    write(out/'figure-metadata.json',dict(alt='Two views of the same S1R projected rotation map with independent NS boundary samples for hidden starts 1.10 and 0.95. Samples are not joined because branch identity is unestablished.',color_limits=[0,0.5],branch_connections_drawn=False,unresolved_status_is_not_quasiperiodicity=True))


def verify(receipt_path):
    receipt=json.loads(receipt_path.read_text())
    if receipt['schema']!='s1r-rotation-audit-v1': raise ValueError('unknown schema')
    if set(receipt['sources'])!=set(SOURCE_FILES): raise ValueError('incomplete source inventory')
    for name,digest in receipt['sources'].items():
        if sha(ROOT/name)!=digest: raise ValueError('source hash mismatch: '+name)
    if set(receipt['inputs'])!={'map','metadata','start_1.10','start_0.95'}: raise ValueError('input inventory mismatch')
    for record in receipt['inputs'].values():
        if sha(record['path'])!=record['sha256']: raise ValueError('input hash mismatch')
    out=Path(receipt['output_directory'])
    if set(receipt['outputs'])!=set(OUTPUT_FILES): raise ValueError('output inventory mismatch')
    if {p.name for p in out.iterdir() if p.is_file()}!=set(OUTPUT_FILES)|{'receipt.json'}:
        raise ValueError('raw directory inventory mismatch')
    for name,digest in receipt['outputs'].items():
        if sha(out/name)!=digest: raise ValueError('output hash mismatch: '+name)
    m,meta,families=load({k:v['path'] for k,v in receipt['inputs'].items()})
    report=analyze(m,families,receipt['config'],int(meta['args']['samples']))
    if report!=json.loads((out/'analysis.json').read_text()): raise ValueError('replay mismatch')
    if json.loads((out/'receipt.json').read_text())!=receipt: raise ValueError('receipt mirror mismatch')
    print('PASS hash inventory and full analysis replay')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map'); parser.add_argument('--main-trace'); parser.add_argument('--lower-trace')
    parser.add_argument('--out'); parser.add_argument('--verify',type=Path)
    args=parser.parse_args()
    if args.verify:
        verify(args.verify); return
    if not all((args.map,args.main_trace,args.lower_trace,args.out)): parser.error('missing inputs/out')
    started=time.perf_counter(); out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    inputs={'map':str(Path(args.map).resolve()),'metadata':str(Path(args.map).with_name('metadata.json').resolve()),
            'start_1.10':str(Path(args.main_trace).resolve()),'start_0.95':str(Path(args.lower_trace).resolve())}
    sources={name:sha(ROOT/name) for name in SOURCE_FILES}
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--',*SOURCE_FILES],cwd=ROOT,text=True).strip()
    if dirty: raise ValueError('analysis sources must be committed')
    m,meta,families=load(inputs)
    report=analyze(m,families,DEFAULTS,int(meta['args']['samples']))
    write(out/'analysis.json',report); render(m,families,out)
    write(out/'runtime.json',dict(python=platform.python_version(),numpy=np.__version__,
          seconds=time.perf_counter()-started,hostname=platform.node(),
          threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}))
    receipt=dict(schema='s1r-rotation-audit-v1',source_commit=source_commit,sources=sources,
                 inputs={k:dict(path=v,sha256=sha(v)) for k,v in inputs.items()},config=DEFAULTS,
                 output_directory=str(out.resolve()),outputs={name:sha(out/name) for name in OUTPUT_FILES},
                 evidence='exploratory float diagnostics; neither confirmed branch connectivity nor tongue ancestry',
                 upstream_provenance='Historical map run card: 959beee68b3a dirty tree. Metadata binds native source/library, not the complete producer Python tree. This receipt binds saved inputs and analysis only.',
                 stage_mismatch='Period/status/FTLE are from 20000-step screen, before 256 PCA samples and 6000 rotation samples, each over two GD steps. No post-rotation period or FTLE recheck.',
                 root_scope='Interpolated roots on geometric segments only; candidate period 2q and rotation agreement are descriptive filters, not locking evidence. Search covers all map cells, with no asserted beyond-boundary relation or null p-value.')
    write(out/'receipt.json',receipt)
    print(json.dumps({k:dict(row_types=v['row_types'],edge_counts=v['edge_counts'],roots=len(v['roots']),comparisons=[{a:b for a,b in c.items() if a!='pairs'} for c in v['comparisons']]) for k,v in report['families'].items()},indent=2))


if __name__=='__main__': main()
