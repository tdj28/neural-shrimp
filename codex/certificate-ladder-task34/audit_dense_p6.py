"""Task34 cascade-P6 saved dense audit; source preparation, no model execution.

The numerical replay block is unchanged from accepted audit_dense.py
(cbd60cab0193b92af82893440d3bed26e5bc920b43f577b683252e1d2b84ab64).
Only the exact producer allocation and original evidence interfaces are adapted.
"""
import argparse,json,hashlib,subprocess,sys,os
from pathlib import Path
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp')
D=R/'codex/certificate-ladder-task34'
B=Path('/data2/neural-shrimp-data/codex/certificate-ladder')
SOURCE='19b63a2002b72ff5bcdaea441a141b58ed05e272'
ATTEMPT='dense-cascade-p6-001'
CELL='cascade-p6'
PERIOD=6
RAW=B/ATTEMPT
SUP=RAW.with_name(RAW.name+'-supervision')
CARD=D/'dense-cascade-p3-001-card.json'  # Original shared five-cell card, not a P3 substitute.
PLAN=D/'dense-cascade-p6-001-plan.json'
CARD_SHA='fad1beeb1eda8afaed84bcf315d27ffc1f1e446f3c2694b9690c1c702e87aa4a'
PLAN_SHA='3217b86fe016cccab366f519bda3bcf1f5ce0b882efe5a62d728bf90d2f2bbb5'
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
sys.path.insert(0,str(R/'codex/certify-chaos'));import supervise as s

def require(ok,message):
 if not ok:raise ValueError(message)

def arguments(argv=None):
 ap=argparse.ArgumentParser()
 for name in ('attempt','source','cell','output'):ap.add_argument('--'+name,required=True)
 a=ap.parse_args(argv)
 require(a.attempt==ATTEMPT and a.source==SOURCE and a.cell==CELL,'wrong original producer allocation')
 out=Path(a.output)
 require(out==B/'dense-cascade-p6-001-audit-001/result.json','fresh owned saved-audit allocation required')
 require(not any(out==p or p in out.parents for p in (RAW,SUP)),'audit output cannot overwrite original producer evidence')
 return a

def expected_cell(card):
 require(card['repository']==str(R),'foreign card repository')
 cells=[x for x in card['cells'] if x['name']==CELL]
 require(len(cells)==1,'missing or duplicate named cascade-P6 allocation')
 c=cells[0]
 require(c['period']==PERIOD and c['alpha']==.065693 and c['log10T']==-.14798,'wrong named-cell parameters or period')
 require(c['alpha_ratio']==[147927485160425,2251799813685248] and c['log10T_ratio']==[-333221336429143,2251799813685248],'wrong exact named-cell parameter ratios')
 require(c['states']==str(B/'settle-001/cascade-p6-states.npy') and c['settle_report']==str(B/'settle-001/cascade-p6-report.json'),'wrong cascade-P6 saved history authority')
 require(tuple(c['alpha'].as_integer_ratio())==tuple(c['alpha_ratio']) and tuple(c['log10T'].as_integer_ratio())==tuple(c['log10T_ratio']),'binary64 parameter mismatch')
 return c

def allocation(rc,ad,plan,card):
 """Pure metadata gate; no target, process, Git, or model access."""
 c=expected_cell(card);lane=plan['lanes']['dense']
 require(plan['repository']==str(R) and plan['raw_root']==str(B) and plan['status']=='reviewed_for_execution','foreign or unreviewed execution plan')
 cmd=['/data2/neural-shrimp/.venv/bin/python','-I','-B',str(D/'dense_qualify.py'),'--card',str(CARD),'--cell',CELL,'--out',str(RAW)]
 require([plan['python'],'-I','-B',lane['entry'],*lane['arguments']]==cmd,'wrong exact planned command')
 require(lane['output']==str(RAW) and lane['supervision']==str(SUP),'wrong planned output or supervision')
 require(rc['source_commit']==ad['source_commit']==SOURCE,'wrong full producer source')
 require(rc['lane']==ad['lane']=='dense' and rc['argv']==ad['argv']==cmd,'wrong receipt/admission lane or exact command')
 require(rc['returncode']==0 and rc['failure'] is None,'original did not close successfully')
 require(all(rc[k] is True for k in ('worker_reaped','worker_identity_absent','worker_group_session_absent')),'original closure flags incomplete')
 require(plan['cpu_threads']==1 and all(ad['threads'][k]=='1' for k in THREADS),'original thread allocation differs')
 require(ad['limits']==lane['limits'],'original limits differ from frozen plan')
 planpath=Path(ad['plan']['path']);planpath=planpath if planpath.is_absolute() else R/planpath
 require(planpath==PLAN and ad['plan']['sha256']==PLAN_SHA and ad['plan']['bytes']==7176,'original plan receipt differs')
 expected={str(Path(e['path']).relative_to(R)):{k:e[k] for k in ('bytes','sha256')} for e in plan['source_files']}
 actual={e['relative_path']:{k:e[k] for k in ('bytes','sha256')} for e in ad['source_blobs']}
 require(len(expected)==len(plan['source_files']) and len(actual)==len(ad['source_blobs']) and actual==expected,'original complete source set differs')
 inputs={e['path']:e for e in card['inputs']};preflight={e['path']:e for e in lane['preflight_pins']}
 require(len(inputs)==len(card['inputs']) and all(preflight.get(k)==v for k,v in inputs.items()),'original card inputs not joined to preflight pins')
 for path in (c['states'],c['settle_report']):require(path in inputs,'missing original selected-cell input pin')
 files={e['path']:e for e in rc['output_files']}
 require(len(files)==len(rc['output_files']),'duplicate original output paths')
 for text in files:
  p=Path(text);require(p.is_absolute() and '..' not in p.parts and RAW in p.parents,'foreign original output path')
 for name in ('report.json',CELL+'/summary.json',CELL+'/monodromy.npy',CELL+'/eigensystem.npz',CELL+'/critical-direct-complex.npy'):
  require(str(RAW/name) in files,'missing original consumed output '+name)
 for k in range(0,896,8):require(str(RAW/CELL/f'columns-{k:03d}.npy') in files,'missing original column block')
 return c

def row_authority(row,cell,files):
 require(row['cell']==cell and row['period']==PERIOD and row['state_index']==513-PERIOD and row['columns']==896,'saved row differs from exact cascade-P6 allocation')
 require([v['part'] for v in row['critical_checks']]==['real','imag'],'both original critical components required')
 for check in row['critical_checks']:
  require(type(check['zero_component']) is bool,'invalid component status')
  if not check['zero_component']:require(str(RAW/CELL/('critical-'+check['part']+'.npz')) in files,'critical component file not in original inventory')

def pinned_source(path,expected):
 entry=s.pin(path);require(entry['sha256']==expected,'frozen source/card hash differs')
 data=subprocess.check_output(['git','show',SOURCE+':'+str(path.relative_to(R))],cwd=R)
 require(len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==expected,'frozen source Git bytes differ')
 return entry

def main():
 require(__debug__,'assertions must remain enabled')
 a=arguments();out=Path(a.output);require(not out.exists() and not out.parent.exists(),'fresh saved-audit output required')
 out.parent.mkdir(exist_ok=False)
 require(all(os.environ.get(k)=='1' for k in THREADS),'one BLAS/OpenMP thread required before NumPy import')
 import numpy as np
 cardpin=pinned_source(CARD,CARD_SHA);planpin=pinned_source(PLAN,PLAN_SHA)
 card=json.loads(CARD.read_text());plan=json.loads(PLAN.read_text())
 raw=RAW;sup=SUP
 rc=json.loads((sup/'receipt.json').read_text());ad=json.loads((sup/'admission.json').read_text())
 cell=allocation(rc,ad,plan,card)
 for e in rc['output_files']:s.verify(e)
 require({str(p)for p in raw.rglob('*')if p.is_file()}=={e['path']for e in rc['output_files']},'whole original output inventory differs')
 for e in ad['source_blobs']:
  b=subprocess.check_output(['git','show',SOURCE+':'+e['relative_path']],cwd=R)
  require(len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256'],'original producer source blob differs')
 for e in card['inputs']:s.verify(e)
 for pid in [ad['supervisor']['pid'],rc['worker_pid']]:require(s.ident(pid)is None and not s.members(pid),'original host process/group closure pending')
 report=json.loads((raw/'report.json').read_text());require(len(report['rows'])==1,'exactly one original scientific row required')
 row=report['rows'][0];folder=raw/CELL;row_authority(row,cell,{e['path']:e for e in rc['output_files']})
 require(row==json.loads((folder/'summary.json').read_text()),'original report and summary disagree')
 p=row['period'];assert row['state_index']==513-p
 history=np.load(row['cell']['states'],allow_pickle=False);assert history.shape==(513,896)
 m=np.load(folder/'monodromy.npy',allow_pickle=False)
 blocks=[np.load(folder/f'columns-{k:03d}.npy',allow_pickle=False)for k in range(0,896,8)]
 assert all(b.shape==(8,896) and np.isfinite(b).all()for b in blocks) and np.array_equal(m,np.concatenate(blocks).T)
 z=np.load(folder/'eigensystem.npz',allow_pickle=False);v=z['values'];U=z['vectors']
 assert v.shape==(896,) and U.shape==(896,896) and np.isfinite(v).all() and np.isfinite(U).all() and np.all(np.abs(v)[:-1]>=np.abs(v)[1:])
 residuals=np.max(np.abs(m@U-U*v[None,:]),axis=0)
 assert np.array_equal(residuals,z['residuals']) and float(residuals.max())==row['maximum_eigen_residual_inf']
 assert float(abs(v[0]))==row['spectral_radius'] and [float(v[0].real),float(v[0].imag)]==row['critical_multiplier'] and float(abs(v[1]))==row['next_modulus']
 assert row['numerically_inside_unit_circle']==bool(np.max(np.abs(v))<1)
 fdcount=0
 for check in row['critical_checks']:
  part=check['part'];direction=getattr(U[:,0],part);norm=float(np.linalg.norm(direction))
  if check['zero_component']:assert norm==0;continue
  z=np.load(folder/('critical-'+part+'.npz'),allow_pickle=False)
  assert all(np.isfinite(z[k]).all()for k in z.files) and np.array_equal(z['direction'],direction/norm)
  assert np.array_equal(z['dense'],m@z['direction'])
  assert float(np.max(np.abs(z['dense']-z['direct'])))==check['dense_vs_direct_inf']
  assert [f['h']for f in check['fd_ladder']]==[1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5]
  for f in check['fd_ladder']:
   h=f['h'];err=float(np.max(np.abs((z['plus_'+str(h)]-z['minus_'+str(h)])/(2*h)-z['direct'])))
   assert err==f['absolute_inf_error'];fdcount+=1
 d=np.load(folder/'critical-direct-complex.npy',allow_pickle=False)
 assert np.isfinite(d).all() and float(np.max(np.abs(d-v[0]*U[:,0])))==row['critical_direct_eigen_residual_inf']
 cycle=history[-p:];assert row['adjacent_separations']==[float(np.max(np.abs(cycle[k]-cycle[(k+1)%p])))for k in range(p)]
 result=dict(receipt=s.pin(sup/'receipt.json'),admission=s.pin(sup/'admission.json'),producer_plan=planpin,producer_card=cardpin,source_commit=SOURCE,cell=CELL,period=PERIOD,source_blobs_checked=len(ad['source_blobs']),outputs_checked=len(rc['output_files']),host_closure=True,columns_checked=896,eigen_residuals_checked=896,fd_rows_checked=fdcount,scientific_row=row,elapsed_seconds=rc['external_elapsed_seconds'],certificate_passed=False,scope='Saved composed matrix, complete eigensystem residuals, critical direct/FD comparisons and phase separation independently recomputed. AD and forward closure inherit reviewed original execution; no model rerun or interval certificate.')
 with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(cell=CELL,rho=row['spectral_radius'],fdrows=fdcount)))
if __name__=='__main__':main()
