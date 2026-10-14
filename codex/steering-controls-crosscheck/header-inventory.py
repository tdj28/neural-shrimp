import os,resource,time
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));os.nice(max(0,10-os.nice(0)))
from pathlib import Path
import hashlib,json,zipfile,subprocess,datetime,numpy as np
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');S='d83ceaa2bcd3050d3f9b2cfaaf5e1ad991365d47';B=Path('/data2/neural-shrimp-data/explore/llm-soft-loop/pod64');O=Path('/tmp/task16-input-metadata-001.json');assert not O.exists();tic=time.monotonic();hashsec=0;hashbytes=0
specs=[('q05-alpha0','scan64/q05-alpha0','scan.npz',0),('llama8-alpha0','llama8b-hires/alpha0','scan.npz',0),('q7-alpha0','controls/q7-alpha0','scan.npz',0),('olmo7-alpha0','controls/o-alpha0c','scan.npz',0)]
for k in [1,2,3]:
 specs.extend([(f'q05-random{k}',f'scan64/q05-rand{k}','scan-partial.npz',0),(f'q05-random{k}',f'scan64/q05-rand{k}-top','scan.npz',96)])
specs.extend([('q05-philosophy','pairmap-qwen0.5b/map128-philosophy','scan.npz',0),('q05-gratitude','pairmap-qwen0.5b/map128-gratitude','scan.npz',0),('llama8-layer16-concept','llama8b-b/coarse-64-hostility-vs-kindness','scan.npz',0),('llama8-layer16-random1','llama8b-hires/rand16-1','scan.npz',0),('llama8-layer16-random2','llama8b-hires/rand16-2','scan.npz',0)])
for x in ['concept','saeH','saeR','gauss']:specs.append((f'llama8-layer19-{x}',f'controls/l19-{x}/scan','scan.npz',0))
def pin(p):
 global hashsec,hashbytes
 t=time.monotonic();before=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1024**2):h.update(b)
 after=p.stat();assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)
 hashsec+=time.monotonic()-t;hashbytes+=before.st_size
 return {'path':str(p),'sha256':h.hexdigest(),'bytes':before.st_size,'mtime_ns':before.st_mtime_ns,'inode':before.st_ino,'mode':oct(before.st_mode&0o777),'same_stat_before_after_hash':True}
segments=[];axes={}
for i,(id,rel,filename,offset) in enumerate(specs):
 p=B/rel/filename;row={'dataset_id':id,'segment_id':f'segment-{i:02d}','relative_directory':rel,'filename':filename,'proposed_fixed_row_offset':offset,**pin(p)}
 headers={}
 with zipfile.ZipFile(p) as z:
  for n in z.namelist():
   assert n.endswith('.npy') and '/' not in n
   with z.open(n) as f:
    v=np.lib.format.read_magic(f)
    assert v in [(1,0),(2,0)],v
    shape,fortran,dtype=(np.lib.format.read_array_header_1_0(f) if v==(1,0) else np.lib.format.read_array_header_2_0(f))
   headers[n[:-4]]={'shape':list(shape),'dtype':dtype.str,'fortran_order':fortran,'object_dtype':bool(dtype.hasobject),'zip_uncompressed_bytes':z.getinfo(n).file_size}
 row['members']=headers
 with np.load(p,allow_pickle=False) as z:
  lt=z['log10_temperature'];a=z['alpha'];assert lt.ndim==a.ndim==1 and np.isfinite(lt).all() and np.isfinite(a).all()
  done=int(z['done']) if 'done' in z.files else None
 row['axes']={k:{'values':v.tolist(),'float_hex':[float(x).hex() for x in v],'dtype':v.dtype.str,'shape':list(v.shape),'contiguous_bytes_sha256':hashlib.sha256(v.tobytes()).hexdigest()} for k,v in [('log10_temperature',lt),('alpha',a)]}
 row['done']=done;row['flat_cells']=int(len(lt)*len(a));row['scientific_values_decoded']=False
 info=B/rel/'scan.json'
 if info.exists():
  row['scan_info']=pin(info);b=info.read_text();idx=b.index('"args"');colon=b.index(':',idx);args,end=json.JSONDecoder().raw_decode(b[colon+1:].lstrip());row['scan_args']=args
 else:row['scan_info']=None;row['scan_args']=None
 row['completion_basis']='scan.npz final-shape writer plus scan.json metadata; original run authority still to bind' if done is None else 'Exact completed flat prefix [0, done) from partial writer; original termination/timeout still to bind'
 segments.append(row);axes[row['segment_id']]=(lt,a)
joins=[]
for k in [1,2,3]:
 low,top=[x for x in segments if x['dataset_id']==f'q05-random{k}'];lo,al=axes[low['segment_id']];hi,ah=axes[top['segment_id']];expected=lo[96:]
 mism=np.flatnonzero(expected!=hi);alpha_mism=np.flatnonzero(al!=ah)
 joins.append({'dataset_id':f'q05-random{k}','low_segment':low['segment_id'],'top_segment':top['segment_id'],'lower_done':low['done'],'lower_completed_full_rows':low['done']//len(al),'lower_completed_row_remainder':low['done']%len(al),'fixed_top_start_row':96,'expected_top_rows':len(expected),'actual_top_rows':len(hi),'alpha_exact_equal':bool(np.array_equal(al,ah)),'temperature_exact_equal':bool(np.array_equal(expected,hi)),'temperature_mismatch_rows':[{'global_row':int(j+96),'expected':float(expected[j]),'actual':float(hi[j]),'expected_hex':float(expected[j]).hex(),'actual_hex':float(hi[j]).hex(),'difference':float(hi[j]-expected[j])} for j in mism],'alpha_mismatch_columns':alpha_mism.tolist(),'mapping_selected':False,'note':'Metadata-only observation; no tolerant join or canonical replacement has been performed.'})
ref=[x for x in segments if x['dataset_id']=='q05-philosophy'][0];rlt,ra=axes[ref['segment_id']]
comparisons=[]
for x in segments:
 if x['dataset_id'].startswith('q05-') and x['dataset_id']!='q05-alpha0':
  lt,a=axes[x['segment_id']];off=x['proposed_fixed_row_offset'];e=rlt[off:off+len(lt)]
  comparisons.append({'segment_id':x['segment_id'],'reference_dataset':'q05-philosophy','alpha_exact_equal':bool(np.array_equal(a,ra)),'log_temperature_at_fixed_rows_exact_equal':bool(np.array_equal(lt,e)),'reference_rows':[off,off+len(lt)]})
o={'schema':'task16-header-readiness-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':S,'writer_sha256':'177fae50175d716e060aaddc41a4377e4207aa84d6f047dae5254f7c16cd966d','datasets':16,'segments':segments,'joins':joins,'q05_reference_comparisons':comparisons,'read_scope':'NPZ headers, coordinate axes and completion scalar only; scan.json args only. No period, Lyapunov, entropy, family or token values decoded. Full file hashes are streamed, not scientific analysis.','input_authentication_complete':False,'historical_source_writer_termination_pending':True,'hash_io':{'bytes':hashbytes,'seconds':hashsec},'elapsed_seconds':time.monotonic()-tic}
O.write_text(json.dumps(o,indent=2,allow_nan=False)+'\n');O.chmod(0o444)
print(json.dumps({'path':str(O),'sha256':hashlib.sha256(O.read_bytes()).hexdigest(),'segments':len(segments),'datasets':16,'hash_io':o['hash_io'],'elapsed_seconds':o['elapsed_seconds'],'joins':[{'id':j['dataset_id'],'done':j['lower_done'],'rows':j['lower_completed_full_rows'],'remainder':j['lower_completed_row_remainder'],'alpha_exact':j['alpha_exact_equal'],'temperature_exact':j['temperature_exact_equal'],'mismatch_rows':len(j['temperature_mismatch_rows']),'max_difference':max([abs(r['difference']) for r in j['temperature_mismatch_rows']],default=0)} for j in joins]},indent=2))
