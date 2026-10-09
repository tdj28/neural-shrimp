import cmath,hashlib,json,math,pathlib,numpy as np
raw=pathlib.Path('/data2/neural-shrimp-data/explore/curvature-mechanism/attempt-001-plane-explanation');old=raw.with_name('attempt-001-panel')
for name in ('eig','eigh','eigvals','eigvalsh','svd','solve','inv','pinv','qr','norm','cond'):
 setattr(np.linalg,name,lambda *a,**k:(_ for _ in ()).throw(AssertionError('forbidden new decomposition/solve')))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fro(x):return math.sqrt(float(np.sum(np.abs(x)**2)))
def col(x):return np.sqrt(np.sum(np.abs(x)**2,axis=0))
def ck(x,y):
 if not np.allclose(x,y,rtol=2e-11,atol=2e-12):raise AssertionError('saved arithmetic mismatch')
def decode(root,name):
 r=json.loads((root/name).read_text());ap=root/r['arrays_file'];assert digest(ap)==r['arrays_sha256']
 with np.load(ap,allow_pickle=False) as z:
  def d(x):
   if isinstance(x,dict) and 'array' in x:
    a=z[x['array']];assert list(a.shape)==x['shape'] and a.dtype.str==x['dtype'] and hashlib.sha256(a.tobytes(order='C')).hexdigest()==x['sha256'];return a.copy()
   if isinstance(x,dict):return {k:d(v) for k,v in x.items()}
   if isinstance(x,list):return [d(v) for v in x]
   return x
  return d(r['payload'])
assert digest(old/'manifest.json')=='898880f0bc1755896251f39e5abf806cc89f32a1558ed102fc0c954bd6d9ff4b'
manifest=json.loads((raw/'manifest.json').read_text())
for name,pin in manifest['files'].items():assert digest(raw/name)==pin['sha256'] and (raw/name).stat().st_size==pin['bytes']
rows=json.loads((raw/'result.json').read_text())['rows'];events=[json.loads(x) for x in (raw/'journal.jsonl').read_text().splitlines()];assert len(events)==len(rows)==6
out=[]
for e,summary in zip(events,rows):
 assert e['kind']=='plane_diagnostic';d=decode(raw,e['file']);a=decode(old,d['source_record'])['analysis'];assert d['slot']==summary['slot'];assert digest(old/d['source_record'])==d['source_record_identity']['sha256']
 M=a['target'];f,g=a['phases'];u=f['u']/fro(f['u']);v=g['u']/fro(g['u']);c=float(u@v);rem=v-u*c;Q=u[:,None] if 1-c*c<=1e-10 else np.column_stack([u,rem/fro(rem)]);ck(Q,d['Q']);ck(rem,d['Gram_Schmidt_remainder']);ck(c,d['cosine']);assert Q.shape[1]==d['dimension']==2
 ortho=fro(Q.T@Q-np.eye(2));ck(ortho,d['orthogonality_residual']);assert ortho<1e-12
 ids=a['target_pair']['indices'];R=a['target_eigensystem']['right'][:,ids];values=a['target_eigensystem']['values'];t=values[ids];ck(ids,d['target_indices']);ck(t,d['target_values']);ck(R,d['target_right_vectors']);E=col(R)**2;coeff=Q.T@R;proj=Q@coeff;rest=R-proj;capture=col(coeff)**2/E;uncaptured=col(rest)**2/E
 for x,y in [(E,d['right_energy']),(coeff,d['captured_coefficients']),(proj,d['captured_vectors']),(rest,d['uncaptured_vectors']),(capture,d['capture_fractions']),(uncaptured,d['uncaptured_energy_fractions']),(capture+uncaptured,np.ones(2)),(capture,summary['capture_fractions'])]:ck(x,y)
 MQ=M@Q;C=Q.T@MQ;leak=MQ-Q@C
 for x,y in [(MQ,d['MQ']),(C,d['compression']),(leak,d['leakage']),(fro(leak),d['leakage_frobenius']),(fro(leak)/max(1,fro(MQ)),d['relative_leakage_frobenius'])]:ck(x,y)
 tr=float(C[0,0]+C[1,1]);det=float(C[0,0]*C[1,1]-C[0,1]*C[1,0]);disc=tr*tr-4*det;roots=np.array([(tr+cmath.sqrt(disc))/2,(tr-cmath.sqrt(disc))/2]);dist=[max(abs(roots-t)),max(abs(roots-t[::-1]))];swap=int(dist[1]<dist[0]);err=dist[swap]/max(1,max(abs(values)))
 for x,y in [(tr,d['compression_trace']),(det,d['compression_determinant']),(disc,d['compression_discriminant']),(roots,d['compression_roots']),(err,d['compression_pair_error']),(err,summary['compression_pair_error']),([1,0] if swap else [0,1],d['unordered_assignment'])]:ck(x,y)
 terms={'native_formation':(M-a['pair_product'])@R,'first_phase_omission':g['S0']@(f['E']@R),'second_phase_omission':g['E']@(f['S0']@R),'interacting_omissions':g['E']@(f['E']@R)};ra=d['residual_action'];total=sum(terms.values());actual=(M-a['model_product'])@R;res=actual-total
 for k,z in terms.items():ck(z,ra['terms'][k]);ck(col(z),ra['term_norms'][k]);ck(col(z)/np.sqrt(E),ra['term_unit_vector_norms'][k])
 for x,y in [(total,ra['sum']),(actual,ra['actual_difference_action']),(res,ra['identity_residual']),(col(total),ra['sum_norms']),(col(res),ra['identity_residual_norms'])]:ck(x,y)
 assert max(col(res))<1e-12
 record={'slot':d['slot'],'orthogonality':ortho,'energy_partition_error':float(max(abs(capture+uncaptured-1))),'capture':capture.tolist(),'leakage_relative':d['relative_leakage_frobenius'],'compression_pair_error':err,'identity_error':float(max(col(res))),'term_unit_norms':{k:(col(z)/np.sqrt(E)).tolist() for k,z in terms.items()},'direct_action_unit_norm':(col(actual)/np.sqrt(E)).tolist()};out.append(record);print(json.dumps(record))
print(json.dumps({'status':'saved_arithmetic_pass','rows':len(out),'raw_manifest_sha256':digest(raw/'manifest.json'),'max_orthogonality':max(x['orthogonality'] for x in out),'max_energy_partition_error':max(x['energy_partition_error'] for x in out),'max_action_identity_error':max(x['identity_error'] for x in out),'all_compression_errors_above_original_02_threshold':all(x['compression_pair_error']>.02 for x in out),'new_model_derivative_decomposition_calls':0}))
