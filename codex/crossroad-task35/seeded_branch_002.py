"""Explicit saved-state branch pilot; never replaces a seed with a fixed-start transient.

Prepared caller only. Execution requires reviewed source/input/resource card.
"""
import argparse,hashlib,json,sys,time,os
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def validate(c):
    if c['period']!=5 or c['dtype']!='float64' or c['log_temperature']!=-.1:
        raise ValueError('unreviewed family')
    if c['free'] not in ('alpha','alpha2') or c['direction'] not in (-1,1):
        raise ValueError('invalid continuation direction')
    if type(c['steps']) is not int or not 1<=c['steps']<=24:
        raise ValueError('bounded24point pilot required')
    if type(c['seed_index']) is not int or c['seed_index']<0:
        raise ValueError('explicit seed index required')
    if c['tol']!=1e-10 or c['alpha_scale']!=10 or c['chunk']!=8:
        raise ValueError('unreviewed numerical settings')
    if not 0<c['ds_min']<=c['ds']<=c['ds_max']<=.002:
        raise ValueError('unreviewed step bounds')
    if len(c['bounds'])!=2 or not c['bounds'][0]<c['bounds'][1]:
        raise ValueError('finite window required')
    if not c['bounds'][0]<=c[c['free']]<=c['bounds'][1]:
        raise ValueError('seed outside window')


def capture_seed_return(cell, X, out, card):
    """Durably retain uncorrected X0..X5 from the one original seed-check pass.

    This is isolated preparation, not an admitted runtime or qualified seed.
    The frozen capture helper performs exactly five one-step calls. Six journal
    rows mean retained seed iterates only: indices 0..4 are uncorrected phases,
    and 5 is the closing image even when it fails to close. The original scalar
    gates are still applied by main after this function returns.

    Each file and the append-only journal are exclusively created. Successful
    callbacks flush/fsync their file, journal and containing directory before
    another map call. Serializer/write/hash/fsync exceptions propagate; partial
    files/rows remain and cannot be resumed or overwritten. A failed fragment
    has no successful-durability claim. These checks provide neither an atomic
    snapshot against external mutation nor future immutability/source authority.
    """
    from seed_phase_capture import capture_seed_phases
    import stat

    out = Path(out)
    if not out.is_absolute() or out.resolve(strict=True) != out:
        raise ValueError('canonical existing absolute output directory required')
    directory = os.open(out, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        identity = os.fstat(directory)

        def check_directory():
            current = os.stat(out, follow_symlinks=False)
            if ((current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino)
                    or not stat.S_ISDIR(current.st_mode) or out.resolve(strict=True) != out):
                raise ValueError('seed output directory identity changed')

        check_directory()
        # Main has just created out: persist that new entry in its parent too.
        parent = os.open(out.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            parent_identity = os.fstat(parent)
            def check_parent():
                current = os.stat(out.parent, follow_symlinks=False)
                child = os.stat(out.name, dir_fd=parent, follow_symlinks=False)
                if ((current.st_dev, current.st_ino) != (parent_identity.st_dev, parent_identity.st_ino)
                        or not stat.S_ISDIR(current.st_mode)
                        or (child.st_dev, child.st_ino) != (identity.st_dev, identity.st_ino)):
                    raise ValueError('seed output parent/entry identity changed')
                check_directory()
            check_parent(); os.fsync(parent); check_parent()
        finally:
            os.close(parent)
        origin = dict(path=card['seed_path'], sha256=card['seed_sha256'], index=card['seed_index'])
        parameters = {key: card[key] for key in ('alpha', 'alpha2', 'log_temperature')}
        ratios = {key: list(float(value).as_integer_ratio()) for key, value in parameters.items()}
        journal_fd = os.open('seed-phases.jsonl', os.O_WRONLY | os.O_CREAT | os.O_EXCL
                             | os.O_APPEND | os.O_NOFOLLOW, 0o600, dir_fd=directory)
        with os.fdopen(journal_fd, 'wb') as journal:
            journal.flush(); os.fsync(journal.fileno()); os.fsync(directory)
            next_index = 0

            def retain(index, snapshot):
                nonlocal next_index
                if type(index) is not int or index != next_index or not 0 <= index <= 5:
                    raise ValueError('exact ordered seed iterate index required')
                check_directory()
                array = snapshot.numpy()
                if tuple(array.shape) != (1, 1, 896) or array.dtype != np.float64:
                    raise ValueError('unchanged CPU binary64 seed window required for serialization')
                name = f'seed-phase-{index:03d}.npy'
                fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=directory)
                with os.fdopen(fd, 'w+b') as stream:
                    np.save(stream, array, allow_pickle=False)
                    stream.flush(); os.fsync(stream.fileno()); os.fsync(directory)
                    before = os.fstat(stream.fileno())
                    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                        raise ValueError('exclusive regular seed file required')
                    stream.seek(0)
                    digest = hashlib.sha256()
                    size = 0
                    for block in iter(lambda: stream.read(65536), b''):
                        digest.update(block); size += len(block)
                    after = os.fstat(stream.fileno())
                    linked = os.stat(name, dir_fd=directory, follow_symlinks=False)
                    version = lambda item: (item.st_dev, item.st_ino, item.st_size,
                                            item.st_mtime_ns, item.st_ctime_ns, item.st_nlink)
                    if (version(before) != version(after) or version(after) != version(linked)
                            or not stat.S_ISREG(linked.st_mode) or size != after.st_size):
                        raise ValueError('saved seed file changed while hashing')
                check_directory()
                record = dict(iterate=index,
                              role='uncorrected_seed_phase' if index < 5 else 'closing_image',
                              phase_index=index if index < 5 else None,
                              file=dict(path=str(out/name), bytes=size, sha256=digest.hexdigest()),
                              shape=[1, 1, 896], dtype='float64', seed_origin=origin,
                              parameters=parameters, parameter_binary64_ratios=ratios,
                              period=5, corrected=False, seed_qualified=False,
                              scientific_attempt_complete=False,
                              scope='Retained uncorrected seed iterate only; scalar gates and original execution authority remain separate.')
                journal.write((json.dumps(record, sort_keys=True, allow_nan=False)+'\n').encode('utf-8'))
                journal.flush(); os.fsync(journal.fileno()); os.fsync(directory)
                check_directory()
                next_index += 1

            returns = capture_seed_phases(cell, X, retain, period=5)
            if next_index != 6:
                raise ValueError('complete ordered seed capture missing')
            check_directory()
        return returns
    finally:
        os.close(directory)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--card',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();c=json.loads(Path(a.card).read_text());validate(c)
    for e in c['inputs']:
        if sha(e['path'])!=e['sha256']:raise ValueError('input hash mismatch')
    if sha(c['seed_path'])!=c['seed_sha256']:raise ValueError('seed hash mismatch')
    states=np.load(c['seed_path'],allow_pickle=False)
    if states.ndim!=2 or states.shape[1]!=896 or states.dtype!=np.float64 or not np.isfinite(states).all() or c['seed_index']>=len(states):
        raise ValueError('invalid seed array')
    os.environ['NEURAL_SHRIMP_DATA']=c['data_root'];os.environ['HF_HUB_OFFLINE']='1'
    root=Path(c['repository']);sys.path.insert(0,str(root/'tools'))
    import explore_llm_soft_loop as esl
    import explore_llm_phase_map as elpm
    import bifurcation_curves as bc
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from branch_trace import trace_branch
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    def write(name,value):(out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    start=time.monotonic()
    loop=esl.SoftLoop('qwen0.5b',4,c['threads'],'open','Hello!',1,'float64',c['device'])
    loop.sm.add_steer_layer(16)
    v=elpm.load_vectors(c['vectors'])['gratitude'];w=elpm.load_vectors(c['vectors2'])['hostility']
    loop.extra=(16,np.array([c['alpha2']*w]))
    cell=esl.SoftOrbit(loop,c['alpha']*v,10.0**c['log_temperature'])
    X=loop.torch.as_tensor(states[c['seed_index']].reshape(1,1,896),dtype=loop.state_dtype,device=loop.device)
    returns=capture_seed_return(cell,X,out,c)
    write('seed-check.json',{'return_distances':returns,'seed_index':c['seed_index'],'seed_sha256':c['seed_sha256']})
    if not returns[4]<=1e-10 or not returns[0]>1e-7:
        raise ValueError('seed residual or primitive-period gate failed')
    fam=esl.soft_family(loop,v,5,10,c['chunk'],reference=loop,exact_jvp=True,second=(16,w),log_temperature=-.1)
    def on_point(rec):
        with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(rec,allow_nan=False)+'\n')
    def on_attempt(rec):
        # Infinity is an explicit failed-residual marker, never a JSON NaN/Infinity token.
        if not np.isfinite(rec['relative_residual']):rec['relative_residual']='nonfinite'
        with (out/'attempts.jsonl').open('a') as f:f.write(json.dumps(rec,allow_nan=False)+'\n')
    def on_state(index,state):np.save(out/f'accepted-state-{index:03d}.npy',state)
    def on_matrix(index,matrix):np.save(out/f'monodromy-{index:03d}.npy',matrix)
    records,reason,ys=trace_branch(fam,fam.to_np(X),[10*c['alpha'],10*c['alpha2']],free=0 if c['free']=='alpha' else 1,direction=c['direction'],steps=c['steps'],ds=c['ds'],ds_min=c['ds_min'],ds_max=c['ds_max'],tol=1e-10,max_newton=10,beyond=.005,bounds=tuple(10*b for b in c['bounds']),on_point=on_point,k=896,refine_k=4,on_attempt=on_attempt,on_state=on_state,on_matrix=on_matrix)
    for row in records:
        row['lambda']/=10
        if 'turn_lambda_estimate' in row:row['turn_lambda_estimate']/=10
    np.save(out/'states.npy',np.array(ys).reshape((len(ys),1,896)))
    # Explicit producer-compatible metadata for the existing curve reader.
    args=dict(model='qwen0.5b',layer=4,threads=c['threads'],dtype='float64',reference_dtype='none',device=c['device'],prompt='open',opener='Hello!',window=1,vectors=c['vectors'],concept='gratitude',log_temperature=-.1,alpha=c['alpha'],alpha2=c['alpha2'],block2=16,vectors2=c['vectors2'],concept2='hostility')
    write('branch.json',{'args':args,'free':c['free'],'period':5,'seed_origin':{'path':c['seed_path'],'sha256':c['seed_sha256'],'index':c['seed_index']},'seed_return_distances':returns,'points':records,'crossings':bc.classify_crossings(records),'stop':reason,'seconds':time.monotonic()-start,'crossroad_claim':False})
    for e in c['inputs']:
        if sha(e['path'])!=e['sha256']:raise ValueError('postrun input mismatch')

if __name__=='__main__':main()
