"""Local exclusive records for the bounded period-two exploratory pilot."""
import hashlib
import io
import json
import os
from pathlib import Path
import numpy as np
from .period2_branch import PilotStop


def canonical(obj):
    return (json.dumps(obj,sort_keys=True,indent=2,allow_nan=False,default=lambda x: x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n').encode()


def digest(data): return hashlib.sha256(data).hexdigest()


def identity(array):
    array=np.ascontiguousarray(array)
    return dict(dtype=array.dtype.str,shape=list(array.shape),order='C',sha256=digest(array.tobytes()))


def file_entry(path):
    data=Path(path).read_bytes()
    return dict(bytes=len(data),sha256=digest(data))


class Store:
    """Append-only payloads plus atomic latest checkpoint; reserve finalization space."""
    def __init__(self,root,limit=64*1024*1024):
        if type(limit) is not int or limit<8192: raise ValueError('invalid storage budget')
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=False)
        self.limit=limit;self.reserve=limit//4;self.used=0;self.count=0
        self.latest=None
        self._write('journal.jsonl',b'')

    def _write(self,name,data,replace=False,final=False):
        path=self.root/name
        if not replace and path.exists(): raise FileExistsError(path)
        old=path.stat().st_size if path.exists() else 0
        # Account for simultaneous temporary and previous versions during replace.
        ceiling=self.limit if final else self.limit-self.reserve
        if self.used+len(data)>ceiling: raise PilotStop('raw_byte_limit')
        tmp=path.with_name(path.name+'.tmp')
        with tmp.open('xb') as h:h.write(data);h.flush();os.fsync(h.fileno())
        os.replace(tmp,path);self.used+=len(data)-old
        fd=os.open(self.root,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)

    def json(self,name,obj,**kwargs):self._write(name,canonical(obj),**kwargs)

    def emit(self,kind,payload):
        number=self.count;self.count+=1
        base=f'{number:05d}-{kind}'
        arrays={}
        def encode(x):
            if isinstance(x,np.ndarray):
                key=f'a{len(arrays):04d}';arrays[key]=x
                return {'array':key,**identity(x)}
            if isinstance(x,np.generic):return x.item()
            if isinstance(x,dict):return {k:encode(v) for k,v in x.items()}
            if isinstance(x,(list,tuple)):return [encode(v) for v in x]
            return x
        obj=dict(kind=kind,sequence=number,payload=encode(payload))
        parts=[]
        if arrays:
            stream=io.BytesIO();np.savez(stream,**arrays)
            data=stream.getvalue();obj['arrays_file']=base+'.npz';obj['arrays_sha256']=digest(data)
            parts.append((base+'.npz',data))
        parts.append((base+'.json',canonical(obj)))
        line=canonical(dict(sequence=number,kind=kind,file=base+'.json')).replace(b'\n',b'')+b'\n'
        checkpoint=canonical(dict(sequence=number,kind=kind,file=base+'.json',status='in_progress'))
        if self.used+sum(len(v) for _,v in parts)+len(line)+len(checkpoint)>self.limit-self.reserve:
            raise PilotStop('raw_byte_limit')
        for name,data in parts:self._write(name,data)
        with (self.root/'journal.jsonl').open('ab') as h:h.write(line);h.flush();os.fsync(h.fileno())
        self.used+=len(line)
        self._write('checkpoint.json',checkpoint,replace=True)
        self.latest=dict(sequence=number,kind=kind,file=base+'.json')

    def finish(self,result):
        self.json('result.json',result,final=True)
        self.json('checkpoint.json',dict(status=result['status'],latest=self.latest),replace=True,final=True)
        files={p.name:file_entry(p) for p in self.root.iterdir() if p.is_file()}
        self.json('manifest.json',dict(schema='s1r-branch-manifest-v1',status=result['status'],files=files),final=True)
        if sum(p.stat().st_size for p in self.root.iterdir())>self.limit:raise AssertionError('storage accounting')


def load_record(root,name):
    root=Path(root)
    if Path(name).name!=name:raise ValueError('unsafe record path')
    obj=json.loads((root/name).read_text());arrays={}
    if 'arrays_file' in obj:
        path=root/obj['arrays_file']
        if Path(obj['arrays_file']).name!=obj['arrays_file'] or digest(path.read_bytes())!=obj['arrays_sha256']:
            raise ValueError('record array binding')
        with np.load(path,allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
    used=set()
    def decode(x):
        if isinstance(x,dict) and 'array' in x:
            a=arrays[x['array']];used.add(x['array'])
            if identity(a)!={k:v for k,v in x.items() if k!='array'}:raise ValueError('array identity mismatch')
            return a
        if isinstance(x,dict):return {k:decode(v) for k,v in x.items()}
        if isinstance(x,list):return [decode(v) for v in x]
        return x
    payload=decode(obj['payload'])
    if used!=set(arrays):raise ValueError('unreferenced arrays')
    return obj['kind'],payload
