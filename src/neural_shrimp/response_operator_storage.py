"""Lossless one-array-bundle per durable event; complete compact event journal."""
import io
import numpy as np

class DurabilityError(RuntimeError):pass

def emitter(store,sequence,slot):
    def emit(kind,value):
        try:
            sequence[0]+=1;tag=f'{sequence[0]:05d}-{slot}-{kind}';record={'kind':kind,'slot':slot,'sequence':sequence[0],'arrays':{}};arrays={}
            for name,v in value.items():
                if isinstance(v,np.ndarray):
                    if v.dtype.hasobject:raise ValueError('object array forbidden')
                    arrays[name]=v;record['arrays'][name]={'key':name,'dtype':v.dtype.str,'shape':list(v.shape)}
                else:record[name]=v
            if arrays:
                stream=io.BytesIO();np.savez_compressed(stream,**arrays);file=tag+'.npz';entry=store.write(file,stream.getvalue());record['array_bundle']={'file':file,**entry}
            store.event(record)
        except Exception as exc:raise DurabilityError('durable bundled event failed: '+str(exc))from exc
    return emit
