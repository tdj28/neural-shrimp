import json, pathlib, resource, signal, zipfile
resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
resource.setrlimit(resource.RLIMIT_CPU,(45,45));signal.alarm(60)
import numpy as np
D=pathlib.Path('/data2/neural-shrimp-data/explore/llm-soft-loop');P=D/'pod64'
def describe(x):
    if isinstance(x,dict):return {k:describe(v) for k,v in x.items()}
    if isinstance(x,list):return {'length':len(x),'first':describe(x[0]) if x else None,'last':describe(x[-1]) if len(x)>1 else None}
    return x
for name in ['hr10/scan','hr11/scan','hr12/scan','hp5/map','dss/p6-body/grid']:
    p=P/(name+'.npz');z=np.load(p,allow_pickle=False)
    print('NPZ',str(p),{k:(z[k].shape,str(z[k].dtype)) for k in z.files})
for p in [P/'p6s/locate-tu-minus-34/locate.json',P/'p6s/locate-wide2-a-64/locate.json',P/'p6s/locate-wide2-b-82/locate.json',P/'kite/locate-la-plus-14/locate.json',P/'kite/locate-la-plus-94/locate.json',P/'kite/locate-la-plus-112/locate.json',D/'kite/newton-at-R/newton_at_R.json',D/'kite/basin-check/basin-check.json',P/'kite/kite-la-branch/branch.json',P/'kite/kite-ut-branch/branch.json']:
    print('JSON',str(p),json.dumps(describe(json.loads(p.read_text()))))
