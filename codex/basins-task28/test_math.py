import importlib.util,numpy as np
s=importlib.util.spec_from_file_location('b',str(__import__('pathlib').Path(__file__).resolve().parents[2]/'tools/task28_basins.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
for p in [1,2,3,6,32]:
 t=np.zeros((129,1,2));t[:,:,0]=1;t[:,0,1]=np.arange(129)%p
 assert m.period(t)[0].tolist()==[p]
t=np.zeros((129,1,2));t[:,:,0]=1;t[:,0,1]=np.arange(129);assert m.period(t)[0].tolist()==[0]
try:m.period(np.zeros((129,1,2)));raise RuntimeError('zero accepted')
except AssertionError:pass
assert m.wilson(0,128)[0]>=-1e-15 and m.wilson(128,128)[1]<=1+1e-15
print('8 manufactured full-state/interval controls passed')
