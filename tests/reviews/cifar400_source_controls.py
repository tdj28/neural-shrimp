from pathlib import Path
import importlib.util,time,os
import numpy as np
root=Path(os.environ['CIFAR400_REVIEW_ROOT'])
s=importlib.util.spec_from_file_location('review_cifar400',root/'tools/diagnose_cifar_large_space.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
# Every operation's declared final attempt is allowed once; a subsequent attempt
# fails before the function, and a failed operation consumes its attempt.
for kind,limit in m.CAPS.items():
 events=[];b=m.Budget400(events.append);b.attempted[kind]=limit-1
 assert 89 < b.deadline-time.monotonic() <=90
 assert b.call(kind,lambda:7)==7
 try:b.call(kind,lambda:(_ for _ in ()).throw(AssertionError('called beyond cap')))
 except ValueError:pass
 else:raise AssertionError('cap accepted')
 assert b.attempted[kind]==limit and b.returned[kind]==1
# NaN/Inf in any matched baseline slot fail after being durably emitted.
for value in (float('nan'),float('inf')):
 ev=[];p=m.Prefix400(np.ones(200),ev.append)
 try:p(dict(kind='gmres_inner',iteration=1,estimated_relative_residual=value))
 except ValueError:pass
 else:raise AssertionError('nonfinite prefix accepted')
 assert len(ev)==1
# Exactly 200 matching callbacks may qualify the control; no 400-count success is
# presumed or forced. Actual true residual and merit gates remain in the core.
p=m.Prefix400(np.ones(200),lambda _:None)
for i in range(200):p(dict(kind='gmres_inner',iteration=i+1,estimated_relative_residual=1.))
p.complete()
print('7 independent source controls passed; no target input/model calls')
