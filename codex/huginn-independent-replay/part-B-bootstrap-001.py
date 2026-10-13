"""Minimal one-worker bootstrap; external timeout owns the wall deadline."""
import hashlib,json,os,resource,runpy,sys
from pathlib import Path
script=Path(sys.argv[1]);card_path=Path(sys.argv[3]);card_bytes=card_path.read_bytes();card=json.loads(card_bytes)
assert sys.argv[2]=='--card'
assert card['status']=='accepted_by_root' and card['attempt']=='part-B-001'
assert os.environ['TASK8_CARD_SHA256']==hashlib.sha256(card_bytes).hexdigest()
assert os.environ['TASK8_SOURCE_COMMIT']==card['source_commit']
assert os.getpriority(os.PRIO_PROCESS,0)==10
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:assert os.environ.get(key)=='1'
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(300,305))
for name,expected in card['sources'].items():assert hashlib.sha256((script.parent/name).read_bytes()).hexdigest()==expected
assert script.name=='replay.py'
authority=Path('/data2/neural-shrimp-data/codex/huginn-independent-replay/part-B-001-supervision')
stat=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
identity={'pid':os.getpid(),'start':int(stat[19]),'pgid':os.getpgid(0),'sid':os.getsid(0),'uid':os.getuid(),'nice':os.getpriority(os.PRIO_PROCESS,0),'address_space_limits':resource.getrlimit(resource.RLIMIT_AS),'source_commit':card['source_commit'],'card_sha256':hashlib.sha256(card_bytes).hexdigest(),'bootstrap_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
with (authority/'worker-identity.json').open('x') as f:json.dump(identity,f,indent=2)
sys.argv=sys.argv[1:]
runpy.run_path(str(script),run_name='__main__')
