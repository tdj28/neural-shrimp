#!/usr/bin/env python3
"""Target-free isolated import check of EXP008's exact bound source allowlist."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import hashlib
ROOT=Path(__file__).resolve().parents[1]


def main():
    plan=json.loads((ROOT/'experiments/hopfield/EXP-008-plan.json').read_text())
    entries=plan['source_files']
    names=[entry['path'] for entry in entries]
    assert len(names)==len(set(names))
    for entry in entries:
        path=ROOT/entry['path']
        assert not Path(entry['path']).is_absolute() and '..' not in Path(entry['path']).parts
        assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
    with tempfile.TemporaryDirectory(prefix='EXP008-source-bundle-') as location:
        destination=Path(location)
        for name in names:
            path=destination/name;path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,path)
        actual={str(path.relative_to(destination)) for path in destination.rglob('*') if path.is_file()}
        assert actual==set(names), 'Missing or unexpected sealed source file'
        env={key:value for key,value in os.environ.items() if key not in ['PYTHONPATH','PYTHONHOME']}
        env['PYTHONDONTWRITEBYTECODE']='1'
        checked=[]
        owned_modules=[Path(name).stem for name in names if name.startswith('tools/') and name.endswith('.py')]
        wrapper = '''import json,pathlib,runpy,sys
script,bundle,names=sys.argv[1:]
owned=set(json.loads(names))|{'neural_shrimp'}
sys.argv=[script,'--help']
try:
    runpy.run_path(script,run_name='__main__')
except SystemExit as error:
    assert error.code in (None,0)
for name,module in list(sys.modules.items()):
    origin=getattr(module,'__file__',None)
    if origin and name.split('.')[0] in owned:
        assert pathlib.Path(origin).resolve().is_relative_to(pathlib.Path(bundle).resolve()),(name,origin)
'''
        for entrypoint in ['tools/run_hopfield_information.py','tools/prepare_hopfield_information.py']:
            result=subprocess.run([sys.executable,'-I','-B','-c',wrapper,str(destination/entrypoint),
                                   str(destination),json.dumps(owned_modules)],
                                  cwd=destination,env=env,check=True,capture_output=True,text=True)
            assert 'usage:' in result.stdout
            checked.append(entrypoint)
        assert {str(path.relative_to(destination)) for path in destination.rglob('*') if path.is_file()}==set(names)
        assert not list(destination.rglob('*.pyc'))
    print(json.dumps(dict(verified=True,source_files=len(names),entrypoints=checked,
                         isolated_interpreter=True,bytecode_written=False,new_trajectories=0)))

if __name__=='__main__':main()
