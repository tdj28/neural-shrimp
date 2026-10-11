#!/usr/bin/env python3
"""Independent numeric-packet codec and strict arithmetic comparisons. No producer imports.

Written before EXP014 outcomes. Original scientific thresholds are unchanged.
Independent arithmetic comparison: rtol1e-10/atol1e-12 for scalar readouts;
rtol1e-10/atol1e-11 for independently interpolated arrays. Categories,
identities, counts, hashes and shape/dtype are exact. CubicHermiteSpline and
math.erf provide implementations independent of the producer's Hermite/CDF.
Raw evidence is read-only; the requested report must be a new external file.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from scipy.interpolate import CubicHermiteSpline

SETTINGS={'core':(1.064,10.260722222222222),'failed-near-edge':(1.0661333333333334,10.297166666666667)}
METHODS=('rk4-0.0025','rk4-0.00125','dop853')
DIRECTIONS=('phase_write','phase_null','neuron_one')
DOSES=(1e-4,1e-3,1e-2);DELAYS=(1,2,4,8);NOISES=(1e-5,1e-4,1e-3)
TIMES=np.arange(12801,dtype=np.float64)*.025;TIMES[-1]=320.


def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def norm(v):return float(np.linalg.norm(v))


class Audit:
    def __init__(self):self.checks=0;self.failures=[];self.max_scalar_error=0.;self.max_array_error=0.
    def check(self,passed,label,details=None):
        self.checks+=1
        if not bool(passed):self.failures.append(dict(check=label,details=details))
        return bool(passed)
    def near(self,a,b,label,*,array=False):
        a,b=np.asarray(a),np.asarray(b)
        finite=a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        error=float(np.max(np.abs(a-b))) if finite and a.size else 0. if finite else None
        if error is not None:
            if array:self.max_array_error=max(self.max_array_error,error)
            else:self.max_scalar_error=max(self.max_scalar_error,error)
        return self.check(finite and np.allclose(a,b,rtol=1e-10,atol=1e-11 if array else 1e-12),label,dict(maximum_absolute_difference=error))
    def exact(self,a,b,label):return self.check(type(a) is type(b) and a==b,label)


def decode_packet(path,audit):
    """Independent inverse of the declared numeric JSON/NPZ storage contract."""
    path=Path(path);metadata=read(path);entry=metadata['arrays'];raw=path.parent/entry['path']
    if Path(entry['path']).name!=entry['path'] or not raw.resolve().is_relative_to(path.parent.resolve()):raise ValueError('Packet path escape')
    audit.check(metadata['schema']=='numeric-packet-v1','packet schema:'+str(path))
    audit.check(raw.stat().st_size==entry['bytes'] and sha(raw)==entry['sha256'],'packet bytes:'+str(path))
    with np.load(raw,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    used=set()
    def walk(v):
        if isinstance(v,list):return [walk(x) for x in v]
        if isinstance(v,dict):
            if '__array__' in v:
                key=v['__array__'];audit.check(set(v)=={'__array__','dtype','shape'} and key not in used,'unique array marker')
                used.add(key);array=arrays[key]
                audit.check(array.dtype.str==v['dtype'] and list(array.shape)==v['shape'] and not array.dtype.hasobject,'array dtype/shape')
                return array
            if '__nonfinite__' in v:
                audit.check(set(v)=={'__nonfinite__'} and v['__nonfinite__'] in ('nan','inf','-inf'),'nonfinite marker')
                return float(v['__nonfinite__'])
            return {k:walk(x) for k,x in v.items()}
        return v
    value=walk(metadata['tree']);audit.check(used==set(arrays),'NPZ complete member census')
    digest=hashlib.sha256(canonical(dict(tree=metadata['tree'],arrays={k:hashlib.sha256(v.tobytes(order='C')).hexdigest() for k,v in arrays.items()}))).hexdigest()
    audit.check(digest==metadata['semantic_sha256'],'packet semantic hash')
    return value

