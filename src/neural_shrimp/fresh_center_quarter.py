"""Exact fixed cover around the already proved central subinterval.

Only bookkeeping and conservative measured admission; no model acquisition or
proof of root/stability is made here. Each cell retains native gauge authority.
"""
from fractions import Fraction as F


def require(v,message):
 if not v:raise ValueError(message)


def partition():
 left=[-49,-47,-45,-43,-41,-39,-37,-35,-34]
 right=[-32,-30,-28,-26,-24,-22,-20,-18,-17]
 new=[(F(a,2048),F(b,2048))for points in(left,right)for a,b in zip(points,points[1:])]
 existing=(F(-34,2048),F(-32,2048));cover=sorted(new+[existing])
 require(len(new)==16 and cover[0][0]==F(-49,2048)and cover[-1][1]==F(-17,2048),'literal first-quarter cover')
 require(all(a<b and b-a<=F(1,1024)for a,b in cover)and all(x[1]==y[0]for x,y in zip(cover,cover[1:])),'closed union with disjoint interiors')
 require(existing not in new,'no reacquisition of existing central cell')
 return [dict(index=i+1,endpoints=[str(a),str(b)],center=str((a+b)/2),halfwidth=str((b-a)/2))for i,(a,b)in enumerate(new)]


def complete_union(outcomes,prior):
 expected=partition();require(prior['passed']and prior['endpoints']==['-17/1024','-1/64'],'literal previously qualified central cell')
 require(len(outcomes)==len(expected),'all16 newly selected cells complete')
 for got,want in zip(outcomes,expected,strict=True):
  require(got['index']==want['index']and got['endpoints']==want['endpoints'],'literal ordered selected cells')
  require(got['passed']and got['native_calls']==126 and got['dimension']==705 and got['complement_dimension']==11,'complete same-root/full11 native cell')
 return dict(passed=True,endpoints=['-49/2048','-17/2048'],new_cells=16,reused_cells=1,total_cells=17)


def forecast(profile_seconds,profile_bytes,prior_seconds,prior_bytes,elapsed,peak_kib):
 import math
 vals=[profile_seconds,profile_bytes,prior_seconds,prior_bytes,elapsed,peak_kib]
 require(all(math.isfinite(v)and v>=0 for v in vals)and min(vals[:4])>0,'actual positive complete measured costs')
 # Keep the full prior one-cell runtime/bytes, including its profile and decode,
 # as a conservative per-cell floor despite reusing loaded data only once.
 seconds=elapsed+2*16*max(prior_seconds,profile_seconds)+60
 raw=profile_bytes+2*16*prior_bytes+64*1024**2
 return dict(passed=seconds<1500 and raw<3*1024**3 and peak_kib+512*1024<4*1024**2,forecast_seconds=seconds,forecast_bytes=raw,profile_native_calls=144,target_native_calls=16*126,total_native_calls=2160)
