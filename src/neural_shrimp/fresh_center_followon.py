"""Exact bounded second-quarter partition, retaining the native gauge."""
from fractions import Fraction as F


def need(v,message):
 if not v:raise ValueError(message)


def domains():
 return [dict(index=i+1,endpoints=[str(F(a,2048)),str(F(b,2048))])for i,(a,b)in enumerate([(-17,-1),(-1,15)])]


def cell(domain,depth=0,index=0):
 need(domain in(1,2)and type(depth)is int and 0<=depth<=3 and type(index)is int and 0<=index<2**depth,'fixed exact binary cell')
 lo,hi=map(F,domains()[domain-1]['endpoints']);width=(hi-lo)/2**depth;a=lo+index*width;b=a+width
 return dict(domain=domain,depth=depth,index=index,endpoints=[str(a),str(b)],center=str((a+b)/2),halfwidth=str((b-a)/2))


def children(row,outcome):
 need(row==cell(row['domain'],row['depth'],row['index']),'literal cell geometry')
 need(outcome['completed']and outcome['native_calls']==126 and outcome['dimension']==705 and outcome['complement_dimension']==11 and outcome['passed']is False,'only complete negative full11 metric may split')
 need(row['depth']<3,'fixed minimum halfwidth1/2048')
 return [cell(row['domain'],row['depth']+1,2*row['index']+i)for i in(0,1)]


def complete(rows):
 need(rows and len(rows)<=16,'finite accepted leaf inventory')
 intervals=[]
 for r in rows:
  c=r['cell'];need(c==cell(c['domain'],c['depth'],c['index'])and r['passed']and r['completed']and r['native_calls']==126 and r['dimension']==705 and r['complement_dimension']==11,'each complete native full11 leaf')
  intervals.append(tuple(map(F,c['endpoints'])))
 intervals.sort();need(intervals[0][0]==F(-17,2048)and intervals[-1][1]==F(15,2048)and all(a[1]==b[0]for a,b in zip(intervals,intervals[1:])),'exact closed second-quarter union without gaps or duplicated interiors')
 return dict(passed=True,endpoints=['-17/2048','15/2048'],accepted_cells=len(rows),domain_count=2)


def forecast(profile_seconds,profile_bytes,pilot_seconds,pilot_bytes,elapsed,peak_kib):
 import math
 need(all(math.isfinite(v)and v>0 for v in[profile_seconds,profile_bytes,pilot_seconds,pilot_bytes,elapsed,peak_kib]),'actual positive full profile and completed pilot costs')
 seconds=elapsed+2*(30*max(profile_seconds,pilot_seconds)+2*profile_seconds)+2*61.466+30
 raw=profile_bytes+2*(30*pilot_bytes+2*profile_bytes)+64*1024**2
 return dict(passed=seconds<3000 and raw<6*1024**3 and peak_kib+512*1024<4*1024**2,forecast_seconds=seconds,forecast_bytes=raw,maximum_centers=30,profile_calls=144,maximum_target_calls=3960,maximum_total_calls=4104)
