"""Exact fixed whole34 partition; unchanged cubic proof kernels are caller-owned."""
from fractions import Fraction as F
from .rigorous_connection import require
from . import cubic_complement_geometry as G
ENDPOINTS=['83/512','115/512']
DOMAINS=64
CENTERS=192
CALLS=144+90*DOMAINS+126*CENTERS

def partition():
 a,b=map(F,ENDPOINTS);w=(b-a)/DOMAINS
 require(w==F(1,1024),'fixed narrowed actual derivative width')
 return [dict(index=i+1,endpoints=[str(a+i*w),str(a+(i+1)*w)])for i in range(DOMAINS)]

def children(rows,row,outcome):
 require(rows==partition()and row['depth']==0,'fixed maximum cell depth1')
 return G.children(rows,row,outcome)

def complete(rows,accepted):
 require(rows==partition()and len(accepted)<=128 and all(v['cell']['depth']<=1 for v in accepted),'fixed complete64-domain/depth1 cover')
 return G.complete(rows,accepted)

def reserve(calls,domains,centers,kind):
 require(calls==144+90*domains+126*centers and kind in('domain','cell'),'complete preceding phase grammar')
 require(centers<CENTERS and (kind!='domain'or domains<DOMAINS),'fixed full worst-case operation limit')
