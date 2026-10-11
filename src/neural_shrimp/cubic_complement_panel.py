"""Bounded multi-chart cubic complement bookkeeping, with no proof arithmetic."""
from fractions import Fraction as F
from .rigorous_connection import require
from . import cubic_complement_geometry as H


def partition(endpoints):
    left,right=map(F,endpoints);count=(right-left)*128
    require(count.denominator==1 and 1<=count<=8,'one to eight exact1/128 derivative domains')
    return H.domains(str(left),str(right),int(count))


def call_cap(domains,centers):
    require(type(domains)is int and 1<=domains<=48 and type(centers)is int and 1<=centers<=256,'declared finite panel')
    return 144+90*domains+126*centers


def reserve(calls,domains_done,centers_done,kind,max_domains,max_centers):
    require(calls==144+90*domains_done+126*centers_done,'complete preceding native accounting')
    require(kind in ('domain','cell'),'literal next unit')
    require(centers_done<max_centers and (kind!='domain'or domains_done<max_domains),'fixed global operation ceiling')
    return 90 if kind=='domain'else 126


def chart_complete(domains,accepted):
    return H.complete(domains,accepted)
