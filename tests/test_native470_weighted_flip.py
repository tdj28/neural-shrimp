import sys,copy
from pathlib import Path
from fractions import Fraction as F
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import probe_native470_weighted_flip as D

def test_exact_positive_weights_and_floor():
 card=D.C.read(ROOT/D.CARD);diag=D.C.read(Path(card['diagnosis']['root'])/'run/diagnostic.json');w=D.weights_from(diag);assert w==D.C.WEIGHTS and len(w)==470 and min(map(F,w))==F(1,2**20)and max(map(F,w))==1
 bad=copy.deepcopy(diag);bad['scaling']['collatz_max']=1.1
 with pytest.raises(ValueError):D.weights_from(bad)

def test_actual_retained_binding_and_different_weight_reject(tmp_path,monkeypatch):
 card=D.C.read(ROOT/D.CARD);monkeypatch.setattr(D.C,'OUT',tmp_path);w=D.C.Work();D.retained(w,card);assert w.native==0
 monkeypatch.setattr(D.C,'WEIGHTS',['1']*470)
 with pytest.raises(ValueError,match='literal exact'):D.retained(w,card)

def test_original_default_preserved_and_new_path():
 assert D.C.OUT==Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/unit-pair/pal/signed-flip-first001/attempt002/run')
 source=(ROOT/'tools/probe_native470_first_flip.py').read_text();assert "WEIGHTS=['1']*470"in source and 'weights=WEIGHTS,**SETTINGS'in source and 'if EXTRA_BINDING is not None'in source
