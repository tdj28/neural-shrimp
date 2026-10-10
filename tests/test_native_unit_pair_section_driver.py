import sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).parents[1]/'tools'),str(Path(__file__).parents[1]/'src')]
import scout_native_unit_pair_section as S

def test_fixed_native_section_settings_and_measured_caps():
 assert S.SETTINGS=={'max_points':48,'max_evaluations':360,'hidden_step':.25}
 assert S.BASE.name=='section'
 assert S.forecast([.1]*12,50000,[44000]*12)['admitted']
 assert not S.forecast([1.]*12,50000,[44000]*12)['admitted']
 assert S.SOURCES.count('src/neural_shrimp/unit_pair_section.py')==1
