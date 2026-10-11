import copy,json,sys
from pathlib import Path
import pytest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'tools/reviews')]
from review_unit_pair_forward_driver import validate_card

def test_literal_prefix_and_owned_path_controls():
 c=json.loads((Path(__file__).resolve().parents[1]/'docs/plans/native705-forward-independent-inputs.json').read_text());validate_card(c)
 for key,value in [('qualified_prefix',9),('expected_workers',0),('source','abc'),('out','/tmp/unowned')]:
  b=copy.deepcopy(c);b[key]=value
  with pytest.raises(ValueError):validate_card(b)
 b=copy.deepcopy(c);b['charts'][0]['path']='../other'
 with pytest.raises(ValueError):validate_card(b)
 b=copy.deepcopy(c);b['prior']['root']='/data2/neural-shrimp-data/audits/../../unowned'
 with pytest.raises(ValueError):validate_card(b)
