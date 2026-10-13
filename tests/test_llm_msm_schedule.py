import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


def test_schedule_factors():
    import explore_llm_msm as m

    assert np.allclose(m.schedule_factors(None, 5), 1.0)
    f = m.schedule_factors("0.5:1:0.5", 41)
    assert abs(f[0] - 0.5) < 1e-12 and abs(f[20] - 1.0) < 1e-12 and abs(f[40] - 0.5) < 1e-12 and abs(f[10] - 0.75) < 1e-12
