"""Exploratory diagnostics for periodic windows in neural training maps.

Finding a stable periodic orbit does not establish shrimp geometry or utility.
"""

from .controls import HenonMap, LogisticMap
from .models import RegressionDataset, TanhRegressor, TrainingMap, make_teacher_regression

__all__ = ["HenonMap", "LogisticMap", "RegressionDataset", "TanhRegressor",
           "TrainingMap", "make_teacher_regression"]
