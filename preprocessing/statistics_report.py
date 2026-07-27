from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class StatisticsReport:
    """
    Stores descriptive statistics of a climate dataset.

    This class is an immutable data container.
    It does not compute statistics itself.
    """

    # Dataset structure
    shape: Tuple[int, ...]
    dimensions: int
    dtype: str
    total_values: int
    memory_usage_bytes: int

    # Missing data
    missing_values: int
    missing_percentage: float

    # Descriptive statistics
    minimum: float
    maximum: float
    mean: float
    median: float
    standard_deviation: float