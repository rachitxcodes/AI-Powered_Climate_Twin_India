from dataclasses import dataclass
import numpy as np

from preprocessing.statistics_report import StatisticsReport
from preprocessing.validation_report import ValidationReport


@dataclass(frozen=True)
class PreprocessingResult:
    """
    Stores the complete output of the preprocessing pipeline.
    """

    data: np.ndarray
    
    statistics: StatisticsReport

    validation: ValidationReport

