from dataclasses import dataclass
import numpy as np

from preprocessing.statistics_report import StatisticsReport
from preprocessing.validation_report import ValidationReport
from core.climate_dataset import ClimateDataset

@dataclass(frozen=True, slots=True)
class PreprocessingResult:

    climate_dataset: ClimateDataset

    data: np.ndarray

    statistics: StatisticsReport

    validation: ValidationReport