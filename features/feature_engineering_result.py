from dataclasses import dataclass

from core.climate_dataset import ClimateDataset


@dataclass(frozen=True, slots=True)
class FeatureEngineeringResult:

    climate_dataset: ClimateDataset

    generated_features: list[str]