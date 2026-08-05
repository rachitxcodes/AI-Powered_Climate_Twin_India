from __future__ import annotations

from core.climate_dataset import ClimateDataset
from features.config import FeatureEngineeringConfig
from features.feature_engineering_result import FeatureEngineeringResult
from features.generators.temporal_generator import TemporalGenerator


class FeatureEngineeringPipeline:
    """
    Orchestrates the Feature Engineering pipeline.
    """

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.temporal_generator = TemporalGenerator()

    def run(
        self,
        preprocessing_result,
    ) -> FeatureEngineeringResult:

        # Extract ClimateDataset from preprocessing
        climate_dataset = preprocessing_result.climate_dataset

        # Extract the underlying xarray.Dataset
        dataset = climate_dataset.dataset

        generated_features: list[str] = []

        # Generate temporal features
        if self.config.enable_temporal_features:

            dataset = self.temporal_generator.generate(dataset)

            generated_features.extend(
                [
                    "year",
                    "month",
                    "day",
                    "day_of_year",
                ]
            )

        # Wrap the engineered xarray.Dataset back into a ClimateDataset
        engineered_climate_dataset = ClimateDataset(dataset)

        return FeatureEngineeringResult(
            climate_dataset=engineered_climate_dataset,
            generated_features=generated_features,
        )