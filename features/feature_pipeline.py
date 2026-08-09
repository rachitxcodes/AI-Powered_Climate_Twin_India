from __future__ import annotations

from core.climate_dataset import ClimateDataset
from features.config import FeatureEngineeringConfig
from features.feature_engineering_result import FeatureEngineeringResult
from features.generators.temporal_generator import TemporalGenerator
from features.generators.seasonal_generator import SeasonalGenerator
from features.generators.lag_generator import LagGenerator
from features.generators.rolling_generator import RollingGenerator

class FeatureEngineeringPipeline:
    """
    Orchestrates the Feature Engineering pipeline.
    """

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.temporal_generator = TemporalGenerator()
        self.seasonal_generator = SeasonalGenerator()
        self.lag_generator = LagGenerator()
        self.rolling_generator = RollingGenerator()

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

        # Generate cyclical features
        if self.config.enable_cyclical_features:

            dataset = self.temporal_generator.generate_cyclical(dataset)

            generated_features.extend(
                [
                    "month_sin",
                    "month_cos",
                    "day_of_year_sin",
                    "day_of_year_cos",  
                ]
            )

        # Generate seasonal features
        if self.config.enable_seasonal_features:

            dataset = self.seasonal_generator.generate(dataset)

            generated_features.extend(
                [
                    "season",
                    "season_id",
                ]
            )

        # Generate lag features
        if self.config.enable_lag_features:

            dataset = self.lag_generator.generate(
                dataset,
                self.config.lag_days,
            )

            generated_features.extend(
                [
                    f"rainfall_lag_{lag}"
                    for lag in self.config.lag_days
                ]
            )

        # Generate rolling features
        if self.config.enable_rolling_features:

            dataset = self.rolling_generator.generate(
                dataset,
                self.config.rolling_windows,
            )

            generated_features.extend(
                [
                    f"rainfall_roll_mean_{window}"
                    for window in self.config.rolling_windows
                ]
            )
        
        # Wrap the engineered xarray.Dataset back into a ClimateDataset
        engineered_climate_dataset = ClimateDataset(dataset)

        return FeatureEngineeringResult(
            climate_dataset=engineered_climate_dataset,
            generated_features=generated_features,
        )