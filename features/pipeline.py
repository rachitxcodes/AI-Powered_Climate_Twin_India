from __future__ import annotations

import pandas as pd

from features.config import FeatureEngineeringConfig
from features.generators.temporal_generator import TemporalGenerator
from features.feature_engineering_result import FeatureEngineeringResult


class FeatureEngineeringPipeline:
    """
    Orchestrates the Feature Engineering pipeline.
    """

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.temporal_generator = TemporalGenerator()

    def run(self, preprocessing_result) -> FeatureEngineeringResult:
        """
        Execute the Feature Engineering pipeline.
        """

        # Step 1: Extract data and year from preprocessing result
        data = preprocessing_result.data
        year = preprocessing_result.year
        metadata = preprocessing_result.metadata

        # Step 2: Create Date column
        dates = pd.date_range(
            start=f"{year}-01-01",
            periods=len(data),
            freq="D",
        )

        df = pd.DataFrame(
            {
                "Date": dates,
                "Rainfall": data,
            }
        )

        # Step 3: Generate Features
        generated_features: list[str] = []

        if self.config.enable_temporal_features:
            df = self.temporal_generator.generate(df)

            generated_features.extend(
                [
                    "year",
                    "month",
                    "day",
                    "day_of_year",
                ]
            )

        # Future
        #
        # if self.config.enable_lag_features:
        #     ...
        #
        # if self.config.enable_rolling_features:
        #     ...
        #
        # if self.config.enable_seasonal_features:
        #     ...

        return FeatureEngineeringResult(
            engineered_data=df,
            generated_features=generated_features,
        )