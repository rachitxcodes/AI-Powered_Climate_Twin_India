# features/pipeline.py

from __future__ import annotations

import pandas as pd

from features.config import FeatureEngineeringConfig
from features.generators import temporal_generator
from features.generators.temporal_generator import TemporalGenerator
from features.result import FeatureEngineeringResult


class FeatureEngineeringPipeline:
    """
    Orchestrates the Feature Engineering pipeline.
    """

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config

    def run(self, preprocessing_result) -> FeatureEngineeringResult:
        """
        Execute the Feature Engineering pipeline.
        """

        # Step 1
        data = preprocessing_result.processed_data

        # Step 2
        df = pd.DataFrame(
            data,
            columns=["rainfall"]  # temporary
        )

        # Step 3
        generated_features: list[str] = []

        # Future:
        #
        # if self.config.enable_temporal_features:
        #     ...
        if self.config.enable_temporal_features:
            df = temporal_generator.generate(df)
            generated_features.extend(
                ["year", "month", "day", "day_of_year"]
            )
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