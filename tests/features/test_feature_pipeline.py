import numpy as np

from features.config import FeatureEngineeringConfig
from features.pipeline import FeatureEngineeringPipeline


class DummyPreprocessingResult:
    def __init__(self):
        self.processed_data = np.array(
            [
                [10.0],
                [20.0],
                [15.0],
            ]
        )


def test_pipeline_returns_feature_engineering_result():

    config = FeatureEngineeringConfig()

    pipeline = FeatureEngineeringPipeline(config)

    result = pipeline.run(DummyPreprocessingResult())

    assert result is not None

    assert result.engineered_data.shape == (3, 1)

    assert list(result.engineered_data.columns) == ["rainfall"]

    assert result.generated_features == []