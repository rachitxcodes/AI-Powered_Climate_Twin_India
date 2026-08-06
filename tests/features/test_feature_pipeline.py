from ingestion.data_loader import DataLoader
from preprocessing.preprocessing_pipeline import PreprocessingPipeline
from features.config import FeatureEngineeringConfig
from features.pipeline import FeatureEngineeringPipeline


loader = DataLoader()

climate = loader.load(
    "rainfall",
    2025,
)

preprocessing = PreprocessingPipeline()

preprocessing_result = preprocessing.run(
    climate,
)

pipeline = FeatureEngineeringPipeline(
    FeatureEngineeringConfig()
)

feature_result = pipeline.run(
    preprocessing_result,
)

print("\nFeature Engineered Dataset")
print("--------------------------------")

print(feature_result.climate_dataset.dataset)

print()

print("Generated Features")

print(feature_result.generated_features)