from ingestion.data_loader import DataLoader
from preprocessing.preprocessing_pipeline import PreprocessingPipeline
from features.config import FeatureEngineeringConfig
from features.pipeline import FeatureEngineeringPipeline
from features.builders.spatial_dataframe_builder import SpatialDataFrameBuilder
from pprint import pprint


loader = DataLoader()
data = loader.load("rainfall", 2025)
pipeline = PreprocessingPipeline()
result = pipeline.run(data)

print(result.data.shape)


# feature_pipeline = FeatureEngineeringPipeline(
#     FeatureEngineeringConfig()
# )
# feature_result = feature_pipeline.run(result)


builder = SpatialDataFrameBuilder()

latitudes, longitudes = builder.build(
    data=result.data,
    metadata=result.metadata,
    year=result.year,
)

print("\nSpatial Coordinates")
print("---------------------------")

print("First 5 Latitudes :")
print(latitudes[:5])

print()

print("First 5 Longitudes :")
print(longitudes[:5])

print()

print("Latitude Count :", len(latitudes))
print("Longitude Count:", len(longitudes))


print("\nMetadata")
pprint(result.metadata)

stats = result.statistics
validation = result.validation

print("Statistics")
print(f"  Shape               : {stats.shape}")
print(f"  Dimensions          : {stats.dimensions}")
print(f"  Data Type           : {stats.dtype}")
print(f"  Total Values        : {stats.total_values}")
print(f"  Memory Usage (bytes): {stats.memory_usage_bytes}")
print(f"  Missing Values      : {stats.missing_values}")
print(f"  Missing (%)         : {stats.missing_percentage:.2f}%")
print(f"  Minimum Value       : {stats.minimum}")
print(f"  Maximum Value       : {stats.maximum}")
print(f"  Mean Value          : {stats.mean}")
print(f"  Median Value        : {stats.median}")
print(f"  Std Dev             : {stats.standard_deviation}")

print("Validation")
print(f"  Valid   : {validation.valid}")
print(f"  Errors  : {validation.errors if validation.errors else 'None'}")
print(f"  Warnings: {validation.warnings if validation.warnings else 'None'}")

print("Processed Data")
print(f"  Shape   : {result.data.shape}")
# print("\nEngineered Data")
# print(feature_result.engineered_data.head())
# print("\nGenerated Features")
# print(feature_result.generated_features)