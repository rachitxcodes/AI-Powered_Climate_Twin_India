from ingestion.data_loader import DataLoader
from preprocessing.preprocessing_pipeline import PreprocessingPipeline
from pprint import pprint


loader = DataLoader()

climate = loader.load(
    "rainfall",
    2025,
)

pipeline = PreprocessingPipeline()

result = pipeline.run(climate)

print(result.data.shape)

print("\nClimate Dataset")
print("-----------------------")
print(result.climate_dataset.dataset)

print("\nStatistics")
print(result.statistics)

print("\nValidation")
print(result.validation)