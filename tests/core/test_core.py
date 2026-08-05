from ingestion.data_loader import DataLoader
from preprocessing.preprocessing_pipeline import PreprocessingPipeline
from core.climate_dataset_factory import ClimateDatasetFactory


# -----------------------------
# Load Dataset
# -----------------------------
loader = DataLoader()

loaded_dataset = loader.load(
    "rainfall",
    2025,
)

# -----------------------------
# Run Preprocessing
# -----------------------------
preprocessing = PreprocessingPipeline()

result = preprocessing.run(
    loaded_dataset,
)

# -----------------------------
# Build ClimateDataset
# -----------------------------
factory = ClimateDatasetFactory()

climate_dataset = factory.create(
    data=result.data,
    metadata=result.metadata,
    year=result.year,
)

# -----------------------------
# Inspect xarray Dataset
# -----------------------------
print("\nClimate Dataset")
print("---------------------------")
print(climate_dataset.dataset)