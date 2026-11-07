from pathlib import Path

from ingestion.data_loader import DataLoader
from preprocessing.climate_dataset_validator import (
    ClimateDatasetValidator,
)


# 1. Locate the combined climate dataset
output_path = Path(
    "data/processed/climate/"
    "climate_2024-06-18_to_2024-06-19.nc"
)

# 2. Load the reference IMD dataset
imd_dataset = DataLoader().load(
    dataset="rainfall",
    year=2024,
).dataset

# 3. Validate the saved climate dataset
validator = ClimateDatasetValidator()

report = validator.validate_file(
    file_path=output_path,
    reference_dataset=imd_dataset,
)

# 4. Display results
print("\n========== CLIMATE DATASET VALIDATION ==========")

print("File:", output_path)
print("Validation status:", report.valid)

print("\nErrors:")
for error in report.errors:
    print("-", error)

print("\nWarnings:")
for warning in report.warnings:
    print("-", warning)

if report.valid:
    print("\nCLIMATE DATASET VALIDATION PASSED")
else:
    print("\nCLIMATE DATASET VALIDATION FAILED")
    raise RuntimeError("Climate dataset validation failed.")