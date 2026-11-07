
from pathlib import Path

import numpy as np

from ingestion.data_loader import DataLoader
from preprocessing.climate_pipeline import ClimatePipeline


# ---------------------------------------------------------
# 1. Locate the project and API-downloaded INSAT files
# ---------------------------------------------------------
project_root = Path(__file__).resolve().parents[2]

insat_dir = project_root / "data" / "raw" / "insat" / "api_validation"

insat_files = sorted(insat_dir.glob("*.h5"))

print("INSAT files found:", len(insat_files))

if len(insat_files) != 23:
    raise RuntimeError(
        f"Expected 23 INSAT files, found {len(insat_files)}"
    )


# ---------------------------------------------------------
# 2. Load real IMD 2024 rainfall data
# ---------------------------------------------------------
print("\nLoading IMD 2024 rainfall dataset...")

imd_dataset = DataLoader().load(
    dataset="rainfall",
    year=2024,
).dataset

print("IMD dataset loaded.")
print("IMD dimensions:", dict(imd_dataset.sizes))


# ---------------------------------------------------------
# 3. Run the existing ClimatePipeline
# ---------------------------------------------------------
print("\nRunning ClimatePipeline with API-downloaded INSAT files...")

pipeline = ClimatePipeline()

result = pipeline.process(
    imd_dataset=imd_dataset,
    insat_files=[str(path) for path in insat_files],
)


# ---------------------------------------------------------
# 4. Validate the unified climate dataset
# ---------------------------------------------------------
required_variables = [
    "rainfall",
    "lst_mean",
    "valid_observation_count",
]

for variable in required_variables:
    assert variable in result, f"Missing variable: {variable}"

assert result.sizes["time"] == 1

assert result.time.values[0] == np.datetime64("2024-06-18")

assert result["rainfall"].dims == (
    "time",
    "latitude",
    "longitude",
)

assert result["lst_mean"].dims == (
    "time",
    "latitude",
    "longitude",
)

assert result["valid_observation_count"].dims == (
    "time",
    "latitude",
    "longitude",
)

valid_lst_count = int(
    np.isfinite(result["lst_mean"].values).sum()
)

valid_observation_count = int(
    (result["valid_observation_count"].values > 0).sum()
)

assert valid_lst_count > 0


# ---------------------------------------------------------
# 5. Print final result
# ---------------------------------------------------------
print("\n========== INTEGRATION SUCCESS ==========")

print("INSAT files processed:", len(insat_files))
print("Output dimensions:", dict(result.sizes))
print("Output variables:", list(result.data_vars))
print("Output date:", result.time.values[0])

print("Valid LST grid cells:", valid_lst_count)
print("Grid cells with INSAT observations:", valid_observation_count)

print("\nClimatePipeline successfully processed")
print("MOSDAC-acquired INSAT data with real IMD rainfall.")
