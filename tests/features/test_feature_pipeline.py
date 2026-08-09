from ingestion.data_loader import DataLoader
from preprocessing.preprocessing_pipeline import PreprocessingPipeline
from features.config import FeatureEngineeringConfig
from features.feature_pipeline import FeatureEngineeringPipeline

import numpy as np


# ============================================================
# Load Climate Dataset
# ============================================================

loader = DataLoader()

climate = loader.load(
    "rainfall",
    2025,
)


# ============================================================
# Preprocessing
# ============================================================

preprocessing = PreprocessingPipeline()

preprocessing_result = preprocessing.run(
    climate,
)


# ============================================================
# Feature Engineering
# ============================================================

pipeline = FeatureEngineeringPipeline(
    FeatureEngineeringConfig()
)

feature_result = pipeline.run(
    preprocessing_result,
)

dataset = feature_result.climate_dataset.dataset


# ============================================================
# Inspect Feature Engineered Dataset
# ============================================================

print("\nFeature Engineered Dataset")
print("--------------------------------")

print(dataset)


print("\nGenerated Features")
print("--------------------------------")

print(feature_result.generated_features)


# ============================================================
# Verify Rolling Features Exist
# ============================================================

assert "rainfall_roll_mean_7" in dataset
assert "rainfall_roll_mean_30" in dataset

print("\nRolling Features Exist: PASS")


# ============================================================
# Verify Rolling Feature Dimensions
# ============================================================

assert dataset["rainfall_roll_mean_7"].dims == (
    "time",
    "latitude",
    "longitude",
)

assert dataset["rainfall_roll_mean_30"].dims == (
    "time",
    "latitude",
    "longitude",
)

print("Rolling Feature Dimensions: PASS")


# ============================================================
# Verify Cyclical Features Exist
# ============================================================

cyclical_features = [
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]

for feature in cyclical_features:
    assert feature in dataset

print("Cyclical Features Exist: PASS")


# ============================================================
# Verify Cyclical Feature Dimensions
# ============================================================

for feature in cyclical_features:

    assert dataset[feature].dims == (
        "time",
    )

print("Cyclical Feature Dimensions: PASS")


# ============================================================
# Verify January 1
# ============================================================

jan_1 = dataset.sel(
    time="2025-01-01"
)

print("\nCyclical Encoding Verification")
print("--------------------------------")

print("January 1")

print(
    "month_sin:",
    jan_1["month_sin"].item()
)

print(
    "month_cos:",
    jan_1["month_cos"].item()
)

print(
    "day_of_year_sin:",
    jan_1["day_of_year_sin"].item()
)

print(
    "day_of_year_cos:",
    jan_1["day_of_year_cos"].item()
)


assert np.isclose(
    jan_1["month_sin"].item(),
    0.0,
    atol=1e-6,
)

assert np.isclose(
    jan_1["month_cos"].item(),
    1.0,
    atol=1e-6,
)

assert np.isclose(
    jan_1["day_of_year_sin"].item(),
    0.0,
    atol=1e-6,
)

assert np.isclose(
    jan_1["day_of_year_cos"].item(),
    1.0,
    atol=1e-6,
)

print("January 1 Cyclical Encoding: PASS")


# ============================================================
# Verify July 1
# ============================================================

jul_1 = dataset.sel(
    time="2025-07-01"
)

print("\nJuly 1")

print(
    "month_sin:",
    jul_1["month_sin"].item()
)

print(
    "month_cos:",
    jul_1["month_cos"].item()
)


assert np.isclose(
    jul_1["month_sin"].item(),
    0.0,
    atol=1e-6,
)

assert np.isclose(
    jul_1["month_cos"].item(),
    -1.0,
    atol=1e-6,
)

print("July 1 Month Cyclical Encoding: PASS")


# ============================================================
# Verify Day-of-Year Progression
# ============================================================

jan_2 = dataset.sel(
    time="2025-01-02"
)

assert jan_2["day_of_year"].item() == 2

assert jan_2["day_of_year_sin"].item() > 0

print("Day-of-Year Cyclical Progression: PASS")


# ============================================================
# Verify Cyclical Values Stay Within [-1, 1]
# ============================================================

for feature in [
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]:

    values = dataset[feature].values

    assert np.all(values >= -1.0 - 1e-6)
    assert np.all(values <= 1.0 + 1e-6)

print("Cyclical Value Range [-1, 1]: PASS")


# ============================================================
# Find a Grid Cell With Valid Data
# ============================================================

rainfall = dataset["rainfall"]

valid_cell = None

for lat_index in range(
    rainfall.sizes["latitude"]
):

    for lon_index in range(
        rainfall.sizes["longitude"]
    ):

        values = rainfall.isel(
            latitude=lat_index,
            longitude=lon_index,
            time=slice(0, 7),
        ).values

        if (
            np.isfinite(values).all()
            and np.any(values != 0)
        ):

            valid_cell = (
                lat_index,
                lon_index,
            )

            break

    if valid_cell is not None:
        break


assert valid_cell is not None

lat_index, lon_index = valid_cell

print(
    f"\nTesting grid cell: "
    f"latitude index={lat_index}, "
    f"longitude index={lon_index}"
)


# ============================================================
# Extract Rainfall and Rolling Mean
# ============================================================

rainfall_point = rainfall.isel(
    latitude=lat_index,
    longitude=lon_index,
)

rolling_7_point = dataset[
    "rainfall_roll_mean_7"
].isel(
    latitude=lat_index,
    longitude=lon_index,
)


# ============================================================
# Manually Calculate Expected Values
# ============================================================

expected_day_1 = rainfall_point.isel(
    time=0
).values

expected_day_2 = np.mean(
    rainfall_point.isel(
        time=slice(0, 2)
    ).values
)

expected_day_7 = np.mean(
    rainfall_point.isel(
        time=slice(0, 7)
    ).values
)


# ============================================================
# Compare Actual vs Expected
# ============================================================

actual_day_1 = rolling_7_point.isel(
    time=0
).values

actual_day_2 = rolling_7_point.isel(
    time=1
).values

actual_day_7 = rolling_7_point.isel(
    time=6
).values


print("\nRolling Mean Verification")
print("--------------------------------")

print("Day 1")
print("Expected:", expected_day_1)
print("Actual  :", actual_day_1)

print("\nDay 2")
print("Expected:", expected_day_2)
print("Actual  :", actual_day_2)

print("\nDay 7")
print("Expected:", expected_day_7)
print("Actual  :", actual_day_7)


# ============================================================
# Rolling Assertions
# ============================================================

assert np.isclose(
    actual_day_1,
    expected_day_1,
)

assert np.isclose(
    actual_day_2,
    expected_day_2,
)

assert np.isclose(
    actual_day_7,
    expected_day_7,
)

print("\nRolling Mean Calculation: PASS")


# ============================================================
# Final Result
# ============================================================

print("\n========================================")
print("FEATURE ENGINEERING INTEGRATION TEST: PASS")
print("========================================")