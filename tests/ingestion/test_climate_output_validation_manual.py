
from pathlib import Path

import numpy as np
import xarray as xr


project_root = Path(__file__).resolve().parents[2]

output_path = (
    project_root
    / "data"
    / "processed"
    / "climate"
    / "climate_2024-06-18.nc"
)

# 1. Confirm the file exists
assert output_path.is_file(), f"Output not found: {output_path}"

print("Output file size:", output_path.stat().st_size, "bytes")

# 2. Reopen the saved NetCDF file
with xr.open_dataset(output_path) as dataset:

    print("\nReopened NetCDF successfully.")
    print(dataset)

    # 3. Validate expected variables
    expected_variables = {
        "rainfall",
        "lst_mean",
        "valid_observation_count",
    }

    assert expected_variables.issubset(dataset.data_vars)

    # 4. Validate dimensions
    assert dataset.sizes["time"] == 1
    assert dataset.sizes["latitude"] == 129
    assert dataset.sizes["longitude"] == 135

    # 5. Validate date
    assert dataset.time.values[0] == np.datetime64("2024-06-18")

    # 6. Validate actual data
    rainfall = dataset["rainfall"].values
    lst = dataset["lst_mean"].values
    observation_count = dataset["valid_observation_count"].values

    valid_rainfall = int(np.isfinite(rainfall).sum())
    valid_lst = int(np.isfinite(lst).sum())
    observed_cells = int((observation_count > 0).sum())

    assert valid_lst > 0
    assert observed_cells > 0
    assert valid_rainfall > 0

    # 7. Check that LST and observation-count coverage agree
    np.testing.assert_array_equal(
        np.isfinite(lst),
        observation_count > 0,
    )

    print("\n========== SAVED DATA VALIDATION PASSED ==========")
    print("Valid rainfall cells:", valid_rainfall)
    print("Valid LST cells:", valid_lst)
    print("Grid cells with observations:", observed_cells)
    print("Date:", dataset.time.values[0])
    print("Variables:", list(dataset.data_vars))
