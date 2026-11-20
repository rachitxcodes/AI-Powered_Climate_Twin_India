
import numpy as np
import pandas as pd
import xarray as xr

from core.climate_dataset import ClimateDataset
from storage.zarr_store import ZarrClimateStore


def test_features_cross_year_boundary(tmp_path):
    # Create 10 days spanning New Year.
    dates = pd.date_range(
        "2023-12-25",
        periods=10,
        freq="D",
    )

    # Different rainfall values make errors detectable.
    values = np.arange(1, 11, dtype=np.float32)

    ds = xr.Dataset(
        {
            "rainfall": (
                ("time", "latitude", "longitude"),
                values.reshape(10, 1, 1),
            )
        },
        coords={
            "time": dates,
            "latitude": [25.0],
            "longitude": [80.0],
        },
    )

    store = ZarrClimateStore()
    path = tmp_path / "boundary.zarr"

    store.save(ClimateDataset(ds), path)

    rainfall = store.load(path).dataset["rainfall"]

    lag_1 = rainfall.shift(time=1)
    rolling_7 = rainfall.rolling(
        time=7,
        min_periods=1,
    ).mean()

    # January 1 is the eighth observation.
    lag_value = lag_1.sel(
        time="2024-01-01"
    ).compute().item()

    rolling_value = rolling_7.sel(
        time="2024-01-01"
    ).compute().item()

    # December 31 rainfall is 7.
    assert lag_value == 7.0

    # Dec 26–Jan 1 values are 2 through 8.
    np.testing.assert_allclose(
        rolling_value,
        np.mean(np.arange(2, 9)),
    )

    # Both feature calculations remain Dask-backed.
    assert hasattr(lag_1.data, "dask")
    assert hasattr(rolling_7.data, "dask")
