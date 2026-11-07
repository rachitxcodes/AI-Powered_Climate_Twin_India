
import numpy as np
import pandas as pd
import xarray as xr

from features.generators.lag_generator import LagGenerator
from features.generators.rolling_generator import RollingGenerator
from features.generators.seasonal_generator import SeasonalGenerator


def create_test_dataset():
    """Create a small, predictable climate dataset."""

    time = pd.date_range(
        start="2024-01-01",
        periods=10,
        freq="D",
    )

    rainfall = np.arange(
        1, 41, dtype=float
    ).reshape(10, 2, 2)

    dataset = xr.Dataset(
        data_vars={
            "rainfall": (
                ("time", "latitude", "longitude"),
                rainfall,
            ),
            "month": (
                "time",
                time.month,
            ),
        },
        coords={
            "time": time,
            "latitude": [26.0, 27.0],
            "longitude": [77.0, 78.0],
        },
    )

    return dataset


def test_lag_generator():
    dataset = create_test_dataset()

    result = LagGenerator().generate(
        dataset,
        lag_days=(1, 3),
    )

    assert "rainfall_lag_1" in result
    assert "rainfall_lag_3" in result

    # Day 2's lag-1 value must equal Day 1's rainfall.
    np.testing.assert_allclose(
        result["rainfall_lag_1"].isel(time=1).values,
        dataset["rainfall"].isel(time=0).values,
    )

    # Day 4's lag-3 value must equal Day 1's rainfall.
    np.testing.assert_allclose(
        result["rainfall_lag_3"].isel(time=3).values,
        dataset["rainfall"].isel(time=0).values,
    )

    # Initial lag positions should contain NaN.
    assert np.isnan(
        result["rainfall_lag_1"].isel(time=0).values
    ).all()

    assert result["rainfall_lag_1"].dims == (
        "time", "latitude", "longitude"
    )


def test_rolling_generator():
    dataset = create_test_dataset()

    result = RollingGenerator().generate(
        dataset,
        windows=(3,),
    )

    assert "rainfall_roll_mean_3" in result

    # First observation: mean of [1] = 1.
    np.testing.assert_allclose(
        result["rainfall_roll_mean_3"].isel(time=0).values,
        dataset["rainfall"].isel(time=0).values,
    )

    # Third observation: mean of [1, 5, 9] = 5
    np.testing.assert_allclose(
        result["rainfall_roll_mean_3"].isel(time=2).values,
        np.array([[5.0, 6.0], [7.0, 8.0]]),
    )

    # Fourth observation: mean of [5, 9, 13] = 9
    np.testing.assert_allclose(
        result["rainfall_roll_mean_3"].isel(time=3).values,
        np.array([[9.0, 10.0], [11.0, 12.0]]),
    )

    assert result["rainfall_roll_mean_3"].dims == (
        "time", "latitude", "longitude"
    )


def test_seasonal_generator():
    dataset = create_test_dataset()

    # Include months representing all four seasons.
    months = np.array([1, 2, 3, 5, 6, 8, 9, 10, 11, 12])

    dataset = dataset.assign(
        month=("time", months)
    )

    result = SeasonalGenerator().generate(dataset)

    expected_ids = np.array([
        0, 0, 1, 1, 2, 2, 2, 3, 3, 3
    ])

    expected_names = np.array([
        "Winter",
        "Winter",
        "Pre-Monsoon",
        "Pre-Monsoon",
        "Southwest Monsoon",
        "Southwest Monsoon",
        "Southwest Monsoon",
        "Post-Monsoon",
        "Post-Monsoon",
        "Post-Monsoon",
    ])

    np.testing.assert_array_equal(
        result["season_id"].values,
        expected_ids,
    )

    np.testing.assert_array_equal(
        result["season"].values,
        expected_names,
    )

    assert result["season"].dims == ("time",)
    assert result["season_id"].dims == ("time",)
