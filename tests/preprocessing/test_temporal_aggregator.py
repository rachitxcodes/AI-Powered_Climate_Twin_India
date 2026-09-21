import numpy as np
import pandas as pd
import pytest
import xarray as xr

from preprocessing.temporal.temporal_aggregator import TemporalAggregator


def test_daily_mean_aggregation():
    times = pd.to_datetime(
        [
            "2024-06-18 06:00",
            "2024-06-18 06:30",
            "2024-06-18 07:00",
            "2024-06-19 06:00",
            "2024-06-19 06:30",
        ]
    )

    dataset = xr.Dataset(
        {
            "lst": (
                "time",
                np.array(
                    [300.0, 302.0, 304.0, 310.0, 312.0],
                    dtype=np.float32,
                ),
            )
        },
        coords={"time": times},
    )

    aggregator = TemporalAggregator()

    result = aggregator.aggregate(
        dataset,
        variable="lst",
    )

    assert "lst_mean" in result
    assert "valid_observation_count" in result

    np.testing.assert_allclose(
        result["lst_mean"].values,
        [302.0, 311.0],
    )


def test_daily_aggregation_ignores_missing_values():
    times = pd.to_datetime(
        [
            "2024-06-18 06:00",
            "2024-06-18 06:30",
            "2024-06-18 07:00",
        ]
    )

    dataset = xr.Dataset(
        {
            "lst": (
                "time",
                np.array(
                    [300.0, np.nan, 306.0],
                    dtype=np.float32,
                ),
            )
        },
        coords={"time": times},
    )

    aggregator = TemporalAggregator()

    result = aggregator.aggregate(
        dataset,
        variable="lst",
    )

    np.testing.assert_allclose(
        result["lst_mean"].values,
        [303.0],
    )

    assert result["valid_observation_count"].values[0] == 2


def test_daily_aggregation_preserves_spatial_dimensions():
    times = pd.to_datetime(
        [
            "2024-06-18 06:00",
            "2024-06-18 06:30",
        ]
    )

    dataset = xr.Dataset(
        {
            "lst": (
                ("time", "latitude", "longitude"),
                np.array(
                    [
                        [
                            [300.0, 301.0],
                            [302.0, 303.0],
                        ],
                        [
                            [304.0, 305.0],
                            [306.0, 307.0],
                        ],
                    ],
                    dtype=np.float32,
                ),
            )
        },
        coords={
            "time": times,
            "latitude": [25.0, 25.25],
            "longitude": [78.0, 78.25],
        },
    )

    aggregator = TemporalAggregator()

    result = aggregator.aggregate(
        dataset,
        variable="lst",
    )

    assert result["lst_mean"].dims == (
        "time",
        "latitude",
        "longitude",
    )

    assert result["lst_mean"].shape == (1, 2, 2)

    np.testing.assert_allclose(
        result["lst_mean"].values,
        [
            [
                [302.0, 303.0],
                [304.0, 305.0],
            ]
        ],
    )


def test_only_daily_frequency_is_supported():
    with pytest.raises(ValueError):
        TemporalAggregator(frequency="1H")


def test_missing_variable_is_rejected():
    dataset = xr.Dataset(
        coords={
            "time": pd.to_datetime(["2024-06-18 06:00"])
        }
    )

    aggregator = TemporalAggregator()

    with pytest.raises(ValueError, match="Variable 'lst' not found"):
        aggregator.aggregate(
            dataset,
            variable="lst",
        )

def test_daily_aggregation_preserves_geolocation():
    times = pd.to_datetime(
        [
            "2024-06-18 06:00",
            "2024-06-18 06:30",
        ]
    )

    dataset = xr.Dataset(
        {
            "lst": (
                ("time", "y", "x"),
                np.array(
                    [
                        [
                            [300.0, 301.0],
                            [302.0, 303.0],
                        ],
                        [
                            [304.0, 305.0],
                            [306.0, 307.0],
                        ],
                    ],
                    dtype=np.float32,
                ),
            )
        },
        coords={
            "time": times,
            "y": [0, 1],
            "x": [0, 1],
            "latitude": (
                ("y", "x"),
                np.array(
                    [
                        [25.0, 25.0],
                        [25.25, 25.25],
                    ]
                ),
            ),
            "longitude": (
                ("y", "x"),
                np.array(
                    [
                        [78.0, 78.25],
                        [78.0, 78.25],
                    ]
                ),
            ),
        },
    )

    aggregator = TemporalAggregator()

    result = aggregator.aggregate(
        dataset,
        variable="lst",
    )

    assert "latitude" in result.coords
    assert "longitude" in result.coords

    assert result["latitude"].dims == ("y", "x")
    assert result["longitude"].dims == ("y", "x")

    np.testing.assert_array_equal(
        result["latitude"].values,
        dataset["latitude"].values,
    )

    np.testing.assert_array_equal(
        result["longitude"].values,
        dataset["longitude"].values,
    )