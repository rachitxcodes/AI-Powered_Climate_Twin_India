import numpy as np
import pandas as pd
import pytest
import xarray as xr

from preprocessing.fusion.dataset_fusion import DatasetFusion


def create_dataset(
    variable_name,
    values,
    times=None,
):
    if times is None:
        times = pd.to_datetime(
            [
                "2024-06-18",
                "2024-06-19",
            ]
        )

    latitude = np.array([25.0, 25.25])
    longitude = np.array([78.0, 78.25])

    return xr.Dataset(
        {
            variable_name: (
                ("time", "latitude", "longitude"),
                np.asarray(values, dtype=np.float32),
            )
        },
        coords={
            "time": times,
            "latitude": latitude,
            "longitude": longitude,
        },
    )


def test_fuses_aligned_datasets():
    rainfall = create_dataset(
        "rainfall",
        [
            [
                [10.0, 11.0],
                [12.0, 13.0],
            ],
            [
                [20.0, 21.0],
                [22.0, 23.0],
            ],
        ],
    )

    lst = create_dataset(
        "lst_mean",
        [
            [
                [300.0, 301.0],
                [302.0, 303.0],
            ],
            [
                [305.0, 306.0],
                [307.0, 308.0],
            ],
        ],
    )

    fusion = DatasetFusion()

    result = fusion.fuse(
        base=rainfall,
        additions=[lst],
    )

    assert "rainfall" in result
    assert "lst_mean" in result

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

    assert result.sizes["time"] == 2
    assert result.sizes["latitude"] == 2
    assert result.sizes["longitude"] == 2


def test_fuses_multiple_additional_datasets():
    rainfall = create_dataset(
        "rainfall",
        np.ones((2, 2, 2)),
    )

    lst = create_dataset(
        "lst_mean",
        np.ones((2, 2, 2)) * 300,
    )

    temperature = create_dataset(
        "temperature",
        np.ones((2, 2, 2)) * 25,
    )

    fusion = DatasetFusion()

    result = fusion.fuse(
        base=rainfall,
        additions=[
            lst,
            temperature,
        ],
    )

    assert set(result.data_vars) == {
        "rainfall",
        "lst_mean",
        "temperature",
    }


def test_mismatched_time_coordinates_are_rejected():
    rainfall = create_dataset(
        "rainfall",
        np.ones((2, 2, 2)),
    )

    different_times = pd.to_datetime(
        [
            "2024-06-19",
            "2024-06-20",
        ]
    )

    lst = create_dataset(
        "lst_mean",
        np.ones((2, 2, 2)),
        times=different_times,
    )

    fusion = DatasetFusion()

    with pytest.raises(
        AssertionError,
        match="time coordinates do not match",
    ):
        fusion.fuse(
            base=rainfall,
            additions=[lst],
        )


def test_mismatched_spatial_coordinates_are_rejected():
    rainfall = create_dataset(
        "rainfall",
        np.ones((2, 2, 2)),
    )

    lst = create_dataset(
        "lst_mean",
        np.ones((2, 2, 2)),
    ).assign_coords(
        latitude=np.array([26.0, 26.25])
    )

    fusion = DatasetFusion()

    with pytest.raises(
        AssertionError,
        match="latitude coordinates do not match",
    ):
        fusion.fuse(
            base=rainfall,
            additions=[lst],
        )


def test_variable_collision_is_rejected():
    rainfall_1 = create_dataset(
        "rainfall",
        np.ones((2, 2, 2)),
    )

    rainfall_2 = create_dataset(
        "rainfall",
        np.ones((2, 2, 2)) * 5,
    )

    fusion = DatasetFusion()

    with pytest.raises(
        ValueError,
        match="variable collision",
    ):
        fusion.fuse(
            base=rainfall_1,
            additions=[rainfall_2],
        )


def test_invalid_input_type_is_rejected():
    fusion = DatasetFusion()

    with pytest.raises(
        TypeError,
        match="Base dataset must be an xarray Dataset",
    ):
        fusion.fuse(
            base="not a dataset",
            additions=[],
        )