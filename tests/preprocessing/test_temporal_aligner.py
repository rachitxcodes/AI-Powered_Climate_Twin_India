import numpy as np
import pytest
import xarray as xr
import pandas as pd
from preprocessing.alignment.temporal_aligner import TemporalAligner


def test_nearest_temporal_mapping():

    source = xr.Dataset(
        {
            "lst": (
                ("time",),
                np.array([100.0, 200.0, 300.0]),
            )
        },
        coords={
            "time": np.array(
                [
                    "2024-06-18T06:00",
                    "2024-06-18T06:30",
                    "2024-06-18T07:00",
                ],
                dtype="datetime64[m]",
            )
        },
    )

    target = xr.Dataset(
        coords={
            "time": np.array(
                [
                    "2024-06-18T06:10",
                    "2024-06-18T06:40",
                    "2024-06-18T07:20",
                ],
                dtype="datetime64[m]",
            )
        }
    )

    aligner = TemporalAligner(
        method="nearest",
        tolerance="30min",
    )

    mapping = aligner.build_mapping(
        source=source,
        target=target,
    )

    np.testing.assert_array_equal(
        mapping["source_indices"],
        np.array([0, 1, 2]),
    )

    np.testing.assert_array_equal(
        mapping["matched"],
        np.array([True, True, True]),
    )

def test_temporal_mapping_respects_tolerance():

    source = xr.Dataset(
        {
            "lst": (
                ("time",),
                np.array([100.0, 200.0]),
            )
        },
        coords={
            "time": np.array(
                [
                    "2024-06-18T06:00",
                    "2024-06-18T07:00",
                ],
                dtype="datetime64[m]",
            )
        }
    )

    target = xr.Dataset(
        coords={
            "time": np.array(
                [
                    "2024-06-18T06:10",
                ],
                dtype="datetime64[m]",
            )
        }
    )

    aligner = TemporalAligner(
        method="nearest",
        tolerance="5min",
    )

    mapping = aligner.build_mapping(
        source=source,
        target=target,
    )

    np.testing.assert_array_equal(
        mapping["source_indices"],
        np.array([-1]),
    )

    np.testing.assert_array_equal(
        mapping["matched"],
        np.array([False]),
    )

def test_apply_temporal_mapping():

    source = xr.Dataset(
        {
            "lst": (
                ("time",),
                np.array([100.0, 200.0, 300.0]),
            )
        },
        coords={
            "time": np.array(
                [
                    "2024-06-18T06:00",
                    "2024-06-18T06:30",
                    "2024-06-18T07:00",
                ],
                dtype="datetime64[m]",
            )
        }
    )

    target = xr.Dataset(
        coords={
            "time": np.array(
                [
                    "2024-06-18T06:10",
                    "2024-06-18T06:40",
                    "2024-06-18T07:20",
                ],
                dtype="datetime64[m]",
            )
        }
    )

    aligner = TemporalAligner(
        method="nearest",
        tolerance="30min",
    )

    mapping = aligner.build_mapping(
        source=source,
        target=target,
    )

    result = aligner.apply_mapping(
        source=source,
        target=target,
        variable="lst",
        mapping=mapping,
    )

    np.testing.assert_allclose(
        result["lst"].values,
        np.array([100.0, 200.0, 300.0]),
    )

    np.testing.assert_array_equal(
        result["time"].values,
        target["time"].values,
    )


def test_apply_temporal_mapping_fills_unmatched_with_nan():

    source = xr.Dataset(
        {
            "lst": (
                ("time",),
                np.array([100.0, 200.0]),
            )
        },
        coords={
            "time": np.array(
                [
                    "2024-06-18T06:00",
                    "2024-06-18T07:00",
                ],
                dtype="datetime64[m]",
            )
        }
    )

    target = xr.Dataset(
        coords={
            "time": np.array(
                [
                    "2024-06-18T06:10",
                    "2024-06-18T06:30",
                ],
                dtype="datetime64[m]",
            )
        }
    )

    aligner = TemporalAligner(
        method="nearest",
        tolerance="15min",
    )

    mapping = aligner.build_mapping(
        source=source,
        target=target,
    )

    result = aligner.apply_mapping(
        source=source,
        target=target,
        variable="lst",
        mapping=mapping,
    )

    assert result["lst"].values[0] == 100.0
    assert np.isnan(result["lst"].values[1])


def test_temporal_alignment_end_to_end():

    source = xr.Dataset(
        {
            "lst": (
                ("time",),
                np.array([100.0, 200.0, 300.0]),
            )
        },
        coords={
            "time": np.array(
                [
                    "2024-06-18T06:00",
                    "2024-06-18T06:30",
                    "2024-06-18T07:00",
                ],
                dtype="datetime64[m]",
            )
        }
    )

    target = xr.Dataset(
        coords={
            "time": np.array(
                [
                    "2024-06-18T06:10",
                    "2024-06-18T06:40",
                    "2024-06-18T07:20",
                ],
                dtype="datetime64[m]",
            )
        }
    )

    aligner = TemporalAligner(
        method="nearest",
        tolerance="30min",
    )

    result = aligner.align(
        source=source,
        target=target,
        variable="lst",
    )

    np.testing.assert_allclose(
        result["lst"].values,
        np.array([100.0, 200.0, 300.0]),
    )

    np.testing.assert_array_equal(
        result["time"].values,
        target["time"].values,
    )


def test_source_time_must_be_sorted():
    source = xr.Dataset(
        {
            "lst": (
                "time",
                np.array([300.0, 301.0, 302.0], dtype=np.float32),
            )
        },
        coords={
            "time": pd.to_datetime(
                [
                    "2024-06-18 07:00",
                    "2024-06-18 06:00",
                    "2024-06-18 06:30",
                ]
            )
        },
    )

    target = xr.Dataset(
        coords={
            "time": pd.to_datetime(["2024-06-18 06:10"])
        }
    )

    aligner = TemporalAligner()

    with pytest.raises(ValueError, match="Source.*sorted"):
        aligner.build_mapping(source, target)


def test_duplicate_source_times_are_rejected():
    source = xr.Dataset(
        {
            "lst": (
                "time",
                np.array([300.0, 301.0, 302.0], dtype=np.float32),
            )
        },
        coords={
            "time": pd.to_datetime(
                [
                    "2024-06-18 06:00",
                    "2024-06-18 06:30",
                    "2024-06-18 06:30",
                ]
            )
        },
    )

    target = xr.Dataset(
        coords={
            "time": pd.to_datetime(["2024-06-18 06:10"])
        }
    )

    aligner = TemporalAligner()

    with pytest.raises(ValueError, match="duplicate"):
        aligner.build_mapping(source, target)