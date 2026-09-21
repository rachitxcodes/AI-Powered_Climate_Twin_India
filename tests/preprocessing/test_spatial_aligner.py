import numpy as np
import xarray as xr
from pathlib import Path

from ingestion.data_loader import DataLoader
from ingestion.readers.insat_reader import INSATReader
from preprocessing.alignment.spatial_aligner import SpatialAligner
from preprocessing.temporal.temporal_aggregator import TemporalAggregator


def test_nearest_spatial_alignment():
    source_latitude = np.array([
        [10.0, 10.0, 10.0],
        [11.0, 11.0, 11.0],
        [12.0, 12.0, 12.0],
    ])

    source_longitude = np.array([
        [20.0, 21.0, 22.0],
        [20.0, 21.0, 22.0],
        [20.0, 21.0, 22.0],
    ])

    source_lst = np.array([
        [10.0, 11.0, 12.0],
        [20.0, 21.0, 22.0],
        [30.0, 31.0, 32.0],
    ])

    source = xr.Dataset(
        {
            "lst": (("y", "x"), source_lst)
        },
        coords={
            "latitude": (("y", "x"), source_latitude),
            "longitude": (("y", "x"), source_longitude),
        },
    )

    target_latitude = np.array([10.1, 11.8])
    target_longitude = np.array([20.1, 21.8])

    target = xr.Dataset(
        coords={
            "latitude": target_latitude,
            "longitude": target_longitude,
        }
    )

    aligner = SpatialAligner(method="nearest")

    result = aligner.align(
        source=source,
        target=target,
        variable="lst",
    )

    expected = np.array([
        [10.0, 12.0],
        [30.0, 32.0],
    ])

    np.testing.assert_allclose(
        result["lst"].values,
        expected,
    )


def test_real_insat_spatial_alignment():
    # --------------------------------------------------
    # Locate the real INSAT sample
    # --------------------------------------------------

    insat_file = Path(
        "tests/fixtures/3DIMG_18JUN2024_0600_L2B_LST_V01R00.h5"
    )

    assert insat_file.exists(), (
        f"INSAT test file not found: {insat_file}"
    )

    # --------------------------------------------------
    # Read real INSAT data
    # --------------------------------------------------

    insat_reader = INSATReader()
    insat_dataset = insat_reader.read(insat_file)

    # --------------------------------------------------
    # Load real IMD rainfall dataset
    # --------------------------------------------------

    loader = DataLoader()
    imd_climate_dataset = loader.load(
        dataset="rainfall",
        year=2025,
    )

    imd_dataset = imd_climate_dataset.dataset

    # --------------------------------------------------
    # Perform spatial alignment
    # --------------------------------------------------

    aligner = SpatialAligner(method="nearest")

    result = aligner.align(
        source=insat_dataset,
        target=imd_dataset,
        variable="lst",
    )

    # --------------------------------------------------
    # Basic output validation
    # --------------------------------------------------

    assert "lst" in result

    assert result["lst"].dims == (
        "time",
        "latitude",
        "longitude",
    )

    assert result.sizes["latitude"] == imd_dataset.sizes["latitude"]
    assert result.sizes["longitude"] == imd_dataset.sizes["longitude"]

    # --------------------------------------------------
    # Coordinate validation
    # --------------------------------------------------

    np.testing.assert_array_equal(
        result["latitude"].values,
        imd_dataset["latitude"].values,
    )

    np.testing.assert_array_equal(
        result["longitude"].values,
        imd_dataset["longitude"].values,
    )

    # --------------------------------------------------
    # Value validation
    # --------------------------------------------------

    lst_values = result["lst"].values

    assert np.isfinite(lst_values).any()

    assert np.nanmin(lst_values) >= 200
    assert np.nanmax(lst_values) <= 400


def test_real_insat_daily_aggregation_then_spatial_alignment():
    # --------------------------------------------------
    # Locate the real INSAT sample
    # --------------------------------------------------

    insat_file = Path(
        "tests/fixtures/3DIMG_18JUN2024_0600_L2B_LST_V01R00.h5"
    )

    assert insat_file.exists(), (
        f"INSAT test file not found: {insat_file}"
    )

    # --------------------------------------------------
    # Read real INSAT data
    # --------------------------------------------------

    insat_reader = INSATReader()
    insat_dataset = insat_reader.read(insat_file)

    # --------------------------------------------------
    # Verify real INSAT geolocation
    # --------------------------------------------------

    assert "latitude" in insat_dataset.coords
    assert "longitude" in insat_dataset.coords

    # --------------------------------------------------
    # Aggregate INSAT to daily resolution
    # --------------------------------------------------

    aggregator = TemporalAggregator()

    daily_insat = aggregator.aggregate(
        dataset=insat_dataset,
        variable="lst",
    )

    # --------------------------------------------------
    # Validate daily aggregation output
    # --------------------------------------------------

    assert "lst_mean" in daily_insat
    assert "valid_observation_count" in daily_insat

    assert "latitude" in daily_insat.coords
    assert "longitude" in daily_insat.coords

    assert daily_insat["lst_mean"].dims == (
        "time",
        "y",
        "x",
    )

    assert daily_insat.sizes["time"] == 1

    # --------------------------------------------------
    # Load real IMD rainfall dataset
    # --------------------------------------------------

    loader = DataLoader()

    imd_climate_dataset = loader.load(
        dataset="rainfall",
        year=2025,
    )

    imd_dataset = imd_climate_dataset.dataset

    # --------------------------------------------------
    # Spatially align daily INSAT to IMD grid
    # --------------------------------------------------

    aligner = SpatialAligner(method="nearest")

    result = aligner.align(
        source=daily_insat,
        target=imd_dataset,
        variable="lst_mean",
    )

    # --------------------------------------------------
    # Validate aligned output
    # --------------------------------------------------

    assert "lst_mean" in result

    assert result["lst_mean"].dims == (
        "time",
        "latitude",
        "longitude",
    )

    assert result.sizes["time"] == 1

    assert result.sizes["latitude"] == (
        imd_dataset.sizes["latitude"]
    )

    assert result.sizes["longitude"] == (
        imd_dataset.sizes["longitude"]
    )

    # --------------------------------------------------
    # Validate target coordinates
    # --------------------------------------------------

    np.testing.assert_array_equal(
        result["latitude"].values,
        imd_dataset["latitude"].values,
    )

    np.testing.assert_array_equal(
        result["longitude"].values,
        imd_dataset["longitude"].values,
    )

    # --------------------------------------------------
    # Validate LST values
    # --------------------------------------------------

    lst_values = result["lst_mean"].values

    assert np.isfinite(lst_values).any()

    assert np.nanmin(lst_values) >= 200
    assert np.nanmax(lst_values) <= 400


def test_spatial_mapping_can_be_reused():
    source_latitude = np.array([
        [10.0, 10.0, 10.0],
        [11.0, 11.0, 11.0],
        [12.0, 12.0, 12.0],
    ])

    source_longitude = np.array([
        [20.0, 21.0, 22.0],
        [20.0, 21.0, 22.0],
        [20.0, 21.0, 22.0],
    ])

    first_values = np.array([
        [10.0, 11.0, 12.0],
        [20.0, 21.0, 22.0],
        [30.0, 31.0, 32.0],
    ])

    second_values = np.array([
        [100.0, 110.0, 120.0],
        [200.0, 210.0, 220.0],
        [300.0, 310.0, 320.0],
    ])

    source_first = xr.Dataset(
        {
            "lst": (("y", "x"), first_values)
        },
        coords={
            "latitude": (("y", "x"), source_latitude),
            "longitude": (("y", "x"), source_longitude),
        },
    )

    source_second = xr.Dataset(
        {
            "lst": (("y", "x"), second_values)
        },
        coords={
            "latitude": (("y", "x"), source_latitude),
            "longitude": (("y", "x"), source_longitude),
        },
    )

    target = xr.Dataset(
        coords={
            "latitude": np.array([10.1, 11.8]),
            "longitude": np.array([20.1, 21.8]),
        }
    )

    aligner = SpatialAligner(method="nearest")

    # Build the geographic mapping only once.
    mapping = aligner.build_mapping(
        source=source_first,
        target=target,
    )

    # Reuse the same mapping with the first dataset.
    first_result = aligner.apply_mapping(
        source=source_first,
        target=target,
        variable="lst",
        mapping=mapping,
    )

    # Reuse the SAME mapping with the second dataset.
    second_result = aligner.apply_mapping(
        source=source_second,
        target=target,
        variable="lst",
        mapping=mapping,
    )

    expected_first = np.array([
        [10.0, 12.0],
        [30.0, 32.0],
    ])

    expected_second = np.array([
        [100.0, 120.0],
        [300.0, 320.0],
    ])

    np.testing.assert_allclose(
        first_result["lst"].values,
        expected_first,
    )

    np.testing.assert_allclose(
        second_result["lst"].values,
        expected_second,
    )