import numpy as np
import pandas as pd
import pytest
import xarray as xr

from preprocessing.climate_pipeline import ClimatePipeline


def create_imd_dataset():
    """
    Create a small synthetic IMD rainfall dataset.

    Grid:
        latitude  = [26, 27]
        longitude = [78, 79]

    Dates:
        18 June 2024
        19 June 2024
    """

    time = pd.date_range(
        "2024-06-18",
        periods=2,
        freq="1D",
    )

    latitude = np.array(
        [26.0, 27.0]
    )

    longitude = np.array(
        [78.0, 79.0]
    )

    rainfall = np.array(
        [
            [
                [10.0, 20.0],
                [30.0, 40.0],
            ],
            [
                [15.0, 25.0],
                [35.0, 45.0],
            ],
        ],
        dtype=np.float32,
    )

    return xr.Dataset(
        {
            "rainfall": (
                ("time", "latitude", "longitude"),
                rainfall,
            )
        },
        coords={
            "time": time,
            "latitude": latitude,
            "longitude": longitude,
        },
    )


def create_insat_dataset():
    """
    Create a small synthetic INSAT dataset.

    INSAT observations occur at:

        18 June 06:00
        18 June 06:30
        19 June 06:00

    This lets us verify that the pipeline converts
    high-frequency observations into daily values.
    """

    time = pd.to_datetime(
        [
            "2024-06-18 06:00",
            "2024-06-18 06:30",
            "2024-06-19 06:00",
        ]
    )

    latitude = np.array(
        [
            [26.0, 26.0],
            [27.0, 27.0],
        ]
    )

    longitude = np.array(
        [
            [78.0, 79.0],
            [78.0, 79.0],
        ]
    )

    lst = np.array(
        [
            [
                [300.0, 301.0],
                [302.0, 303.0],
            ],
            [
                [304.0, 305.0],
                [306.0, 307.0],
            ],
            [
                [308.0, 309.0],
                [310.0, 311.0],
            ],
        ],
        dtype=np.float32,
    )

    return xr.Dataset(
        {
            "lst": (
                ("time", "y", "x"),
                lst,
            )
        },
        coords={
            "time": time,
            "y": np.arange(2),
            "x": np.arange(2),
            "latitude": (
                ("y", "x"),
                latitude,
            ),
            "longitude": (
                ("y", "x"),
                longitude,
            ),
        },
    )


class FakeINSATReader:
    """
    Fake reader used only for testing the pipeline.

    It avoids reading an actual HDF file because the purpose
    of these tests is to verify orchestration.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def read(self, path):
        return self.dataset


def test_pipeline_processes_single_insat_file():
    imd = create_imd_dataset()
    insat = create_insat_dataset()

    pipeline = ClimatePipeline(
        insat_reader=FakeINSATReader(insat),
    )

    result = pipeline.process(
        imd_dataset=imd,
        insat_files=["fake_file.h5"],
    )

    assert "rainfall" in result
    assert "lst_mean" in result
    assert "valid_observation_count" in result

    assert result.sizes["time"] == 2


def test_pipeline_preserves_matching_dates():
    imd = create_imd_dataset()
    insat = create_insat_dataset()

    pipeline = ClimatePipeline(
        insat_reader=FakeINSATReader(insat),
    )

    result = pipeline.process(
        imd_dataset=imd,
        insat_files=["fake_file.h5"],
    )

    expected_dates = pd.to_datetime(
        [
            "2024-06-18",
            "2024-06-19",
        ]
    )

    np.testing.assert_array_equal(
        result.time.values,
        expected_dates.values,
    )


def test_pipeline_produces_daily_lst():
    imd = create_imd_dataset()
    insat = create_insat_dataset()

    pipeline = ClimatePipeline(
        insat_reader=FakeINSATReader(insat),
    )

    result = pipeline.process(
        imd_dataset=imd,
        insat_files=["fake_file.h5"],
    )

    # June 18 has two observations.
    #
    # First observation:
    # 300, 301, 302, 303
    #
    # Second observation:
    # 304, 305, 306, 307
    #
    # Daily mean:
    # 302, 303, 304, 305

    expected = np.array(
        [
            [
                [302.0, 303.0],
                [304.0, 305.0],
            ],
            [
                [308.0, 309.0],
                [310.0, 311.0],
            ],
        ]
    )

    np.testing.assert_allclose(
        result["lst_mean"].values,
        expected,
    )


def test_pipeline_preserves_spatial_grid():
    imd = create_imd_dataset()
    insat = create_insat_dataset()

    pipeline = ClimatePipeline(
        insat_reader=FakeINSATReader(insat),
    )

    result = pipeline.process(
        imd_dataset=imd,
        insat_files=["fake_file.h5"],
    )

    np.testing.assert_array_equal(
        result.latitude.values,
        imd.latitude.values,
    )

    np.testing.assert_array_equal(
        result.longitude.values,
        imd.longitude.values,
    )


def test_pipeline_requires_insat_files():
    imd = create_imd_dataset()

    pipeline = ClimatePipeline()

    with pytest.raises(ValueError):
        pipeline.process(
            imd_dataset=imd,
            insat_files=[],
        )


def test_pipeline_rejects_invalid_imd_dataset():
    invalid_imd = xr.Dataset()

    pipeline = ClimatePipeline()

    with pytest.raises(ValueError):
        pipeline.process(
            imd_dataset=invalid_imd,
            insat_files=["fake_file.h5"],
        )


def test_real_insat_imd_pipeline():
    """
    End-to-end test using real IMD 2024 rainfall data
    and the real INSAT-3D LST HDF fixture.
    """

    from pathlib import Path

    from ingestion.data_loader import DataLoader

    # ---------------------------------------------------------
    # 1. Load real IMD 2024 rainfall
    # ---------------------------------------------------------
    imd_dataset = DataLoader().load(
        dataset="rainfall",
        year=2024,
    ).dataset

    # ---------------------------------------------------------
    # 2. Locate the real INSAT fixture
    # ---------------------------------------------------------
    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "3DIMG_18JUN2024_0600_L2B_LST_V01R00.h5"
    )

    assert fixture_path.exists()

    # ---------------------------------------------------------
    # 3. Run the complete pipeline
    # ---------------------------------------------------------
    pipeline = ClimatePipeline()

    result = pipeline.process(
        imd_dataset=imd_dataset,
        insat_files=[str(fixture_path)],
    )

    # ---------------------------------------------------------
    # 4. Verify variables
    # ---------------------------------------------------------
    assert "rainfall" in result
    assert "lst_mean" in result
    assert "valid_observation_count" in result

    # ---------------------------------------------------------
    # 5. Only 18 June should remain because our INSAT
    #    fixture contains observations for that day only.
    # ---------------------------------------------------------
    assert result.sizes["time"] == 1

    assert result.time.values[0] == np.datetime64(
        "2024-06-18"
    )

    # ---------------------------------------------------------
    # 6. Verify that INSAT has been mapped to the IMD grid
    # ---------------------------------------------------------
    assert result["lst_mean"].dims == (
        "time",
        "latitude",
        "longitude",
    )

    assert result["rainfall"].dims == (
        "time",
        "latitude",
        "longitude",
    )

    # ---------------------------------------------------------
    # 7. Verify spatial dimensions match IMD
    # ---------------------------------------------------------
    assert (
        result.sizes["latitude"]
        == imd_dataset.sizes["latitude"]
    )

    assert (
        result.sizes["longitude"]
        == imd_dataset.sizes["longitude"]
    )

    # ---------------------------------------------------------
    # 8. Verify that some real LST values exist
    # ---------------------------------------------------------
    assert np.isfinite(
        result["lst_mean"].values
    ).any()

    # ---------------------------------------------------------
    # 9. Verify LST is physically plausible
    # ---------------------------------------------------------
    valid_lst = result["lst_mean"].values[
        np.isfinite(result["lst_mean"].values)
    ]

    assert valid_lst.min() > 200
    assert valid_lst.max() < 400