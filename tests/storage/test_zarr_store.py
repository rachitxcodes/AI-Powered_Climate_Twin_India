
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from core.climate_dataset import ClimateDataset
from storage.zarr_store import ZarrClimateStore


@pytest.fixture
def sample_climate_dataset():
    ds = xr.Dataset(
        {
            "rainfall": (
                ("time", "latitude", "longitude"),
                np.arange(
                    40 * 4 * 4, dtype=np.float32
                ).reshape(40, 4, 4),
            )
        },
        coords={
            "time": pd.date_range(
                "2024-01-01", periods=40, freq="D"
            ),
            "latitude": [25.0, 25.25, 25.5, 25.75],
            "longitude": [80.0, 80.25, 80.5, 80.75],
        },
    )
    return ClimateDataset(ds)


def test_zarr_round_trip(tmp_path, sample_climate_dataset):
    store = ZarrClimateStore()
    path = tmp_path / "climate.zarr"

    store.save(sample_climate_dataset, path)
    restored = store.load(path)

    xr.testing.assert_equal(
        sample_climate_dataset.dataset,
        restored.dataset.compute(),
    )


def test_zarr_load_is_lazy(tmp_path, sample_climate_dataset):
    store = ZarrClimateStore()
    path = tmp_path / "climate.zarr"

    store.save(sample_climate_dataset, path)
    restored = store.load(path)

    assert restored.dataset["rainfall"].chunks is not None


def test_zarr_rejects_overwrite(tmp_path, sample_climate_dataset):
    store = ZarrClimateStore()
    path = tmp_path / "climate.zarr"

    store.save(sample_climate_dataset, path)

    with pytest.raises(FileExistsError):
        store.save(sample_climate_dataset, path)


def test_zarr_rejects_missing_store(tmp_path):
    store = ZarrClimateStore()

    with pytest.raises(FileNotFoundError):
        store.load(tmp_path / "missing.zarr")


def test_zarr_dask_processing(tmp_path, sample_climate_dataset):
    store = ZarrClimateStore()
    path = tmp_path / "climate.zarr"

    store.save(sample_climate_dataset, path)
    restored = store.load(path)

    rainfall = restored.dataset["rainfall"]

    # Verify Dask uses multiple chunks
    assert rainfall.chunks is not None
    assert len(rainfall.chunks[0]) > 1

    # Calculate spatial mean lazily
    daily_mean = rainfall.mean(
        dim=["latitude", "longitude"]
    )

    assert hasattr(daily_mean.data, "dask")

    # Compare Dask results against NumPy results
    expected = sample_climate_dataset.dataset[
        "rainfall"
    ].mean(dim=["latitude", "longitude"])

    xr.testing.assert_allclose(
        daily_mean.compute(),
        expected,
    )
