
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from core.climate_dataset import ClimateDataset
from ingestion.multi_year_loader import MultiYearDataLoader


def make_year(year):
    dates = pd.date_range(
        f"{year}-01-01",
        f"{year}-12-31",
        freq="D",
    )

    return ClimateDataset(
        xr.Dataset(
            {"rainfall": (
                ("time", "latitude", "longitude"),
                np.ones((len(dates), 2, 2)),
            )},
            coords={
                "time": dates,
                "latitude": [25.0, 25.25],
                "longitude": [80.0, 80.25],
            },
        )
    )


def test_combines_years_and_handles_leap_year():
    loader = MultiYearDataLoader()
    loader.loader.load = lambda dataset, year: make_year(year)

    result = loader.load("rainfall", [2024, 2023])
    ds = result.dataset

    assert ds.sizes["time"] == 731
    assert pd.Timestamp(ds.time.values[0]) == pd.Timestamp("2023-01-01")
    assert pd.Timestamp(ds.time.values[-1]) == pd.Timestamp("2024-12-31")


def test_rejects_duplicate_years():
    loader = MultiYearDataLoader()

    with pytest.raises(ValueError, match="Duplicate years"):
        loader.load("rainfall", [2023, 2023])


def test_rejects_empty_years():
    loader = MultiYearDataLoader()

    with pytest.raises(ValueError, match="At least one year"):
        loader.load("rainfall", [])


def test_rejects_spatial_grid_mismatch():
    loader = MultiYearDataLoader()

    def load_mock(dataset, year):
        ds = make_year(year)

        if year == 2024:
            ds.dataset.coords["latitude"] = [26.0, 26.25]

        return ds

    loader.loader.load = load_mock

    with pytest.raises(ValueError, match="Spatial grid mismatch"):
        loader.load("rainfall", [2023, 2024])


def test_rejects_missing_year():
    loader = MultiYearDataLoader()
    loader.loader.load = lambda dataset, year: make_year(year)

    with pytest.raises(ValueError, match="missing or duplicate dates"):
        loader.load("rainfall", [2023, 2025])


def test_rejects_incomplete_year():
    loader = MultiYearDataLoader()

    incomplete = make_year(2025)

    incomplete._dataset = incomplete.dataset.isel(
        time=slice(0, 300)
    )

    loader.loader.load = lambda dataset, year: incomplete

    with pytest.raises(
        ValueError,
        match="Incomplete or invalid daily data",
    ):
        loader.load("rainfall", [2025])
