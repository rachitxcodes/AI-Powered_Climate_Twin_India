
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from core.climate_dataset import ClimateDataset
from features.config import FeatureEngineeringConfig
from features.feature_pipeline import FeatureEngineeringPipeline
from storage.zarr_store import ZarrClimateStore


def test_multi_year_feature_pipeline(tmp_path):
    dates = pd.date_range(
        "2023-12-20",
        periods=20,
        freq="D",
    )

    values = np.arange(1, 21, dtype=np.float32)

    ds = xr.Dataset(
        {
            "rainfall": (
                ("time", "latitude", "longitude"),
                values.reshape(20, 1, 1),
            )
        },
        coords={
            "time": dates,
            "latitude": [25.0],
            "longitude": [80.0],
        },
    )

    store = ZarrClimateStore()
    path = tmp_path / "multi_year.zarr"
    store.save(ClimateDataset(ds), path)

    climate = store.load(path)

    result = FeatureEngineeringPipeline(
        FeatureEngineeringConfig()
    ).run(
        SimpleNamespace(climate_dataset=climate)
    )

    engineered = result.climate_dataset.dataset

    assert engineered.sizes["time"] == 20

    assert "rainfall_lag_1" in engineered
    assert "rainfall_roll_mean_7" in engineered
    assert "month_sin" in engineered
    assert "season_id" in engineered

    assert engineered["rainfall_lag_1"].chunks is not None
    assert engineered["rainfall_roll_mean_7"].chunks is not None

    # January 1 is the 13th observation.
    lag = engineered["rainfall_lag_1"].sel(
        time="2024-01-01"
    ).compute().item()

    rolling = engineered["rainfall_roll_mean_7"].sel(
        time="2024-01-01"
    ).compute().item()

    assert lag == 12.0

    np.testing.assert_allclose(
        rolling,
        np.mean(np.arange(7, 14)),
    )
