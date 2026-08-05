from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from core.climate_dataset import ClimateDataset


class ClimateDatasetFactory:
    """
    Builds the canonical ClimateDataset from
    raw NumPy arrays and metadata.
    """

    def create(
        self,
        data: np.ndarray,
        metadata: dict,
        year: int,
    ) -> ClimateDataset:

        xdef = metadata["xdef"]
        ydef = metadata["ydef"]

        latitudes = np.arange(
            ydef["start"],
            ydef["start"] + ydef["count"] * ydef["increment"],
            ydef["increment"],
        )

        longitudes = np.arange(
            xdef["start"],
            xdef["start"] + xdef["count"] * xdef["increment"],
            xdef["increment"],
        )

        dates = pd.date_range(
            start=f"{year}-01-01",
            periods=data.shape[0],
            freq="D",
        )

        dataset = xr.Dataset(
            data_vars={
                "rainfall": (
                    ("time", "latitude", "longitude"),
                    data,
                )
            },
            coords={
                "time": dates,
                "latitude": latitudes,
                "longitude": longitudes,
            },
            attrs=metadata,
        )

        return ClimateDataset(dataset)