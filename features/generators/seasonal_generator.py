from __future__ import annotations

import numpy as np
import xarray as xr


class SeasonalGenerator:
    """
    Generates IMD seasonal features from the month variable.
    """

    def generate(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:

        dataset = dataset.copy()

        months = dataset["month"].values

        season_id = np.zeros_like(months)

        # Winter
        season_id[np.isin(months, [1, 2])] = 0

        # Pre-Monsoon
        season_id[np.isin(months, [3, 4, 5])] = 1

        # Southwest Monsoon
        season_id[np.isin(months, [6, 7, 8, 9])] = 2

        # Post-Monsoon
        season_id[np.isin(months, [10, 11, 12])] = 3

        season_names = np.array(
            [
                "Winter",
                "Pre-Monsoon",
                "Southwest Monsoon",
                "Post-Monsoon",
            ],
            dtype=object,
        )

        season = season_names[season_id]

        dataset["season_id"] = ("time", season_id)

        dataset["season"] = ("time", season)

        return dataset