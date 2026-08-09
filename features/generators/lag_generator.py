from __future__ import annotations

import xarray as xr


class LagGenerator:
    """
    Generates lag features for climate variables.
    """

    def generate(
        self,
        dataset: xr.Dataset,
        lag_days: tuple[int, ...],
    ) -> xr.Dataset:

        dataset = dataset.copy()

        for lag in lag_days:

            dataset[f"rainfall_lag_{lag}"] = (
                dataset["rainfall"]
                .shift(time=lag)
            )

        return dataset