from __future__ import annotations

import xarray as xr


class TemporalGenerator:
    """
    Generates temporal features directly on an xarray Dataset.
    """

    def generate(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:

        dataset = dataset.copy()

        dataset["year"] = ("time", dataset.time.dt.year.data)

        dataset["month"] = ("time", dataset.time.dt.month.data)

        dataset["day"] = ("time", dataset.time.dt.day.data)

        dataset["day_of_year"] = (
            "time",
            dataset.time.dt.dayofyear.data,
        )

        return dataset