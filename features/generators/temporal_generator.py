from __future__ import annotations

import numpy as np
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

        dataset["year"] = (
            "time",
            dataset.time.dt.year.data,
        )

        dataset["month"] = (
            "time",
            dataset.time.dt.month.data,
        )

        dataset["day"] = (
            "time",
            dataset.time.dt.day.data,
        )

        dataset["day_of_year"] = (
            "time",
            dataset.time.dt.dayofyear.data,
        )

        return dataset

    def generate_cyclical(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:

        dataset = dataset.copy()

        # Month cycle
        month_angle = (
            2 * np.pi * (dataset.time.dt.month - 1) / 12
        )

        dataset["month_sin"] = (
            "time",
            np.sin(month_angle.data),
        )

        dataset["month_cos"] = (
            "time",
            np.cos(month_angle.data),
        )

        # Day-of-year cycle
        day_of_year = dataset.time.dt.dayofyear

        days_in_year = xr.where(
            dataset.time.dt.is_leap_year,
            366,
            365,
        )

        day_angle = (
            2
            * np.pi
            * (day_of_year - 1)
            / days_in_year
        )

        dataset["day_of_year_sin"] = (
            "time",
            np.sin(day_angle.data),
        )

        dataset["day_of_year_cos"] = (
            "time",
            np.cos(day_angle.data),
        )

        return dataset