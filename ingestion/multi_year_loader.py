
import numpy as np
import pandas as pd
import xarray as xr

from core.climate_dataset import ClimateDataset
from ingestion.data_loader import DataLoader


class MultiYearDataLoader:
    """Combine multiple annual climate datasets."""

    def __init__(self):
        self.loader = DataLoader()

    def load(self, dataset: str, years: list[int]) -> ClimateDataset:
        if not years:
            raise ValueError("At least one year is required.")

        if len(years) != len(set(years)):
            raise ValueError("Duplicate years are not allowed.")

        datasets = []

        for year in sorted(years):
            climate_data = self.loader.load(dataset, year)
            current = climate_data.dataset

            # Validate complete daily coverage for this year
            expected_dates = pd.date_range(
                start=f"{year}-01-01",
                end=f"{year}-12-31",
                freq="D",
            )

            actual_dates = pd.DatetimeIndex(
                current.time.values
            )

            if not actual_dates.equals(expected_dates):
                raise ValueError(
                    f"Incomplete or invalid daily data for year {year}. "
                    f"Expected {len(expected_dates)} days, "
                    f"found {len(actual_dates)}."
                )

            if datasets:
                reference = datasets[0]

                for coord in ("latitude", "longitude"):
                    if not np.array_equal(
                        reference[coord].values,
                        current[coord].values,
                    ):
                        raise ValueError(
                            f"Spatial grid mismatch in year {year}"
                        )

            datasets.append(current)

        combined = xr.concat(
            datasets,
            dim="time",
            join="exact",
        )

        actual_dates = pd.DatetimeIndex(combined.time.values)

        expected_dates = pd.date_range(
            start=actual_dates[0],
            periods=len(actual_dates),
            freq="D",
        )

        if not actual_dates.equals(expected_dates):
            raise ValueError(
                "Combined dataset contains missing or duplicate dates."
            )

        return ClimateDataset(combined)
