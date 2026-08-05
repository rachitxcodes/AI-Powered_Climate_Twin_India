from __future__ import annotations

import xarray as xr


class ClimateDataset:
    """
    Canonical in-memory representation of climate data.

    This class wraps an xarray.Dataset and serves as the
    central data object used throughout the system.
    """

    def __init__(self, dataset: xr.Dataset):
        self._dataset = dataset

    @property
    def dataset(self) -> xr.Dataset:
        return self._dataset

    @property
    def variables(self) -> list[str]:
        """Return all climate variables."""
        return list(self._dataset.data_vars)

    @property
    def coordinates(self) -> list[str]:
        """Return coordinate names."""
        return list(self._dataset.coords)

    @property
    def attrs(self) -> dict:
        """Return dataset metadata."""
        return self._dataset.attrs

    def get_variable(self, name: str):
        """Return a climate variable."""
        return self._dataset[name]

    def add_variable(self, name: str, value):
        """Add a new variable to the dataset."""
        self._dataset[name] = value

    def copy(self):
        """Return a deep copy of the ClimateDataset."""
        return ClimateDataset(
            self._dataset.copy(deep=True)
        )