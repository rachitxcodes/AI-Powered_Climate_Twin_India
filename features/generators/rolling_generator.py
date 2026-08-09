from __future__ import annotations

import xarray as xr


class RollingGenerator:
    """
    Generates rolling statistical features for climate variables.
    """

    def generate(
        self,
        dataset: xr.Dataset,
        windows: tuple[int, ...],
    ) -> xr.Dataset:
        """
        Generate rolling mean rainfall features.

        Parameters
        ----------
        dataset : xr.Dataset
            Climate dataset containing rainfall data.

        windows : tuple[int, ...]
            Rolling window sizes in days.

        Returns
        -------
        xr.Dataset
            Dataset containing the generated rolling features.
        """

        dataset = dataset.copy()

        for window in windows:

            dataset[f"rainfall_roll_mean_{window}"] = (
                dataset["rainfall"]
                .rolling(
                    time=window,
                    min_periods=1,
                )
                .mean()
            )

        return dataset