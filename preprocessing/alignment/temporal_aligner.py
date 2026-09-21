import pandas as pd
import xarray as xr
import numpy as np


class TemporalAligner:

    def __init__(self, method="nearest", tolerance="30min"):

        if method != "nearest":
            raise ValueError(
                "Only 'nearest' temporal alignment is supported."
            )

        self.method = method
        self.tolerance = pd.Timedelta(tolerance)

    def build_mapping(self, source, target):
        """
        Build a mapping from target timestamps
        to the nearest source timestamps.
        """

        self._validate_source(source)
        self._validate_target(target)

        source_times = pd.DatetimeIndex(
            source["time"].values
        )

        target_times = pd.DatetimeIndex(
            target["time"].values
        )

        source_indices = source_times.get_indexer(
            target_times,
            method=self.method,
            tolerance=self.tolerance,
        )

        matched = source_indices != -1

        time_differences = []

        for target_index, source_index in enumerate(source_indices):

            if source_index == -1:
                time_differences.append(pd.NaT)

            else:
                difference = (
                    source_times[source_index]
                    - target_times[target_index]
                )

                time_differences.append(difference)

        return {
            "source_indices": source_indices,
            "matched": matched,
            "time_differences": pd.to_timedelta(
                time_differences
            ),
        }

    def apply_mapping(self, source, target, variable, mapping):
        """
        Apply a previously calculated temporal mapping
        to the source variable.
        """

        self._validate_source(source)
        self._validate_target(target)

        if variable not in source:
            raise ValueError(
                f"Variable '{variable}' not found in source dataset."
            )

        source_indices = mapping["source_indices"]

        source_data = source[variable]

        aligned_values = []

        for source_index in source_indices:

            if source_index == -1:
                aligned_values.append(
                    xr.full_like(
                        source_data.isel(time=0),
                        np.nan,
                    )
                )

            else:
                aligned_values.append(
                    source_data.isel(time=source_index)
                )

        aligned_data = xr.concat(
            aligned_values,
            dim=target["time"],
        )

        result = aligned_data.to_dataset(
            name=variable
        )

        result[variable].attrs.update(
            source[variable].attrs
        )

        return result

    def align(self, source, target, variable):
        """
        Convenience method.

        Builds the temporal mapping and immediately
        applies it.
        """

        mapping = self.build_mapping(
            source=source,
            target=target,
        )

        return self.apply_mapping(
            source=source,
            target=target,
            variable=variable,
            mapping=mapping,
        )

    @staticmethod
    def _validate_source(source):

        if not isinstance(source, xr.Dataset):
            raise TypeError(
                "Source must be an xarray Dataset."
            )

        if "time" not in source:
            raise ValueError(
                "Source dataset must contain time."
            )

        TemporalAligner._validate_time_index(source, "Source")

    @staticmethod
    def _validate_target(target):

        if not isinstance(target, xr.Dataset):
            raise TypeError(
                "Target must be an xarray Dataset."
            )

        if "time" not in target:
            raise ValueError(
                "Target dataset must contain time."
            )

        TemporalAligner._validate_time_index(target, "Target")
        
    @staticmethod
    def _validate_time_index(dataset, name):
        time_index = pd.DatetimeIndex(dataset["time"].values)

        if time_index.has_duplicates:
            raise ValueError(f"{name} dataset contains duplicate timestamps.")

        if not time_index.is_monotonic_increasing:
            raise ValueError(
                f"{name} dataset time coordinate must be sorted."
            )

    

