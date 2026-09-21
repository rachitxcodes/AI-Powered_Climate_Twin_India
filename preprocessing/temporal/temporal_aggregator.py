import xarray as xr


class TemporalAggregator:
    """Aggregate high-frequency climate observations to daily resolution."""

    def __init__(self, frequency="1D"):
        if frequency != "1D":
            raise ValueError("Only daily aggregation ('1D') is supported.")

        self.frequency = frequency

    def aggregate(self, dataset, variable):
        self._validate_dataset(dataset, variable)

        data = dataset[variable]

        daily_mean = data.resample(
            time=self.frequency
        ).mean(
            dim="time",
            skipna=True,
        )

        valid_observation_count = data.resample(
            time=self.frequency
        ).count(dim="time")

        result = xr.Dataset(
            {
                f"{variable}_mean": daily_mean,
                "valid_observation_count": valid_observation_count,
            }
        )

        result[f"{variable}_mean"].attrs.update(data.attrs)

        result[f"{variable}_mean"].attrs["aggregation"] = "daily_mean"

        result["valid_observation_count"].attrs.update(
            {
                "long_name": (
                    "Number of valid observations used in daily aggregation"
                ),
                "description": (
                    "Count of non-missing observations contributing "
                    "to the daily mean."
                ),
            }
        )

        return result

    @staticmethod
    def _validate_dataset(dataset, variable):
        if not isinstance(dataset, xr.Dataset):
            raise TypeError("Dataset must be an xarray Dataset.")

        if "time" not in dataset:
            raise ValueError("Dataset must contain a time dimension.")

        if variable not in dataset:
            raise ValueError(
                f"Variable '{variable}' not found in dataset."
            )