import numpy as np
import xarray as xr


class DatasetFusion:
    """Fuse already aligned climate datasets."""

    def fuse(self, base, additions):
        self._validate_dataset(base, "Base")

        if not isinstance(additions, (list, tuple)):
            additions = [additions]

        for index, dataset in enumerate(additions):
            self._validate_dataset(
                dataset,
                f"Addition {index}",
            )

            self._validate_compatible_coordinates(
                base,
                dataset,
            )

            self._validate_variable_collisions(
                base,
                dataset,
            )

        datasets = [base, *additions]

        return xr.merge(
            datasets,
            join="exact",
            compat="equals",
        )

    @staticmethod
    def _validate_dataset(dataset, name):
        if not isinstance(dataset, xr.Dataset):
            raise TypeError(
                f"{name} dataset must be an xarray Dataset."
            )

        required_coordinates = {
            "time",
            "latitude",
            "longitude",
        }

        missing_coordinates = required_coordinates - set(
            dataset.coords
        )

        if missing_coordinates:
            raise ValueError(
                f"{name} dataset is missing required coordinates: "
                f"{sorted(missing_coordinates)}"
            )

        time = dataset["time"]

        if not np.issubdtype(time.dtype, np.datetime64):
            raise TypeError(
                f"{name} dataset time coordinate must contain "
                "datetime values."
            )

    @staticmethod
    def _validate_compatible_coordinates(
        base,
        addition,
    ):
        np.testing.assert_array_equal(
            base["time"].values,
            addition["time"].values,
            err_msg="Dataset time coordinates do not match.",
        )

        np.testing.assert_allclose(
            base["latitude"].values,
            addition["latitude"].values,
            err_msg="Dataset latitude coordinates do not match.",
        )

        np.testing.assert_allclose(
            base["longitude"].values,
            addition["longitude"].values,
            err_msg="Dataset longitude coordinates do not match.",
        )

    @staticmethod
    def _validate_variable_collisions(
        base,
        addition,
    ):
        collisions = set(base.data_vars) & set(addition.data_vars)

        if collisions:
            raise ValueError(
                "Dataset variable collision detected: "
                f"{sorted(collisions)}"
            )