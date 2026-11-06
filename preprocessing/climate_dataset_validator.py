from pathlib import Path

import numpy as np
import xarray as xr

from preprocessing.validation_report import ValidationReport


class ClimateDatasetValidator:
    """
    Validates the structure and integrity of a processed
    climate dataset.
    """

    REQUIRED_COORDINATES = {
        "time",
        "latitude",
        "longitude",
    }

    REQUIRED_VARIABLES = {
        "rainfall",
        "lst_mean",
        "valid_observation_count",
    }

    def validate_dataset(
        self,
        dataset: xr.Dataset,
        reference_dataset: xr.Dataset | None = None,
    ) -> ValidationReport:

        errors = []
        warnings = []

        # 1. Check input type
        if not isinstance(dataset, xr.Dataset):
            return ValidationReport(
                valid=False,
                errors=("Input must be an xarray Dataset.",),
                warnings=(),
            )

        # 2. Check that the dataset is not empty
        if not dataset.sizes or any(
            size == 0 for size in dataset.sizes.values()
        ):
            errors.append("Dataset is empty.")

        # 3. Check required coordinates
        missing_coordinates = (
            self.REQUIRED_COORDINATES - set(dataset.coords)
        )

        if missing_coordinates:
            errors.append(
                f"Missing coordinates: {sorted(missing_coordinates)}"
            )

        # 4. Check required variables
        missing_variables = (
            self.REQUIRED_VARIABLES - set(dataset.data_vars)
        )

        if missing_variables:
            errors.append(
                f"Missing variables: {sorted(missing_variables)}"
            )

        # Stop coordinate-dependent checks if coordinates are missing.
        if missing_coordinates:
            return ValidationReport(
                valid=False,
                errors=tuple(errors),
                warnings=tuple(warnings),
            )

        # 5. Check time coordinate
        time_values = dataset["time"].values

        if time_values.size == 0:
            errors.append("Time coordinate is empty.")

        elif np.issubdtype(time_values.dtype, np.datetime64):

            if np.isnat(time_values).any():
                errors.append("Time coordinate contains invalid dates.")

            if not np.all(
                np.diff(time_values) > np.timedelta64(0, "ns")
            ):
                errors.append(
                    "Dates must be chronological and unique."
                )

        else:
            errors.append(
                "Time coordinate must contain datetime values."
            )

        # 6. Check spatial coordinates
        for coordinate in ("latitude", "longitude"):
            values = dataset[coordinate].values

            if not np.isfinite(values).all():
                errors.append(
                    f"{coordinate} contains invalid values."
                )

        # 7. Check dimensions of climate variables
        expected_dimensions = (
            "time",
            "latitude",
            "longitude",
        )

        for variable in self.REQUIRED_VARIABLES:
            if variable in dataset.data_vars:

                if dataset[variable].dims != expected_dimensions:
                    errors.append(
                        f"{variable} has incorrect dimensions: "
                        f"{dataset[variable].dims}"
                    )

        # 8. Check observation counts
        if "valid_observation_count" in dataset:

            counts = dataset["valid_observation_count"].values
            finite_counts = counts[np.isfinite(counts)]

            if (finite_counts < 0).any():
                errors.append(
                    "Observation counts cannot be negative."
                )

        # 9. Compare spatial grid against IMD reference
        if reference_dataset is not None:

            for coordinate in ("latitude", "longitude"):

                if coordinate not in reference_dataset.coords:
                    errors.append(
                        f"Reference dataset is missing {coordinate}."
                    )
                    continue

                if not np.array_equal(
                    dataset[coordinate].values,
                    reference_dataset[coordinate].values,
                ):
                    errors.append(
                        f"{coordinate} does not match the IMD grid."
                    )

        return ValidationReport(
            valid=len(errors) == 0,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    def validate_file(
        self,
        file_path: Path,
        reference_dataset: xr.Dataset | None = None,
    ) -> ValidationReport:
        """
        Open a saved NetCDF file and validate its contents.
        """

        file_path = Path(file_path)

        if not file_path.is_file():
            return ValidationReport(
                valid=False,
                errors=(f"Dataset file not found: {file_path}",),
                warnings=(),
            )

        try:
            with xr.open_dataset(file_path) as dataset:
                loaded_dataset = dataset.load()

        except Exception as exc:
            return ValidationReport(
                valid=False,
                errors=(
                    f"Could not open saved NetCDF file: {exc}",
                ),
                warnings=(),
            )

        return self.validate_dataset(
            dataset=loaded_dataset,
            reference_dataset=reference_dataset,
        )