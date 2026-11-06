import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from ingestion.clients.mosdac_client import MOSDACClient
from ingestion.data_loader import DataLoader
from preprocessing.climate_pipeline import ClimatePipeline
from preprocessing.climate_dataset_validator import (
    ClimateDatasetValidator,
)


class ClimateDataWorkflow:
    """
    Coordinate MOSDAC acquisition, IMD loading,
    climate dataset processing, and validation.
    """

    def __init__(self):
        self.mosdac_client = MOSDACClient()
        self.data_loader = DataLoader()
        self.climate_pipeline = ClimatePipeline()
        self.climate_dataset_validator = ClimateDatasetValidator()

        self.project_root = Path(__file__).resolve().parents[1]
        self._imd_cache = {}

    # ---------------------------------------------------------
    # 1. LOAD AND CACHE IMD DATA
    # ---------------------------------------------------------

    def _get_imd_dataset(self, year: int):
        """
        Load an IMD dataset once per year and reuse it.
        """

        if year not in self._imd_cache:

            print(f"\nLoading IMD rainfall data for {year}...")

            self._imd_cache[year] = self.data_loader.load(
                dataset="rainfall",
                year=year,
            ).dataset

            print(f"IMD {year} dataset loaded and cached.")

        else:
            print(f"\nReusing cached IMD dataset for {year}.")

        return self._imd_cache[year]

    # ---------------------------------------------------------
    # 2. VALIDATE SAVED CLIMATE DATASET
    # ---------------------------------------------------------

    def _validate_saved_dataset(
        self,
        output_path: Path,
        reference_dataset: xr.Dataset,
    ):
        """
        Validate a saved NetCDF climate dataset.
        """

        print("\nValidating saved climate dataset...")

        report = self.climate_dataset_validator.validate_file(
            file_path=output_path,
            reference_dataset=reference_dataset,
        )

        print("Validation status:", report.valid)

        if report.errors:
            print("\nValidation errors:")

            for error in report.errors:
                print("-", error)

        if report.warnings:
            print("\nValidation warnings:")

            for warning in report.warnings:
                print("-", warning)

        if not report.valid:
            raise RuntimeError(
                f"Climate dataset validation failed: {output_path}"
            )

        print("CLIMATE DATASET VALIDATION PASSED")

        return report

    # ---------------------------------------------------------
    # 3. PROCESS ONE DAY
    # ---------------------------------------------------------

    def run_day(
        self,
        target_date: date,
        username: str,
        password: str,
    ):
        """
        Acquire and process climate data for one day.
        """

        if not username or not password:
            raise ValueError("MOSDAC credentials cannot be empty.")

        # -----------------------------------------------------
        # 1. Define storage locations
        # -----------------------------------------------------

        insat_dir = (
            self.project_root
            / "data"
            / "raw"
            / "insat"
            / "api_validation"
        )

        processed_dir = (
            self.project_root
            / "data"
            / "processed"
            / "climate"
        )

        insat_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        # -----------------------------------------------------
        # 2. Discover available INSAT records
        # -----------------------------------------------------

        print(f"\nSearching MOSDAC for {target_date}...")

        records = self.mosdac_client.search_records(
            start_date=target_date,
            end_date=target_date,
        )

        if not records:
            raise RuntimeError(
                f"No INSAT records found for {target_date}."
            )

        print(f"MOSDAC records found: {len(records)}")

        # -----------------------------------------------------
        # 3. Download missing INSAT files
        # -----------------------------------------------------

        print("\nDownloading available INSAT records...")

        download_result = self.mosdac_client.download_records(
            records=records,
            username=username,
            password=password,
            output_dir=insat_dir,
        )

        if download_result["failed"]:
            raise RuntimeError(
                "Some INSAT downloads failed: "
                f"{download_result['failed']}"
            )

        # Build paths from API records, not every file
        # present in the directory.

        insat_files = sorted(
            {
                insat_dir / Path(record["identifier"]).name
                for record in records
                if record.get("identifier")
            }
        )

        missing_files = [
            path
            for path in insat_files
            if not path.is_file() or path.stat().st_size == 0
        ]

        if missing_files:
            raise FileNotFoundError(
                f"Some API records are missing locally: {missing_files}"
            )

        print(f"Available INSAT files ready: {len(insat_files)}")

        # -----------------------------------------------------
        # 4. Load IMD rainfall data
        # -----------------------------------------------------

        imd_dataset = self._get_imd_dataset(target_date.year)

        # -----------------------------------------------------
        # 5. Run ClimatePipeline
        # -----------------------------------------------------

        print("\nRunning ClimatePipeline...")

        result = self.climate_pipeline.process(
            imd_dataset=imd_dataset,
            insat_files=[str(path) for path in insat_files],
        )

        # -----------------------------------------------------
        # 6. Validate requested date
        # -----------------------------------------------------

        expected_date = np.datetime64(target_date.isoformat())

        result_dates = result["time"].dt.floor("D").values

        if not np.any(result_dates == expected_date):
            raise RuntimeError(
                f"ClimatePipeline did not produce data for {target_date}."
            )

        # -----------------------------------------------------
        # 7. Save daily climate dataset
        # -----------------------------------------------------

        output_path = (
            processed_dir
            / f"climate_{target_date.isoformat()}.nc"
        )

        netcdf_dataset = self._make_netcdf_compatible(result)

        netcdf_dataset.to_netcdf(output_path)

        print("\nDaily dataset saved:", output_path)

        # -----------------------------------------------------
        # 8. Automatically validate saved daily dataset
        # -----------------------------------------------------

        self._validate_saved_dataset(
            output_path=output_path,
            reference_dataset=imd_dataset,
        )

        # -----------------------------------------------------
        # 9. Display completion summary
        # -----------------------------------------------------

        print("\n========== WORKFLOW COMPLETED ==========")

        print("Date:", target_date)
        print("INSAT files:", len(insat_files))
        print("Output dimensions:", dict(result.sizes))
        print("Output variables:", list(result.data_vars))
        print("Saved dataset:", output_path)

        return result, output_path

    # ---------------------------------------------------------
    # 4. PROCESS A DATE RANGE
    # ---------------------------------------------------------

    def run_range(
        self,
        start_date: date,
        end_date: date,
        username: str,
        password: str,
    ):
        """
        Process each day in an inclusive date range.
        """

        if end_date < start_date:
            raise ValueError(
                "end_date cannot be earlier than start_date"
            )

        results = {}
        skipped_dates = []

        current_date = start_date

        while current_date <= end_date:

            print("\n" + "=" * 55)
            print(f"PROCESSING DATE: {current_date}")
            print("=" * 55)

            try:
                result, output_path = self.run_day(
                    target_date=current_date,
                    username=username,
                    password=password,
                )

                results[current_date] = {
                    "dataset": result,
                    "output_path": output_path,
                }

            except RuntimeError as error:

                if str(error).startswith("No INSAT records found"):
                    print(
                        f"Skipping {current_date}: no INSAT records."
                    )

                    skipped_dates.append(current_date)

                else:
                    raise

            current_date += timedelta(days=1)

        # -----------------------------------------------------
        # Date range summary
        # -----------------------------------------------------

        print("\n========== DATE RANGE SUMMARY ==========")

        print("Requested:", start_date, "to", end_date)
        print("Successfully processed:", len(results))
        print("Skipped dates:", skipped_dates)

        # -----------------------------------------------------
        # Combine successful daily datasets
        # -----------------------------------------------------

        if results:

            combined_path = self.combine_daily_outputs(
                results=results,
                start_date=start_date,
                end_date=end_date,
            )

            print("\nCombined climate dataset:", combined_path)

        return results

    # ---------------------------------------------------------
    # 5. COMBINE DAILY DATASETS
    # ---------------------------------------------------------

    def combine_daily_outputs(
        self,
        results: dict,
        start_date: date,
        end_date: date,
    ) -> Path:
        """
        Combine successfully processed daily NetCDF files
        into one chronological climate dataset.
        """

        if not results:
            raise ValueError(
                "No successfully processed daily datasets to combine."
            )

        datasets = []

        try:

            # -------------------------------------------------
            # 1. Open and load daily outputs
            # -------------------------------------------------

            for target_date in sorted(results):

                output_path = results[target_date]["output_path"]

                if not output_path.is_file():
                    raise FileNotFoundError(output_path)

                with xr.open_dataset(output_path) as dataset:
                    datasets.append(dataset.load())

            # -------------------------------------------------
            # 2. Verify spatial grids
            # -------------------------------------------------

            reference = datasets[0]

            for dataset in datasets[1:]:

                for coordinate in ("latitude", "longitude"):

                    if not reference[coordinate].equals(
                        dataset[coordinate]
                    ):
                        raise ValueError(
                            f"Inconsistent {coordinate} grid "
                            "between daily datasets."
                        )

            # -------------------------------------------------
            # 3. Combine along time
            # -------------------------------------------------

            combined = xr.concat(
                datasets,
                dim="time",
                data_vars="all",
                coords="minimal",
                compat="equals",
                join="exact",
                combine_attrs="override",
            )

            # -------------------------------------------------
            # 4. Sort chronologically
            # -------------------------------------------------

            combined = combined.sortby("time")

            # -------------------------------------------------
            # 5. Reject duplicate timestamps
            # -------------------------------------------------

            if combined.get_index("time").has_duplicates:
                raise ValueError(
                    "Duplicate timestamps found in combined dataset."
                )

            # -------------------------------------------------
            # 6. Save combined dataset
            # -------------------------------------------------

            output_path = (
                self.project_root
                / "data"
                / "processed"
                / "climate"
                / (
                    f"climate_{start_date.isoformat()}"
                    f"_to_{end_date.isoformat()}.nc"
                )
            )

            compatible_dataset = self._make_netcdf_compatible(
                combined
            )

            compatible_dataset.to_netcdf(output_path)

            print("\n========== COMBINED DATASET SAVED ==========")

            print("Output:", output_path)
            print("Dimensions:", dict(combined.sizes))
            print("Variables:", list(combined.data_vars))
            print("Dates:", combined.time.values)

            # -------------------------------------------------
            # 7. Automatically validate combined dataset
            # -------------------------------------------------

            self._validate_saved_dataset(
                output_path=output_path,
                reference_dataset=reference,
            )

            return output_path

        finally:

            for dataset in datasets:
                dataset.close()

    # ---------------------------------------------------------
    # 6. NETCDF ATTRIBUTE COMPATIBILITY
    # ---------------------------------------------------------

    @staticmethod
    def _make_netcdf_compatible(dataset):
        """
        Convert dictionary attributes into JSON strings
        so xarray can serialize them to NetCDF.
        """

        dataset = dataset.copy(deep=False)

        # Dataset-level attributes

        dataset.attrs = {
            key: json.dumps(value) if isinstance(value, dict) else value
            for key, value in dataset.attrs.items()
        }

        # Variable-level attributes

        for variable in dataset.variables:

            dataset[variable].attrs = {
                key: json.dumps(value) if isinstance(value, dict) else value
                for key, value in dataset[variable].attrs.items()
            }

        return dataset
