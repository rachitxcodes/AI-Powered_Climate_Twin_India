import xarray as xr

from ingestion.readers.insat_reader import INSATReader
from preprocessing.alignment.spatial_aligner import SpatialAligner
from preprocessing.fusion.dataset_fusion import DatasetFusion
from preprocessing.temporal.temporal_aggregator import TemporalAggregator


class ClimatePipeline:
    """
    Orchestrates the processing of IMD rainfall and INSAT LST data.

    The pipeline does not implement ingestion, temporal aggregation,
    spatial alignment, or dataset fusion itself. It coordinates
    those components in the correct order.
    """

    def __init__(
        self,
        insat_reader=None,
        temporal_aggregator=None,
        spatial_aligner=None,
        dataset_fusion=None,
    ):
        self.insat_reader = insat_reader or INSATReader()
        self.temporal_aggregator = (
            temporal_aggregator or TemporalAggregator()
        )
        self.spatial_aligner = spatial_aligner or SpatialAligner()
        self.dataset_fusion = dataset_fusion or DatasetFusion()

    def process(self, imd_dataset, insat_files):
        """
        Process INSAT observations and fuse them with IMD rainfall.

        Parameters
        ----------
        imd_dataset : xarray.Dataset
            IMD rainfall dataset on the target grid.

        insat_files : list[str]
            Paths to INSAT HDF files.

        Returns
        -------
        xarray.Dataset
            Unified daily climate dataset.
        """

        # ---------------------------------------------------------
        # 1. Validate inputs
        # ---------------------------------------------------------
        self._validate_imd_dataset(imd_dataset)

        if not insat_files:
            raise ValueError(
                "At least one INSAT file is required."
            )

        # ---------------------------------------------------------
        # 2. Read all INSAT files
        # ---------------------------------------------------------
        insat_datasets = [
            self.insat_reader.read(path)
            for path in insat_files
        ]

        # ---------------------------------------------------------
        # 3. Combine INSAT observations along time
        # ---------------------------------------------------------
        insat_dataset = self._combine_insat_datasets(
            insat_datasets
        )

        # ---------------------------------------------------------
        # 4. Convert high-frequency INSAT observations
        #    into daily data
        # ---------------------------------------------------------
        daily_insat = self.temporal_aggregator.aggregate(
            insat_dataset,
            variable="lst",
        )

        # ---------------------------------------------------------
        # 5. Spatially align daily LST to the IMD grid
        # ---------------------------------------------------------
        mapping = self.spatial_aligner.build_mapping(
            source=daily_insat,
            target=imd_dataset,
        )

        aligned_lst = self.spatial_aligner.apply_mapping(
            source=daily_insat,
            target=imd_dataset,
            variable="lst_mean",
            mapping=mapping,
        )

        aligned_count = self.spatial_aligner.apply_mapping(
            source=daily_insat,
            target=imd_dataset,
            variable="valid_observation_count",
            mapping=mapping,
        )

        # ---------------------------------------------------------
        # 7. Combine the aligned INSAT variables
        # ---------------------------------------------------------
        aligned_insat = xr.merge(
            [
                aligned_lst,
                aligned_count,
            ],
            join="exact",
            compat="equals",
        )

        # ---------------------------------------------------------
        # 8. Keep only dates present in both datasets
        # ---------------------------------------------------------
        imd_subset = self._select_matching_dates(
            dataset=imd_dataset,
            reference=aligned_insat,
        )

        aligned_insat = self._select_matching_dates(
            dataset=aligned_insat,
            reference=imd_subset,
        )

        # ---------------------------------------------------------
        # 9. Fuse IMD rainfall + INSAT variables
        # ---------------------------------------------------------
        return self.dataset_fusion.fuse(
            base=imd_subset,
            additions=[aligned_insat],
        )
    
    @staticmethod
    def _combine_insat_datasets(datasets):
        """
        Combine multiple INSAT datasets along the time dimension.
        """

        if not datasets:
            raise ValueError(
                "No INSAT datasets were provided."
            )

        if len(datasets) == 1:
            return datasets[0]

        return xr.concat(
            datasets,
            dim="time",
        )

    @staticmethod
    def _select_matching_dates(dataset, reference):
        """
        Keep dates from `dataset` that also exist in `reference`.

        The comparison is performed at daily resolution, so timestamps
        such as:

            2024-06-18 00:00
            2024-06-18 06:00

        are treated as belonging to the same calendar day.
        """

        dataset_dates = dataset["time"].dt.floor("D")
        reference_dates = reference["time"].dt.floor("D")

        matching_dates = dataset_dates.isin(reference_dates)

        return dataset.isel(time=matching_dates)

    @staticmethod
    def _validate_imd_dataset(dataset):
        """
        Validate that the IMD dataset has the coordinates required
        by the downstream alignment and fusion components.
        """

        if not isinstance(dataset, xr.Dataset):
            raise TypeError(
                "IMD dataset must be an xarray Dataset."
            )

        required_coordinates = {
            "time",
            "latitude",
            "longitude",
        }

        missing = required_coordinates - set(dataset.coords)

        if missing:
            raise ValueError(
                "IMD dataset is missing required coordinates: "
                f"{sorted(missing)}"
            )