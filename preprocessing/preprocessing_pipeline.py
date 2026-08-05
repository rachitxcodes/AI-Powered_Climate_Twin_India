from preprocessing.dataset_statistics import DatasetStatistics
from preprocessing.validator import DatasetValidator
from preprocessing.missing_value_handler import MissingValueHandler
from preprocessing.missing_value_strategy import MissingValueStrategy
from preprocessing.preprocessing_result import PreprocessingResult


class PreprocessingPipeline:
    """
    Executes preprocessing on a ClimateDataset.
    """

    def __init__(self):
        self._statistics = DatasetStatistics()
        self._validator = DatasetValidator()
        self._missing_handler = MissingValueHandler()

    def run(self, climate_dataset) -> PreprocessingResult:
        """
        Run the preprocessing pipeline on a ClimateDataset.
        """

        # Extract rainfall data from the ClimateDataset
        raw_data = climate_dataset.dataset["rainfall"].values

        # Compute statistics
        statistics = self._statistics.compute(raw_data)

        # Validate statistics
        validation = self._validator.validate(statistics)

        # Handle missing values
        processed = self._missing_handler.handle(
            raw_data,
            MissingValueStrategy.KEEP,
        )

        return PreprocessingResult(
            climate_dataset=climate_dataset,
            data=processed,
            statistics=statistics,
            validation=validation,
        )