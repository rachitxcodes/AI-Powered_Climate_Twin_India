from preprocessing.dataset_statistics import DatasetStatistics
from preprocessing.validator import DatasetValidator
from preprocessing.missing_value_handler import MissingValueHandler
from preprocessing.missing_value_strategy import MissingValueStrategy
from preprocessing.preprocessing_result import PreprocessingResult

class PreprocessingPipeline:

    def __init__(self):

        self._statistics = DatasetStatistics()
        self._validator = DatasetValidator()
        self._missing_handler = MissingValueHandler()

    def run(self, loaded_dataset):

        # Extract the raw NumPy array
        raw_data = loaded_dataset.data

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
            data=processed,
            metadata=loaded_dataset.metadata,
            statistics=statistics,
            validation=validation,
            year=loaded_dataset.year
        )