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

    def run(self, data):

        statistics = self._statistics.compute(data)

        validation = self._validator.validate(statistics)

        processed = self._missing_handler.handle(
            data,
            MissingValueStrategy.KEEP,
        )

        return PreprocessingResult(
            data=processed,
            statistics=statistics,
            validation=validation,
        )