from preprocessing.statistics_report import StatisticsReport
from preprocessing.validation_report import ValidationReport


class DatasetValidator:

    MAX_MISSING_PERCENTAGE = 10.0

    def validate(
        self,
        statistics: StatisticsReport,
    ) -> ValidationReport:

        errors = []
        warnings = []

        if statistics.total_values == 0:
            errors.append("Dataset is empty.")

        if statistics.dimensions != 3:
            errors.append(
                f"Expected 3 dimensions, found {statistics.dimensions}."
            )

        if statistics.missing_percentage > self.MAX_MISSING_PERCENTAGE:
            warnings.append(
                f"Missing values exceed {self.MAX_MISSING_PERCENTAGE}%."
            )

        return ValidationReport(
            valid=len(errors) == 0,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )