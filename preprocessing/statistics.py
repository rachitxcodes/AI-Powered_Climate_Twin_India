from __future__ import annotations
import numpy as np
from preprocessing.statistics_report import StatisticsReport

class DatasetStatistics:
    """
    Computes descriptive statistics for climate datasets.

    This class analyzes a NumPy array and returns a
    StatisticsReport. It never modifies the input data.
    """

    def compute(self, data: np.ndarray) -> StatisticsReport:
        """
        Compute descriptive statistics for a climate dataset.

        Parameters
        ----------
        data : np.ndarray
            Climate dataset loaded into memory.

        Returns
        -------
        StatisticsReport
            Summary statistics describing the dataset.

        Raises
        ------
        ValueError
            If the dataset is empty.
        """

        if data.size == 0:
            raise ValueError("Cannot compute statistics for an empty dataset.")

        missing_values = int(np.isnan(data).sum())
        missing_percentage = (missing_values / data.size) * 100

        return StatisticsReport(
            shape=data.shape,
            dimensions=data.ndim,
            dtype=str(data.dtype),
            total_values=data.size,
            memory_usage_bytes=data.nbytes,
            missing_values=missing_values,
            missing_percentage=missing_percentage,
            minimum=float(np.nanmin(data)),
            maximum=float(np.nanmax(data)),
            mean=float(np.nanmean(data)),
            median=float(np.nanmedian(data)),
            standard_deviation=float(np.nanstd(data)),
        )