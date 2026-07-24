"""
IMD GRD Reader

Reads IMD gridded binary (.grd) files and converts them
into NumPy arrays.

Responsibilities:
- Read binary GRD files
- Use metadata supplied by CTLParser
- Replace missing values
- Return NumPy arrays

Does NOT:
- Parse CTL files
- Perform preprocessing
- Train models
- Create visualizations
- Convert to xarray
"""

from pathlib import Path
import numpy as np


class IMDReader:
    """Reads IMD GRD files using parsed CTL metadata."""

    def __init__(self):
        pass

    def read(self, file_path, metadata):
        """
        Read an IMD .grd file.

        Parameters
        ----------
        file_path : str | Path
            Path to the .grd file.

        metadata : dict
            Metadata returned by CTLParser.

        Returns
        -------
        numpy.ndarray
            Array of shape (days, height, width)
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() != ".grd":
            raise ValueError("Expected a .grd file")

        width = metadata["xdef"]["count"]
        height = metadata["ydef"]["count"]
        missing_value = metadata["missing_value"]

        with file_path.open("rb") as file:
            data = np.fromfile(file, dtype=np.float32)

        values_per_day = width * height

        if data.size % values_per_day != 0:
            raise ValueError(
                f"Data size ({data.size}) is not divisible by "
                f"grid size ({values_per_day})."
            )

        num_days = data.size // values_per_day

        rainfall = data.reshape(num_days, height, width)

        rainfall[rainfall == missing_value] = np.nan

        return rainfall