"""
IMD GRD Reader -
Reads IMD gridded rainfall binary (.grd) files and converts
them into standard Python numerical arrays.

Responsibilities:
- Read binary rainfall files
- Handle IMD-specific format
- Replace missing values
- Return NumPy arrays

Does NOT:
- Perform preprocessing
- Train models
- Create visualizations
- Convert to xarray
"""

from pathlib import Path
import numpy as np

GRID_WIDTH = 135
GRID_HEIGHT = 129
MISSING_VALUE = -999.0

class IMDReader:
    """Reads IMD rainfall GRD files."""
    def __init__(self):
        pass

    def read(self, file_path):
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix != ".grd":
            raise ValueError("Expected a .grd file")

        with open(file_path, "rb") as f:
            data = np.fromfile(f, dtype=np.float32)

        values_per_day = GRID_WIDTH * GRID_HEIGHT
        num_days = data.size // values_per_day

        # Verify axis order using IMD documentation
        if data.size % values_per_day != 0:
            raise ValueError(f"Data size {data.size} is not a multiple of {values_per_day}.")

        # Reshape the data into the grid dimensions
        rainfall = data.reshape(num_days, GRID_HEIGHT, GRID_WIDTH)
        
        # Replace missing values
        rainfall[rainfall == MISSING_VALUE] = np.nan
        
        return rainfall