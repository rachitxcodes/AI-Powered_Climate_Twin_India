"""
Dataset Registry -
Central catalog of all datasets supported by ClimateTwin.

Responsibilities:
- Store dataset metadata
- Describe providers
- Describe formats
- Describe storage locations

This module does NOT:
- Download data
- Read data
- Process data
"""

DATASETS = {
    "imd_rainfall": {
        "name": "IMD Gridded Rainfall",
        "provider": "India Meteorological Department",
        "format": "NetCDF",
        "storage": "data/raw/imd"
    },
    "imd_temperature": {
        "name": "IMD Temperature",
        "provider": "India Meteorological Department",
        "format": "NetCDF",
        "storage": "data/raw/imd"
    },
    "insat_lst": {
        "name": "INSAT Land Surface Temperature",
        "provider": "MOSDAC",
        "format": "NetCDF",
        "storage": "data/raw/insat"
    }
}

from pathlib import Path

from ingestion.dataset_config import DatasetConfig
from ingestion.readers.imd_reader import IMDReader

class DatasetRegistry:
    """
    Stores configuration for all supported datasets.
    """
    def __init__(self):
        self._datasets = {
            "rainfall": DatasetConfig(

                name="rainfall",

                base_directory=Path("data/raw/imd"),

                ctl_file=Path("data/raw/imd/rainfall.ctl"),

                file_pattern="Rainfall_ind{year}_rfp25.grd",

                reader=IMDReader
            )
        }   

    def get(self, dataset_name: str) -> DatasetConfig:
        if dataset_name not in self._datasets:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        return self._datasets[dataset_name]
   
