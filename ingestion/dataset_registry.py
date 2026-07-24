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