from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xarray as xr


class INSATReader:
    """
    Reads INSAT-3D L2B Land Surface Temperature HDF5 files.

    The reader preserves the native INSAT spatial geometry.
    Spatial alignment to the IMD grid is handled separately.
    """

    def read(self, file_path: Path) -> xr.Dataset:

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"INSAT file not found: {file_path}"
            )

        with h5py.File(file_path, "r") as file:

            # --------------------------------------------------
            # Read LST
            # --------------------------------------------------

            lst = file["LST"][:]

            # --------------------------------------------------
            # Read latitude
            # --------------------------------------------------

            latitude = file["Latitude"][:].astype(np.float32)

            latitude_scale = float(
                file["Latitude"].attrs["scale_factor"][0]
            )

            latitude_fill = int(
                file["Latitude"].attrs["_FillValue"][0]
            )

            latitude = np.where(
                latitude == latitude_fill,
                np.nan,
                latitude * latitude_scale,
            )

            # --------------------------------------------------
            # Read longitude
            # --------------------------------------------------

            longitude = file["Longitude"][:].astype(np.float32)

            longitude_scale = float(
                file["Longitude"].attrs["scale_factor"][0]
            )

            longitude_fill = int(
                file["Longitude"].attrs["_FillValue"][0]
            )

            longitude = np.where(
                longitude == longitude_fill,
                np.nan,
                longitude * longitude_scale,
            )

            # --------------------------------------------------
            # Handle LST missing values
            # --------------------------------------------------

            lst_fill = float(
                file["LST"].attrs["_FillValue"][0]
            )

            lst = np.where(
                lst == lst_fill,
                np.nan,
                lst,
            ).astype(np.float32)

            # --------------------------------------------------
            # Read time
            # --------------------------------------------------

            time_value = float(file["time"][0])

            time = pd.to_datetime(
                "2000-01-01"
            ) + pd.to_timedelta(
                time_value,
                unit="m",
            )

            # --------------------------------------------------
            # Read GeoX / GeoY
            # --------------------------------------------------

            geo_x = file["GeoX"][:]
            geo_y = file["GeoY"][:]

        # ------------------------------------------------------
        # Build xarray Dataset
        # ------------------------------------------------------

        dataset = xr.Dataset(
            data_vars={
                "lst": (
                    ("time", "y", "x"),
                    lst,
                ),
            },
            coords={
                "time": [time],
                "y": geo_y,
                "x": geo_x,
                "latitude": (
                    ("y", "x"),
                    latitude,
                ),
                "longitude": (
                    ("y", "x"),
                    longitude,
                ),
            },
        )

        # ------------------------------------------------------
        # Dataset metadata
        # ------------------------------------------------------

        dataset["lst"].attrs.update(
            {
                "long_name": "Land Surface Temperature",
                "units": "K",
            }
        )

        return dataset