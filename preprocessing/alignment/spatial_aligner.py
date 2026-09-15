import numpy as np
import xarray as xr

from pyresample import geometry
from pyresample import kd_tree


class SpatialAligner:

    def __init__(self, method="nearest", radius_of_influence=50000):
        if method != "nearest":
            raise ValueError(
                "Only 'nearest' spatial alignment is supported."
            )

        self.method = method
        self.radius_of_influence = radius_of_influence

    def build_mapping(self, source, target):
        """
        Build the geographic mapping from source pixels
        to target grid points.

        The mapping can be reused for multiple source
        datasets that share the same spatial grid.
        """

        self._validate_source_coordinates(source)
        self._validate_target_coordinates(target)

        source_latitude = source["latitude"].values
        source_longitude = source["longitude"].values

        target_latitude = target["latitude"].values
        target_longitude = target["longitude"].values

        target_latitude, target_longitude = self._create_target_grid(
            target_latitude,
            target_longitude,
        )

        source_grid = geometry.SwathDefinition(
            lons=source_longitude,
            lats=source_latitude,
        )

        target_grid = geometry.SwathDefinition(
            lons=target_longitude,
            lats=target_latitude,
        )

        neighbour_info = kd_tree.get_neighbour_info(
            source_grid,
            target_grid,
            radius_of_influence=self.radius_of_influence,
            neighbours=1,
            epsilon=0,
            reduce_data=True,
        )

        return {
            "neighbour_info": neighbour_info,
            "target_shape": target_latitude.shape,
        }

    def apply_mapping(self, source, target, variable, mapping):
        """
        Apply a previously calculated spatial mapping
        to source data.
        """

        self._validate_source(source, variable)

        neighbour_info = mapping["neighbour_info"]
        target_shape = mapping["target_shape"]

        source_data = source[variable]

        if "time" in source_data.dims:

            aligned_values = []

            for time_index in range(source_data.sizes["time"]):

                values = source_data.isel(
                    time=time_index
                ).values

                aligned = kd_tree.get_sample_from_neighbour_info(
                    "nn",
                    target_shape,
                    values,
                    neighbour_info[0],
                    neighbour_info[1],
                    neighbour_info[2],
                    fill_value=np.nan,
                )

                aligned_values.append(aligned)

            aligned_values = np.stack(aligned_values)

            result = xr.Dataset(
                {
                    variable: (
                        ("time", "latitude", "longitude"),
                        aligned_values,
                    )
                },
                coords={
                    "time": source["time"],
                    "latitude": target["latitude"],
                    "longitude": target["longitude"],
                },
            )

        else:

            aligned_values = kd_tree.get_sample_from_neighbour_info(
                "nn",
                target_shape,
                source_data.values,
                neighbour_info[0],
                neighbour_info[1],
                neighbour_info[2],
                fill_value=np.nan,
            )

            result = xr.Dataset(
                {
                    variable: (
                        ("latitude", "longitude"),
                        aligned_values,
                    )
                },
                coords={
                    "latitude": target["latitude"],
                    "longitude": target["longitude"],
                },
            )

        result[variable].attrs.update(
            source[variable].attrs
        )

        return result

    def align(self, source, target, variable):
        """
        Convenience method.

        Builds the spatial mapping and immediately
        applies it.
        """

        mapping = self.build_mapping(
            source=source,
            target=target,
        )

        return self.apply_mapping(
            source=source,
            target=target,
            variable=variable,
            mapping=mapping,
        )

    @staticmethod
    def _create_target_grid(latitude, longitude):

        if latitude.ndim == 1 and longitude.ndim == 1:

            latitude, longitude = np.meshgrid(
                latitude,
                longitude,
                indexing="ij",
            )

        elif latitude.ndim == 2 and longitude.ndim == 2:
            pass

        else:
            raise ValueError(
                "Target latitude and longitude must both be "
                "1D or both be 2D."
            )

        return latitude, longitude

    @staticmethod
    def _validate_source_coordinates(source):

        if not isinstance(source, xr.Dataset):
            raise TypeError(
                "Source must be an xarray Dataset."
            )

        if "latitude" not in source:
            raise ValueError(
                "Source dataset must contain latitude."
            )

        if "longitude" not in source:
            raise ValueError(
                "Source dataset must contain longitude."
            )

        if source["latitude"].ndim != 2:
            raise ValueError(
                "Source latitude must be 2D."
            )

        if source["longitude"].ndim != 2:
            raise ValueError(
                "Source longitude must be 2D."
            )

    @staticmethod
    def _validate_target_coordinates(target):

        if not isinstance(target, xr.Dataset):
            raise TypeError(
                "Target must be an xarray Dataset."
            )

        if "latitude" not in target:
            raise ValueError(
                "Target dataset must contain latitude."
            )

        if "longitude" not in target:
            raise ValueError(
                "Target dataset must contain longitude."
            )

    @staticmethod
    def _validate_source(source, variable):

        SpatialAligner._validate_source_coordinates(source)

        if variable not in source:
            raise ValueError(
                f"Variable '{variable}' not found in source dataset."
            )