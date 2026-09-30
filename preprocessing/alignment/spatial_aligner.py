import numpy as np
import xarray as xr

from pyresample import geometry
from pyresample import kd_tree


class SpatialAligner:
    """
    Align a source dataset from its geographic grid/swath
    onto a target latitude/longitude grid.

    The geographic mapping is built once and can then be
    reused for multiple datasets having the same source grid.
    """

    def __init__(
        self,
        method="nearest",
        radius_of_influence=50000,
    ):
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

        Invalid source latitude/longitude pixels are excluded
        from the mapping.
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

        # ---------------------------------------------------------
        # Validate target coordinates
        # ---------------------------------------------------------

        if not np.isfinite(target_latitude).all():
            raise ValueError(
                "Target latitude contains invalid values."
            )

        if not np.isfinite(target_longitude).all():
            raise ValueError(
                "Target longitude contains invalid values."
            )

        # ---------------------------------------------------------
        # Handle invalid INSAT geolocation pixels
        #
        # Real INSAT swaths contain NaN latitude/longitude values
        # outside the valid satellite observation area.
        #
        # pyresample should not be given those pixels as normal
        # geographic coordinates.
        # ---------------------------------------------------------

        valid_source_mask = (
            np.isfinite(source_latitude)
            & np.isfinite(source_longitude)
        )

        if not np.any(valid_source_mask):
            raise ValueError(
                "Source dataset contains no valid geographic coordinates."
            )

        source_lats = source_latitude[valid_source_mask]
        source_lons = source_longitude[valid_source_mask]

        # ---------------------------------------------------------
        # Build source and target geographic definitions
        # ---------------------------------------------------------

        source_grid = geometry.SwathDefinition(
            lons=source_lons,
            lats=source_lats,
        )

        target_grid = geometry.SwathDefinition(
            lons=target_longitude,
            lats=target_latitude,
        )

        # ---------------------------------------------------------
        # Build nearest-neighbour mapping
        # ---------------------------------------------------------

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
            "source_valid_mask": valid_source_mask,
        }

    def apply_mapping(
        self,
        source,
        target,
        variable,
        mapping,
    ):
        """
        Apply a previously calculated geographic mapping
        to a source variable.
        """

        self._validate_source(source, variable)

        neighbour_info = mapping["neighbour_info"]
        target_shape = mapping["target_shape"]
        source_valid_mask = mapping["source_valid_mask"]

        source_data = source[variable]

        # ---------------------------------------------------------
        # The mapping was created using only valid geographic
        # source pixels.
        #
        # Therefore every source variable must be reduced using
        # exactly the same valid-pixel mask before resampling.
        # ---------------------------------------------------------

        if source_data.dims[-2:] != ("y", "x"):
            if "time" not in source_data.dims:
                raise ValueError(
                    "Source variable must use ('y', 'x') spatial dimensions."
                )

        if "time" in source_data.dims:

            aligned_values = []

            for time_index in range(
                source_data.sizes["time"]
            ):

                values = source_data.isel(
                    time=time_index
                ).values

                values = self._prepare_values_for_resampling(
                    values
                )

                values = values[source_valid_mask]

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

            aligned_values = np.stack(
                aligned_values
            )

            result = xr.Dataset(
                {
                    variable: (
                        (
                            "time",
                            "latitude",
                            "longitude",
                        ),
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

            values = source_data.values

            values = self._prepare_values_for_resampling(
                values
            )

            values = values[source_valid_mask]

            aligned_values = kd_tree.get_sample_from_neighbour_info(
                "nn",
                target_shape,
                values,
                neighbour_info[0],
                neighbour_info[1],
                neighbour_info[2],
                fill_value=np.nan,
            )

            result = xr.Dataset(
                {
                    variable: (
                        (
                            "latitude",
                            "longitude",
                        ),
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

    def align(
        self,
        source,
        target,
        variable,
    ):
        """
        Build the geographic mapping and immediately
        apply it.
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
    def _prepare_values_for_resampling(values):
        """
        Convert integer/bool arrays to float32 because
        spatial resampling may need NaN as a fill value.
        """

        values = np.asarray(values)

        if np.issubdtype(
            values.dtype,
            np.integer,
        ):
            return values.astype(
                np.float32
            )

        if np.issubdtype(
            values.dtype,
            np.bool_,
        ):
            return values.astype(
                np.float32
            )

        return values

    @staticmethod
    def _create_target_grid(
        latitude,
        longitude,
    ):
        """
        Convert 1D target latitude/longitude coordinates
        into a 2D geographic grid.
        """

        if (
            latitude.ndim == 1
            and longitude.ndim == 1
        ):

            latitude, longitude = np.meshgrid(
                latitude,
                longitude,
                indexing="ij",
            )

        elif (
            latitude.ndim == 2
            and longitude.ndim == 2
        ):
            pass

        else:
            raise ValueError(
                "Target latitude and longitude must both "
                "be 1D or both be 2D."
            )

        return latitude, longitude

    @staticmethod
    def _validate_source_coordinates(source):

        if not isinstance(
            source,
            xr.Dataset,
        ):
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

        if (
            source["latitude"].shape
            != source["longitude"].shape
        ):
            raise ValueError(
                "Source latitude and longitude must "
                "have the same shape."
            )

    @staticmethod
    def _validate_target_coordinates(target):

        if not isinstance(
            target,
            xr.Dataset,
        ):
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

        if target["latitude"].ndim not in (1, 2):
            raise ValueError(
                "Target latitude must be 1D or 2D."
            )

        if target["longitude"].ndim not in (1, 2):
            raise ValueError(
                "Target longitude must be 1D or 2D."
            )

    @staticmethod
    def _validate_source(
        source,
        variable,
    ):

        SpatialAligner._validate_source_coordinates(
            source
        )

        if variable not in source:
            raise ValueError(
                f"Variable '{variable}' not found "
                "in source dataset."
            )

        if "y" not in source[variable].dims:
            raise ValueError(
                f"Variable '{variable}' must contain "
                "the 'y' spatial dimension."
            )

        if "x" not in source[variable].dims:
            raise ValueError(
                f"Variable '{variable}' must contain "
                "the 'x' spatial dimension."
            )