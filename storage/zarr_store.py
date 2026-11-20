
from pathlib import Path

import xarray as xr

from core.climate_dataset import ClimateDataset


class ZarrClimateStore:
    """Save and load chunked climate datasets using Zarr."""

    def save(
        self,
        climate_dataset: ClimateDataset,
        path: str | Path,
    ) -> None:
        path = Path(path)

        if path.exists():
            raise FileExistsError(
                f"Zarr store already exists: {path}"
            )

        path.parent.mkdir(parents=True, exist_ok=True)

        dataset = climate_dataset.dataset.chunk(
            {
                "time": 30,
                "latitude": 32,
                "longitude": 32,
            }
        )

        dataset.to_zarr(
            str(path),
            mode="w-",
            zarr_format=3,
            consolidated=False,
        )

    def load(self, path: str | Path) -> ClimateDataset:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Zarr store not found: {path}"
            )

        dataset = xr.open_zarr(
            str(path),
            zarr_format=3,
            consolidated=False,
            chunks={
                "time": 30,
                "latitude": 32,
                "longitude": 32,
            },
        )

        return ClimateDataset(dataset)
