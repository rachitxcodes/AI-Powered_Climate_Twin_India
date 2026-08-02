from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    """
    Represents a dataset loaded from disk together
    with its metadata.
    """

    data: np.ndarray

    metadata: dict

    year: int