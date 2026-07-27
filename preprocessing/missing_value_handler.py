import numpy as np
from preprocessing.missing_value_strategy import MissingValueStrategy

class MissingValueHandler:

    def handle(
        self,
        data: np.ndarray,
        strategy: MissingValueStrategy,
    ) -> np.ndarray:

        if strategy == MissingValueStrategy.KEEP:
            return data.copy()

        raise ValueError(
            f"Unsupported missing value strategy: {strategy}"
        )