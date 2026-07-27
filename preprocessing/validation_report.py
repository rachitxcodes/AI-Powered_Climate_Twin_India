from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ValidationReport:
    """
    Stores the validation result for a climate dataset.
    """

    valid: bool

    errors: Tuple[str, ...]

    warnings: Tuple[str, ...]