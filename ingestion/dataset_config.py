from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    """
    Configuration describing a climate dataset.
    """

    name: str
    base_directory: Path
    ctl_file: Path
    file_pattern: str
    reader: type