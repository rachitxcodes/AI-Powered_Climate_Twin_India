from ingestion.dataset_registry import DatasetRegistry
from ingestion.readers.ctl_parser import CTLParser
from ingestion.loaded_dataset import LoadedDataset


class DataLoader:

    def __init__(self):
        self.registry = DatasetRegistry()
        self.parser = CTLParser()
        self._metadata_cache = {}

    def load(self, dataset: str, year: int):

        config = self.registry.get(dataset)

        filename = config.file_pattern.format(year=year)
        grd_path = config.base_directory / filename

        # Parse metadata only once
        if dataset not in self._metadata_cache:
            self._metadata_cache[dataset] = self.parser.parse(config.ctl_file)

        metadata = self._metadata_cache[dataset]

        # Read dataset
        reader = config.reader()
        data = reader.read(grd_path, metadata)

        # Return BOTH data and metadata
        return LoadedDataset(
            data=data,
            metadata=metadata,
        )