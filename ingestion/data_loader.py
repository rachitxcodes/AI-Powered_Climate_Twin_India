from ingestion.dataset_registry import DatasetRegistry
from ingestion.readers.ctl_parser import CTLParser

class DataLoader:

    def __init__(self):
        self.registry = DatasetRegistry()
        self.parser = CTLParser()
        self._metadata_cache = {}


    def load(self, dataset: str, year: int):
        config = self.registry.get(dataset)
        filename = config.file_pattern.format(year=year)
        grd_path = config.base_directory / filename

        if dataset not in self._metadata_cache:
            self._metadata_cache[dataset] = self.parser.parse(config.ctl_file)

        metadata = self._metadata_cache[dataset]
        reader = config.reader()

        return reader.read(grd_path, metadata)