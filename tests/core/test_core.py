from ingestion.data_loader import DataLoader


loader = DataLoader()

climate_dataset = loader.load(
    "rainfall",
    2025,
)

print(type(climate_dataset))

print()

print(climate_dataset.dataset)