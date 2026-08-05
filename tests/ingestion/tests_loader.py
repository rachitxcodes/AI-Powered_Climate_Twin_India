import numpy as np  
from ingestion.data_loader import DataLoader

loader = DataLoader()

dataset = loader.load("rainfall", 2025)

print(type(dataset))
print(dataset.dataset)