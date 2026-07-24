import numpy as np  
from ingestion.data_loader import DataLoader

loader = DataLoader()

for year in [2025, 2024, 2023]:
    data = loader.load("rainfall", year)
    print(f"Year           : {year}")
    print(f"Shape          : {data.shape}")
    print(f"Data Type      : {data.dtype}")
    print(f"Missing Values : {np.isnan(data).sum()}")
    print(f"Minimum Value  : {np.nanmin(data)}")
    print(f"Maximum Value  : {np.nanmax(data)}")
    print("-" * 40)