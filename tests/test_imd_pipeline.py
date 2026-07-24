from pathlib import Path
import numpy as np
from ingestion.readers.ctl_parser import CTLParser
from ingestion.readers.imd_reader import IMDReader

ctl_file_path = Path("data/raw/imd/rainfall.ctl")
grd_file_path = Path("data/raw/imd/Rainfall_ind2025_rfp25.grd")

parser = CTLParser()
metadata = parser.parse(ctl_file_path)

reader = IMDReader()
data = reader.read(grd_file_path, metadata)

total_values = data.size
missing_values = np.isnan(data).sum()
missing_pct = missing_values / total_values * 100

print(f"Shape          : {data.shape}")
print(f"Data Type      : {data.dtype}")
print(f"Total Values   : {total_values}")
print(f"Missing Values : {missing_values}")
print(f"Missing (%)    : {missing_pct:.2f}%")
print(f"Minimum Value  : {np.nanmin(data)}")
print(f"Maximum Value  : {np.nanmax(data)}")