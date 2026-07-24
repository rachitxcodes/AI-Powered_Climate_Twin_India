from ingestion.readers.imd_reader import IMDReader
import numpy as np

reader = IMDReader()

rainfall = reader.read("data/raw/imd/Rainfall_ind2025_rfp25.grd")

print(rainfall.shape)
print(rainfall.dtype)
print(np.nanmin(rainfall))
print(np.nanmax(rainfall))