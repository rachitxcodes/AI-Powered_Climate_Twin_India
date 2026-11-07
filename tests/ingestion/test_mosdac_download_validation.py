from pathlib import Path

from ingestion.readers.insat_reader import INSATReader


file_path = Path(
    "data/raw/insat/api_validation/"
    "3DIMG_18JUN2024_2300_L2B_LST_V01R00.h5"
)

if not file_path.exists():
    raise FileNotFoundError(f"Downloaded file not found: {file_path}")

print("File size:", file_path.stat().st_size, "bytes")

reader = INSATReader()
dataset = reader.read(file_path)

print("\nINSATReader validation successful.")
print(dataset)

print("\nDimensions:", dict(dataset.sizes))
print("Variables:", list(dataset.data_vars))
