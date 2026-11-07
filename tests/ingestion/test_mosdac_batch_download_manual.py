
from datetime import date
from getpass import getpass
from pathlib import Path

from ingestion.clients.mosdac_client import MOSDACClient


client = MOSDACClient()

# Discover all available records for 18 June 2024.
records = client.search_records(
    start_date=date(2024, 6, 18),
    end_date=date(2024, 6, 18),
)

print(f"\nMOSDAC records found: {len(records)}")

if not records:
    raise RuntimeError("No MOSDAC records found.")

# Keep API downloads separate from manually downloaded files.
output_dir = Path("data/raw/insat/api_validation")

username = input("MOSDAC username/email: ").strip()
password = getpass("MOSDAC password: ")

if not username or not password:
    raise ValueError("MOSDAC credentials cannot be empty.")

# Download all missing records.
result = client.download_records(
    records=records,
    username=username,
    password=password,
    output_dir=output_dir,
)

print("\n========== BATCH DOWNLOAD SUMMARY ==========")
print("Records found:", len(records))
print("Downloaded:", len(result["downloaded"]))
print("Skipped:", len(result["skipped"]))
print("Failed:", len(result["failed"]))

print("\nDownloaded files:")
for filename in result["downloaded"]:
    print(filename)

print("\nSkipped files:")
for filename in result["skipped"]:
    print(filename)

print("\nFailed files:")
for failure in result["failed"]:
    print(failure)

actual_files = list(output_dir.glob("*.h5"))

print("\nHDF5 files currently in validation folder:", len(actual_files))
