
from datetime import date
from pathlib import Path

from ingestion.clients.mosdac_client import MOSDACClient


client = MOSDACClient()

records = client.search_records(
    start_date=date(2024, 6, 21),
    end_date=date(2024, 6, 21),
)

print(f"API records returned: {len(records)}")

api_files = {
    record["identifier"]
    for record in records
    if record.get("identifier")
}

raw_dir = Path("data/raw/insat")

local_files = {
    path.name
    for path in raw_dir.glob("*.h5")
}

print(f"Local HDF5 files: {len(local_files)}")
print(f"Unique API filenames: {len(api_files)}")

print("\nAPI records missing locally:")
for filename in sorted(api_files - local_files):
    print(filename)

print("\nLocal files missing from API results:")
for filename in sorted(local_files - api_files):
    print(filename)

if api_files == local_files:
    print("\nPASS: API filenames match local filenames.")
else:
    print("\nCHECK REQUIRED: API and local filenames differ.")
