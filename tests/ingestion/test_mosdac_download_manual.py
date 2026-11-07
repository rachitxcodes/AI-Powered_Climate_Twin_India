
from datetime import date
from getpass import getpass
from pathlib import Path

from ingestion.clients.mosdac_client import MOSDACClient


def main():
    client = MOSDACClient()

    # Step 1: Search MOSDAC for available INSAT-3D LST records.
    print("Searching MOSDAC for records: 18–30 June 2024...")

    records = client.search_records(
        start_date=date(2024, 6, 18),
        end_date=date(2024, 6, 30),
    )

    # Step 2: Display search results.
    print(f"\nTotal MOSDAC records found: {len(records)}")

    if not records:
        raise RuntimeError(
            "MOSDAC returned no records for 18–30 June 2024."
        )

    print("\nAvailable files:")

    for index, item in enumerate(records, start=1):
        print(
            f"{index}. {item.get('identifier')} "
            f"(Record ID: {item.get('id')})"
        )

    # Step 3: Select the first available record.
    record = records[0]

    print("\nSelected file:", record.get("identifier"))
    print("MOSDAC record ID:", record.get("id"))

    # Step 4: Collect MOSDAC credentials.
    username = input("MOSDAC username/email: ").strip()
    password = getpass("MOSDAC password: ")

    if not username or not password:
        raise ValueError("MOSDAC credentials cannot be empty.")

    # Step 5: Download into a separate validation directory.
    output_dir = Path(
        "data/raw/insat/api_validation_june_range"
    )

    downloaded_path = client.download_record(
        record=record,
        username=username,
        password=password,
        output_dir=output_dir,
    )

    # Step 6: Verify the downloaded file.
    print("\nDownload completed successfully.")
    print("Saved to:", downloaded_path)
    print("File size:", downloaded_path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
