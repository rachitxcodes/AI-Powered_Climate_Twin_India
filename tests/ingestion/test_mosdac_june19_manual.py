from datetime import date

from ingestion.clients.mosdac_client import MOSDACClient

client = MOSDACClient()

records = client.search_records(
    start_date=date(2024, 6, 19),
    end_date=date(2024, 6, 19),
)

print("Records found:", len(records))

for record in records:
    print(record.get("identifier"))