
from datetime import date
from getpass import getpass

from ingestion.climate_workflow import ClimateDataWorkflow


username = input("MOSDAC username: ").strip()
password = getpass("MOSDAC password: ")

workflow = ClimateDataWorkflow()

results = workflow.run_range(
    start_date=date(2024, 6, 18),
    end_date=date(2024, 6, 19),
    username=username,
    password=password,
)

print("\n========== FINAL RANGE RESULTS ==========")

for target_date, item in results.items():
    print(
        target_date,
        "->",
        item["output_path"],
        "| dimensions:",
        dict(item["dataset"].sizes),
    )
