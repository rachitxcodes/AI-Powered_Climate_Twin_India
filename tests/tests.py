from ingestion.data_loader import DataLoader
from preprocessing.statistics import DatasetStatistics

loader = DataLoader()

data = loader.load("rainfall", 2025)

statistics = DatasetStatistics()

report = statistics.compute(data)

print(f"Shape               : {report.shape}")
print(f"Dimensions          : {report.dimensions}")
print(f"Data Type           : {report.dtype}")
print(f"Total Values        : {report.total_values}")
print(f"Memory Usage (bytes): {report.memory_usage_bytes}")
print(f"Missing Values      : {report.missing_values}")
print(f"Missing (%)         : {report.missing_percentage:.2f}%")
print(f"Minimum Value       : {report.minimum}")
print(f"Maximum Value       : {report.maximum}")
print(f"Mean Value          : {report.mean}")
print(f"Median Value        : {report.median}")
print(f"Std Dev             : {report.standard_deviation}")