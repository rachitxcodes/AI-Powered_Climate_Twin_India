from ingestion.data_loader import DataLoader
from preprocessing.pipeline import PreprocessingPipeline


loader = DataLoader()

data = loader.load("rainfall", 2025)

pipeline = PreprocessingPipeline()

result = pipeline.run(data)

stats = result.statistics
validation = result.validation

print("Statistics")
print(f"  Shape               : {stats.shape}")
print(f"  Dimensions          : {stats.dimensions}")
print(f"  Data Type           : {stats.dtype}")
print(f"  Total Values        : {stats.total_values}")
print(f"  Memory Usage (bytes): {stats.memory_usage_bytes}")
print(f"  Missing Values      : {stats.missing_values}")
print(f"  Missing (%)         : {stats.missing_percentage:.2f}%")
print(f"  Minimum Value       : {stats.minimum}")
print(f"  Maximum Value       : {stats.maximum}")
print(f"  Mean Value          : {stats.mean}")
print(f"  Median Value        : {stats.median}")
print(f"  Std Dev             : {stats.standard_deviation}")

print("Validation")
print(f"  Valid   : {validation.valid}")
print(f"  Errors  : {validation.errors if validation.errors else 'None'}")
print(f"  Warnings: {validation.warnings if validation.warnings else 'None'}")

print("Processed Data")
print(f"  Shape   : {result.data.shape}")