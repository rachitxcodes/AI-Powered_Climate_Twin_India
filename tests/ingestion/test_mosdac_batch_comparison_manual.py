
from pathlib import Path
import hashlib


reference_dir = Path("data/raw/insat")
api_dir = Path("data/raw/insat/api_validation")


def get_sha256(file_path: Path) -> str:
    """Calculate a file's SHA256 checksum."""

    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


reference_files = {
    file.name: file
    for file in reference_dir.glob("*.h5")
}

api_files = {
    file.name: file
    for file in api_dir.glob("*.h5")
}

print("Reference files:", len(reference_files))
print("API files:", len(api_files))

missing_from_api = set(reference_files) - set(api_files)
missing_from_reference = set(api_files) - set(reference_files)

print("\nMissing from API folder:")
for filename in sorted(missing_from_api):
    print(filename)

print("\nMissing from reference folder:")
for filename in sorted(missing_from_reference):
    print(filename)

if missing_from_api or missing_from_reference:
    print("\nFAIL: Filename comparison failed.")
    raise SystemExit(1)

print("\nAll filenames match. Comparing file contents...")

mismatches = []

for filename in sorted(reference_files):
    reference_hash = get_sha256(reference_files[filename])
    api_hash = get_sha256(api_files[filename])

    if reference_hash != api_hash:
        mismatches.append(filename)

if mismatches:
    print("\nFiles with different contents:")
    for filename in mismatches:
        print(filename)

    print("\nCHECK REQUIRED: Content comparison failed.")

else:
    print("\nPASS: All 23 files match by filename and SHA256.")
