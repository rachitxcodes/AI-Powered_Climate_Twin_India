from datetime import date
from pathlib import Path
import requests


class MOSDACClient:
    """Search MOSDAC for available INSAT-3D LST records."""

    BASE_URL = "https://mosdac.gov.in/apios/datasets.json"
    DATASET_ID = "3DIMG_L2B_LST"
    PAGE_SIZE = 100

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def search_records(
        self,
        start_date: date,
        end_date: date,
    ) -> list[dict]:
        """Return records available within the requested date range."""

        if end_date < start_date:
            raise ValueError("end_date cannot be earlier than start_date")

        params = {
            "datasetId": self.DATASET_ID,
            "startTime": start_date.isoformat(),
            "endTime": end_date.isoformat(),
        }

        records = []
        start_index = 1

        with requests.Session() as session:
            while True:
                page_params = {
                    **params,
                    "startIndex": start_index,
                }

                response = session.get(
                    self.BASE_URL,
                    params=page_params,
                    timeout=self.timeout,
                )

                if response.status_code >= 400:
                    print("\nMOSDAC API ERROR")
                    print("Status code:", response.status_code)
                    print("Response body:", response.text[:2000])

                try:
                    result = response.json()
                except ValueError:
                    response.raise_for_status()
                    raise ValueError("MOSDAC returned an invalid JSON response.")

                # MOSDAC sometimes returns HTTP 500 when no records match.
                if response.status_code == 500:
                    message = result.get("message", [])

                    if (
                        message == ["Data unavailable for given parameters"]
                        or message == "Data unavailable for given parameters"
                    ):
                        print("MOSDAC reports no data for this date.")
                        return []

                # Raise all other HTTP errors normally.
                response.raise_for_status()

                if not isinstance(result, dict):
                    raise ValueError(
                        "Unexpected MOSDAC response: expected a JSON object."
                    )

                entries = result.get("entries", [])

                if not entries:
                    break

                records.extend(entries)

                total_results = int(result.get("totalResults", 0))

                if len(records) >= total_results:
                    break

                start_index += self.PAGE_SIZE


        return records


    TOKEN_URL = "https://mosdac.gov.in/download_api/gettoken"
    DOWNLOAD_URL = "https://mosdac.gov.in/download_api/download"

    def _get_access_token(self, username: str, password: str) -> str:
        """Authenticate with MOSDAC and return an access token."""

        response = requests.post(
            self.TOKEN_URL,
            json={
                "username": username,
                "password": password,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        token_data = response.json()
        access_token = token_data.get("access_token")

        if not access_token:
            raise RuntimeError(
                "MOSDAC authentication succeeded without returning "
                "an access token."
            )

        return access_token


    def _download_with_token(
        self,
        record: dict,
        access_token: str,
        output_dir: Path,
    ) -> Path:
        """Download one record using an existing access token."""

        record_id = record.get("id")
        filename = record.get("identifier")

        if record_id is None or not filename:
            raise ValueError(
                "Record must contain both 'id' and 'identifier'."
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        destination = output_dir / Path(filename).name
        temporary_file = destination.with_suffix(
            destination.suffix + ".part"
        )

        if destination.exists():
            raise FileExistsError(
                f"File already exists: {destination}"
            )

        headers = {
            "Authorization": f"Bearer {access_token}"
        }

        params = {"id": record_id}

        try:
            with requests.get(
                self.DOWNLOAD_URL,
                headers=headers,
                params=params,
                stream=True,
                timeout=(10, 120),
            ) as response:
                response.raise_for_status()

                with temporary_file.open("wb") as file:
                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):
                        if chunk:
                            file.write(chunk)

            if (
                not temporary_file.exists()
                or temporary_file.stat().st_size == 0
            ):
                raise RuntimeError("MOSDAC returned an empty download.")

            temporary_file.replace(destination)
            return destination

        except Exception:
            temporary_file.unlink(missing_ok=True)
            raise

    def download_record(
        self,
        record: dict,
        username: str,
        password: str,
        output_dir: Path,
    ) -> Path:
        """Authenticate and download one MOSDAC record."""

        access_token = self._get_access_token(username, password)

        return self._download_with_token(
            record=record,
            access_token=access_token,
            output_dir=output_dir,
        )

    def download_records(
        self,
        records: list[dict],
        username: str,
        password: str,
        output_dir: Path,
    ) -> dict:
        """Download multiple records using one authentication session."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        access_token = self._get_access_token(username, password)

        downloaded = []
        skipped = []
        failed = []

        for record in records:
            filename = record.get("identifier")

            if not filename:
                failed.append({
                    "record": record,
                    "error": "Missing filename identifier",
                })
                continue

            destination = output_dir / Path(filename).name

            if destination.exists():
                skipped.append(destination.name)
                continue

            try:
                path = self._download_with_token(
                    record=record,
                    access_token=access_token,
                    output_dir=output_dir,
                )
                downloaded.append(path.name)
                print(f"Downloaded: {path.name}")

            except Exception as error:
                failed.append({
                    "filename": filename,
                    "error": str(error),
                })
                print(f"Failed: {filename} — {error}")

        return {
            "downloaded": downloaded,
            "skipped": skipped,
            "failed": failed,
        }
