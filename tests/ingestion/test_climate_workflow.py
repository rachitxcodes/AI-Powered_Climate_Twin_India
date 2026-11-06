from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from ingestion.climate_workflow import ClimateDataWorkflow
from preprocessing.validation_report import ValidationReport


def make_climate_dataset(day: str = "2024-06-18") -> xr.Dataset:
    coordinates = {
        "time": [np.datetime64(day)],
        "latitude": [26.0],
        "longitude": [78.0],
    }
    dimensions = ("time", "latitude", "longitude")

    return xr.Dataset(
        {
            "rainfall": (dimensions, np.array([[[10.0]]], dtype=np.float32)),
            "lst_mean": (dimensions, np.array([[[300.0]]], dtype=np.float32)),
            "valid_observation_count": (
                dimensions,
                np.array([[[1]]], dtype=np.int16),
            ),
        },
        coords=coordinates,
    )


class FakeMOSDACClient:
    def __init__(self, *, records=None, download_result=None, write_files=True):
        self.records = records if records is not None else []
        self.download_result = download_result or {
            "downloaded": [],
            "skipped": [],
            "failed": [],
        }
        self.write_files = write_files
        self.search_calls = []
        self.download_calls = []

    def search_records(self, start_date, end_date):
        self.search_calls.append((start_date, end_date))
        return self.records

    def download_records(self, *, records, username, password, output_dir):
        self.download_calls.append((records, username, password, output_dir))

        if self.write_files:
            for record in records:
                identifier = record.get("identifier")
                if identifier:
                    destination = Path(output_dir) / Path(identifier).name
                    destination.write_bytes(b"fake hdf data")

        return self.download_result


class FakeDataLoader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.calls = []

    def load(self, *, dataset, year):
        self.calls.append((dataset, year))
        return SimpleNamespace(dataset=self.dataset)


class FakeClimatePipeline:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def process(self, *, imd_dataset, insat_files):
        self.calls.append((imd_dataset, insat_files))
        return self.result


class FakeClimateDatasetValidator:
    def __init__(self, report):
        self.report = report
        self.calls = []

    def validate_file(self, *, file_path, reference_dataset):
        self.calls.append((file_path, reference_dataset))
        return self.report


@pytest.fixture
def imd_dataset():
    return make_climate_dataset()[["rainfall"]]


@pytest.fixture
def valid_report():
    return ValidationReport(valid=True, errors=(), warnings=())


@pytest.fixture
def workflow(tmp_path, imd_dataset, valid_report):
    workflow = ClimateDataWorkflow()
    workflow.project_root = tmp_path
    workflow.mosdac_client = FakeMOSDACClient(
        records=[{"id": 7, "identifier": "3DIMG_18JUN2024_0600_L2B_LST.h5"}]
    )
    workflow.data_loader = FakeDataLoader(imd_dataset)
    workflow.climate_pipeline = FakeClimatePipeline(make_climate_dataset())
    workflow.climate_dataset_validator = FakeClimateDatasetValidator(valid_report)
    return workflow


def test_run_day_orchestrates_dependencies_saves_and_validates(workflow, tmp_path):
    result, output_path = workflow.run_day(
        target_date=date(2024, 6, 18), username="user", password="secret"
    )

    expected_insat_path = (
        tmp_path
        / "data"
        / "raw"
        / "insat"
        / "api_validation"
        / "3DIMG_18JUN2024_0600_L2B_LST.h5"
    )
    expected_output_path = (
        tmp_path / "data" / "processed" / "climate" / "climate_2024-06-18.nc"
    )

    assert result is workflow.climate_pipeline.result
    assert output_path == expected_output_path
    assert output_path.is_file()
    assert workflow.mosdac_client.search_calls == [(date(2024, 6, 18), date(2024, 6, 18))]
    assert workflow.mosdac_client.download_calls == [
        (
            workflow.mosdac_client.records,
            "user",
            "secret",
            expected_insat_path.parent,
        )
    ]
    assert workflow.data_loader.calls == [("rainfall", 2024)]
    assert workflow.climate_pipeline.calls == [
        (workflow.data_loader.dataset, [str(expected_insat_path)])
    ]
    assert workflow.climate_dataset_validator.calls == [
        (expected_output_path, workflow.data_loader.dataset)
    ]

    with xr.open_dataset(output_path) as saved:
        assert set(saved.data_vars) == {
            "rainfall",
            "lst_mean",
            "valid_observation_count",
        }


def test_get_imd_dataset_caches_each_year(workflow):
    first = workflow._get_imd_dataset(2024)
    second = workflow._get_imd_dataset(2024)

    assert first is second
    assert workflow.data_loader.calls == [("rainfall", 2024)]


def test_run_day_rejects_empty_credentials_before_dependencies(workflow):
    with pytest.raises(ValueError, match="credentials cannot be empty"):
        workflow.run_day(date(2024, 6, 18), username="", password="secret")

    assert workflow.mosdac_client.search_calls == []
    assert workflow.mosdac_client.download_calls == []
    assert workflow.data_loader.calls == []


def test_run_day_stops_when_mosdac_returns_no_records(workflow):
    workflow.mosdac_client.records = []

    with pytest.raises(RuntimeError, match="No INSAT records found"):
        workflow.run_day(date(2024, 6, 19), username="user", password="secret")

    assert workflow.mosdac_client.download_calls == []
    assert workflow.data_loader.calls == []
    assert workflow.climate_pipeline.calls == []


def test_run_day_stops_before_processing_when_downloads_fail(workflow):
    workflow.mosdac_client.download_result = {
        "downloaded": [],
        "skipped": [],
        "failed": [{"filename": "missing.h5", "error": "network failure"}],
    }

    with pytest.raises(RuntimeError, match="Some INSAT downloads failed"):
        workflow.run_day(date(2024, 6, 18), username="user", password="secret")

    assert workflow.data_loader.calls == []
    assert workflow.climate_pipeline.calls == []


def test_run_day_rejects_pipeline_output_without_requested_date(workflow, tmp_path):
    workflow.climate_pipeline.result = make_climate_dataset("2024-06-17")

    with pytest.raises(RuntimeError, match="did not produce data"):
        workflow.run_day(date(2024, 6, 18), username="user", password="secret")

    assert workflow.climate_dataset_validator.calls == []
    assert not (tmp_path / "data" / "processed" / "climate" / "climate_2024-06-18.nc").exists()


def test_run_day_raises_when_saved_output_validation_fails(workflow, valid_report):
    workflow.climate_dataset_validator.report = ValidationReport(
        valid=False,
        errors=("latitude does not match the IMD grid.",),
        warnings=(),
    )

    with pytest.raises(RuntimeError, match="Climate dataset validation failed"):
        workflow.run_day(date(2024, 6, 18), username="user", password="secret")

    assert len(workflow.climate_dataset_validator.calls) == 1


def test_run_range_skips_no_data_and_combines_successful_days(workflow, monkeypatch, tmp_path):
    successful_date = date(2024, 6, 18)
    skipped_date = date(2024, 6, 19)
    daily_output = tmp_path / "climate_2024-06-18.nc"
    daily_output.write_bytes(b"placeholder")
    calls = []

    def fake_run_day(target_date, username, password):
        calls.append((target_date, username, password))
        if target_date == skipped_date:
            raise RuntimeError(f"No INSAT records found for {target_date}.")
        return make_climate_dataset(), daily_output

    combined_path = tmp_path / "climate_2024-06-18_to_2024-06-19.nc"
    combine_calls = []

    def fake_combine_daily_outputs(*, results, start_date, end_date):
        combine_calls.append((results, start_date, end_date))
        return combined_path

    monkeypatch.setattr(workflow, "run_day", fake_run_day)
    monkeypatch.setattr(workflow, "combine_daily_outputs", fake_combine_daily_outputs)

    results = workflow.run_range(successful_date, skipped_date, "user", "secret")

    assert list(results) == [successful_date]
    assert calls == [
        (successful_date, "user", "secret"),
        (skipped_date, "user", "secret"),
    ]
    assert combine_calls == [(results, successful_date, skipped_date)]


def test_run_range_rejects_reversed_dates(workflow):
    with pytest.raises(ValueError, match="end_date cannot be earlier"):
        workflow.run_range(date(2024, 6, 19), date(2024, 6, 18), "user", "secret")
