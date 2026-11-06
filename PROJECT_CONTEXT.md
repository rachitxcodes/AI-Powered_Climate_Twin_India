# Project Context

## Purpose and authoritative project brief

The original objectives are defined by `PS5_Digital_Twin_Overview.docx` (provided at `C:\Users\Rachit Dubey\Downloads\PS5_Digital_Twin_Overview.docx`). It describes a proof of concept for an AI-powered digital twin of India's climate, piloted in Uttar Pradesh.

The intended scope is rainfall and temperature analysis and short-term prediction using national data; IMD and INSAT data fusion; a PyTorch LSTM forecasting model (with Transformer identified as a stretch goal); a digital-twin simulator; a what-if scenario engine; and an interactive Streamlit geospatial dashboard. The brief also identifies xarray, NetCDF, pandas, GeoPandas, Streamlit, Folium/Leaflet, and Plotly as candidate tools.

This document records what is actually implemented and verified in this repository. The brief's target architecture is not evidence that every target component is complete.

## Current architecture

The implemented data path is a Python/xarray pipeline for IMD gridded rainfall and INSAT-3D Land Surface Temperature (LST):

1. `ingestion.readers.ctl_parser.CTLParser` parses IMD CTL metadata.
2. `ingestion.readers.imd_reader.IMDReader` reads IMD GRD rainfall arrays.
3. `core.climate_dataset_factory.ClimateDatasetFactory` constructs an xarray rainfall dataset with `time`, `latitude`, and `longitude` coordinates, wrapped by `core.climate_dataset.ClimateDataset`.
4. `ingestion.readers.insat_reader.INSATReader` reads INSAT-3D L2B LST HDF5 observations on their native `y`/`x` geometry with two-dimensional latitude and longitude coordinates.
5. `preprocessing.temporal.temporal_aggregator.TemporalAggregator` converts high-frequency INSAT observations into daily `lst_mean` and `valid_observation_count` values.
6. `preprocessing.alignment.spatial_aligner.SpatialAligner` builds reusable nearest-neighbor mappings and maps daily INSAT variables to the IMD latitude/longitude grid.
7. `preprocessing.climate_pipeline.ClimatePipeline` keeps only overlapping calendar dates and uses `preprocessing.fusion.dataset_fusion.DatasetFusion` to combine rainfall and aligned LST outputs.
8. `preprocessing.climate_dataset_validator.ClimateDatasetValidator` validates saved climate datasets. `ingestion.climate_workflow.ClimateDataWorkflow` coordinates MOSDAC acquisition, IMD loading, processing, saving, and validation; both are currently untracked worktree files.

The fused output contract used by the validator requires `time`, `latitude`, and `longitude` coordinates and the variables `rainfall`, `lst_mean`, and `valid_observation_count`, each on `(time, latitude, longitude)`.

## Important engineering decisions

- `xarray.Dataset` is the primary multidimensional interchange object.
- INSAT native swath geometry is preserved until explicit alignment to the IMD target grid.
- Spatial alignment supports nearest-neighbor mapping, with the mapping reusable for multiple variables that share the same source geometry.
- Fusion is limited to calendar dates available in both the IMD and aligned INSAT datasets.
- NetCDF serialization converts dictionary attributes to JSON strings for compatibility.
- Feature engineering remains separate from ingestion and preprocessing. Existing generators cover temporal, seasonal, lag, and rolling features.

## Temperature data semantics

The current satellite temperature variable is INSAT **Land Surface Temperature** (LST), represented as `lst` before aggregation and `lst_mean` after daily aggregation. It measures satellite-derived land-surface thermal conditions and must not be described as near-surface air temperature.

The original brief calls for IMD rainfall and temperature data. IMD near-surface air-temperature observations or gridded temperature products are intended to represent air temperature. They are scientifically distinct from LST, must retain separate names and metadata, and must not be silently substituted for one another. The current tracked dataset registry and loader support IMD rainfall only; an IMD air-temperature ingestion path and its validated role in preprocessing/modeling are pending.

## Implementation and verification status

| Area | Status | Evidence and boundary |
| --- | --- | --- |
| Core climate dataset wrapper and rainfall factory | Implemented; automated-test coverage present | `ClimateDataset` wraps xarray datasets; the factory creates rainfall datasets from parsed metadata and arrays. |
| IMD CTL/GRD rainfall ingestion | Implemented; automated-test coverage present | CTL parser, `IMDReader`, registry, and loader exist. The registry currently exposes only `rainfall`. |
| INSAT-3D LST reader | Implemented; automated-test coverage present | HDF5 reader and an INSAT fixture/test exist. |
| Temporal aggregation, spatial alignment, and fusion | Implemented; automated-test coverage present | Components and focused preprocessing tests exist. |
| Climate pipeline orchestration | Implemented; automated-test coverage present | `ClimatePipeline` coordinates reading, aggregation, alignment, matching dates, and fusion. |
| MOSDAC client, workflow, and saved-output validator | Implemented in untracked worktree files; manually verified for one day | Do not treat this as committed baseline until the user elects to manage the untracked files. |
| MOSDAC real-data run: 18 June 2024 | Manually verified | 23 INSAT observations were processed with IMD rainfall through the real-data pipeline. The resulting NetCDF dataset passed validation. |
| MOSDAC real-data run: 19 June 2024 | Manually verified operational outcome | Processing was skipped because MOSDAC returned no data. This is not a successful data-processing run. |
| Automated MOSDAC workflow testing and multi-date verification | Pending | Manual scripts exist in the untracked worktree, but broader automated coverage and verification remain pending. |
| Production robustness | Pending | Credential handling, provider variability, retries, observability, and repeatable operational validation are not established. |
| IMD air-temperature ingestion | Pending | Required by the original brief; not implemented in the active registry/loader. |
| Forecasting, digital-twin simulation, what-if engine, API, and Streamlit dashboard | Pending | Placeholder directories/configuration do not constitute implementation. |

## Tests and known limitations

Tests exist across `core`, `ingestion`, `preprocessing`, and `features`, including manual MOSDAC scripts. Their existence alone does not establish passing or production readiness.

Known issue: `tests/preprocessing/test_climate_pipeline.py` currently has `IndentationError: unexpected indent` at line 1, preventing normal collection of the full test suite. This documentation task does not change that file.

`requirements.txt` currently lists only PyYAML and xarray, while repository code and tests import additional packages such as NumPy, pandas, h5py, pyresample, requests, and pytest. The complete reproducible dependency baseline is therefore pending documentation and validation.

## Repository boundaries

- `models/`, `twin/`, `api/`, `dashboard/`, `notebooks/`, and `docs/` are presently placeholders or empty from an implementation perspective.
- Raw data belongs under `data/raw/`; processed outputs belong under `data/processed/`. These locations are ignored by Git.
- Repository configuration identifies ClimateTwin version `0.1`, Uttar Pradesh as the default region, a seven-day forecast horizon, and LSTM as the baseline. These settings express intent, not completed forecasting functionality.
