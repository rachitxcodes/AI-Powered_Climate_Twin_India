# Project Roadmap

## Original objective

The authoritative original scope is `PS5_Digital_Twin_Overview.docx`. The proof of concept targets an AI-powered climate digital twin for Uttar Pradesh, covering rainfall and temperature, IMD and INSAT data use, short-term forecasting, simulation, a what-if module, and an interactive Streamlit geospatial dashboard.

The envisioned pipeline is data acquisition, preprocessing, feature engineering and fusion, model training and validation, digital-twin simulation, scenario analysis, and dashboard visualization. The original brief proposes PyTorch LSTM forecasting, with Transformer as a stretch goal, and evaluation with RMSE/MAE against IMD ground truth.

## Current development checkpoint

**Checkpoint: rainfall-plus-LST data foundation with one manually verified real-data day.**

The repository implements a preprocessing path that fuses IMD gridded rainfall with INSAT satellite-derived LST after daily aggregation and spatial alignment to the IMD grid. On 18 June 2024, 23 INSAT observations were processed with IMD rainfall through the real-data MOSDAC workflow; the resulting NetCDF dataset passed validation. This is manually verified.

On 19 June 2024, the workflow was skipped because MOSDAC returned no data. This is a manually verified provider outcome, not validation of a second processing day.

This checkpoint does not establish multi-date correctness, automated MOSDAC workflow coverage, production readiness, IMD air-temperature integration, forecasting, simulation, scenario analysis, or dashboard delivery.

## Scientific data boundary

INSAT LST is land surface temperature, not near-surface air temperature. LST is a satellite-derived surface variable that can be fused as an additional climate signal but must retain its own variable name, metadata, validation, and modeling interpretation.

IMD air-temperature observations or gridded temperature products are the intended source for air-temperature variables in the original brief. Their ingestion, alignment, and validated modeling role remain pending. Future work must not replace air temperature with LST without an explicit, scientifically justified method.

## Delivery stages

### Stage 0 - Repository and verification baseline

Status: In progress.

- Preserve existing implementation and in-progress worktree files.
- Resolve the dependency specification into a reproducible environment.
- Address the known test-collection issue separately: `tests/preprocessing/test_climate_pipeline.py` has an `IndentationError` at line 1.
- Establish a clear automated/manual/integration test taxonomy.

Completion evidence: documented dependency setup and a deliberately scoped test run with recorded results. Fixing the known collection error is outside this documentation change.

### Stage 1 - Robust data acquisition and climate-data validation

Status: Partially implemented; broader verification pending.

- Retain and test the IMD rainfall plus INSAT LST flow.
- Convert manual MOSDAC scenarios into automated, credential-safe tests where feasible.
- Verify multiple dates, no-data dates, download failures, output validation failures, and repeatability.
- Improve operational resilience around provider responses, retries, partial downloads, credentials, logging, and output retention.

Completion evidence: automated coverage and documented manual verification over a representative multi-date sample. The 18 June 2024 run alone is insufficient.

### Stage 2 - Pilot-region data preparation and feature datasets

Status: Partially implemented foundations; pending end-to-end delivery.

- Apply Uttar Pradesh pilot-region masking/boundary logic.
- Define missing-data, normalization, and time-based train/validation/test policies.
- Produce reproducible aligned rainfall, LST, and IMD air-temperature datasets.
- Extend and validate lag, rolling, seasonal, and cyclical features for forecasting.

Completion evidence: versioned dataset contract, reproducible feature-generation run, and validation reports.

### Stage 3 - Forecasting model and evaluation

Status: Pending.

- Implement and train the PyTorch LSTM baseline for rainfall and air-temperature forecasting.
- Keep LST scientifically distinct; document any use as an auxiliary predictive feature.
- Use time-respecting train/validation/test splits.
- Measure RMSE and MAE against appropriate IMD ground truth; retain baselines and experiment records.
- Consider Transformer only after the LSTM baseline and evaluation protocol are established.

Completion evidence: reproducible training/evaluation artifacts, reported metrics, and documented limits.

### Stage 4 - Digital-twin state and simulation

Status: Pending.

- Define the climate-state representation for the Uttar Pradesh pilot.
- Combine observed, fused, and forecast signals into a time-aware simulated state.
- Specify simulation assumptions, uncertainty boundaries, and update behavior.

Completion evidence: tested simulation interface and a documented mapping from data/model outputs to twin state.

### Stage 5 - What-if scenario engine

Status: Pending.

- Define valid rainfall and air-temperature perturbations.
- Re-run the model or simulator under controlled scenarios.
- Report outputs, assumptions, bounds, and non-causal interpretation clearly.

Completion evidence: repeatable scenario runs with tests and user-facing explanation of limitations.

### Stage 6 - Streamlit dashboard and deployment readiness

Status: Pending.

- Build the Streamlit dashboard with a map layer, time-series charts, region selection, forecast views, and what-if controls.
- Integrate validated data/model/twin interfaces rather than bypassing pipeline contracts.
- Add API/service boundaries only where warranted by the dashboard architecture.
- Establish logging, configuration, secrets handling, monitoring, and deployment documentation.

Completion evidence: demonstrated end-to-end pilot workflow, UI verification, and operational runbook.

## Roadmap maintenance rule

Move an item forward only when its stated completion evidence exists. Record whether evidence is automated, manual, or pending; do not infer completion from placeholder directories, configuration values, or source-file presence.
