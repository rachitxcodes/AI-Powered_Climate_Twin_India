# AGENTS.md

## Repository operating instructions

This repository is an AI-powered climate-twin proof of concept for India, with Uttar Pradesh as the intended pilot region. Work carefully from the documented architecture and current evidence; do not infer that a feature is complete solely because a directory, configuration entry, or source file exists.

Before changing implementation, inspect `git status`, `PROJECT_CONTEXT.md`, `PROJECT_ROADMAP.md`, relevant source, tests, and applicable handoff notes. Explain significant changes and run validation appropriate to the risk.

Do not modify, delete, move, stage, reset, discard, commit, or push existing work without explicit user approval. Treat modified and untracked files as in-progress, user-owned work unless the user specifically places them in scope.

Use evidence-based status terms:

- **Implemented**: source code exists with a defined responsibility.
- **Automated-test coverage present**: relevant automated tests exist; this does not imply they currently pass.
- **Manually verified**: a documented real or manual run produced the stated result.
- **Pending**: not yet implemented, tested, or verified.
- **Known issue**: an observed limitation or failure that is not being addressed in the current task.

Preserve module boundaries. Keep acquisition, dataset readers, preprocessing, temporal aggregation, spatial alignment, fusion, validation, feature engineering, modeling, simulation, API, and dashboard concerns separate. Prefer established naming and data contracts over parallel abstractions.

## Data and architecture conventions

### NetCDF

NetCDF (Network Common Data Form) stores multidimensional scientific data, such as temperature, humidity, wind speed, rainfall, and satellite observations, across space and time. Use it for persisted climate datasets when its coordinate-aware, multidimensional representation is appropriate.

### xarray

Climate data varies across latitude, longitude, altitude, and time. `xarray` is the primary in-memory interchange format because it handles N-dimensional labeled arrays and datasets natively; pandas is principally a two-dimensional tabular tool. Keep coordinate names and dimension order explicit. The existing climate pipeline expects `time`, `latitude`, and `longitude` coordinates, with fused variables on `(time, latitude, longitude)`.

Use lazy loading deliberately when datasets and supported backends permit it; do not assume every current in-memory operation is lazy.

### GeoPandas

`xarray` is suited to raster/gridded or swath-like data. GeoPandas is suited to vector geometries such as points, lines, polygons, administrative boundaries, and station locations. Use GeoPandas when exact boundary operations, shapefile/GeoJSON/GeoPackage handling, reprojection of vector data, or geometric queries are needed. Do not substitute it for xarray in grid-based processing.

### Temperature semantics

INSAT LST is satellite-derived **land surface temperature**: it describes the radiometric temperature of the land surface under satellite retrieval conditions. It is not equivalent to near-surface air temperature.

IMD air-temperature observations or gridded products are the intended source for near-surface air-temperature variables. Preserve separate variable names, metadata, units, validation ranges, and modeling roles for LST and air temperature. Never label `lst` or `lst_mean` as air temperature, and do not use it as a direct replacement for IMD air-temperature observations without an explicitly documented scientific method.

## Data, security, and verification practices

- Keep raw and processed climate data out of version control unless the user explicitly changes that policy.
- Never put MOSDAC credentials, tokens, or other secrets in source, tests, documentation, or output.
- Treat external-provider behavior as variable. Distinguish reproducible automated tests from credentialed/manual MOSDAC runs.
- Retain the native INSAT swath geometry until the established spatial-alignment step maps it to the target IMD grid.
- Preserve the pipeline rule that fusion uses dates shared by the source datasets.
- When architecture, verification evidence, or project stage changes, update `PROJECT_CONTEXT.md` and `PROJECT_ROADMAP.md` in the same scoped change.

## Current known issue

`tests/preprocessing/test_climate_pipeline.py` currently fails static parsing with `IndentationError: unexpected indent` at line 1. Record and respect this issue during test reporting. Do not fix it as part of documentation-only work unless explicitly asked.
