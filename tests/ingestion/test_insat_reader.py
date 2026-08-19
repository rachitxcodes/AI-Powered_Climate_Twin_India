from pathlib import Path

import numpy as np

from ingestion.readers.insat_reader import INSATReader


def test_insat_reader():

    file_path = Path(
        "tests/fixtures/"
        "3DIMG_18JUN2024_0600_L2B_LST_V01R00.h5"
    )

    reader = INSATReader()

    dataset = reader.read(file_path)

    # --------------------------------------------------
    # Variables
    # --------------------------------------------------

    assert "lst" in dataset
    assert "latitude" in dataset
    assert "longitude" in dataset

    print("Variables: PASS")

    # --------------------------------------------------
    # Dimensions
    # --------------------------------------------------

    assert dataset["lst"].dims == (
        "time",
        "y",
        "x",
    )

    assert dataset["lst"].shape == (
        1,
        2816,
        2805,
    )

    print("LST Dimensions: PASS")

    # --------------------------------------------------
    # Coordinate dimensions
    # --------------------------------------------------

    assert dataset["latitude"].dims == (
        "y",
        "x",
    )

    assert dataset["longitude"].dims == (
        "y",
        "x",
    )

    print("Geolocation Dimensions: PASS")

    # --------------------------------------------------
    # Time
    # --------------------------------------------------

    assert len(dataset.time) == 1

    print("Time: PASS")

    # --------------------------------------------------
    # Units
    # --------------------------------------------------

    assert dataset["lst"].attrs["units"] == "K"

    print("LST Units: PASS")

    # --------------------------------------------------
    # Missing values
    # --------------------------------------------------

    assert not np.any(
        dataset["lst"].values == -999
    )

    print("Missing Value Handling: PASS")

    # --------------------------------------------------
    # Valid values exist
    # --------------------------------------------------

    assert np.isfinite(
        dataset["lst"].values
    ).any()

    print("Valid LST Values: PASS")

    print("\nINSAT READER TEST: PASS")