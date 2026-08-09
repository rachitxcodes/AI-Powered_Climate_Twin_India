import numpy as np
import pandas as pd
import xarray as xr

from features.generators.temporal_generator import TemporalGenerator


def test_cyclical_encoding_handles_leap_year():

    time = pd.to_datetime(
        [
            "2024-02-28",
            "2024-02-29",
            "2024-03-01",
        ]
    )

    dataset = xr.Dataset(
        coords={
            "time": time,
        }
    )

    generator = TemporalGenerator()

    dataset = generator.generate(dataset)

    dataset = generator.generate_cyclical(dataset)

    # --------------------------------------------------------
    # Verify leap year
    # --------------------------------------------------------

    assert dataset.time.dt.is_leap_year.all()

    # --------------------------------------------------------
    # Verify day-of-year
    # --------------------------------------------------------

    assert (
        dataset["day_of_year"].values
        == np.array([59, 60, 61])
    ).all()

    # --------------------------------------------------------
    # Independently calculate expected February 29 value
    # --------------------------------------------------------

    expected_angle = (
        2
        * np.pi
        * (60 - 1)
        / 366
    )

    expected_sin = np.sin(expected_angle)
    expected_cos = np.cos(expected_angle)

    actual_sin = dataset.sel(
        time="2024-02-29"
    )["day_of_year_sin"].item()

    actual_cos = dataset.sel(
        time="2024-02-29"
    )["day_of_year_cos"].item()

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    assert np.isclose(
        actual_sin,
        expected_sin,
        atol=1e-6,
    )

    assert np.isclose(
        actual_cos,
        expected_cos,
        atol=1e-6,
    )

    print(
        "\nLeap-Year Cyclical Encoding: PASS"
    )