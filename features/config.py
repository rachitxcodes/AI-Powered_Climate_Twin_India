# features/config.py

from dataclasses import dataclass, field


@dataclass(slots=True)
class FeatureEngineeringConfig:
    """
    Configuration for the Feature Engineering pipeline.
    """

    enable_temporal_features: bool = True
    enable_lag_features: bool = True
    enable_rolling_features: bool = True
    enable_seasonal_features: bool = True

    lag_days: list[int] = field(default_factory=lambda: [1, 7, 30])

    rolling_windows: list[int] = field(default_factory=lambda: [7, 30])
    
    enable_lag_features: bool = True

    lag_days: tuple[int, ...] = (
        1,
        3,
        7,
    )