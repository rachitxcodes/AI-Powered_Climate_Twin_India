from dataclasses import dataclass, field
import pandas as pd


@dataclass(slots=True)
class FeatureEngineeringResult:
    """
    Stores the output of the Feature Engineering pipeline.
    """

    engineered_data: pd.DataFrame

    generated_features: list[str] = field(default_factory=list)