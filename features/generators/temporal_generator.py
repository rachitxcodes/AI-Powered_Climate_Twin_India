import pandas as pd


class TemporalGenerator:

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        df["year"] = df["Date"].dt.year
        df["month"] = df["Date"].dt.month
        df["day"] = df["Date"].dt.day
        df["day_of_year"] = df["Date"].dt.dayofyear

        return df