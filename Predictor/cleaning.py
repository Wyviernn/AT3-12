import numpy as np
import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize NaN values and clamp numeric scores to [0, 100].

    - Replaces string 'NaN' with actual np.nan
    - Coerces numeric columns to numbers
    - Sets values <0 or >100 to NaN

    Returns a cleaned copy of the DataFrame.
    """
    df = df.copy()
    df.replace('NaN', np.nan, inplace=True)

    numeric_cols = [c for c in df.columns if c != 'Student_Name']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df.loc[(df[col] < 0) | (df[col] > 100), col] = np.nan

    return df
