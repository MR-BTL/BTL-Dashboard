"""General helper functions for data processing."""

import re
import pandas as pd
import numpy as np


def normalize_colname(c: str) -> str:
    """Normalize column names to snake_case."""
    c = str(c).strip()
    c = c.replace("\n", " ")
    c = re.sub(r"\s+", " ", c)
    c = c.lower().strip()
    c = c.replace("-", "_")
    c = c.replace(" ", "_")
    return c


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize DataFrame column names to snake_case."""
    if df is None or df.empty:
        return df
    df = df.copy()
    df.columns = [normalize_colname(c) for c in df.columns]
    return df


def clean_user_id(series: pd.Series) -> pd.Series:
    """Clean user ID series by removing trailing .0 and empty values."""
    if series is None:
        return pd.Series(dtype="string")
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.replace({"nan": "", "None": ""})
    return s


def to_datetime_any(s: pd.Series) -> pd.Series:
    """Convert series to datetime, handling multiple formats."""
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def pick_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name from candidates that exists in the DataFrame."""
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def ensure_numeric(df: pd.DataFrame, col: str, default=0.0) -> pd.Series:
    """Ensure a column is numeric, returning default if column doesn't exist."""
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype="float")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def safe_value_counts(df: pd.DataFrame, col: str, top=20) -> pd.DataFrame:
    """Safely get value counts for a column."""
    if df.empty or col not in df.columns:
        return pd.DataFrame(columns=[col, "count"])
    vc = df[col].astype(str).replace("nan", "").replace("None", "")
    vc = vc[vc.str.strip() != ""].value_counts().head(top).reset_index()
    vc.columns = [col, "count"]
    return vc
