"""Data filtering utilities."""

import pandas as pd


def filter_date(df: pd.DataFrame, col: str, dr: tuple | None) -> pd.DataFrame:
    """Filter DataFrame by date range."""
    if df.empty or dr is None or col not in df.columns:
        return df
    start, end = dr
    if df[col].dtype == 'datetime64[ns]':
        mask = (df[col].dt.date >= start) & (df[col].dt.date <= end)
    else:
        s = pd.to_datetime(df[col], errors="coerce").dt.date
        mask = (s >= start) & (s <= end)
    return df.loc[mask]


def filter_in(df: pd.DataFrame, col: str, values: list) -> pd.DataFrame:
    """Filter DataFrame by column values. Empty list means no filter."""
    if df.empty or col not in df.columns:
        return df
    if not values:
        return df
    
    if df[col].dtype.name == 'category':
        values_set = set(str(v) for v in values)
        mask = df[col].astype(str).isin(values_set)
    else:
        values_set = set(str(v) for v in values)
        mask = df[col].astype(str).isin(values_set)
    return df.loc[mask]


def filter_text_contains(df: pd.DataFrame, col: str, q: str) -> pd.DataFrame:
    """Filter DataFrame by text contains."""
    if df.empty or col not in df.columns or not q:
        return df
    if df[col].dtype.name == 'category':
        mask = df[col].astype(str).str.contains(q, case=False, na=False)
    else:
        mask = df[col].astype(str).str.contains(q, case=False, na=False)
    return df.loc[mask]
