"""Data type optimization and performance utilities."""

import pandas as pd


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame data types to reduce memory usage."""
    if df.empty:
        return df
    
    df = df.copy()
    
    for col in df.select_dtypes(include=['object']).columns:
        if len(df) > 0 and df[col].nunique() / len(df) < 0.5:
            try:
                df[col] = df[col].astype('category')
            except (TypeError, ValueError, KeyError):
                pass
    
    for col in df.select_dtypes(include=['int64']).columns:
        try:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        except (TypeError, ValueError, KeyError):
            pass
    
    for col in df.select_dtypes(include=['float64']).columns:
        try:
            df[col] = pd.to_numeric(df[col], downcast='float')
        except (TypeError, ValueError, KeyError):
            pass
    
    return df


def limit_display_rows(df: pd.DataFrame, max_rows: int = 1000) -> pd.DataFrame:
    """Limit DataFrame rows for display to improve UI performance."""
    if df.empty or len(df) <= max_rows:
        return df
    return df.head(max_rows)


def optimize_chart_data(df: pd.DataFrame, max_points: int = 1000) -> pd.DataFrame:
    """Limit data points for charts to improve rendering performance."""
    if df.empty or len(df) <= max_points:
        return df
    if len(df) > max_points:
        step = len(df) // max_points
        return df.iloc[::step].head(max_points)
    return df


def safe_fillna(df: pd.DataFrame, value=0) -> pd.DataFrame:
    """Safely fill NaN values, handling categorical columns properly."""
    if df.empty:
        return df
    df = df.copy()
    for col in df.columns:
        if df[col].dtype.name == 'category':
            try:
                if pd.api.types.is_numeric_dtype(df[col].cat.categories.dtype):
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(value)
                else:
                    df[col] = df[col].astype('object')
            except (AttributeError, TypeError):
                df[col] = df[col].astype('object')
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(value)
    return df
