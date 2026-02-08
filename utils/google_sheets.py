"""Google Sheets integration utilities."""

import re
import requests
from io import BytesIO
import pandas as pd


def convert_google_sheets_url(url: str) -> str | None:
    """Convert Google Sheets share URL to export URL."""
    pattern = r"/spreadsheets/d/([a-zA-Z0-9-_]+)"
    match = re.search(pattern, url)
    if not match:
        return None
    sheet_id = match.group(1)
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"


def load_file_from_url(url: str) -> BytesIO:
    """Download Google Sheets file as Excel."""
    export_url = convert_google_sheets_url(url)
    if not export_url:
        raise ValueError("Invalid Google Sheets URL format")
    resp = requests.get(export_url, timeout=60)
    resp.raise_for_status()
    return BytesIO(resp.content)


def safe_read_sheet(xls: pd.ExcelFile, name: str) -> pd.DataFrame:
    """Read Excel sheet with error handling."""
    try:
        df = pd.read_excel(xls, name, engine='openpyxl')
        return df
    except Exception:
        return pd.DataFrame()
