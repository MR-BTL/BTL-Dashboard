"""Data export utilities."""

from io import BytesIO
import pandas as pd


def to_excel_download(sheets: dict[str, pd.DataFrame], filename="export.xlsx") -> BytesIO:
    """Create Excel file from multiple DataFrames."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name[:31], index=False)
    output.seek(0)
    return output
