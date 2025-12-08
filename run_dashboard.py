import os
import sys
import subprocess
from pathlib import Path

def main():
    # Locate app.py relative to this file (works in normal + bundled modes)
    if getattr(sys, "frozen", False):
        # PyInstaller / frozen executable
        base_path = Path(sys.executable).parent
    else:
        base_path = Path(__file__).parent

    app_path = base_path / "app.py"

    # Streamlit command
    cmd = [
        sys.executable,
        "-m", "streamlit",
        "run",
        str(app_path),
        "--server.headless=false",
        "--browser.gatherUsageStats=false",
    ]

    # Start streamlit server
    subprocess.Popen(cmd)

    # Optional: on some systems, we may want to wait so the process doesn't exit immediately
    # But Streamlit keeps the process alive, so we don't block here.

if __name__ == "__main__":
    main()
