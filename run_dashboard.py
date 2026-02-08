"""
Dashboard Launcher Script

This script provides a convenient way to launch the Streamlit dashboard.
It handles path resolution for both normal execution and bundled executables.

Usage:
    python run_dashboard.py
    
Or make it executable and run directly:
    chmod +x run_dashboard.py
    ./run_dashboard.py
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    """
    Launch the Streamlit dashboard application.
    
    This function:
    1. Locates the app.py file relative to this script
    2. Constructs the Streamlit command with appropriate flags
    3. Launches the Streamlit server
    
    Supports both normal Python execution and PyInstaller-bundled executables.
    """
    # Locate app.py relative to this file (works in normal + bundled modes)
    if getattr(sys, "frozen", False):
        # PyInstaller / frozen executable mode
        base_path = Path(sys.executable).parent
    else:
        # Normal Python execution mode
        base_path = Path(__file__).parent

    app_path = base_path / "app.py"
    
    # Verify app.py exists
    if not app_path.exists():
        print(f"Error: app.py not found at {app_path}")
        print("Please ensure app.py is in the same directory as this script.")
        sys.exit(1)

    # Streamlit command with configuration flags
    cmd = [
        sys.executable,
        "-m", "streamlit",
        "run",
        str(app_path),
        "--server.headless=false",      # Show browser automatically
        "--browser.gatherUsageStats=false",  # Disable usage statistics
    ]

    # Start streamlit server
    print("Starting Streamlit dashboard...")
    print(f"Dashboard will open at: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.Popen(cmd)
        # Streamlit keeps the process alive, so we don't need to block here
    except Exception as e:
        print(f"Error launching dashboard: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Streamlit is installed: pip install streamlit")
        print("2. Verify app.py exists in the current directory")
        print("3. Check that all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
