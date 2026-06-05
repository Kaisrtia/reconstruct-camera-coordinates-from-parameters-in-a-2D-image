from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import os


if __name__ == "__main__":
    app_path = Path(__file__).resolve().parents[1] / "src" / "camera_reconstruction" / "web" / "streamlit_app.py"
    port = os.environ.get("CAMERA_RECONSTRUCTION_PORT", "8501")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.address",
            "127.0.0.1",
            "--server.port",
            port,
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ],
        check=True,
    )
