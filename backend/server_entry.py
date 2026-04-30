"""Entry point for PyInstaller-frozen backend.

Run directly as a standalone binary; starts uvicorn in-process against
backend.app.main:app. Host/port can be overridden via env vars.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    # When frozen, _MEIPASS points at the temp extraction dir. Ensure
    # bundled Python can find the lib/project_remedy package and data dirs.
    if getattr(sys, "frozen", False):
        bundle_root = Path(sys._MEIPASS)  # type: ignore[attr-defined]
        lib_dir = bundle_root / "lib"
        if lib_dir.exists() and str(lib_dir) not in sys.path:
            sys.path.insert(0, str(lib_dir))

    host = os.getenv("PROJECT_REMEDY_BACKEND_HOST", "127.0.0.1")
    port = int(os.getenv("PROJECT_REMEDY_BACKEND_PORT", "8000"))

    import uvicorn
    from backend.app.main import app

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
