# PyInstaller spec for the Remedy PDF Desktop backend sidecar.
# Build with: pyinstaller --clean backend/project_remedy_backend.spec
#
# Produces a directory (onedir mode) with:
#   remedy-pdf-desktop-backend  -- launcher executable
#   _internal/                   -- Python runtime + deps + bundled libs
# Tauri registers the executable as an externalBin sidecar.

from pathlib import Path

project_root = Path(SPECPATH).parent.resolve()
lib_dir = project_root / "lib"

a = Analysis(
    [str(project_root / "backend" / "server_entry.py")],
    pathex=[str(project_root), str(lib_dir)],
    binaries=[],
    datas=[
        (str(lib_dir / "project_remedy"), "lib/project_remedy"),
    ],
    hiddenimports=[
        "backend.app.main",
        "backend.app.config",
        "backend.app.remediation",
        "backend.app.routes_api",
        "project_remedy",
        "project_remedy.pdf_fixer",
        "project_remedy.pdf_vision",
        "project_remedy.pdf_checker",
        "project_remedy.compliance_report",
        "project_remedy.office_remediator",
        "project_remedy.faithful_rebuild",
        "project_remedy.math.extractor",
        "project_remedy.math.converter",
        "project_remedy.math.tagger",
        "latex2mathml",
        "latex2mathml.converter",
        "pikepdf",
        "pypdf",
        "fitz",
        "fontTools",
        "docx",
        "pptx",
        "openpyxl",
        "bs4",
        "httpx",
        "requests",
        "uvicorn",
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "notebook"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="remedy-pdf-desktop-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="remedy-pdf-desktop-backend",
)
