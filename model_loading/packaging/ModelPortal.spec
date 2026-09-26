# PyInstaller recipe for a single-file ModelPortal executable.
#
# Build on the operating system you are building for — PyInstaller does not
# cross-compile. On Windows that produces dist\ModelPortal.exe. See
# build_exe.bat, or `make exe` on Linux/macOS.

from pathlib import Path

root = Path(SPECPATH).parent
package = root / "modelportal"

a = Analysis(
    [str(root / "packaging" / "launch.py")],
    pathex=[str(root)],
    datas=[
        (str(package / "static"), "modelportal/static"),
        (str(package / "templates"), "modelportal/templates"),
    ],
    hiddenimports=["modelportal.web"],
    excludes=["tkinter", "numpy", "pytest"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name="ModelPortal",
    # The console window is deliberate: it shows the address and it is the
    # off switch. Closing it stops the portal.
    console=True,
    upx=False,
)
