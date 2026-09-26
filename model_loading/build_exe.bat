@echo off
rem Builds dist\ModelPortal.exe. Run this on Windows, from this folder.
rem Needs Python 3.10+ from python.org (tick "Add python.exe to PATH").

cd /d "%~dp0"

rem Prefer the "py" launcher that python.org installs: plain "python" can be
rem the Microsoft Store shortcut, which is not Python.
set PYTHON=python
where py >nul 2>nul && set PYTHON=py -3

if not exist .venv-win (
    echo Creating the build environment...
    %PYTHON% -m venv .venv-win || goto :fail
)
.venv-win\Scripts\python -m pip install --upgrade pip || goto :fail
.venv-win\Scripts\python -m pip install -e ".[exe]" || goto :fail
.venv-win\Scripts\pyinstaller --noconfirm --clean --distpath dist --workpath build packaging\ModelPortal.spec || goto :fail

echo.
echo Built: %cd%\dist\ModelPortal.exe
echo Copy that one file to any Windows computer and double-click it.
pause
exit /b 0

:fail
echo.
echo The build failed. See the messages above.
pause
exit /b 1
