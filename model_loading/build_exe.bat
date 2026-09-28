@echo off
rem Builds dist\ModelPortal.exe. Run this on Windows, from this folder.
rem Needs Python 3.10+ from python.org (tick "Add python.exe to PATH").

cd /d "%~dp0"

rem The script builds from the files next to it. Run straight out of a ZIP, or
rem copied on its own, it finds none of them.
if not exist pyproject.toml goto :nofiles
if not exist packaging\ModelPortal.spec goto :nofiles

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

:nofiles
echo This script must sit in the model_loading folder, next to pyproject.toml
echo and the packaging and modelportal folders. It is running from:
echo   %cd%
echo If you downloaded a ZIP, right-click it, choose "Extract All", then run
echo build_exe.bat from the extracted model_loading folder.
pause
exit /b 1

:fail
echo.
echo The build failed. See the messages above.
pause
exit /b 1
