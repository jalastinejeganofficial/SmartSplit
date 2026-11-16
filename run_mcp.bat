@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
set PYTHONPATH=%SCRIPT_DIR%
python -m smartsplit.server
endlocal
