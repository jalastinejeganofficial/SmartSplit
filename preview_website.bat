@echo off
setlocal
echo Starting SmartSplit Website Preview...
cd /d "%~dp0\docs"
start "" index.html
echo Opened docs\index.html in default browser.
pause
