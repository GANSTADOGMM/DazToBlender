@echo off
REM Start DazConv and open it in your browser.  Windows.
REM Nothing to install beyond Python 3.8+ from python.org or the Microsoft Store.
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    py -3 -m dazconv serve %*
    goto :end
)
where python >nul 2>nul
if %errorlevel%==0 (
    python -m dazconv serve %*
    goto :end
)

echo DazConv needs Python 3.8 or newer.
echo Install it from https://www.python.org/downloads/ and run this again.
pause
:end
endlocal
