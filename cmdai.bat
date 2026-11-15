@echo off
setlocal
set "CMDAI_ROOT=%~dp0"
if "%CMDAI_ROOT:~-1%"=="\" set "CMDAI_ROOT=%CMDAI_ROOT:~0,-1%"
set "PYTHONIOENCODING=utf-8"
python "%~dp0cmdai.py" %*
endlocal & exit /b %ERRORLEVEL%
