@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "ICBC_LOCAL_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%ICBC_LOCAL_PYTHON%" (
  "%ICBC_LOCAL_PYTHON%" run_rag.py --config config/service_rag.json --mode retrieve
) else (
  python run_rag.py --config config/service_rag.json --mode retrieve
)
pause
