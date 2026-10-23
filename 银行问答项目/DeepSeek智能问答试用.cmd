@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo DeepSeek real API mode: each question sends one paid request. Type exit to quit.
set "ICBC_LOCAL_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%ICBC_LOCAL_PYTHON%" (
  "%ICBC_LOCAL_PYTHON%" run_rag.py --config config/service_rag.json --mode api
) else (
  python run_rag.py --config config/service_rag.json --mode api
)
pause
