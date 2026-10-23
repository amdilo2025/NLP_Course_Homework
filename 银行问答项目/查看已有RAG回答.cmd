@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "ICBC_REPLAY_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%ICBC_REPLAY_PYTHON%" (
  "%ICBC_REPLAY_PYTHON%" tools/show_saved_answer.py --details
) else (
  python tools/show_saved_answer.py --details
)
pause
