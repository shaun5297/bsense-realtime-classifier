@echo off
setlocal
set "PROJECT_DIR=%~dp0"
set "ENV_PYTHON=D:\ProgramData\miniforge3\envs\bci-gpu\python.exe"
if exist "%ENV_PYTHON%" (
  set "PYTHON_EXE=%ENV_PYTHON%"
) else (
  set "PYTHON_EXE=python"
)
set "PYTHONPATH=%PROJECT_DIR%src"
"%PYTHON_EXE%" -m bsense_classifier.app %*
endlocal
