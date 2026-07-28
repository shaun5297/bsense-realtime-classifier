@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
set "PYTHONPATH=%PROJECT_ROOT%src"
set "PREFERRED_PYTHON=D:\ProgramData\miniforge3\envs\bci-gpu\python.exe"

if exist "%PREFERRED_PYTHON%" (
  "%PREFERRED_PYTHON%" -m bsense_classifier.robot_app %*
) else (
  python -m bsense_classifier.robot_app %*
)
