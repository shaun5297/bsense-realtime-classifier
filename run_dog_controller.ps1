$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$preferredPython = "D:\ProgramData\miniforge3\envs\bci-gpu\python.exe"

if (Test-Path $preferredPython) {
    $python = $preferredPython
} else {
    $python = "python"
}

$env:PYTHONPATH = Join-Path $projectRoot "src"
& $python -m bsense_classifier.robot_app @args
