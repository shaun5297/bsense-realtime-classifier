$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$environmentPython = "D:\ProgramData\miniforge3\envs\bci-gpu\python.exe"
$pythonExecutable = if (Test-Path -LiteralPath $environmentPython) {
    $environmentPython
} else {
    "python"
}
$env:PYTHONPATH = Join-Path $projectDir "src"
& $pythonExecutable -m bsense_classifier.app @args
