$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceRoot = Split-Path -Parent $projectRoot
$python = "D:\ProgramData\miniforge3\envs\bci-gpu\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}

$env:PYTHONPATH = Join-Path $projectRoot "src"
$dataRoot = Join-Path $workspaceRoot "data\bsense"
$outputDir = Join-Path $projectRoot "models\m7_p300"

& $python -m bsense_classifier.p300_training `
    --data-root $dataRoot `
    --output-dir $outputDir

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "M7 P300 model ready:"
Write-Host (Join-Path $outputDir "model.joblib")
