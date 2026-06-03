$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$config = Join-Path $repoRoot "experiment\configs\followup_cpu_rapid.yaml"
$runDir = Join-Path $repoRoot "experiment\runs\followup_cpu_rapid_20m"

if (-not (Test-Path $python)) {
    throw "Virtual environment not found. Run scripts\setup_windows_xpu.ps1 first."
}

Write-Host "Running rapid CPU follow-up profile"
Write-Host "Config: $config"
Write-Host "Run dir: $runDir"

& $python -m tinystories_xpu.prepare_data --config $config
& $python -m tinystories_xpu.train --config $config
& $python -m tinystories_xpu.telemetry --run-dir $runDir
& $python -m tinystories_xpu.analyze --run-dir $runDir

Write-Host "Done. See: $runDir"
