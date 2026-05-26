$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$config = Join-Path $repoRoot "experiment\configs\smoke.yaml"

if (-not (Test-Path $python)) {
    throw "Virtual environment not found. Run scripts\setup_windows_xpu.ps1 first."
}

& $python -m tinystories_xpu.prepare_data --config $config
& $python -m tinystories_xpu.train --config $config
& $python -m tinystories_xpu.telemetry --run-dir (Join-Path $repoRoot "experiment\runs\smoke_xpu")
& $python -m tinystories_xpu.analyze --run-dir (Join-Path $repoRoot "experiment\runs\smoke_xpu")
