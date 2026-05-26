param(
    [string]$PythonLauncher = "py -3.11",
    [switch]$NightlyTorch
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $repoRoot ".venv"
$experimentPath = Join-Path $repoRoot "experiment"

Write-Host "Creating Python virtual environment at $venvPath"
if (-not (Test-Path $venvPath)) {
    Invoke-Expression "$PythonLauncher -m venv `"$venvPath`""
}

$python = Join-Path $venvPath "Scripts\python.exe"

Write-Host "Upgrading pip"
& $python -m pip install --upgrade pip setuptools wheel

if ($NightlyTorch) {
    Write-Host "Installing PyTorch XPU nightly wheels"
    & $python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu
}
else {
    Write-Host "Installing PyTorch XPU stable wheels"
    & $python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
}

Write-Host "Installing experiment package"
& $python -m pip install -e $experimentPath

Write-Host "Verifying torch.xpu availability"
& $python -m tinystories_xpu.verify_xpu

Write-Host ""
Write-Host "Done. Activate with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
