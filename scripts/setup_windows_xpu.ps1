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
$verifyExit = $LASTEXITCODE

if ($verifyExit -ne 0) {
    Write-Warning "torch.xpu verification failed. Falling back to CPU-only PyTorch wheels."
    & $python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) {
        throw "CPU fallback installation failed."
    }

    & $python -c "import torch; print('cpu fallback torch:', torch.__version__); print('cuda available:', torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) {
        throw "CPU fallback verification failed."
    }

    Write-Warning "XPU is unavailable on this machine. Training will run on CPU unless you enable XPU later."
}

Write-Host ""
Write-Host "Done. Activate with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
