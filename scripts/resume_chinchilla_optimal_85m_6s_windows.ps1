$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$config = Join-Path $repoRoot "experiment\configs\chinchilla_optimal_85m_6s.yaml"
$runDir = Join-Path $repoRoot "experiment\runs\chinchilla_optimal_85m_6s"
$metaPath = Join-Path $repoRoot "experiment\data\tinystories_smoke\meta.json"

$allSeeds = @(101, 202, 303, 404, 505, 606)
$targetTokens = 85000000

if (-not (Test-Path $python)) {
    throw "Virtual environment not found. Run scripts\setup_windows_xpu.ps1 first."
}

if (-not (Test-Path $metaPath)) {
    Write-Host "Prepared data cache missing. Running prepare stage first..."
    & $python -m tinystories_xpu.prepare_data --config $config
}

function Get-LatestCheckpointTokens {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SeedDir,
        [Parameter(Mandatory = $true)]
        [int]$Seed
    )

    $checkpointDir = Join-Path $SeedDir "checkpoints"
    if (-not (Test-Path $checkpointDir)) {
        return 0
    }

    $files = Get-ChildItem -Path $checkpointDir -Filter "seed_${Seed}_tokens_*.pt" -ErrorAction SilentlyContinue
    if (-not $files) {
        return 0
    }

    $latest = 0
    foreach ($file in $files) {
        if ($file.Name -match "_tokens_(\d+)\.pt$") {
            $tokens = [int64]$Matches[1]
            if ($tokens -gt $latest) {
                $latest = $tokens
            }
        }
    }
    return $latest
}

$resumeSeeds = @()
foreach ($seed in $allSeeds) {
    $seedDir = Join-Path $runDir "seed_$seed"
    $statusPath = Join-Path $seedDir "status.json"

    $statusStage = ""
    $statusTokens = 0
    if (Test-Path $statusPath) {
        try {
            $status = Get-Content $statusPath -Raw | ConvertFrom-Json
            if ($null -ne $status.stage) { $statusStage = [string]$status.stage }
            if ($null -ne $status.tokens_seen) { $statusTokens = [int64]$status.tokens_seen }
        }
        catch {
            Write-Host "Warning: could not parse $statusPath. Treating seed $seed as incomplete."
        }
    }

    $checkpointTokens = Get-LatestCheckpointTokens -SeedDir $seedDir -Seed $seed
    $bestTokens = [Math]::Max($statusTokens, $checkpointTokens)
    $isDone = ($statusStage -eq "done") -and ($bestTokens -ge $targetTokens)

    if (-not $isDone) {
        $resumeSeeds += $seed
    }

    Write-Host "seed=$seed stage=$statusStage tokens=$bestTokens done=$isDone"
}

if ($resumeSeeds.Count -eq 0) {
    Write-Host "All seeds already reached target tokens. Rebuilding telemetry + analysis only..."
}
else {
    $seedArgs = $resumeSeeds -join " "
    Write-Host "Resuming incomplete seeds: $seedArgs"
    & $python -m tinystories_xpu.train --config $config --seeds $resumeSeeds
}

& $python -m tinystories_xpu.telemetry --run-dir $runDir
& $python -m tinystories_xpu.analyze --run-dir $runDir

Write-Host "Done. See: $runDir"
