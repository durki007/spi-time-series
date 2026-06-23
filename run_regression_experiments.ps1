#Requires -Version 5.1
<#
.SYNOPSIS
    Run all regression experiment configs in experiments/regression/.
.PARAMETER Force
    Pass --force to each run, bypassing cached checkpoints.
.PARAMETER Jobs
    Number of parallel jobs for hyperparameter search (passed as --jobs).
#>
param(
    [switch]$Force,
    [int]$Jobs = -1
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = $PSScriptRoot
$VenvActivate = Join-Path $Root ".venv\Scripts\Activate.ps1"
$ExperimentsDir = Join-Path $Root "experiments\regression"

# ---------------------------------------------------------------------------
# Activate virtual environment
# ---------------------------------------------------------------------------
if (-not (Test-Path $VenvActivate)) {
    Write-Error "Virtual environment not found at $VenvActivate"
    exit 1
}
& $VenvActivate

# ---------------------------------------------------------------------------
# Collect configs
# ---------------------------------------------------------------------------
$Configs = Get-ChildItem -Path $ExperimentsDir -Filter "*.yaml" | Sort-Object Name

if ($Configs.Count -eq 0) {
    Write-Warning "No YAML configs found in $ExperimentsDir"
    exit 0
}

Write-Host ""
Write-Host "Found $($Configs.Count) experiment(s):" -ForegroundColor Cyan
$Configs | ForEach-Object { Write-Host "  - $($_.Name)" }
Write-Host ""

# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------
$Results = @()
$TotalStart = Get-Date

foreach ($Config in $Configs) {
    $Stem       = $Config.BaseName
    $OutputDir  = Join-Path $Root "output\$Stem"
    $ExtraArgs  = @()
    if ($Force)      { $ExtraArgs += "--force" }
    if ($Jobs -ne 0) { $ExtraArgs += "--jobs"; $ExtraArgs += "$Jobs" }

    Write-Host ("=" * 70) -ForegroundColor DarkGray
    Write-Host "Running: $($Config.Name)" -ForegroundColor Yellow
    Write-Host "Output:  $OutputDir"
    Write-Host ("=" * 70) -ForegroundColor DarkGray

    $Start = Get-Date
    $ExitCode = 0

    try {
        python -m spi_time_series.main $Config.FullName --output-dir $OutputDir @ExtraArgs
        $ExitCode = $LASTEXITCODE
    } catch {
        $ExitCode = 1
        Write-Warning "Exception: $_"
    }

    $Elapsed = (Get-Date) - $Start
    $Status  = if ($ExitCode -eq 0) { "OK" } else { "FAILED" }
    $Color   = if ($ExitCode -eq 0) { "Green" } else { "Red" }

    Write-Host ""
    Write-Host "[$Status] $($Config.Name)  ($([int]$Elapsed.TotalSeconds)s)" -ForegroundColor $Color
    Write-Host ""

    $Results += [PSCustomObject]@{
        Config  = $Config.Name
        Status  = $Status
        Seconds = [int]$Elapsed.TotalSeconds
        Output  = $OutputDir
    }
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
$TotalElapsed = (Get-Date) - $TotalStart
Write-Host ("=" * 70) -ForegroundColor DarkGray
Write-Host "SUMMARY  (total: $([int]$TotalElapsed.TotalSeconds)s)" -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor DarkGray

$Results | ForEach-Object {
    $Color = if ($_.Status -eq "OK") { "Green" } else { "Red" }
    Write-Host ("  [{0,-6}]  {1,-40}  {2,4}s" -f $_.Status, $_.Config, $_.Seconds) -ForegroundColor $Color
}

$Failed = $Results | Where-Object { $_.Status -ne "OK" }
if ($Failed) {
    Write-Host ""
    Write-Host "$($Failed.Count) experiment(s) failed." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "All experiments completed successfully." -ForegroundColor Green
