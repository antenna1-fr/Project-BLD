Param(
  [switch]$NoVenv
)

$ErrorActionPreference = 'Stop'

Write-Host 'Setting up Python environment for BLD...' -ForegroundColor Cyan

# Resolve repo root
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

if (-not $NoVenv) {
  $venvPath = Join-Path $RepoRoot 'venv'
  if (-not (Test-Path $venvPath)) {
    Write-Host 'Creating virtual environment (venv) ...'
    python -m venv $venvPath
  }
  Write-Host 'Activating venv ...'
  . "$venvPath\Scripts\Activate.ps1"
}

Write-Host 'Upgrading pip ...'
python -m pip install --upgrade pip

Write-Host 'Installing requirements ...'
python -m pip install -r requirements.txt

Write-Host 'Done.' -ForegroundColor Green

