<#
run.ps1 - simple helper to create/use a venv and run main.py

Usage examples (from anywhere):
  # create venv if missing, install requirements, run
  Set-Location -Path "C:\Users\USER\OneDrive\Desktop\neural_network"; ./run.ps1 -InstallRequirements

  # just run (create venv if missing, but don't install requirements)
  Set-Location -Path "C:\Users\USER\OneDrive\Desktop\neural_network"; ./run.ps1

  # pass args to main.py
  ./run.ps1 -- --config .\config\hyperparams.json

Notes:
 - This script calls the venv python directly (no ActivationPolicy issues).
 - To run this script in PowerShell, you may need to allow script execution: Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
#>

param(
    [switch]$InstallRequirements,
    [string]$PythonPath = "python"
)

try {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
} catch {
    $scriptDir = Get-Location
}

Set-Location -Path $scriptDir
$venvPath = Join-Path $scriptDir ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path -Path $venvPath)) {
    Write-Host "Creating virtual environment at '$venvPath'..."
    & $PythonPath -m venv $venvPath
    if ($LASTEXITCODE -ne 0) { throw "Failed to create venv using '$PythonPath'" }
} else {
    Write-Host "Using existing virtual environment at '$venvPath'"
}

# Ensure pip/tools are up-to-date
Write-Host "Upgrading pip/setuptools/wheel in venv..."
& $venvPython -m pip install --upgrade pip setuptools wheel | Write-Host

if ($InstallRequirements) {
    $reqFile = Join-Path $scriptDir "requirements.txt"
    if (Test-Path $reqFile) {
        Write-Host "Installing requirements from $reqFile into venv..."
        & $venvPython -m pip install -r $reqFile
        if ($LASTEXITCODE -ne 0) { throw "Failed to install requirements" }
    } else {
        Write-Host "No requirements.txt found at $reqFile"
    }
}

# Run main.py and forward any remaining args (use -- to separate run.ps1 params from script args)
$mainFile = Join-Path $scriptDir "main.py"
if (-not (Test-Path $mainFile)) { throw "main.py not found in $scriptDir" }

Write-Host "Running main.py using venv python..."
& $venvPython $mainFile @args
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    Write-Host "main.py exited with code $exitCode" -ForegroundColor Yellow
} else {
    Write-Host "main.py finished successfully" -ForegroundColor Green
}

exit $exitCode
