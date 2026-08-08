$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BackendPython = Join-Path $ProjectRoot "backend\.venv\Scripts\python.exe"
$Ruff = Join-Path $ProjectRoot "backend\.venv\Scripts\ruff.exe"

function Assert-LastExitCode {
    param([string]$Step)

    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

Push-Location (Join-Path $ProjectRoot "backend")
try {
    & $BackendPython -m pytest -v
    Assert-LastExitCode "Backend tests"
    & $Ruff check src tests
    Assert-LastExitCode "Backend lint"
}
finally {
    Pop-Location
}

Push-Location (Join-Path $ProjectRoot "frontend")
try {
    npm test
    Assert-LastExitCode "Frontend tests"
    npm run build
    Assert-LastExitCode "Frontend build"
}
finally {
    Pop-Location
}

Write-Host "All project checks passed." -ForegroundColor Green
