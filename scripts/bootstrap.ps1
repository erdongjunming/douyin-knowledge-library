$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ToolPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Uv = Join-Path $ProjectRoot ".venv\Scripts\uv.exe"
$env:UV_CACHE_DIR = Join-Path $ProjectRoot ".uv-cache"

if (-not (Test-Path -LiteralPath $Uv)) {
    python -m venv (Join-Path $ProjectRoot ".venv")
    & $ToolPython -m pip install uv
}

Push-Location (Join-Path $ProjectRoot "backend")
try {
    & $Uv sync --all-groups
}
finally {
    Pop-Location
}

Push-Location (Join-Path $ProjectRoot "frontend")
try {
    npm install
}
finally {
    Pop-Location
}

Write-Host "Project dependencies are ready." -ForegroundColor Green

