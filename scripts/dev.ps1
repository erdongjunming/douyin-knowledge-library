$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BackendPython = Join-Path $ProjectRoot "backend\.venv\Scripts\python.exe"

$backend = Start-Process -FilePath $BackendPython `
    -ArgumentList "-m", "uvicorn", "douyin_library.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload" `
    -WorkingDirectory (Join-Path $ProjectRoot "backend") `
    -PassThru

try {
    Push-Location (Join-Path $ProjectRoot "frontend")
    npm run dev
}
finally {
    Pop-Location
    if (-not $backend.HasExited) {
        Stop-Process -Id $backend.Id
    }
}

