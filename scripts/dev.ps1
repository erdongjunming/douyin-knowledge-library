$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BackendRoot = Join-Path $ProjectRoot "backend"
$FrontendRoot = Join-Path $ProjectRoot "frontend"
$RuntimeRoot = Join-Path $ProjectRoot ".runtime"
$LogRoot = Join-Path $RuntimeRoot "logs"
$BackendPython = Join-Path $BackendRoot ".venv\Scripts\python.exe"
. (Join-Path $PSScriptRoot "local-runtime.ps1")
Import-ProviderUserEnvironment
$BackendUrl = "http://127.0.0.1:8002"
$FrontendUrl = "http://127.0.0.1:5175"
$OllamaUrl = "http://127.0.0.1:11434"
# The launcher owns the service it starts, so bind it to the same loopback
# endpoint that readiness checks and the application use.
$env:OLLAMA_HOST = $OllamaUrl
$OllamaModel = if ([string]::IsNullOrWhiteSpace($env:DKL_NOTE_OLLAMA_MODEL)) { "qwen3:4b" } else { $env:DKL_NOTE_OLLAMA_MODEL }
$StopBackendOnExit = $env:DKL_STOP_BACKEND_ON_EXIT -ne "0"
$ManagedProcesses = @()

if (-not (Test-Path -LiteralPath $BackendPython)) {
    throw "Backend environment not found: $BackendPython"
}

New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null

function Invoke-LocalRequest {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url
    )

    $request = [System.Net.WebRequest]::Create($Url)
    $request.Proxy = $null
    $request.Timeout = 2000
    $response = $null
    $reader = $null
    try {
        $response = $request.GetResponse()
        $reader = New-Object System.IO.StreamReader($response.GetResponseStream())
        [pscustomobject]@{
            StatusCode = [int]$response.StatusCode
            Body = $reader.ReadToEnd()
        }
    }
    finally {
        if ($null -ne $reader) { $reader.Dispose() }
        if ($null -ne $response) { $response.Dispose() }
    }
}

function Test-BackendReady {
    try {
        $response = Invoke-LocalRequest "$BackendUrl/api/health"
        if ($response.StatusCode -ne 200 -or [string]::IsNullOrWhiteSpace($response.Body)) {
            return $false
        }
        $health = $response.Body | ConvertFrom-Json
        return $health.status -in @("ok", "degraded")
    }
    catch {
        return $false
    }
}

function Test-LocalEndpoint {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url
    )

    try {
        return (Invoke-LocalRequest $Url).StatusCode -eq 200
    }
    catch {
        return $false
    }
}

function Test-OllamaReady {
    try {
        $request = [System.Net.WebRequest]::Create("$OllamaUrl/api/tags")
        $request.Proxy = $null
        $request.Timeout = 2000
        $response = $request.GetResponse()
        try {
            $reader = New-Object System.IO.StreamReader($response.GetResponseStream())
            try {
                $payload = $reader.ReadToEnd() | ConvertFrom-Json
                return @($payload.models) | Where-Object {
                    $_.name -eq $OllamaModel -or $_.model -eq $OllamaModel
                } | Select-Object -First 1
            }
            finally { $reader.Dispose() }
        }
        finally { $response.Dispose() }
    }
    catch { return $false }
}

function Get-OllamaExecutable {
    $command = Get-Command ollama -ErrorAction SilentlyContinue
    if ($null -ne $command) { return $command.Source }
    $standard = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    if (Test-Path -LiteralPath $standard) { return $standard }
    return $null
}

function Wait-OllamaReady {
    if (Test-OllamaReady) {
        Write-Host "Ollama ready: $OllamaModel" -ForegroundColor Green
        return
    }
    $executable = Get-OllamaExecutable
    if ($null -eq $executable) {
        throw "Ollama executable not found. Install Ollama or set PATH before starting the project."
    }
    $script:ollamaProcess = Start-ManagedProcess `
        -FilePath $executable `
        -ArgumentList @("serve") `
        -WorkingDirectory (Split-Path -Parent $executable) `
        -Label "ollama"
    $script:ollamaStartedHere = $true
    Write-Host "Starting Ollama (PID $($ollamaProcess.Id))..." -ForegroundColor Yellow
    $deadline = (Get-Date).AddSeconds(60)
    while ((Get-Date) -lt $deadline) {
        if (Test-OllamaReady) {
            Write-Host "Ollama ready: $OllamaModel" -ForegroundColor Green
            return
        }
        Start-Sleep -Milliseconds 500
    }
    throw "Ollama did not expose $OllamaModel within 60 seconds. Run 'ollama list' to verify the local model."
}

function Get-PortOwner {
    param(
        [Parameter(Mandatory = $true)]
        [int]$Port
    )

    try {
        return Get-NetTCPConnection -State Listen -LocalAddress 127.0.0.1 -LocalPort $Port `
            -ErrorAction SilentlyContinue |
            Select-Object -First 1 -ExpandProperty OwningProcess
    }
    catch {
        return $null
    }
}

function Start-ManagedProcess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(Mandatory = $true)]
        [string[]]$ArgumentList,
        [Parameter(Mandatory = $true)]
        [string]$WorkingDirectory,
        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $stdout = Join-Path $LogRoot "$Label.$stamp.out.log"
    $stderr = Join-Path $LogRoot "$Label.$stamp.err.log"
    $process = Start-Process -FilePath $FilePath `
        -ArgumentList $ArgumentList `
        -WorkingDirectory $WorkingDirectory `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru
    $script:ManagedProcesses += [pscustomobject]@{
        Process = $process
        Label = $Label
        Stdout = $stdout
        Stderr = $stderr
    }
    return $process
}

function Stop-ProcessTree {
    param(
        [Parameter(Mandatory = $true)]
        [int]$ProcessId
    )

    if ($ProcessId -le 0) { return }
    try {
        & taskkill.exe /PID $ProcessId /T /F 2>$null | Out-Null
    }
    catch {
        Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
    }
}

function Wait-BackendReady {
    param(
        [Parameter(Mandatory = $true)]
        [System.Diagnostics.Process]$Process,
        [int]$TimeoutSeconds = 45
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-BackendReady) { return }
        if ($Process.HasExited) {
            throw "Backend exited before becoming ready (exit code $($Process.ExitCode))."
        }
        Start-Sleep -Milliseconds 500
    }
    throw "Backend did not become ready within $TimeoutSeconds seconds."
}

function Wait-EndpointReady {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url,
        [Parameter(Mandatory = $true)]
        [System.Diagnostics.Process]$Process,
        [int]$TimeoutSeconds = 45
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-LocalEndpoint $Url) { return }
        if ($Process.HasExited) {
            throw "Frontend exited before becoming ready (exit code $($Process.ExitCode))."
        }
        Start-Sleep -Milliseconds 500
    }
    throw "Frontend did not become ready within $TimeoutSeconds seconds."
}

$backendProcess = $null
$backendStartedHere = $false
$frontendProcess = $null
$ollamaProcess = $null
$ollamaStartedHere = $false

$LauncherLock = Enter-ProjectLauncherLock -ProjectRoot $ProjectRoot
try {
    # Serialize the check and all subsequent startup work across launchers.
    Assert-NoProjectWorkers -PythonPath $BackendPython
    Wait-OllamaReady
    if (Test-BackendReady) {
        Write-Host "Backend already ready: $BackendUrl" -ForegroundColor Green
    }
    else {
        $backendOwner = Get-PortOwner 8002
        if ($null -ne $backendOwner) {
            throw "Port 8002 is already occupied by PID $backendOwner, but /api/health is not ready."
        }
        $backendProcess = Start-ManagedProcess `
            -FilePath $BackendPython `
            -ArgumentList @("-m", "uvicorn", "douyin_library.main:app", "--host", "127.0.0.1", "--port", "8002", "--log-level", "info") `
            -WorkingDirectory $BackendRoot `
            -Label "backend"
        $backendStartedHere = $true
        Write-Host "Starting backend (PID $($backendProcess.Id))..." -ForegroundColor Yellow
        Wait-BackendReady -Process $backendProcess
        Write-Host "Backend ready: $BackendUrl" -ForegroundColor Green
    }

    $generalProcess = Start-ManagedProcess `
        -FilePath $BackendPython `
        -ArgumentList @("-m", "douyin_library.worker_main", "--pool", "general") `
        -WorkingDirectory $BackendRoot `
        -Label "worker-general"
    $asrProcess = Start-ManagedProcess `
        -FilePath $BackendPython `
        -ArgumentList @("-m", "douyin_library.worker_main", "--pool", "asr") `
        -WorkingDirectory $BackendRoot `
        -Label "worker-asr"
    Write-Host "Workers started (general PID $($generalProcess.Id), ASR PID $($asrProcess.Id))." -ForegroundColor Green

    if (Test-LocalEndpoint "$FrontendUrl/") {
        Write-Host "Frontend already ready: $FrontendUrl/" -ForegroundColor Green
        Start-Process $FrontendUrl
    }
    else {
        $frontendProcess = Start-ManagedProcess `
            -FilePath "npm.cmd" `
            -ArgumentList @("run", "dev", "--", "--host", "127.0.0.1", "--port", "5175") `
            -WorkingDirectory $FrontendRoot `
            -Label "frontend"
        Write-Host "Starting frontend (PID $($frontendProcess.Id))..." -ForegroundColor Yellow
        Wait-EndpointReady -Url "$FrontendUrl/" -Process $frontendProcess
        Write-Host "Frontend ready: $FrontendUrl/" -ForegroundColor Green
        Start-Process $FrontendUrl
    }

    Write-Host "Backend:  $BackendUrl/" -ForegroundColor Green
    Write-Host "Frontend: $FrontendUrl/" -ForegroundColor Green
    Write-Host "Press Ctrl+C to stop processes started by this launcher." -ForegroundColor Cyan

    while ($true) {
        Start-Sleep -Seconds 1
        foreach ($managed in $ManagedProcesses) {
            if ($managed.Process.HasExited) {
                throw "$($managed.Label) exited (code $($managed.Process.ExitCode)). Logs: $($managed.Stdout), $($managed.Stderr)"
            }
        }
    }
}
catch {
    Write-Error $_
    throw
}
finally {
    foreach ($managed in @($ManagedProcesses)) {
        $keepBackend = $managed.Label -eq "backend" -and -not $StopBackendOnExit
        if (-not $keepBackend -and $null -ne $managed.Process -and -not $managed.Process.HasExited) {
            Write-Host "Stopping $($managed.Label) process tree (PID $($managed.Process.Id))..." -ForegroundColor Yellow
            Stop-ProcessTree -ProcessId $managed.Process.Id
        }
    }
    if (-not $StopBackendOnExit -and $backendStartedHere) {
        Write-Host "DKL_STOP_BACKEND_ON_EXIT=0 requested; leaving backend running (PID $($backendProcess.Id))." -ForegroundColor Yellow
    }
    $LauncherLock.ReleaseMutex()
    $LauncherLock.Dispose()
}
