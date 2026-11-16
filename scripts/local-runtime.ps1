function Enter-ProjectLauncherLock {
    param([Parameter(Mandatory = $true)][string]$ProjectRoot)
    $hasher = [System.Security.Cryptography.SHA256]::Create()
    try {
        $canonical = [System.IO.Path]::GetFullPath($ProjectRoot).TrimEnd('\').ToLowerInvariant()
        $digest = $hasher.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($canonical))
        $name = 'Local\DouyinKnowledgeLibrary-' + [BitConverter]::ToString($digest).Replace('-', '')
    }
    finally { $hasher.Dispose() }
    $guard = New-Object System.Threading.Mutex($false, $name)
    try {
        try { $acquired = $guard.WaitOne(0) }
        catch [System.Threading.AbandonedMutexException] { $acquired = $true }
        if (-not $acquired) { throw 'project_launcher_already_running' }
        return $guard
    }
    catch { $guard.Dispose(); throw }
}

function Import-ProviderUserEnvironment {
    param([scriptblock]$ReadUserValue = {
        param($Name)
        [Environment]::GetEnvironmentVariable($Name, 'User')
    })
    foreach ($name in @(
        'DKL_NOTE_PROVIDER_POLICY', 'DKL_NOTE_EXTERNAL_BASE_URL',
        'DKL_NOTE_EXTERNAL_MODEL', 'DKL_NOTE_EXTERNAL_API_KEY'
    )) {
        if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name, 'Process'))) {
            $value = & $ReadUserValue $name
            if (-not [string]::IsNullOrWhiteSpace($value)) {
                [Environment]::SetEnvironmentVariable($name, $value, 'Process')
            }
        }
    }
}

function Assert-NoProjectWorkers {
    param([Parameter(Mandatory = $true)][string]$PythonPath)
    $workers = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
        $_.ExecutablePath -ieq $PythonPath -and
        $_.CommandLine -match '-m\s+douyin_library\.worker_main(?:\s|$)'
    })
    if ($workers.Count -gt 0) {
        throw "project_workers_already_running: $($workers.ProcessId -join ', '). Reuse the running services or stop their launcher before restarting."
    }
}
