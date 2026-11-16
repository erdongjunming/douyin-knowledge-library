$ErrorActionPreference = 'Stop'
$taskRepo = Split-Path -Parent $PSScriptRoot
$taskRoot = 'D:\DouyinKnowledgeLibrary'
$taskPython = Join-Path $taskRepo 'backend\.venv\Scripts\python.exe'
$taskRunner = Join-Path $taskRepo 'scripts\upgrade_legacy_notes.py'
$taskInventory = Join-Path $taskRoot 'audit\legacy-upgrade-inventory-20260918-full.json'
$taskAudit = Join-Path $taskRoot 'audit\legacy-upgrade-batches-20260918-full.jsonl'
$env:DKL_NOTE_PROVIDER_POLICY = 'local'
$env:DKL_NOTE_OLLAMA_MODEL = 'qwen3:4b'
$env:DKL_NOTE_OLLAMA_URL = 'http://127.0.0.1:11434'
$env:DKL_STORAGE_ROOT = $taskRoot
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'

$taskRunnerProcesses = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -match 'upgrade_legacy_notes\.py'
}
if ($taskRunnerProcesses) {
    throw 'legacy_upgrade_already_running'
}

$taskStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$taskOut = Join-Path $taskRoot "audit\legacy-upgrade-run-resumed-$taskStamp.out.log"
$taskErr = Join-Path $taskRoot "audit\legacy-upgrade-run-resumed-$taskStamp.err.log"
$taskProcess = Start-Process -FilePath $taskPython `
    -ArgumentList '-u', $taskRunner, '--inventory', $taskInventory, '--audit', $taskAudit `
    -WorkingDirectory $taskRepo -WindowStyle Hidden `
    -RedirectStandardOutput $taskOut -RedirectStandardError $taskErr -PassThru
[pscustomobject]@{ pid = $taskProcess.Id; stdout = $taskOut; stderr = $taskErr } | ConvertTo-Json
