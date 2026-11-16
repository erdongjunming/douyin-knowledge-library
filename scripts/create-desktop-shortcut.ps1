param(
    [string]$DestinationDirectory = [Environment]::GetFolderPath("Desktop")
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Launcher = Join-Path $ProjectRoot "start-local.cmd"
$ShortcutName = -join [char[]]@(0x6296, 0x97F3, 0x77E5, 0x8BC6, 0x8D44, 0x6599, 0x5E93)
$ShortcutPath = Join-Path $DestinationDirectory ($ShortcutName + ".lnk")

if (-not (Test-Path -LiteralPath $Launcher)) {
    throw "Launcher not found: $Launcher"
}
if (-not (Test-Path -LiteralPath $DestinationDirectory -PathType Container)) {
    throw "Shortcut destination not found: $DestinationDirectory"
}

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($ShortcutPath)
$shortcut.TargetPath = $Launcher
$shortcut.WorkingDirectory = $ProjectRoot
$shortcut.Description = "Start the Douyin Knowledge Library"
$shortcut.IconLocation = "{0}\System32\shell32.dll,14" -f $env:SystemRoot
$shortcut.WindowStyle = 1
$shortcut.Save()

Write-Host "Desktop shortcut created: $ShortcutPath" -ForegroundColor Green
