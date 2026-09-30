# Start the diagnosis service on the local network (Windows) - no internet connection needed.
# The phone must be on the same network: a Wi-Fi router (internet not required), this computer's
# Mobile hotspot, or the iPhone's Personal Hotspot with this computer connected to it.
#
#   pwsh lan/start_lan.ps1                    # http://<address>:8000
#   pwsh lan/start_lan.ps1 -Https             # https with a local certificate (lets the phone install the app)
#   pwsh lan/start_lan.ps1 -OpenFirewall      # once, from an administrator PowerShell: allow the port on
#                                             # private networks in Windows Firewall
#   pwsh lan/start_lan.ps1 -Port 8080 -ModelsDir D:\models
param([switch]$Https, [int]$Port = 8000, [string]$ModelsDir = "", [switch]$OpenFirewall)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "not prepared yet - run once while online: pwsh lan/prepare_offline.ps1" }

if ($OpenFirewall) {
    $admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $admin) { throw "-OpenFirewall needs an administrator PowerShell (right-click > Run as administrator)" }
    $name = "Cancer Diagnosis LAN ($Port)"
    if (-not (Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -DisplayName $name -Direction Inbound -Protocol TCP -LocalPort $Port `
            -Profile Private, Domain -Action Allow | Out-Null
        "Windows Firewall: TCP port $Port allowed on private networks"
    }
}

$argsList = @((Join-Path $root "server\lan.py"), "--port", $Port)
if ($Https) { $argsList += "--https" }
if ($ModelsDir) { $argsList += @("--models-dir", $ModelsDir) }
$env:PYTHONIOENCODING = "utf-8"
& $py @argsList
