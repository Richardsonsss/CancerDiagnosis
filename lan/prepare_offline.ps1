# One-time preparation of the local diagnosis computer (Windows). Run it while the computer is
# online; afterwards the diagnosis service runs without internet (start_lan.ps1).
#
#   pwsh lan/prepare_offline.ps1                  # CPU (INT8 members where faster)
#   pwsh lan/prepare_offline.ps1 -Gpu             # any DirectX 12 GPU (NVIDIA, AMD, Intel) via DirectML
#   pwsh lan/prepare_offline.ps1 -Wheelhouse      # also save every package into lan/wheels so that a
#                                                 # computer that is never online can be set up from it
#   pwsh lan/prepare_offline.ps1 -FromWheelhouse  # set up this (offline) computer from lan/wheels
#
# Needs Python 3.10+ (python.org or the Microsoft Store). The web app (web/dist) is part of the release
# bundle; it is only rebuilt when Node.js is installed and web/dist is missing.
param([switch]$Gpu, [switch]$Wheelhouse, [switch]$FromWheelhouse)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$venv = Join-Path $PSScriptRoot ".venv"
$wheels = Join-Path $PSScriptRoot "wheels"
$req = Join-Path $root ("server\requirements" + ($(if ($Gpu) { "-directml" } else { "" })) + ".txt")

$python = (Get-Command python -ErrorAction SilentlyContinue) ?? (Get-Command py -ErrorAction SilentlyContinue)
if (-not $python) { throw "Python 3.10 or newer is required (https://www.python.org/downloads/)" }
if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) { & $python.Source -m venv $venv }
$py = Join-Path $venv "Scripts\python.exe"

if ($FromWheelhouse) {
    & $py -m pip install --no-index --find-links $wheels -r $req
} else {
    & $py -m pip install --upgrade pip
    & $py -m pip install -r $req
    if ($Wheelhouse) {
        & $py -m pip download -r $req -d $wheels
        "packages saved to $wheels - copy the whole project folder to the offline computer and run: pwsh lan/prepare_offline.ps1 -FromWheelhouse"
    }
}

$dist = Join-Path $root "web\dist"
if (-not (Test-Path (Join-Path $dist "index.html"))) {
    if (Get-Command npm -ErrorAction SilentlyContinue) {
        Push-Location (Join-Path $root "web"); npm ci; npm run build; Pop-Location
    } else { throw "web\dist is missing and Node.js is not installed - use the release bundle, which contains web\dist" }
}
New-Item -ItemType Directory -Force (Join-Path $root "models") | Out-Null
& $py -c "import onnxruntime as o; print('ONNX Runtime', o.__version__, '| providers:', ', '.join(o.get_available_providers()))"
"ready - put the *_package.zip files into $(Join-Path $root 'models') and run: pwsh lan/start_lan.ps1"
