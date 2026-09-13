$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
try {
    if (-not (Test-Path '.venv/Scripts/python.exe')) {
        if (Get-Command py -ErrorAction SilentlyContinue) {
            & py -3.11 -m venv .venv
        } elseif (Get-Command python -ErrorAction SilentlyContinue) {
            & python -c "import sys; assert sys.version_info[:2] == (3,11), 'Python 3.11 is required'"
            if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.11 (64-bit), then retry.' }
            & python -m venv .venv
        } else { throw 'Install Python 3.11 (64-bit) from python.org, then retry.' }
        if ($LASTEXITCODE -ne 0) { throw 'Cannot create Python 3.11 environment.' }
    }
    $stationPython = Join-Path (Get-Location) '.venv/Scripts/python.exe'
    & $stationPython -c "import sys; assert sys.version_info[:2] == (3,11), 'This installer requires Python 3.11'"
    if ($LASTEXITCODE -ne 0) { throw 'Existing .venv uses another Python version. Use a fresh clone.' }
    & $stationPython -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) { throw 'Cannot initialize pip.' }
    & $stationPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'Cannot update pip. Check internet access.' }
    & $stationPython -m pip install -r requirements-gpu.txt
    if ($LASTEXITCODE -ne 0) { throw 'GPU dependency installation failed.' }
    & $stationPython -m pip install -r requirements-windows-lock.txt
    if ($LASTEXITCODE -ne 0) { throw 'Application dependency installation failed.' }
    & $stationPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency verification failed.' }
    & $stationPython scripts/prepare_station.py
    if ($LASTEXITCODE -ne 0) { throw 'Preparation failed. Read the error above.' }
} catch {
    Write-Host $_ -ForegroundColor Red
    exit 1
}
