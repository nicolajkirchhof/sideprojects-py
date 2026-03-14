# Pull-AllDoltDatabases.ps1
# Goes into finance/_data and pulls the latest data for all Dolt databases found there.

$ErrorActionPreference = "Stop"

# Resolve repo-relative path (assumes script is run from repo root; adjust if needed)
$dataDir = Join-Path -Path (Get-Location) -ChildPath "finance\_data"

if (-not (Test-Path -LiteralPath $dataDir)) {
  throw "Path not found: $dataDir"
}

# Ensure dolt is available
if (-not (Get-Command dolt -ErrorAction SilentlyContinue)) {
  throw "Dolt executable not found on PATH. Install Dolt or add it to PATH, then re-run."
}

Push-Location $dataDir
try {
  # A Dolt database is a directory containing a .dolt folder
  $dbDirs = Get-ChildItem -Directory | Where-Object {
    Test-Path -LiteralPath (Join-Path $_.FullName ".dolt")
  }

  if (-not $dbDirs) {
    Write-Host "No Dolt databases found under: $dataDir"
    exit 0
  }

  Write-Host "Found $($dbDirs.Count) Dolt database(s) under: $dataDir"
  Write-Host ""

  foreach ($db in $dbDirs) {
    Write-Host "=== Pulling: $($db.Name) $($db.FullName) ==="
    Push-Location $db.FullName
    try {
      # Pull latest from the default remote/branch configured for this repo
      dolt --doltcfg-dir . pull

      if ($LASTEXITCODE -ne 0) {
        throw "dolt pull failed (exit code $LASTEXITCODE)"
      }
    }
    finally {
      Pop-Location
    }
    Write-Host ""
  }

  Write-Host "Done."
}
finally {
  Pop-Location
}
