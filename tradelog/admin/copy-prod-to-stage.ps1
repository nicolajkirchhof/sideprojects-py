<#
.SYNOPSIS
    Copies all data from the production Tradelog database to the staging database using BCP.
    Works with any two SQL Server instances (local, remote, Azure SQL).

.PARAMETER ProdServer
    Production SQL Server host. Default: 127.0.0.1,1433

.PARAMETER ProdDatabase
    Production database name. Default: Tradelog_Prod

.PARAMETER StageServer
    Staging SQL Server host. Default: 127.0.0.1,1433

.PARAMETER StageDatabase
    Staging database name. Default: Tradelog_Stage

.PARAMETER User
    SQL Server username. Default: sa

.PARAMETER Password
    SQL Server password.

.EXAMPLE
    .\copy-prod-to-stage.ps1 -Password "mypassword"
    .\copy-prod-to-stage.ps1 -ProdServer "prod.database.windows.net" -StageServer "127.0.0.1,1433" -Password "mypassword"
#>
param(
    [string]$ProdServer = "127.0.0.1,1433",
    [string]$ProdDatabase = "Tradelog_Prod",
    [string]$StageServer = "127.0.0.1,1433",
    [string]$StageDatabase = "Tradelog_Stage",
    [string]$User = "sa",
    [Parameter(Mandatory)][string]$Password
)

$ErrorActionPreference = "Stop"

# Verify tools
if (-not (Get-Command "sqlcmd" -ErrorAction SilentlyContinue)) {
    throw "sqlcmd not found. Install via: winget install Microsoft.SqlCmd"
}
if (-not (Get-Command "bcp" -ErrorAction SilentlyContinue)) {
    throw "bcp not found. Install SQL Server command-line utilities."
}

$TempDir = Join-Path $env:TEMP "tradelog-copy-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
New-Item -ItemType Directory -Path $TempDir -Force | Out-Null

Write-Host "=== Tradelog: Copy Production -> Staging ===" -ForegroundColor Cyan
Write-Host "Source: $ProdServer / $ProdDatabase"
Write-Host "Target: $StageServer / $StageDatabase"
Write-Host "Temp:   $TempDir"
Write-Host ""

# Get list of user tables from production
$tablesQuery = "SET NOCOUNT ON; SELECT TABLE_SCHEMA + '.' + TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_NAME != '__EFMigrationsHistory' ORDER BY TABLE_NAME"
$tables = sqlcmd -S $ProdServer -d $ProdDatabase -U $User -P $Password -Q $tablesQuery -h -1 -W
if ($LASTEXITCODE -ne 0) { throw "Failed to query tables from production" }

$tables = $tables | Where-Object { $_.Trim() -ne "" }
Write-Host "Found $($tables.Count) tables to copy." -ForegroundColor Yellow
Write-Host ""

# Disable FK constraints on staging
Write-Host "Disabling foreign key constraints on staging..." -ForegroundColor Gray
sqlcmd -S $StageServer -d $StageDatabase -U $User -P $Password -Q "EXEC sp_MSforeachtable 'ALTER TABLE ? NOCHECK CONSTRAINT ALL'" 2>$null

foreach ($table in $tables) {
    $table = $table.Trim()
    $safeName = $table -replace '[\[\].]', '_'
    $dataFile = Join-Path $TempDir "$safeName.dat"

    Write-Host "  $table" -NoNewline

    # Truncate staging table
    sqlcmd -S $StageServer -d $StageDatabase -U $User -P $Password -Q "TRUNCATE TABLE $table" 2>$null
    if ($LASTEXITCODE -ne 0) {
        # TRUNCATE fails if table has FK references; use DELETE instead
        sqlcmd -S $StageServer -d $StageDatabase -U $User -P $Password -Q "DELETE FROM $table"
    }

    # Export from prod
    bcp $table out $dataFile -S $ProdServer -d $ProdDatabase -U $User -P $Password -n -q 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host " [EXPORT FAILED]" -ForegroundColor Red
        continue
    }

    # Check if file has data
    if (-not (Test-Path $dataFile) -or (Get-Item $dataFile).Length -eq 0) {
        Write-Host " (empty)" -ForegroundColor Gray
        continue
    }

    # Import to staging (BCP -E flag preserves identity values)
    $result = bcp $table in $dataFile -S $StageServer -d $StageDatabase -U $User -P $Password -n -q -E 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host " [IMPORT FAILED]" -ForegroundColor Red
    }
    else {
        # Extract row count from bcp output
        $rows = ($result | Select-String "(\d+) rows copied" | ForEach-Object { $_.Matches[0].Groups[1].Value })
        Write-Host " -> $rows rows" -ForegroundColor Green
    }
}

# Re-enable FK constraints on staging
Write-Host ""
Write-Host "Re-enabling foreign key constraints on staging..." -ForegroundColor Gray
sqlcmd -S $StageServer -d $StageDatabase -U $User -P $Password -Q "EXEC sp_MSforeachtable 'ALTER TABLE ? WITH CHECK CHECK CONSTRAINT ALL'" 2>$null

# Cleanup
Remove-Item $TempDir -Recurse -Force

Write-Host ""
Write-Host "=== Copy complete ===" -ForegroundColor Green
