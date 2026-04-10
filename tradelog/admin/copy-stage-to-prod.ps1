<#
.SYNOPSIS
    Full clone of the staging Tradelog database over production using
    sqlpackage extract + clean + publish. Handles schema changes and
    data replacement in a single flow, without requiring master-level
    database permissions.

    DESTRUCTIVE: wipes every user object in the production database
    before publishing the staging schema + data. Prompts for confirmation
    unless -Force is supplied.

.PARAMETER StageServer
    Staging SQL Server host. Default: ocin.database.windows.net

.PARAMETER StageDatabase
    Staging database name. Default: tradelog_stage

.PARAMETER ProdServer
    Production SQL Server host. Default: ocin.database.windows.net

.PARAMETER ProdDatabase
    Production database name. Default: tradelog-prod

.PARAMETER User
    SQL Server username. Default: ocin

.PARAMETER Password
    SQL Server password.

.PARAMETER Force
    Skip the confirmation prompt. Use with care.

.EXAMPLE
    pwsh .\copy-stage-to-prod.ps1 -Password "mypassword"
#>
param(
    [string]$StageServer = "ocin.database.windows.net",
    [string]$StageDatabase = "tradelog_stage",
    [string]$ProdServer = "ocin.database.windows.net",
    [string]$ProdDatabase = "tradelog-prod",
    [string]$User = "ocin",
    [Parameter(Mandatory)][string]$Password,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command "sqlpackage" -ErrorAction SilentlyContinue)) {
    throw "sqlpackage not found. Install via: dotnet tool install -g microsoft.sqlpackage"
}
if (-not (Get-Command "sqlcmd" -ErrorAction SilentlyContinue)) {
    throw "sqlcmd not found. Install via: winget install Microsoft.SqlCmd"
}

Write-Host "=== Tradelog: Full clone Staging -> Production ===" -ForegroundColor Cyan
Write-Host "Source: $StageServer / $StageDatabase"
Write-Host "Target: $ProdServer / $ProdDatabase  (will be WIPED and REBUILT)" -ForegroundColor Yellow
Write-Host ""

if (-not $Force) {
    $confirm = Read-Host "Type 'yes' to overwrite production"
    if ($confirm -ne "yes") {
        Write-Host "Aborted." -ForegroundColor Red
        exit 1
    }
}

$TempDir = Join-Path $env:TEMP "tradelog-clone-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
$Dacpac = Join-Path $TempDir "$StageDatabase.dacpac"
$CleanSql = Join-Path $PSScriptRoot "clean-db.sql"

if (-not (Test-Path $CleanSql)) {
    throw "clean-db.sql not found next to script: $CleanSql"
}

function Wake-Database {
    param([string]$Server, [string]$Database, [string]$User, [string]$Password)
    Write-Host "  Waking $Database (serverless auto-resume) ..." -ForegroundColor Gray
    $attempts = 0
    $maxAttempts = 6
    while ($attempts -lt $maxAttempts) {
        sqlcmd -S $Server -d $Database -U $User -P $Password -C -l 120 -Q "SELECT 1" 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { return }
        $attempts++
        Start-Sleep -Seconds 10
    }
    throw "Failed to wake $Database after $maxAttempts attempts"
}

try {
    # Step 0: Wake both databases so extract and publish don't hit a cold serverless tier
    Wake-Database -Server $StageServer -Database $StageDatabase -User $User -Password $Password
    Wake-Database -Server $ProdServer -Database $ProdDatabase -User $User -Password $Password

    # Step 1: Extract staging with all table data
    Write-Host "[1/3] Extracting $StageDatabase schema + data ..." -ForegroundColor Yellow
    sqlpackage /Action:Extract `
        /SourceEncryptConnection:False `
        /SourceServerName:$StageServer `
        /SourceDatabaseName:$StageDatabase `
        /SourceUser:$User `
        /SourcePassword:$Password `
        /SourceTimeout:120 `
        /TargetFile:$Dacpac `
        /p:ExtractAllTableData=True
    if ($LASTEXITCODE -ne 0) { throw "sqlpackage extract failed" }
    Write-Host "  Extract complete: $Dacpac" -ForegroundColor Green

    # Re-wake prod in case it went back to sleep during the extract
    Wake-Database -Server $ProdServer -Database $ProdDatabase -User $User -Password $Password

    # Step 2: Clean production (drop every user object)
    Write-Host "[2/3] Cleaning $ProdDatabase ..." -ForegroundColor Yellow
    sqlcmd -S $ProdServer -d $ProdDatabase -U $User -P $Password -C -l 120 -i $CleanSql
    if ($LASTEXITCODE -ne 0) { throw "clean-db.sql failed" }
    Write-Host "  Target cleaned." -ForegroundColor Green

    # Step 3: Publish dacpac into production
    Write-Host "[3/3] Publishing to $ProdDatabase ..." -ForegroundColor Yellow
    sqlpackage /Action:Publish `
        /TargetEncryptConnection:False `
        /TargetServerName:$ProdServer `
        /TargetDatabaseName:$ProdDatabase `
        /TargetUser:$User `
        /TargetPassword:$Password `
        /TargetTimeout:120 `
        /SourceFile:$Dacpac `
        /p:BlockOnPossibleDataLoss=False
    if ($LASTEXITCODE -ne 0) { throw "sqlpackage publish failed" }
    Write-Host "  Publish complete." -ForegroundColor Green

    Write-Host ""
    Write-Host "=== Clone complete ===" -ForegroundColor Green
    Write-Host "Production database '$ProdDatabase' now matches staging '$StageDatabase'."
}
finally {
    Remove-Item $TempDir -Recurse -Force -ErrorAction SilentlyContinue
}
