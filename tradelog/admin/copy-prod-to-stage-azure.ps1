<#
.SYNOPSIS
    Copies the production Azure SQL database to staging using Azure SQL's native database copy.
    Both databases must be on the same Azure SQL server.
    This creates a point-in-time copy, then swaps it in as the staging database.

.PARAMETER Server
    Azure SQL server name (e.g., myserver.database.windows.net)

.PARAMETER ProdDatabase
    Production database name. Default: Tradelog_Prod

.PARAMETER StageDatabase
    Staging database name. Default: Tradelog_Stage

.PARAMETER User
    SQL admin username.

.PARAMETER Password
    SQL admin password.

.EXAMPLE
    .\copy-prod-to-stage-azure.ps1 -Server "myserver.database.windows.net" -User "sqladmin" -Password "mypassword"
#>
param(
    [Parameter(Mandatory)][string]$Server,
    [string]$ProdDatabase = "Tradelog_Prod",
    [string]$StageDatabase = "Tradelog_Stage",
    [Parameter(Mandatory)][string]$User,
    [Parameter(Mandatory)][string]$Password
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command "sqlcmd" -ErrorAction SilentlyContinue)) {
    throw "sqlcmd not found. Install via: winget install Microsoft.SqlCmd"
}

$TempName = "${StageDatabase}_copy_$(Get-Date -Format 'yyyyMMddHHmmss')"

Write-Host "=== Azure SQL: Copy Production -> Staging ===" -ForegroundColor Cyan
Write-Host "Server: $Server"
Write-Host "Source: $ProdDatabase"
Write-Host "Target: $StageDatabase"
Write-Host ""

# Step 1: Create a copy of production
Write-Host "[1/3] Creating database copy: $TempName ..." -ForegroundColor Yellow
Write-Host "  This may take several minutes for large databases."
sqlcmd -S $Server -d master -U $User -P $Password -Q "CREATE DATABASE [$TempName] AS COPY OF [$ProdDatabase]"
if ($LASTEXITCODE -ne 0) { throw "Database copy failed" }

# Wait for copy to complete (Azure SQL copies are async)
Write-Host "  Waiting for copy to complete..." -ForegroundColor Gray
$maxWait = 60  # max iterations (1 per 10 seconds = 10 minutes)
$i = 0
do {
    Start-Sleep -Seconds 10
    $state = sqlcmd -S $Server -d master -U $User -P $Password -Q "SET NOCOUNT ON; SELECT state_desc FROM sys.databases WHERE name = '$TempName'" -h -1 -W
    $i++
    if ($i % 6 -eq 0) { Write-Host "  Still copying... ($($i * 10)s elapsed)" -ForegroundColor Gray }
} while ($state -match "COPYING" -and $i -lt $maxWait)

if ($state -notmatch "ONLINE") {
    throw "Database copy did not complete. Current state: $state"
}
Write-Host "  Copy complete." -ForegroundColor Green

# Step 2: Drop existing staging database
Write-Host "[2/3] Dropping existing staging database: $StageDatabase ..." -ForegroundColor Yellow
sqlcmd -S $Server -d master -U $User -P $Password -Q "IF EXISTS (SELECT 1 FROM sys.databases WHERE name = '$StageDatabase') DROP DATABASE [$StageDatabase]"
if ($LASTEXITCODE -ne 0) { throw "Failed to drop staging database" }

# Step 3: Rename copy to staging
Write-Host "[3/3] Renaming $TempName -> $StageDatabase ..." -ForegroundColor Yellow
sqlcmd -S $Server -d master -U $User -P $Password -Q "ALTER DATABASE [$TempName] MODIFY NAME = [$StageDatabase]"
if ($LASTEXITCODE -ne 0) { throw "Failed to rename database" }

Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Green
Write-Host "Staging database '$StageDatabase' is now a copy of '$ProdDatabase'."
