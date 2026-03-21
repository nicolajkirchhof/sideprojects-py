<#
.SYNOPSIS
    Builds the Tradelog application: Angular frontend + .NET backend into a single standalone executable.

.PARAMETER Runtime
    Target runtime identifier. Default: win-x64. Other options: linux-x64, osx-x64.

.PARAMETER Configuration
    Build configuration. Default: Release.

.EXAMPLE
    .\build.ps1
    .\build.ps1 -Runtime linux-x64
#>
param(
    [string]$Runtime = "win-x64",
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$FrontendDir = Join-Path $Root "frontend"
$BackendDir = Join-Path $Root "backend"
$OutputDir = Join-Path $Root "dist\tradelog-$Runtime"

Write-Host "=== Tradelog Build ===" -ForegroundColor Cyan
Write-Host "Runtime:       $Runtime"
Write-Host "Configuration: $Configuration"
Write-Host "Output:        $OutputDir"
Write-Host ""

# Step 1: Build Angular frontend
Write-Host "[1/3] Building Angular frontend..." -ForegroundColor Yellow
Push-Location $FrontendDir
try {
    npx ng build --configuration=production
    if ($LASTEXITCODE -ne 0) { throw "Angular build failed" }
}
finally { Pop-Location }

# Verify output
$WwwRoot = Join-Path $BackendDir "wwwroot"
if (-not (Test-Path (Join-Path $WwwRoot "index.html"))) {
    # Angular may output to a subdirectory (browser/)
    $BrowserDir = Join-Path $WwwRoot "browser"
    if (Test-Path (Join-Path $BrowserDir "index.html")) {
        Write-Host "  Moving files from browser/ subfolder..." -ForegroundColor Gray
        Get-ChildItem $BrowserDir | Move-Item -Destination $WwwRoot -Force
        Remove-Item $BrowserDir -Recurse -Force
    }
    else {
        throw "Angular build output not found in $WwwRoot"
    }
}
Write-Host "  Frontend built to $WwwRoot" -ForegroundColor Green

# Step 2: Publish .NET backend (self-contained)
Write-Host "[2/3] Publishing .NET backend..." -ForegroundColor Yellow
Push-Location $BackendDir
try {
    dotnet publish -c $Configuration -r $Runtime --self-contained -o $OutputDir
    if ($LASTEXITCODE -ne 0) { throw ".NET publish failed" }
}
finally { Pop-Location }
Write-Host "  Backend published to $OutputDir" -ForegroundColor Green

# Step 3: Copy environment config files
Write-Host "[3/3] Copying configuration files..." -ForegroundColor Yellow
Copy-Item (Join-Path $BackendDir "appsettings.json") $OutputDir -Force
Copy-Item (Join-Path $BackendDir "appsettings.Staging.json") $OutputDir -Force
Copy-Item (Join-Path $BackendDir "appsettings.Production.json") $OutputDir -Force

Write-Host ""
Write-Host "=== Build complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Run with:" -ForegroundColor Cyan
Write-Host "  Staging:    `$env:ASPNETCORE_ENVIRONMENT='Staging'; .\dist\tradelog-$Runtime\backend.exe"
Write-Host "  Production: `$env:ASPNETCORE_ENVIRONMENT='Production'; .\dist\tradelog-$Runtime\backend.exe"
Write-Host ""
