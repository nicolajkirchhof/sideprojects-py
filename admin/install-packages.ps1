# PowerShell script to install all required Python packages using pip
# This script uses the requirements.txt file in the repository root

Write-Host "Installing Python packages from requirements.txt..."
pip install -r ..\requirements.txt

Write-Host "Installation complete. Please check for any errors above."
