# Create a conda environment with Python 3.11 and install packages
conda create -n ds311 python=3.11 ipython matplotlib qt pyqt pandas numpy scipy scikit-learn spyder

# Activate the environment
conda activate ds311

# Install additional packages using pip
pip install yfinance backtesting backtrader ib_async mplfinance influxdb blackscholes python-dateutil pytz

# Print success message
Write-Host "Environment setup complete. All required packages have been installed."
Write-Host "To activate the environment, run: conda activate ds311"
