# Environment Setup Instructions

This directory contains scripts for setting up the development environment for the sideprojects-py repository.

## Installing Required Packages

To install all the required Python packages for this repository, follow these steps:

1. Make sure you have [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.

2. Open a PowerShell terminal.

3. Navigate to the repository root directory:
   ```
   cd path\to\sideprojects-py
   ```

4. Run the installation script:
   ```
   .\admin\install-environment.ps1
   ```

5. The script will:
   - Create a new conda environment named 'ds311' with Python 3.11
   - Install all the required packages listed in requirements.txt
   - Activate the environment

6. After installation, you can activate the environment anytime using:
   ```
   conda activate ds311
   ```

## Installed Packages

The script installs the following packages:

### Core Data Science
- numpy
- pandas
- scipy
- scikit-learn

### Visualization
- matplotlib
- mplfinance

### Finance-specific
- yfinance
- backtesting
- backtrader
- ib_async
- blackscholes

### Data Storage and Processing
- influxdb

### Utilities
- python-dateutil
- pytz

### Development and Interactive Computing
- ipython
- spyder

### GUI
- qt
- pyqt

## Troubleshooting

If you encounter any issues during installation:

1. Make sure you have administrator privileges
2. Check that Anaconda/Miniconda is properly installed and in your PATH
3. Try running the commands manually from the script if the script fails
