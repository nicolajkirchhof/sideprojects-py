import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def safe_log(x, min_value=1e-10):
  """
  Safely compute logarithm by ensuring input is positive

  Parameters:
  -----------
  x : array-like
      Input values
  min_value : float
      Minimum value to clip data to prevent log(0)
  """
  return np.log(np.maximum(x, min_value))

# Modified log function for curve fitting
def log_function_with_offset(x, a, b, c, d):
  """
  Generic logarithmic function with offset: y = a * log(bx + c) + d
  with safety checks for domain
  """
  inside_log = b * x + c
  # Ensure values inside log are positive
  return a * safe_log(inside_log) + d

def fit_log_curve_df(df, x_col, y_col, plot=True):
  """
  Fit a logarithmic curve to data from DataFrame columns

  Parameters:
  -----------
  df : pandas DataFrame
      DataFrame containing the data
  x_col : str
      Name of column for independent variable
  y_col : str
      Name of column for dependent variable
  plot : bool
      Whether to create visualization

  Returns:
  --------
  tuple
      (optimal parameters, R-squared value)
  """
  # Remove any NaN values
  df_clean = df[[x_col, y_col]].dropna()

  # Convert to arrays for curve_fit
  x_data = df_clean[x_col].values
  y_data = df_clean[y_col].values

  # Initial parameter guesses
  p0 = [1.0, 1.0, 1.0, np.mean(y_data)]

  try:
    # Fit the curve
    popt, pcov = curve_fit(log_function_with_offset, x_data, y_data,
                           p0=p0,
                           maxfev=10000)

    # Generate predictions
    y_pred = log_function_with_offset(x_data, *popt)

    # Calculate R-squared
    r_squared = 1 - (np.sum((y_data - y_pred) ** 2) /
                     np.sum((y_data - np.mean(y_data)) ** 2))

    if plot:
      # Create visualization
      plt.figure(figsize=(24, 14))

      # Plot original data using DataFrame
      df_clean.plot.scatter(x=x_col, y=y_col, color='blue',
                            alpha=0.5, label='Data', ax=plt.gca())

      # Plot predicted points
      plt.scatter(x_data, y_pred, color='green',
                  alpha=0.5, label='Predicted Points')

    # Plot fitted curve
      x_smooth = np.linspace(df_clean[x_col].min(),
                             df_clean[x_col].max(),
                             1000)
      y_smooth = log_function_with_offset(x_smooth, *popt)
      plt.plot(x_smooth, y_smooth, 'r-', label='Fitted Curve')

      # Add equation to plot
      equation = f'y = {popt[0]:.3f} * ln({popt[1]:.3f}x + {popt[2]:.3f}) + {popt[3]:.3f}'
      plt.text(0.02, 0.95, equation, transform=plt.gca().transAxes,
               bbox=dict(facecolor='white', alpha=0.8))

      # Add R-squared value
      plt.text(0.02, 0.89, f'R² = {r_squared:.4f}',
               transform=plt.gca().transAxes,
               bbox=dict(facecolor='white', alpha=0.8))

      plt.xlabel(x_col)
      plt.ylabel(y_col)
      plt.title(f'Logarithmic Curve Fit: {y_col} vs {x_col}')
      plt.legend()
      plt.grid(True, alpha=0.3)
      plt.show()

    # Add predictions to DataFrame
    df_clean['predicted'] = y_pred
    df_clean['residuals'] = df_clean[y_col] - df_clean['predicted']

    return popt, r_squared, df_clean

  except Exception as e:
    print(f"Error in curve fitting: {str(e)}")
    return None, None, None

def analyze_residuals_df(df, x_col, actual_col='actual', pred_col='predicted'):
  """
  Analyze residuals using DataFrame

  Parameters:
  -----------
  df : pandas DataFrame
      DataFrame containing actual and predicted values
  x_col : str
      Name of x-axis column
  actual_col : str
      Name of column with actual values
  pred_col : str
      Name of column with predicted values
  """
  # Calculate residuals if not already present
  if 'residuals' not in df.columns:
    df['residuals'] = df[actual_col] - df[pred_col]

  # Create residual plots
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

  # Residuals vs X using DataFrame plot
  df.plot.scatter(x=x_col, y='residuals', alpha=0.5, ax=ax1)
  ax1.axhline(y=0, color='r', linestyle='--')
  ax1.set_xlabel(x_col)
  ax1.set_ylabel('Residuals')
  ax1.set_title('Residuals vs X')
  ax1.grid(True, alpha=0.3)

  # Residual histogram using DataFrame plot
  df['residuals'].plot.hist(bins=20, alpha=0.5, ax=ax2)
  ax2.set_xlabel('Residual Value')
  ax2.set_ylabel('Frequency')
  ax2.set_title('Residual Distribution')
  ax2.grid(True, alpha=0.3)

  plt.tight_layout()
  plt.show()

  # Print residual statistics using DataFrame methods
  print("\nResidual Statistics:")
  print(df['residuals'].describe())
