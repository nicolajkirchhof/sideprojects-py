from scipy.stats import norm
import numpy as np


def calculate_greeks(S, K, T, r, sigma, option_type="C"):
  """
    Calculate Greeks for options using the Black-Scholes model.
    Args:
        S (float): Current underlying stock price
        K (float): Option strike price
        T (float): Time to expiration (in years)
        r (float): Risk-free interest rate (e.g., 0.05 for 5%)
        sigma (float): Implied volatility (as a decimal)
        option_type (str): 'C' for call, 'P' for put
    Returns:
        dict: Greeks (Delta, Gamma, Vega, Theta, Rho)
    """
  # Black-Scholes factors
  d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)

  # Delta
  if option_type == "C":
    delta = norm.cdf(d1)
  elif option_type == "P":
    delta = -norm.cdf(-d1)

  # Gamma (same for calls and puts)
  gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

  # Vega (same for calls and puts)
  vega = S * norm.pdf(d1) * np.sqrt(T)

  # Theta
  if option_type == "C":
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2))
  elif option_type == "P":
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
             + r * K * np.exp(-r * T) * norm.cdf(-d2))

  # Rho
  if option_type == "C":
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)
  elif option_type == "P":
    rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

  return {
    "delta": delta,
    "gamma": gamma,
    "vega": vega / 100,  # Convert to percentage terms
    "theta": theta / 365,  # Convert to daily terms
    "rho": rho / 100  # Convert to percentage terms
  }
