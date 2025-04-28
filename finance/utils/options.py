import numpy as np
import pandas as pd
import blackscholes as bs

from finance.utils.fitlog import log_function_with_offset

SKEW = {
  'DAX': {'ATM': [ 0.40375593,  6.95075883, -0.8375398 ,  0.17255903], 'OTM': [ 3.67816432e-01,  4.30107676e+02, -3.00228684e-01,  4.94081049e-01]},
  'ESTX50': {'ATM': [2.72508214e+02, 2.50274938e-03, 9.96007080e-01, 9.88537431e-01], 'OTM': [6.16779131e-01, 3.69403318e+02, 7.34503531e-02, 3.48376546e-01]}
}

def put_credit_spread_pnl(S, atm_put, wing_put):
  # Net credit received
  net_credit = atm_put.price - wing_put.price

  # Region 1: Stock price is above the higher strike
  if S >= atm_put.strike:
    return net_credit
  # Region 2: Stock price between the strikes
  elif wing_put.strike <= S < atm_put.strike:
    return net_credit - (atm_put.strike - S)
  # Region 3: Stock price is below the lower strike
  else:  # S < K0
    return net_credit - (atm_put.strike - wing_put.strike)

def call_credit_spread_pnl(S, atm_call, wing_call):
  # Net credit received
  net_credit = atm_call.price - wing_call.price

  # Region 1: Stock price is above the higher strike
  if S <= atm_call.strike:
    return net_credit
  # Region 2: Stock price between the strikes
  elif wing_call.strike >= S > atm_call.strike:
    return net_credit - (S - atm_call.strike)
  # Region 3: Stock price is below the lower strike
  else:  # S < K0
    return net_credit - (wing_call.strike - atm_call.strike)

def iron_butterfly_profit_loss(S, wing_call, atm_call, atm_put, wing_put):
  """
    Calculate profit/loss for an asymmetrical Iron Condor at expiration.

    :return: Array of profit/loss at expiration
    """
  return put_credit_spread_pnl(S, atm_put, wing_put) + call_credit_spread_pnl(S, atm_call, wing_call)


def create_option_chain(region, date, iv, underlying, strike_offset, expiry_days, sigma, symbol, debug=False):
  risk_free_rate_year = risk_free_rate(date, region)

  t = expiry_days / 365
  ##%%
  underlying_low = underlying - underlying * sigma * iv * np.sqrt(expiry_days)
  underlying_high = underlying + underlying * sigma * iv * np.sqrt(expiry_days)
  underlying_low_boundary = int(np.floor(underlying_low / strike_offset) * strike_offset)
  underlying_high_boundary = int(np.ceil(underlying_high / strike_offset) * strike_offset)
  strikes = range(underlying_low_boundary, underlying_high_boundary, strike_offset)

  ##%%

  opts = []
  for strike in strikes:
    pct = (underlying - strike) / underlying
    iv_skewed = estimate_skew(pct, iv, strike, symbol)
    call = bs.BlackScholesCall(underlying, strike, t, risk_free_rate_year, iv_skewed)
    put = bs.BlackScholesPut(underlying, strike, t, risk_free_rate_year, iv_skewed)
    v_call = vars(call)
    v_call['right'] = 'C'
    v_put = vars(put)
    v_put['right'] = 'P'
    opts.append({'right':'C', 'delta': call.delta(), 'theta': call.theta(), 'gamma': call.gamma(), 'vega': call.vega(), 'price': call.price(), 'strike': strike, 'pos':0})
    opts.append({'right':'P', 'delta': put.delta(), 'theta': put.theta(), 'gamma': put.gamma(), 'vega': put.vega(), 'price': put.price(), 'strike': strike, 'pos':0})
    if debug:
      print(f'{call.vega():.3f} ν {call.gamma():.3f} Γ {call.theta():.3f} Θ {call.delta():.3f} Δ {call.price():.3f} C -- {strike} -- P {put.price():.3f} Δ {put.delta():.3f} Θ {put.theta():.3f} Γ {put.gamma():.3f} ν {put.vega():.3f} ')

  return opts


def risk_free_rate(date, region):
  risk_free_rate_year = None
  if region == 'EU':
    eu_interest = pd.read_csv('finance/ECB_Interest.csv', index_col='DATE', parse_dates=True)
    risk_free_rate_year = eu_interest[eu_interest.index < str(date.date())].iloc[-1].Main / 100
  if region == 'US':
    us_interest = pd.read_csv('finance/US_Interest.csv', index_col='observation_date', parse_dates=True)
    risk_free_rate_year = us_interest[us_interest.index <= str(date.date())].tail(1)['THREEFY1'].iat[0] / 100
  if risk_free_rate_year is None:
    raise ValueError('No risk free rate found for the given date')
  return risk_free_rate_year


def estimate_volatility_skew(S, K, sigma_ATM, skew_down, skew_up, smile_width):
  """
    Estimate a simple volatility skew for index options.

    Parameters:
        S (float): Underlying price.
        strike_prices (list): List of strike prices.
        sigma_ATM (float): ATM implied volatility.
        skew_down (float): Skew factor for OTM puts (strikes below ATM).
        skew_up (float): Skew factor for OTM calls (strikes above ATM).
        smile_width (float): Controls the smoothness of the smile/skew.

    Returns:
        list: List of estimated volatilities for the given strikes.
    """
  log_moneyness = np.log(K / S)

  if K < S:  # OTM puts
    skew = skew_down
  else:  # OTM calls
    skew = skew_up

  # Adjust volatility based on skew and smile width
  iv = sigma_ATM + skew * log_moneyness + smile_width * log_moneyness ** 2
  return iv

def option_bs_price_correction(bs_price, price_diff_pct):
  if price_diff_pct >= 1:
    return bs_price
  return bs_price / (1 - price_diff_pct)

def estimate_skew(pct, iv, bs_price, symbol):
  skew = SKEW[symbol]
  if iv * pct < 0.0005:
    pct_diff_fct = log_function_with_offset(iv, *skew['ATM'])
    return option_bs_price_correction(bs_price, pct_diff_fct)
  else:
    pct_diff_fct = log_function_with_offset(min(0.008, iv*pct), *skew['OTM'])
    return option_bs_price_correction(bs_price, pct_diff_fct)

def implied_volatility(S, K, T, r, market_price, right, tol=1e-6, max_iter=1000):
  """
  Calculate implied volatility using the bisection method.

  Parameters:
  - S: Current stock price (float)
  - K: Strike price (float)
  - T: Time to maturity in years (float)
  - r: Risk-free interest rate (float, e.g., 0.05 for 5%)
  - market_price: The actual (market) price of the option (float)
  - option_type: "call" or "put" (default="call")
  - tol: Tolerance for stopping the iteration (default=1e-6)
  - max_iter: Maximum number of iterations (default=1000)

  Returns:
  - Implied volatility (float).
  """
  # Initial bounds for volatility
  lower_vol = 1e-5    # Volatility cannot be zero
  upper_vol = 5.0     # Arbitrary high value for initial upper bound

  for i in range(max_iter):
    # Calculate the midpoint
    mid_vol = (lower_vol + upper_vol) / 2.0
    # Calculate the theoretical price with the mid volatility
    option = bs.BlackScholesCall(S, K, T, r, mid_vol) if right == "C" else bs.BlackScholesPut(S, K, T, r, mid_vol)
    theoretical_price = option.price()

    # Check the difference between theoretical price and market price
    price_diff = theoretical_price - market_price

    # If the difference is within the tolerance, return the implied volatility
    if abs(price_diff) < tol:
      return mid_vol

    # Adjust the bounds
    if price_diff > 0:
      # If theoretical price is too high, reduce upper bound
      upper_vol = mid_vol
    else:
      # If theoretical price is too low, raise lower bound
      lower_vol = mid_vol

  # If no result is found, raise an exception
  raise ValueError("Implied volatility did not converge within the given iterations.")
