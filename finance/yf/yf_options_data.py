import yfinance as yf
import pandas as pd

from finance.utils.greeks import calculate_greeks
import numpy as np
import ib_async as ib

#%%
# Define stock symbol and expiration
symbol = "QQQ"
expiration = None  # Set expiration date as 'YYYY-MM-DD' if required

# Fetch the options chain data
# Load ticker data
ticker = yf.Ticker(symbol)

# %%
ib.util.startLoop()
ib_con = ib.IB()
tws_real_port = 7497
tws_paper_port = 7497
api_real_port = 4002
api_paper_port = 4002
ib_con.connect('127.0.0.1', api_paper_port, clientId=12, readonly=True)
# ib_con.connect('127.0.0.1', tws_paper_port, clientId=10, readonly=True)
ib_con.reqMarketDataType(1)  # Use free, delayed, frozen data
# ib_con.reqMarketDataType(2)  # Use free, delayed, frozen data

#%%
contract = ib.Stock(symbol=symbol, exchange='SMART', currency='USD')
details = ib_con.reqContractDetails(contract)
print(details[0].longName)

# Ensure the contract details are validated via IB
ib_con.qualifyContracts(contract)

#%%
# Get available expiration dates
expirations = ticker.options
print(f"Available Expirations for {symbol}: {expirations}")

# Use the first expiration date if one is not provided
if expiration is None:
  expiration = expirations[0]
if expiration not in expirations:
  raise ValueError(f"Invalid expiration date! Choose from: {expirations}")

print(f"Fetching options for {symbol} with expiration: {expiration}")

#%%
# %%
expiration_ib = expiration.replace('-', '')
strike = 485
right = 'C'
# exchange = chain.exchange
option_contract = ib.Option(symbol=symbol, exchange="SMART", strike=strike, lastTradeDateOrContractMonth=expiration_ib, right=right, currency=contract.currency)
# option_contract.includeExpired = True

ib_con.qualifyContracts(option_contract)
ib_con.sleep(1)
#%%
ib_con.reqMarketDataType(2)  # Use free, delayed, frozen data
snapshot = ib_con.reqMktData(option_contract, "100, 101, 104, 105, 106, 225, 233, 375, 588", False, False)
ib_con.sleep(2)
ib_con.reqMarketDataType(1)  # Use free, delayed, frozen data
while ib.util.isNan(snapshot.callOpenInterest):
  ib_con.sleep(2)
ib_con.cancelMktData(option_contract)

#%%
# Fetch calls and puts for the specific expiration
options_chain = {}
options_chain['calls'] = ticker.option_chain(expiration).calls
options_chain['puts'] = ticker.option_chain(expiration).puts

# Current stock price from Yahoo Finance
stock_price = yf.Ticker(symbol).info["regularMarketPrice"]
#%%
# Time to expiration (in years)
today = pd.Timestamp.today().date()
expiration_date = pd.Timestamp(expiration).date()
T = (expiration_date - today).days / 365.0
if T == 0:
  T = 1e-6

#%%
risk_free_rate=0.045

#%%
# Add Greeks to calls and puts
for option_type in ["calls", "puts"]:
  options = options_chain[option_type]
  greeks = options.apply(lambda row: calculate_greeks(
    S=stock_price,
    K=row["strike"],
    T=T,
    r=risk_free_rate,
    sigma=row["impliedVolatility"] if not np.isnan(row["impliedVolatility"]) else 0.2,  # Assume 20% IV if missing
    option_type="C" if option_type == "calls" else "P"
  ), axis=1)

  # Split Greek dict into columns
  greek_df = pd.DataFrame(greeks.tolist())
  options_chain[option_type] = pd.concat([options.reset_index(drop=True), greek_df], axis=1)


#%%

# Display the first few rows of raw options data
print("Calls Options (Raw):")
print(options_chain['calls'].head())
print("Puts Options (Raw):")
print(options_chain['puts'].head())


