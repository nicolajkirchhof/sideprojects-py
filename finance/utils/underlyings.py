from finance import utils

us_stock_symbols = [
  'AAL', 'AAPL', 'ACHR', 'ADBE', 'AFRM', 'AI', 'AMD', 'AMZN', 'APP', 'ASTS', 'AVGO', 'B', 'BA', 'BABA', 'BAC', 'BE',
  'BIDU', 'BMNR', 'BULL', 'C', 'CHWY', 'CIFR', 'COIN', 'CRM', 'CRWD', 'CRWV', 'CSCO', 'CVNA', 'EOSE', 'F', 'FIG', 'GME',
  'GOOGL', 'HIMS', 'HL', 'HOOD', 'HPE', 'INTC', 'IREN', 'JD', 'JNJ', 'JPM', 'KO', 'LLY', 'LULU', 'M', 'MARA', 'META',
  'MRNA', 'MRVL', 'MSFT', 'MSTR', 'MU', 'NBIS', 'NFLX', 'NIO', 'NKE', 'NVDA', 'OKLO', 'OPEN', 'ORCL', 'PCG',
  'PEP', 'PFE', 'PLTR', 'PYPL', 'RDDT', 'RGTI', 'RIOT', 'RKLB', 'SBET', 'SBUX', 'SHOP', 'SMCI', 'SMR', 'SNAP', 'SOFI',
  'SOUN', 'TGT', 'TSLA', 'TSM', 'TTD', 'U', 'UBER', 'UNH', 'UPS', 'VZ', 'WBD', 'WFC', 'WMT', 'WOLF', 'WULF', 'XOM',
  'XYZ']

market_etf_symbols = ['SPY', 'QQQ', 'IWM', 'DIA']
sectors_etf_symbols = ['SMH', 'XBI', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
world_etf_symbols = ['EEM', 'EFA', 'EWZ', 'FXI', 'EWJ', 'EWW', 'EWC']
crypto_etf_symbols = ['IBIT', 'ETHA']
forex_etf_symbols = ['FXY', 'FXE', 'FXF', 'FXC', 'FXA', 'FXB']
metals_etf_symbols = ['GLD', 'GDX', 'SLV', 'COPX', 'SIL']
energy_etf_symbols = ['UNG', 'USO', 'XOP']
agriculture_etf_symbols = ['SOYB', 'CORN', 'WEAT', 'CANE']

cboe_volatility_indices = ['VIX', 'VXN', 'RVX', 'GVZ', 'OVX', 'VXSLV', 'VXEEM', 'VXEFA', 'VXEWZ', 'VXAPL', 'VXGOG',
                           'VXAZN', 'VXIBM', 'VXTLT', 'VXGS']

eu_volatility_indices = ['V2TX', 'V1X']

eu_indices = ['DAX', 'ESTX50'] + eu_volatility_indices

us_indices = ['SPX', 'NDX', 'RUT', 'INDU']

eu_index_future = ['FDAX', 'FESTX50']

us_index_futures = ['FES', 'FNQ', 'FRTY', 'FVXM', 'FZB', 'FZC', 'FZF', 'FZN', 'FZT', 'FZW', 'FSI', 'FGC', 'FCL', 'FNG']

forex = ['EURUSD', 'EURGBP', 'EURCHF', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDJPY', 'CHFUSD']

DAILY_SYMBOLS = {**{symbol: {'EX': utils.exchanges.US_EXCHANGE} for symbol in
                    us_stock_symbols + market_etf_symbols + sectors_etf_symbols + world_etf_symbols + crypto_etf_symbols +
                    forex_etf_symbols + metals_etf_symbols + energy_etf_symbols + agriculture_etf_symbols + cboe_volatility_indices},
                 **{f: {'EX': utils.exchanges.US_NY_EXCHANGE} for f in forex},
                 **{e: {'EX': utils.exchanges.DE_EXCHANGE} for e in eu_indices + eu_index_future + eu_volatility_indices}, }
