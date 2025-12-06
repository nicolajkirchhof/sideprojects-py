from finance import utils

us_stock_symbols = [
  'AAL', 'AAPL', 'ABNB', 'ACHR', 'ADBE', 'AFRM', 'AG', 'AI', 'ALAB', 'AMAT', 'AMD', 'AMZN', 'APLD', 'APP', 'ARM', 'ASTS',
  'AVGO', 'B', 'BA', 'BABA', 'BAC', 'BBAI', 'BE', 'BIDU', 'BMNR', 'BMY', 'BP', 'BULL', 'C', 'CAPR', 'CCJ', 'CCL', 'CELH',
  'CIFR', 'CLF', 'CLSK', 'CMCSA', 'CMG', 'COIN', 'CORZ', 'COST', 'CRCL', 'CRM', 'CRWD', 'CRWV', 'CSCO', 'CTRA', 'CVNA',
  'CVX', 'DAL', 'DASH', 'DDOG', 'DELL', 'DERM', 'DIS', 'DJT', 'DKNG', 'ENPH', 'EOSE', 'EQT', 'ET', 'F', 'FCX', 'FIG',
  'FISI', 'GAP', 'GLW', 'GLXY', 'GM', 'GME', 'GOOG', 'GOOGL', 'GRAB', 'GS', 'HD', 'HIMS', 'HL', 'HOOD', 'HPE', 'HTZ',
  'HUT', 'IBM', 'INTC', 'IONQ', 'IREN', 'JD', 'JNJ', 'JOBY', 'JPM', 'KO', 'KVUE', 'LCID', 'LEN', 'LLY', 'LULU', 'LUMN',
  'LYFT', 'MARA', 'META', 'MP', 'MRK', 'MRNA', 'MRVL', 'MSFT', 'MSTR', 'MU', 'NBIS', 'NEE', 'NEM', 'NFLX', 'NIO', 'NKE',
  'NOK', 'NU', 'NVDA', 'NVO', 'NVTS', 'OKLO', 'ONDS', 'ONON', 'OPEN', 'ORCL', 'OSCR', 'OWL', 'OXY', 'PANW', 'PATH',
  'PBR', 'PDD', 'PFE', 'PINS', 'PLTR', 'POET', 'PYPL', 'QBTS', 'QCOM', 'QS', 'QUBT', 'RDDT', 'RGTI', 'RIOT', 'RIVN',
  'RKLB', 'RKT', 'SBET', 'SBUX', 'SHOP', 'SLB', 'SMCI', 'SMR', 'SNAP', 'SNDK', 'SNOW', 'SOFI', 'SOUN', 'T', 'TEVA',
  'TGT', 'TLRY', 'TMC', 'TSLA', 'TSM', 'TTD', 'U', 'UBER', 'UNH', 'UPS', 'UPST', 'USAR', 'UUUU', 'VALE', 'VRT',
  'VZ', 'WBD', 'WFC', 'WMT', 'WOLF', 'WULF', 'XOM', 'XPEV', 'XYZ']

market_etf_symbols = ['SPY', 'QQQ', 'IWM', 'DIA', 'RSP', 'VTI', 'QQQE']
sectors_etf_symbols = ['SMH', 'XBI', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
world_etf_symbols = ['EEM', 'EFA', 'EWZ', 'FXI', 'EWJ', 'EWW', 'EWC']
crypto_etf_symbols = ['IBIT', 'ETHA']
forex_etf_symbols = ['FXY', 'FXE', 'FXF', 'FXC', 'FXA', 'FXB']
metals_etf_symbols = ['GLD', 'GDX', 'SLV', 'COPX', 'SIL', 'CPER', 'URA', 'URNM', 'PALL']
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
