import os
import time

import pandas as pd
from sqlalchemy import text

from finance.utils import exchanges

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

DAILY_SYMBOLS = {**{symbol: {'EX': exchanges.US_EXCHANGE} for symbol in
                    us_stock_symbols + market_etf_symbols + sectors_etf_symbols + world_etf_symbols + crypto_etf_symbols +
                    forex_etf_symbols + metals_etf_symbols + energy_etf_symbols + agriculture_etf_symbols + cboe_volatility_indices},
                 **{f: {'EX': exchanges.US_NY_EXCHANGE} for f in forex},
                 **{e: {'EX': exchanges.DE_EXCHANGE} for e in eu_indices + eu_index_future + eu_volatility_indices}, }


_STATE_DIR = "finance/_data/state"
_STOCKS_PATH = f"{_STATE_DIR}/liquid_stocks.parquet"
_ETFS_PATH = f"{_STATE_DIR}/liquid_etfs.parquet"

# Screener-aligned defaults (BarchartScreeners.md Global Base Filters)
_DEFAULT_MIN_PRICE = 5.0
_DEFAULT_MIN_AVG_VOLUME = 1_000_000
_DEFAULT_LOOKBACK_DAYS = 60


def refresh_liquid_universe(
    min_price: float = _DEFAULT_MIN_PRICE,
    min_avg_volume: int = _DEFAULT_MIN_AVG_VOLUME,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
) -> tuple[list[str], list[str]]:
    """
    Query Dolt stocks DB for liquid stocks and ETFs matching screener criteria.

    Uses recent {lookback_days}-day average volume and average close price as
    proxies for the screener's 20D avg volume and last price filters.

    Writes liquid_stocks.parquet and liquid_etfs.parquet to _data/state/.
    Returns (stock_symbols, etf_symbols).
    """
    from finance.utils.dolt_data import db_stocks_connection

    t0 = time.time()

    query_stocks = text("""
        SELECT o.act_symbol
        FROM ohlcv o
        JOIN symbol s ON o.act_symbol = s.act_symbol
        WHERE s.is_etf = 0
          AND o.act_symbol NOT LIKE '%$$%'
          AND o.act_symbol NOT LIKE '%.%'
          AND o.date >= DATE_SUB(CURDATE(), INTERVAL :lookback DAY)
        GROUP BY o.act_symbol
        HAVING AVG(o.volume) > :min_vol
           AND AVG(o.close) > :min_price
           AND MAX(o.close) < 5000
    """)

    query_etfs = text("""
        SELECT o.act_symbol
        FROM ohlcv o
        JOIN symbol s ON o.act_symbol = s.act_symbol
        WHERE s.is_etf = 1
          AND o.act_symbol NOT LIKE '%$$%'
          AND o.act_symbol NOT LIKE '%.%'
          AND o.date >= DATE_SUB(CURDATE(), INTERVAL :lookback DAY)
        GROUP BY o.act_symbol
        HAVING AVG(o.volume) > :min_vol
           AND AVG(o.close) > :min_price
           AND MAX(o.close) < 5000
    """)

    params = {"min_vol": min_avg_volume, "min_price": min_price, "lookback": lookback_days}

    df_stocks = pd.read_sql(query_stocks, db_stocks_connection, params=params)
    df_etfs = pd.read_sql(query_etfs, db_stocks_connection, params=params)

    stocks = sorted(df_stocks["act_symbol"].tolist())
    etfs = sorted(df_etfs["act_symbol"].tolist())

    # Remove any overlap (prefer ETF classification)
    stock_set = set(stocks) - set(etfs)
    stocks = sorted(stock_set)

    os.makedirs(_STATE_DIR, exist_ok=True)
    pd.DataFrame({"symbol": stocks}).to_parquet(_STOCKS_PATH, index=False)
    pd.DataFrame({"symbol": etfs}).to_parquet(_ETFS_PATH, index=False)

    elapsed = time.time() - t0
    print(f"[Universe] {len(stocks)} stocks + {len(etfs)} ETFs "
          f"(price>{min_price}, avg_vol>{min_avg_volume:,}, {lookback_days}d lookback) "
          f"in {elapsed:.1f}s")

    return stocks, etfs


def get_liquid_stocks():
    return pd.read_parquet(_STOCKS_PATH)['symbol'].tolist()


def get_liquid_etfs():
    return pd.read_parquet(_ETFS_PATH)['symbol'].tolist()


def get_liquid_underlyings():
    return get_liquid_stocks() + get_liquid_etfs()
