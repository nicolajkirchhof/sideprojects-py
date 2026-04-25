"""Incremental update of intraday Parquet files from IBKR.

Reads the last timestamp from each symbol's Parquet file, fetches new bars
from IBKR in 10-day chunks, and appends them.

Usage:
  uv run python -m finance.ibkr.data.update_intraday
  uv run python -m finance.ibkr.data.update_intraday --schema index
  uv run python -m finance.ibkr.data.update_intraday --schema index --symbol SPX
  uv run python -m finance.ibkr.data.update_intraday --instance paper
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import ib_async as ib
import pandas as pd

from finance.utils import ibkr as ibkr_utils
from finance.utils.intraday import (
    BAR_COLUMNS, get_last, write_bars, sec_type_to_schema,
)

DURATION = '10 D'
OFFSET_DAYS = 9
BAR_SIZE = '1 min'
RTH = False
SKIP_DAYS = 3
FALLBACK_START = datetime(2013, 6, 1)
MAX_CONSECUTIVE_ERRORS = 3

LOG_DIR = Path('finance/_data/logs')

log = logging.getLogger('intraday_update')


def setup_logging() -> None:
    """Configure console + rotating file logging."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f'intraday_{datetime.now():%Y%m%d_%H%M%S}.log'

    formatter = logging.Formatter(
        '%(asctime)s  %(levelname)-5s  %(message)s',
        datefmt='%H:%M:%S',
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(logging.Formatter(
        '%(asctime)s  %(levelname)-5s  %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    ))

    log.setLevel(logging.DEBUG)
    log.addHandler(console)
    log.addHandler(fh)

    log.info('Log file: %s', log_file)


def build_contracts():
    """Return the full contract universe as a list of ib_async contract objects."""
    eu_indices = [ib.Index(x, 'EUREX', 'EUR') for x in ['DAX', 'ESTX50', 'V2TX', 'V1X']]
    us_indices = [
        *[ib.Index(x, 'CBOE', 'USD') for x in ['VIX', 'VXN', 'RVX', 'VXSLV', 'GVZ', 'OVX']],
        ib.Index('SPX', 'CBOE', 'USD'),
        ib.Index('NDX', 'NASDAQ', 'USD'),
        ib.Index('RUT', 'RUSSELL', 'USD'),
        ib.Index('INDU', 'CME', 'USD'),
    ]
    fr_index = ib.Index('CAC40', 'MONEP', 'EUR')
    jp_index = ib.Index('N225', 'OSE.JPN', 'JPY')
    hk_index = ib.Index('HSI', 'HKFE', 'HKD')

    index_cfd_euro = ['IBGB100', 'IBEU50', 'IBDE40', 'IBFR40', 'IBES35', 'IBNL25', 'IBCH20']
    index_cfd_us = ['IBUS500', 'IBUS30', 'IBUST100']
    index_cfd_asia = ['IBHK50', 'IBJP225', 'IBAU200']
    index_cfds = [ib.CFD(symbol=s, exchange='SMART')
                  for s in [*index_cfd_euro, *index_cfd_us, *index_cfd_asia]]

    commodity_cfds = [ib.Commodity('XAUUSD', exchange='SMART'),
                      ib.Commodity('USGOLD', exchange='IBMETAL')]

    forex = [ib.Forex(symbol=sym, exchange='IDEALPRO', currency=cur) for sym, cur in
             [('EUR', 'USD'), ('EUR', 'GBP'), ('EUR', 'CHF'), ('GBP', 'USD'),
              ('AUD', 'USD'), ('USD', 'CAD'), ('USD', 'JPY'), ('CHF', 'USD')]]

    eu_futures = [ib.ContFuture(symbol=x, multiplier='1', exchange='EUREX', currency='EUR')
                  for x in ['DAX', 'ESTX50']]
    us_futures = [ib.ContFuture(symbol=x[0], multiplier=x[1], exchange=x[2], currency='USD')
                  for x in [
                      ('MES', '5', 'CME'), ('MNQ', '2', 'CME'), ('RTY', '50', 'CME'),
                      ('MYM', '0.5', 'CBOT'), ('VXM', '100', 'CFE'),
                      ('ZB', '1000', 'CBOT'), ('ZC', '5000', 'CBOT'), ('ZF', '1000', 'CBOT'),
                      ('ZN', '1000', 'CBOT'), ('ZT', '2000', 'CBOT'), ('ZW', '5000', 'CBOT'),
                      ('SI', '5000', 'COMEX'), ('GC', '100', 'COMEX'),
                      ('CL', '1000', 'NYMEX'), ('NG', '10000', 'NYMEX'),
                  ]]
    jp_futures = [ib.ContFuture(symbol='N225M', multiplier='100', exchange='OSE.JPN', currency='JPY')]
    swe_futures = [ib.ContFuture(symbol='OMXS30', multiplier='100', exchange='OMS', currency='SEK')]

    us_etf_symbols = [
        'EEM', 'EWZ', 'FXI', 'GDX', 'GLD', 'HYG', 'IEFA', 'IWM', 'LQD', 'QQQ',
        'SLV', 'SMH', 'SPY', 'TLT', 'TQQQ', 'UNG', 'USO', 'XLB', 'XLC', 'XLE',
        'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY', 'XOP',
    ]
    us_etfs = [ib.Stock(symbol=x, exchange='SMART', currency='USD') for x in us_etf_symbols]

    return [
        *eu_indices, *us_indices, jp_index, fr_index, hk_index,
        *us_etfs, *commodity_cfds, *index_cfds, *forex,
        *eu_futures, *us_futures, *jp_futures, *swe_futures,
    ]


def contract_to_symbol(contract) -> str:
    """Map an IBKR contract to its Parquet symbol name."""
    if contract.secType in ('IND', 'CFD', 'STK', 'CMDTY'):
        return contract.symbol
    if 'FUT' in contract.secType:
        return f'F{contract.symbol}'
    if contract.secType == 'CASH':
        return contract.symbol + contract.currency
    raise ValueError(f'Unknown secType: {contract.secType}')


def update_contract(ib_con, contract, symbol: str, schema: str) -> dict:
    """Fetch new bars from IBKR and append to the symbol's Parquet file.

    Returns a summary dict: {bars: int, chunks: int, skipped: bool, error: str | None}.
    """
    last_time = get_last(symbol, schema)
    total_bars = 0
    total_chunks = 0

    if last_time is not None:
        current_date = last_time.to_pydatetime() if hasattr(last_time, 'to_pydatetime') else last_time
        if current_date.replace(tzinfo=None).date() > (datetime.now() - timedelta(days=SKIP_DAYS)).date():
            log.info('  %s: up to date (last=%s), skipping', symbol, current_date.date())
            return {'bars': 0, 'chunks': 0, 'skipped': True, 'error': None}
    else:
        current_date = FALLBACK_START

    end_date = datetime.now()
    types = ibkr_utils.TYPES_OF_DATA[contract.secType]

    for typ in types:
        if typ == 'OPTION_IMPLIED_VOLATILITY' and contract.symbol in ibkr_utils.NO_OOI_INDICES:
            continue
        if typ == 'HISTORICAL_VOLATILITY' and contract.symbol in ibkr_utils.NO_HV_INDICES:
            continue

        cursor = current_date.replace(tzinfo=None) if hasattr(current_date, 'tzinfo') and current_date.tzinfo else current_date
        consecutive_errors = 0
        while cursor < end_date:
            cursor = cursor + pd.Timedelta(days=OFFSET_DAYS)
            end_str = cursor.strftime('%Y%m%d %H:%M:%S') if contract.secType != 'CONTFUT' else ''

            try:
                data = ib_con.reqHistoricalData(
                    contract, endDateTime=end_str, durationStr=DURATION,
                    barSizeSetting=BAR_SIZE, whatToShow=typ, useRTH=RTH,
                )
            except Exception as e:
                consecutive_errors += 1
                log.warning('  %s => %s %s ERROR: %s', symbol, typ, cursor.date(), e)
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    log.error('  %s/%s: %d consecutive errors, skipping data type',
                              symbol, typ, MAX_CONSECUTIVE_ERRORS)
                    break
                continue

            if not data:
                consecutive_errors += 1
                log.debug('  %s => %s %s #0 (empty)', symbol, typ, cursor.date())
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    log.warning('  %s/%s: %d consecutive empty responses, skipping data type',
                                symbol, typ, MAX_CONSECUTIVE_ERRORS)
                    break
                continue

            consecutive_errors = 0
            total_chunks += 1
            total_bars += len(data)
            log.info('  %s => %s %s #%d', symbol, typ, cursor.date(), len(data))

            df = ib.util.df(data).rename(columns=ibkr_utils.FIELD_NAME_LU[typ])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.index = df.index.tz_localize('UTC') if df.index.tz is None else df.index.tz_convert('UTC')
            df.index.name = 'time'

            cols = [c for c in BAR_COLUMNS if c in df.columns]
            df = df[cols]

            write_bars(df, symbol, schema)

    return {'bars': total_bars, 'chunks': total_chunks, 'skipped': False, 'error': None}


def main():
    parser = argparse.ArgumentParser(description='Incremental IBKR intraday update')
    parser.add_argument('--schema', help='Update only this schema')
    parser.add_argument('--symbol', help='Update only this symbol (requires --schema)')
    parser.add_argument('--instance', default='real', choices=['real', 'paper', 'api', 'api_paper'],
                        help='TWS instance (default: real)')
    args = parser.parse_args()

    if args.symbol and not args.schema:
        parser.error('--symbol requires --schema')

    setup_logging()

    contracts = build_contracts()

    log.info('Connecting to IBKR (%s)...', args.instance)
    ib_con = ibkr_utils.connect(args.instance, 30, 2)
    log.info('Connected')

    stats = {'updated': 0, 'skipped': 0, 'failed': 0, 'total_bars': 0}
    t_start = datetime.now()

    try:
        for contract in contracts:
            symbol = contract_to_symbol(contract)
            schema = sec_type_to_schema(contract.secType)

            if args.schema and schema != args.schema:
                continue
            if args.symbol and symbol != args.symbol:
                continue

            log.info('[%s/%s]', schema, symbol)
            try:
                ib_con.qualifyContracts(contract)
                result = update_contract(ib_con, contract, symbol, schema)
                if result['skipped']:
                    stats['skipped'] += 1
                else:
                    stats['updated'] += 1
                    stats['total_bars'] += result['bars']
            except Exception as e:
                log.error('  %s FAILED: %s', symbol, e)
                stats['failed'] += 1
                continue
    finally:
        if ib_con.isConnected():
            ib_con.disconnect()

    elapsed = datetime.now() - t_start
    log.info('')
    log.info('=== Summary ===')
    log.info('  Updated:  %d symbols (%d bars)', stats['updated'], stats['total_bars'])
    log.info('  Skipped:  %d symbols (already fresh)', stats['skipped'])
    log.info('  Failed:   %d symbols', stats['failed'])
    log.info('  Elapsed:  %s', str(elapsed).split('.')[0])


if __name__ == '__main__':
    main()
