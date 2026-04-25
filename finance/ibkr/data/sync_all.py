"""Run all IBKR data syncs in succession.

Single entry point for catching up or maintaining all market data:
  1. Intraday 1-min bars (all schemas: stk, index, cfd, forex, future)
  2. Daily OHLCV + volatility (SPY, QQQ, VIX, and DRIFT underlyings)

Usage:
  uv run python -m finance.ibkr.data.sync_all
  uv run python -m finance.ibkr.data.sync_all --instance api_paper
  uv run python -m finance.ibkr.data.sync_all --skip-daily
  uv run python -m finance.ibkr.data.sync_all --skip-intraday
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

LOG_DIR = Path('finance/_data/logs')

log = logging.getLogger('sync_all')

INTRADAY_SCHEMAS = ['stk', 'index', 'cfd', 'forex', 'future']

DAILY_SYMBOLS = [
    'SPY', 'QQQ', 'IWM', '$VIX',
    'GLD', 'SLV', 'TLT', 'EEM', 'FXI', 'EWZ',
    'UNG', 'USO', 'TQQQ',
]


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f'sync_all_{datetime.now():%Y%m%d_%H%M%S}.log'

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

    # Only attach handlers to our loggers, not root (avoids ib_async debug spam)
    for name in ('sync_all', 'intraday_update'):
        lgr = logging.getLogger(name)
        lgr.setLevel(logging.DEBUG)
        lgr.addHandler(console)
        lgr.addHandler(fh)

    # Silence noisy third-party loggers
    logging.getLogger('ib_async').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    log.info('Log file: %s', log_file)


def sync_intraday(instance: str) -> None:
    """Run intraday update for all schemas sequentially."""
    from finance.ibkr.data.update_intraday import (
        build_contracts, contract_to_symbol, update_contract,
    )
    from finance.utils import ibkr as ibkr_utils
    from finance.utils.intraday import sec_type_to_schema

    log.info('=== Intraday sync — all schemas ===')

    log.info('Connecting to IBKR (%s) for intraday...', instance)
    ib_con = ibkr_utils.connect(instance, 30, 2)
    log.info('Connected')

    contracts = build_contracts()
    stats = {'updated': 0, 'skipped': 0, 'failed': 0, 'total_bars': 0}

    try:
        for schema in INTRADAY_SCHEMAS:
            log.info('')
            log.info('--- Schema: %s ---', schema)
            schema_contracts = [
                c for c in contracts
                if sec_type_to_schema(c.secType) == schema
            ]
            for contract in schema_contracts:
                symbol = contract_to_symbol(contract)
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
    finally:
        if ib_con.isConnected():
            ib_con.disconnect()

    log.info('')
    log.info('Intraday: %d updated (%d bars), %d skipped, %d failed',
             stats['updated'], stats['total_bars'], stats['skipped'], stats['failed'])


def sync_daily(instance: str) -> None:
    """Refresh daily OHLCV + volatility cache for key symbols."""
    from finance.utils.ibkr import daily_w_volatility, connect

    log.info('')
    log.info('=== Daily OHLCV sync ===')

    log.info('Connecting to IBKR (%s) for daily...', instance)
    ib_con = connect(instance, 31, 2)
    log.info('Connected')

    updated = 0
    failed = 0

    try:
        for symbol in DAILY_SYMBOLS:
            log.info('[daily/%s]', symbol)
            try:
                df = daily_w_volatility(symbol, offline=False, ib_con=ib_con, refresh_offset_days=0)
                if df is not None and not df.empty:
                    log.info('  %s: %d bars, last=%s', symbol, len(df), df.index[-1].date())
                    updated += 1
                else:
                    log.warning('  %s: no data returned', symbol)
                    failed += 1
            except Exception as e:
                log.error('  %s FAILED: %s', symbol, e)
                failed += 1
    finally:
        if ib_con.isConnected():
            ib_con.disconnect()

    log.info('')
    log.info('Daily: %d updated, %d failed', updated, failed)


def main():
    parser = argparse.ArgumentParser(
        description='Run all IBKR data syncs (intraday + daily)',
    )
    parser.add_argument('--instance', default='real',
                        choices=['real', 'paper', 'api', 'api_paper'],
                        help='TWS instance (default: real)')
    parser.add_argument('--skip-intraday', action='store_true',
                        help='Skip intraday 1-min bar sync')
    parser.add_argument('--skip-daily', action='store_true',
                        help='Skip daily OHLCV sync')
    args = parser.parse_args()

    setup_logging()

    t_start = datetime.now()
    log.info('sync_all started — instance=%s', args.instance)

    if not args.skip_intraday:
        sync_intraday(args.instance)

    if not args.skip_daily:
        sync_daily(args.instance)

    elapsed = datetime.now() - t_start
    log.info('')
    log.info('=== All done — elapsed %s ===', str(elapsed).split('.')[0])


if __name__ == '__main__':
    main()
