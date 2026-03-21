"""
finance.utils._dormant.ibkr_options
=====================================
IBKR options-specific helper functions.
Used by paused options strategies in finance/ibkr/paused/.
"""
from typing import Literal

import ib_async as ib
import numpy as np


def get_options_price(market_data, type):
    return getattr(market_data, type) if getattr(market_data, type) > 0 else getattr(market_data, 'prev' + type.capitalize())


def get_options_data(ib_con, contracts, tick_list="100, 101, 104, 105, 106, 165, 588", signalParameterLive="ask",
                     signalParameterFrozen="last", max_waittime=120):
    contracts = contracts if isinstance(contracts, list) else [contracts]
    ib_con.reqMarketDataType(2)  # Use free, delayed, frozen data
    snapshots = []
    for contract in contracts:
        snapshots.append(ib_con.reqMktData(contract, tick_list, False, False))
    _wait_for_data(ib_con, snapshots, signalParameterFrozen, max_waittime)

    ib_con.reqMarketDataType(1)  # Use free, delayed, frozen data
    _wait_for_data(ib_con, snapshots, signalParameterLive, max_waittime)

    for contract in contracts:
        ib_con.cancelMktData(contract)
    return snapshots


def _wait_for_data(ib_con, objects, indicator_name, max_wait_time):
    wait_time = 0
    while wait_time < max_wait_time and any(
        [ib.util.isNan(getattr(obj, indicator_name)) or getattr(obj, indicator_name) is None for obj in objects]):
        print(f"Waiting {wait_time} / {max_wait_time} for {indicator_name} to be available.")
        ib_con.sleep(1)
        wait_time += 1


def yearly_to_daily_iv(iv):
    if iv is None:
        return None
    return iv / np.sqrt(252)


def get_and_qualify_contract_details(ib_con, contract):
    details = ib_con.reqContractDetails(contract)
    print(details[0].longName)
    ib_con.qualifyContracts(details[0].contract)
    return details[0]


def get_sigma_move(contract_ticker, sigma, num_days):
    sigma_move = contract_ticker.impliedVolatility * np.sqrt(num_days / 365) * contract_ticker.last
    max_value = np.ceil(contract_ticker.last + sigma * sigma_move)
    min_value = np.floor(contract_ticker.last - sigma * sigma_move)
    return sigma_move, max_value, min_value


def get_last_available(ticker, type: Literal["last", "bid", "ask"] = "last"):
    value = getattr(ticker, type)
    if np.isnan(value) or value is None or value < 0:
        value = getattr(ticker, "prev" + type.capitalize())
    return value
