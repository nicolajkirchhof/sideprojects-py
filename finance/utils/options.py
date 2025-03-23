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
