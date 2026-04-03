# Options Strategies

## Trade Management - Guidelines

- Close trade if one of the premises has been changed. Never 'adapt' trades
  - This usually leads to over-management and sabotages your trade.

**Take profit / Roll Vertically**

- @25 - 75% when selling premium (25% straddle, 50% strangle)
- @100-150% for debit spreads

**Take loss / Roll Horizontally**

- @21 DTE (or defined trade target) if not in Profit
- @100 % Loss for short premium trade
- @50 % Loss for long premium trade

**Increase size by**

- Increase Delta
- Add new strategy/underlying
- Increase contract size (greatly increases tail risk)

## XYZ Trade

- Buy X,Y approx 30Δ, 25Δ PDS
  - Wider spread and higher delta lead to better hedge against slow downmoves
  - Expiration based on Chart and Risk portfolio.
- Sell Z approx 20Δ PUT at the same expiration
  - Lower delta gives higher winrate but prevents effective high delta hedge
  - Half gives option to use the PDS as a free hedge for other trades
- X/Y, Z is to be determined. There are different pros and cons. Higher ZΔ increases the risk on volatility spikes & large losses
  - Potential 111 or 221 would be save options to test out
  - TK trades 112 which needs higher margins and has two time the downside risk as 111
- Manage at approx. 200% max loss or 50Δ of lowest/highest short based on underlying factors.
- Trade to expiration if underlying goes into the direction of the lowest/highes short. If the underlying goes into the different direction it makes sense to use the loosing long option as hedge to scale in.

## Overview

| | Directional | Neutral | Volatility |
|---|---|---|---|
| | PMC[CP] | IC | Calendar |
| | Debit/Credit Spread | Straddle/Strangle | |
| | XYZ | BWB | |
| | Jade Lizzard | Iron Fly | |
| | Synthetic Long/Short | | |

## Options Strategies per Price of Underlying

| Price Range | Strategies |
|---|---|
| Low <50 | PMC[CP], Debit/Credit Spread ITM, XYZ, BWB, Synthetic Long/Short, Put/Call, Calendar |
| Modest 50<=x<=1000 | ALL |
| High >1000 | BE CAREFUL!!! |

## XYZ

**Structure**

- Sell Z NPs typically @ 7D to 16D
- Buy PDS or Put @ 30D to 25D with a width of approx 2D - 5D

**Manage**

- Set SL on NP only approx. 200% - 250%
- Alternative manage by tasty rules @ 21 DTE

## Vertical Spread Plays

Well known: PMCC, PMCP

**Typical setup**

- Long term P/C ITM
- Sell P/C against it with shorter time duration to collect premium based on price action

## Strangles

- Best R/R is typically in the 16 - 30Delta range
- SPY best BT setup 16D Put / 10D Call

## PMCC/PMCP

- Sell C/P 120-XXX DTE 70-80Δ
  - Alternatively go synthetic long/short by buying a call and selling a put at the same strike
- Sell short term 7-45 DTE C/P against it

## Paid Call/Put (Jade Lizzard)

Directional trade with a defined R/R and no upfront cost

- Sell a Strangle approx. 20D on the directional side and 30D on the non directional side
- Buy a P/C on the directional side below or directly above the short
  - Above means that the premium of the strangle must be > than the strike difference

## Informative

### Selling Premium

- Usually does not make sense if IVP < 50 or IVR < 30
- Becomes risky if IVP/IVR > 80

**Probability of touch**

- 0.8*Delta for puts / 1* Delta for calls when managing at 21 DTE
- Distribution is skewed towards OTM options, e.g. POT of 10 Delta put is 3%

**Naked vs Spread**

- If you have enough capital to hold the underlying or you can get out at 2.5*premium at a reasonable level, it's fine to go naked. If not, spread it out.

## Swing Trading

- Identify overpriced / underpriced stocks
- Positive is
  - When the sector / neighbouring stocks are also increasing / decreasing
  - When latest earnings support your trading direction
- For option trades make sure you define a SL below the swing low / high
