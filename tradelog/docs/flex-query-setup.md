# IBKR Flex Query Setup Guide

## Overview

The Tradelog app uses IBKR Flex Queries as the authoritative source for trades, positions, and account equity. The TWS API is only used for live Greeks/IV snapshots and stock prices.

You need to configure **one Flex Query** in IBKR Account Management and provide the credentials to the Tradelog app.

---

## Step 1: Create a Flex Web Service Token

1. Log in to **IBKR Account Management** (clientportal.interactivebrokers.com)
2. Go to **Settings → Reporting → Flex Web Service**
   (or search "Flex" in the settings search)
3. Click **Generate Token**
4. Copy the token (64 characters). It expires after 1 year — note the expiry date.

> The token is shown only once. Store it securely.

---

## Step 2: Create the Activity Flex Query

1. Go to **Reports → Flex Queries**
2. Click **Create** (Activity Flex Query, not Legacy)
3. Configure as follows:

### General Settings

| Setting | Value |
|---------|-------|
| Query Name | `Tradelog Sync` |
| Period | `Last 365 Calendar Days` |
| Format | `XML` |
| Date Format | `yyyyMMdd` |
| Time Format | `HHmmss` |

### Sections to Enable

Enable **exactly** these three sections and their fields:

#### Trades

Click "Trades" → select these fields:

| Field | Column in XML | Purpose |
|-------|--------------|---------|
| Trade ID | `tradeID` | Dedup key for import |
| Contract ID | `conid` | Links to option positions |
| Symbol | `symbol` | Ticker |
| Security Type | `assetCategory` | STK / FUT / OPT / FOP |
| Date/Time | `dateTime` | Execution timestamp |
| Quantity | `quantity` | Signed (+buy, -sell) |
| Trade Price | `tradePrice` | Execution price |
| IB Commission | `ibCommission` | Commission charged (negative = cost) |
| Net Cash | `netCash` | Net proceeds |
| Buy/Sell | `buySell` | BUY or SELL |
| Strike | `strike` | Option strike (0 for STK/FUT) |
| Expiry | `expiry` | Option expiration (empty for STK/FUT) |
| Put/Call | `putCall` | P or C (empty for STK/FUT) |
| Multiplier | `multiplier` | 1 for STK, 100 for OPT, etc. |
| Currency | `currency` | Trade currency |
| Order Type | `orderType` | LMT / MKT / etc. (audit) |
| Exchange | `exchange` | Execution venue (audit) |

Leave all other fields unchecked.

#### Open Positions

Click "Open Positions" → select these fields:

| Field | Column in XML | Purpose |
|-------|--------------|---------|
| Contract ID | `conid` | Match DB positions |
| Symbol | `symbol` | Display |
| Security Type | `assetCategory` | Filtering |
| Quantity | `position` | Current size |
| Cost Basis Price | `costBasisPrice` | Per-unit cost |
| Cost Basis Money | `costBasisMoney` | Total cost basis |
| Strike | `strike` | Option strike |
| Expiry | `expiry` | Option expiration |
| Put/Call | `putCall` | P or C |
| Multiplier | `multiplier` | Contract size |
| Unrealized P&L | `fifoPnlUnrealized` | Reconciliation |
| Currency | `currency` | Position currency |

#### Equity Summary in Base Currency

Click "Equity Summary in Base Currency" → select these fields:

| Field | Column in XML | Purpose |
|-------|--------------|---------|
| Report Date | `reportDate` | Date key |
| Total | `total` | Net Liquidation Value |
| Long Option Value | `longOptionValue` | Portfolio composition |
| Short Option Value | `shortOptionValue` | Portfolio composition |

> This section provides daily equity snapshots for up to 365 days of history.

### Delivery Configuration

| Setting | Value |
|---------|-------|
| Delivery | Online (no email) |
| Include header/trailer | Yes (default) |

4. Click **Save**
5. Note the **Query ID** shown in the Flex Queries list (numeric, e.g., `987654`)

---

## Step 3: Configure in Tradelog App

1. Go to **Account Settings** in the Tradelog app
2. Enter:
   - **Flex Token**: The 64-character token from Step 1
   - **Flex Query ID**: The numeric query ID from Step 2
3. Save

---

## Step 4: Test

1. Click **"Sync from Flex"** in the app
2. The first sync may take 30-60 seconds (IBKR generates the report on demand)
3. Verify:
   - Trades appear with accurate commissions
   - Option positions show correct open/close lifecycle
   - Capital history shows daily NAV snapshots

---

## Flex Query XML Structure Reference

The returned XML has this structure (simplified):

```xml
<FlexQueryResponse queryName="Tradelog Sync" type="AF">
  <FlexStatements count="1">
    <FlexStatement accountId="U1234567" fromDate="20250401" toDate="20260331">

      <Trades>
        <Trade tradeID="123456789" conid="12345" symbol="SPY"
               assetCategory="OPT" dateTime="20260315;093000"
               quantity="-1" tradePrice="3.50" ibCommission="-1.05"
               netCash="349.00" buySell="SELL" strike="570"
               expiry="20260418" putCall="P" multiplier="100"
               currency="USD" orderType="LMT" exchange="CBOE" />
        <!-- more Trade elements -->
      </Trades>

      <OpenPositions>
        <OpenPosition conid="12345" symbol="SPY" assetCategory="OPT"
                      position="-1" costBasisPrice="-3.50"
                      costBasisMoney="-350.00" strike="570"
                      expiry="20260418" putCall="P" multiplier="100"
                      fifoPnlUnrealized="50.00" currency="USD" />
        <!-- more OpenPosition elements -->
      </OpenPositions>

      <EquitySummaryInBase>
        <EquitySummaryByReportDateInBase reportDate="20260331"
                      total="125000.00" longOptionValue="5000.00"
                      shortOptionValue="-12000.00" />
        <!-- one per day, up to 365 days -->
      </EquitySummaryInBase>

    </FlexStatement>
  </FlexStatements>
</FlexQueryResponse>
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| "Invalid token" | Token expired or mistyped | Regenerate in Account Management |
| "Query not found" | Wrong query ID | Check Flex Queries list for correct ID |
| "Statement generation in progress" | Normal on first request | App retries automatically (up to 2 min) |
| Missing sections in XML | Query not configured correctly | Re-check fields in Step 2 |
| Empty Trades section | No trades in the query period | Check period setting (365 days) |
| "FlexQueryResponse" with error | IBKR-side issue | Check IBKR system status page |

---

## Notes

- **Token expiry**: Tokens expire after ~1 year. The app will show an auth error when this happens — regenerate the token.
- **Rate limits**: IBKR allows ~1 Flex request per 10 minutes per query. The app enforces a cooldown.
- **Data lag**: Flex reports reflect end-of-previous-day data. Intraday trades may not appear until the next day's report.
- **Commissions**: Flex provides final, settled commissions including exchange fees — more accurate than TWS API's real-time estimates.
