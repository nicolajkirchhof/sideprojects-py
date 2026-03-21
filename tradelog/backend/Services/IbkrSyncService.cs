using IBApi;
using tradelog.Data;
using tradelog.Dtos;
using tradelog.Models;
using Microsoft.EntityFrameworkCore;

namespace tradelog.Services;

public class IbkrSyncService
{
    private readonly DataContext _context;

    public IbkrSyncService(DataContext context)
    {
        _context = context;
    }

    public async Task<IbkrSyncResultDto> SyncAll(Account account)
    {
        var result = new IbkrSyncResultDto();

        using var tws = new IbkrConnectionManager();

        try
        {
            // Step 1: Connect
            tws.Connect(account.Host, account.Port, account.ClientId);

            // Sanity check: verify TWS is managing the expected account
            if (tws.ManagedAccounts.Count > 0 && !string.IsNullOrEmpty(account.IbkrAccountId))
            {
                if (!tws.ManagedAccounts.Contains(account.IbkrAccountId))
                {
                    var managed = string.Join(", ", tws.ManagedAccounts);
                    throw new InvalidOperationException(
                        $"Account mismatch: expected {account.IbkrAccountId} but TWS manages [{managed}]. " +
                        "Check that you selected the correct account and TWS instance.");
                }
            }

            // Step 2: Account summary → Capital snapshot
            result.CapitalCreated = await SyncAccountSummary(tws);

            // Step 3: Positions + Executions
            var (optCreated, optClosed) = await SyncOptionPositions(tws);
            result.OptionPositionsCreated = optCreated;
            result.OptionPositionsClosed = optClosed;

            result.TradesCreated = await SyncTradeExecutions(tws, account.LastSyncAt);

            // Step 4: Greeks + Margin for open option positions
            result.GreeksLogged = await SyncGreeksAndMargin(tws);

            // Step 5: Stock prices for open stock/futures positions
            result.StockPricesUpdated = await SyncStockPrices(tws);

            result.Success = true;
            result.Message = result.ToSummary();
        }
        catch (Exception ex)
        {
            result.Success = false;
            result.Message = $"Sync failed: {ex.Message}";
        }
        finally
        {
            tws.Disconnect();
        }

        return result;
    }

    // ────────────────────────────────────────────
    // Step 2: Account Summary → Capital
    // ────────────────────────────────────────────

    private async Task<int> SyncAccountSummary(IbkrConnectionManager tws)
    {
        var summary = await tws.RequestAccountSummaryAsync();

        decimal Parse(string key) =>
            summary.TryGetValue(key, out var v) && decimal.TryParse(v, out var d) ? d : 0;

        var netLiq = Parse("NetLiquidation");
        var maint = Parse("MaintMarginReq");
        var excess = Parse("AvailableFunds");
        var bpr = Parse("BuyingPower");

        if (netLiq == 0) return 0;

        // Compute portfolio aggregation (reusing existing logic)
        var agg = await PortfolioAggregationService.ComputeAsync(_context);

        var today = DateTime.UtcNow.Date;
        var maintenancePct = netLiq != 0 ? Math.Round(maint * 100 / netLiq, 2) : 0;

        // Upsert: update today's snapshot if it already exists
        var existing = await _context.Capitals.FirstOrDefaultAsync(c => c.Date == today);
        if (existing != null)
        {
            existing.NetLiquidity = netLiq;
            existing.Maintenance = maint;
            existing.ExcessLiquidity = excess;
            existing.Bpr = bpr;
            existing.MaintenancePct = maintenancePct;
            existing.TotalPnl = agg.TotalPnl;
            existing.UnrealizedPnl = agg.UnrealizedPnl;
            existing.RealizedPnl = agg.RealizedPnl;
            existing.NetDelta = agg.NetDelta;
            existing.NetTheta = agg.NetTheta;
            existing.NetVega = agg.NetVega;
            existing.NetGamma = agg.NetGamma;
            existing.AvgIv = agg.AvgIv;
            existing.TotalMargin = agg.TotalMargin;
            existing.TotalCommissions = agg.TotalCommissions;
        }
        else
        {
            _context.Capitals.Add(new Capital
            {
                Date = today,
                NetLiquidity = netLiq,
                Maintenance = maint,
                ExcessLiquidity = excess,
                Bpr = bpr,
                MaintenancePct = maintenancePct,
                TotalPnl = agg.TotalPnl,
                UnrealizedPnl = agg.UnrealizedPnl,
                RealizedPnl = agg.RealizedPnl,
                NetDelta = agg.NetDelta,
                NetTheta = agg.NetTheta,
                NetVega = agg.NetVega,
                NetGamma = agg.NetGamma,
                AvgIv = agg.AvgIv,
                TotalMargin = agg.TotalMargin,
                TotalCommissions = agg.TotalCommissions,
            });
        }

        await _context.SaveChangesAsync();
        return 1;
    }

    // ────────────────────────────────────────────
    // Step 3a: Option Positions
    // ────────────────────────────────────────────

    private async Task<(int created, int closed)> SyncOptionPositions(IbkrConnectionManager tws)
    {
        var portfolioItems = await tws.RequestPositionsAsync();
        var optionItems = portfolioItems
            .Where(p => p.Contract.SecType is "OPT" or "FOP")
            .ToList();

        var existingPositions = await _context.OptionPositions
            .Where(p => p.Closed == null)
            .ToListAsync();

        int created = 0;

        var seenConIds = new HashSet<int>();

        foreach (var item in optionItems)
        {
            var conId = item.Contract.ConId;
            seenConIds.Add(conId);

            // Find existing by ConId or ContractId string
            var existing = existingPositions.FirstOrDefault(p =>
                p.ConId == conId || p.ContractId == conId.ToString());

            if (existing != null)
            {
                // Update ConId/SecType if not set, and position size if changed
                existing.ConId = conId;
                existing.SecType = item.Contract.SecType;
                existing.Pos = (int)item.Position;
            }
            else
            {
                // Parse expiry from contract
                var expiryStr = item.Contract.LastTradeDateOrContractMonth;
                DateTime.TryParseExact(expiryStr, "yyyyMMdd",
                    System.Globalization.CultureInfo.InvariantCulture,
                    System.Globalization.DateTimeStyles.None, out var expiry);

                var multiplier = 100;
                if (int.TryParse(item.Contract.Multiplier, out var m)) multiplier = m;

                var right = item.Contract.Right == "P" ? PositionRight.Put : PositionRight.Call;
                var cost = (decimal)(item.AvgCost / multiplier);

                var newPos = new OptionPosition
                {
                    Symbol = item.Contract.Symbol,
                    ContractId = conId.ToString(),
                    ConId = conId,
                    SecType = item.Contract.SecType,
                    Opened = DateTime.UtcNow.Date,
                    Expiry = expiry,
                    Pos = (int)item.Position,
                    Right = right,
                    Strike = (decimal)item.Contract.Strike,
                    Cost = cost,
                    Multiplier = multiplier,
                };

                _context.OptionPositions.Add(newPos);
                created++;
            }
        }

        // Mark closed: positions in DB that are no longer in portfolio
        int closed = 0;
        foreach (var pos in existingPositions)
        {
            var conId = pos.ConId ?? 0;
            if (conId > 0 && !seenConIds.Contains(conId))
            {
                pos.Closed = DateTime.UtcNow.Date;
                closed++;
            }
        }

        await _context.SaveChangesAsync();
        return (created, closed);
    }

    // ────────────────────────────────────────────
    // Step 3b: Trade Executions (STK/FUT)
    // ────────────────────────────────────────────

    private async Task<int> SyncTradeExecutions(IbkrConnectionManager tws, DateTime? lastSyncAt)
    {
        // Format for TWS: "yyyyMMdd HH:mm:ss" or "yyyyMMdd-HH:mm:ss"
        var sinceStr = lastSyncAt?.ToString("yyyyMMdd-HH:mm:ss") ?? "";
        var executions = await tws.RequestExecutionsAsync(sinceStr);

        var stockFutureExecs = executions
            .Where(e => e.Contract.SecType is "STK" or "FUT")
            .ToList();

        if (stockFutureExecs.Count == 0) return 0;

        // Get existing execution IDs for dedup
        var execIds = stockFutureExecs.Select(e => e.Execution.ExecId).ToList();
        var existingExecIds = await _context.Trades
            .Where(t => t.ExecutionId != null && execIds.Contains(t.ExecutionId))
            .Select(t => t.ExecutionId)
            .ToHashSetAsync();

        int created = 0;
        foreach (var exec in stockFutureExecs)
        {
            if (existingExecIds.Contains(exec.Execution.ExecId))
                continue;

            var multiplier = 1;
            if (int.TryParse(exec.Contract.Multiplier, out var m)) multiplier = m;

            // Parse execution time
            DateTime.TryParseExact(exec.Execution.Time, "yyyyMMdd-HH:mm:ss",
                System.Globalization.CultureInfo.InvariantCulture,
                System.Globalization.DateTimeStyles.None, out var tradeDate);
            if (tradeDate == default) tradeDate = DateTime.UtcNow;

            var posChange = exec.Execution.Side == "BOT"
                ? (int)exec.Execution.Shares
                : -(int)exec.Execution.Shares;

            var trade = new Trade
            {
                Symbol = exec.Contract.Symbol,
                Date = tradeDate,
                PosChange = posChange,
                Price = (decimal)exec.Execution.Price,
                Commission = 0, // Commission comes via separate callback; we'll leave 0 for now
                Multiplier = multiplier,
                ConId = exec.Contract.ConId,
                ExecutionId = exec.Execution.ExecId,
            };

            _context.Trades.Add(trade);
            created++;
        }

        if (created > 0)
            await _context.SaveChangesAsync();

        return created;
    }

    // ────────────────────────────────────────────
    // Step 4: Greeks + Margin
    // ────────────────────────────────────────────

    private async Task<int> SyncGreeksAndMargin(IbkrConnectionManager tws)
    {
        var openPositions = await _context.OptionPositions
            .Where(p => p.Closed == null && p.ConId != null)
            .ToListAsync();

        if (openPositions.Count == 0) return 0;

        int logged = 0;
        var now = DateTime.UtcNow;

        foreach (var pos in openPositions)
        {
            try
            {
                var contract = new Contract
                {
                    ConId = pos.ConId!.Value,
                    Exchange = "SMART",
                    SecType = pos.SecType ?? "OPT",
                };

                // Request market data (snapshot)
                var snapshot = await tws.RequestMarketDataAsync(contract);

                // Determine price
                decimal price = snapshot.Last;
                if (price <= 0 && snapshot.Bid > 0 && snapshot.Ask > 0)
                    price = (snapshot.Bid + snapshot.Ask) / 2;

                // Greeks
                var delta = snapshot.HasGreeks ? snapshot.Delta : 0;
                var theta = snapshot.HasGreeks ? snapshot.Theta : 0;
                var gamma = snapshot.HasGreeks ? snapshot.Gamma : 0;
                var vega = snapshot.HasGreeks ? snapshot.Vega : 0;
                var iv = snapshot.HasGreeks && snapshot.ImpliedVol > 0 ? snapshot.ImpliedVol : 0;
                var underlyingPrice = snapshot.UnderlyingPrice > 0
                    ? (decimal)snapshot.UnderlyingPrice
                    : 0;

                // Compute time value
                decimal moneyness = pos.Right == PositionRight.Put
                    ? pos.Strike - underlyingPrice
                    : underlyingPrice - pos.Strike;
                decimal timeValue = moneyness > 0 ? price - moneyness : price;

                // Margin via whatIfOrder
                decimal margin = 0;
                try
                {
                    var order = new Order
                    {
                        Action = pos.Pos > 0 ? "SELL" : "BUY",
                        TotalQuantity = Math.Abs(pos.Pos),
                        OrderType = "MKT",
                        Tif = "DAY",
                    };
                    var state = await tws.RequestWhatIfOrderAsync(contract, order);
                    if (decimal.TryParse(state.MaintMarginChange, out var marginChange))
                        margin = Math.Abs(marginChange);
                }
                catch
                {
                    // whatIfOrder can fail for some contract types; margin stays 0
                }

                var log = new OptionPositionsLog
                {
                    DateTime = now,
                    ContractId = pos.ContractId,
                    Underlying = underlyingPrice,
                    Iv = (decimal)iv,
                    Price = price,
                    TimeValue = timeValue,
                    Delta = (decimal)delta,
                    Theta = (decimal)theta,
                    Gamma = (decimal)gamma,
                    Vega = (decimal)vega,
                    Margin = margin,
                };

                _context.OptionPositionsLogs.Add(log);
                logged++;

                // Small delay to avoid TWS rate limits
                await Task.Delay(200);
            }
            catch
            {
                // Skip individual position errors, continue with next
            }
        }

        if (logged > 0)
            await _context.SaveChangesAsync();

        return logged;
    }

    // ────────────────────────────────────────────
    // Step 5: Stock Prices
    // ────────────────────────────────────────────

    private async Task<int> SyncStockPrices(IbkrConnectionManager tws)
    {
        // Determine which symbols have open stock/futures positions
        var trades = await _context.Trades
            .OrderBy(t => t.Symbol).ThenBy(t => t.Date).ThenBy(t => t.Id)
            .ToListAsync();

        var tradeDtos = TradeComputations.ComputeRunningFields(trades);
        var openSymbols = tradeDtos
            .GroupBy(d => d.Symbol)
            .Where(g => g.Last().TotalPos != 0)
            .Select(g => new { Symbol = g.Key, Last = g.Last() })
            .ToList();

        if (openSymbols.Count == 0) return 0;

        int updated = 0;
        var now = DateTime.UtcNow;

        foreach (var sym in openSymbols)
        {
            try
            {
                var secType = sym.Last.Multiplier > 1 ? "FUT" : "STK";
                var contract = new Contract
                {
                    Symbol = sym.Symbol,
                    SecType = secType,
                    Exchange = "SMART",
                    Currency = "USD",
                };

                var lastPrice = await tws.RequestLastPriceAsync(contract);
                if (lastPrice <= 0) continue;

                var cached = await _context.StockPriceCaches
                    .FirstOrDefaultAsync(c => c.Symbol == sym.Symbol);

                if (cached != null)
                {
                    cached.LastPrice = lastPrice;
                    cached.UpdatedAt = now;
                }
                else
                {
                    _context.StockPriceCaches.Add(new StockPriceCache
                    {
                        Symbol = sym.Symbol,
                        LastPrice = lastPrice,
                        UpdatedAt = now,
                    });
                }

                updated++;
                await Task.Delay(200);
            }
            catch
            {
                // Skip individual errors
            }
        }

        if (updated > 0)
            await _context.SaveChangesAsync();

        return updated;
    }

}
