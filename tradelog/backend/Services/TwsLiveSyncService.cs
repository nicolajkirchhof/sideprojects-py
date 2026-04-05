using IBApi;
using tradelog.Data;
using tradelog.Dtos;
using tradelog.Models;
using Microsoft.EntityFrameworkCore;

namespace tradelog.Services;

/// <summary>
/// Connects to TWS for live data only: Greeks/IV snapshots for open options
/// and last prices for open stock/futures positions.
/// </summary>
public class TwsLiveSyncService
{
    private readonly DataContext _context;
    private readonly ILogger<TwsLiveSyncService> _logger;
    private readonly ILoggerFactory _loggerFactory;

    public TwsLiveSyncService(DataContext context, ILogger<TwsLiveSyncService> logger, ILoggerFactory loggerFactory)
    {
        _context = context;
        _logger = logger;
        _loggerFactory = loggerFactory;
    }

    public async Task<LiveSyncResultDto> SyncAll(Account account)
    {
        var result = new LiveSyncResultDto();

        using var tws = new IbkrConnectionManager(_loggerFactory.CreateLogger<IbkrConnectionManager>());

        try
        {
            tws.Connect(account.Host, account.Port, account.ClientId);

            if (tws.ManagedAccounts.Count > 0 && !string.IsNullOrEmpty(account.IbkrAccountId))
            {
                if (!tws.ManagedAccounts.Contains(account.IbkrAccountId))
                {
                    var managed = string.Join(", ", tws.ManagedAccounts);
                    throw new InvalidOperationException(
                        $"Account mismatch: expected {account.IbkrAccountId} but TWS manages [{managed}].");
                }
            }

            _logger.LogInformation("Live sync: fetching Greeks for open option positions...");
            result.GreeksLogged = await SyncGreeksAndMargin(tws);

            _logger.LogInformation("Live sync: fetching stock prices...");
            result.StockPricesUpdated = await SyncStockPrices(tws);

            _logger.LogInformation("Live sync: recomputing capital aggregations...");
            result.CapitalAggregationsUpdated = await RecomputeCapitalAggregations();

            result.Success = true;
            result.Message = result.ToSummary();
        }
        catch (Exception ex)
        {
            result.Success = false;
            result.Message = $"Live sync failed: {ex.Message}";
            _logger.LogError(ex, "Live sync failed");
        }
        finally
        {
            tws.Disconnect();
        }

        return result;
    }

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
            _logger.LogInformation("SyncGreeks: {Symbol} conId={ConId} secType={SecType} strike={Strike} {Right} pos={Pos}",
                pos.Symbol, pos.ConId, pos.SecType, pos.Strike, pos.Right, pos.Pos);
            try
            {
                var contract = new Contract
                {
                    ConId = pos.ConId!.Value,
                    Exchange = "SMART",
                    SecType = pos.SecType ?? "OPT",
                };

                var snapshot = await tws.RequestMarketDataAsync(contract);

                decimal price = snapshot.Last;
                if (price <= 0 && snapshot.Bid > 0 && snapshot.Ask > 0)
                    price = (snapshot.Bid + snapshot.Ask) / 2;

                var delta = snapshot.HasGreeks ? snapshot.Delta : 0;
                var theta = snapshot.HasGreeks ? snapshot.Theta : 0;
                var gamma = snapshot.HasGreeks ? snapshot.Gamma : 0;
                var vega = snapshot.HasGreeks ? snapshot.Vega : 0;
                var iv = snapshot.HasGreeks && snapshot.ImpliedVol > 0 ? snapshot.ImpliedVol : 0;
                var underlyingPrice = snapshot.UnderlyingPrice > 0
                    ? (decimal)snapshot.UnderlyingPrice : 0;

                decimal moneyness = pos.Right == PositionRight.Put
                    ? pos.Strike - underlyingPrice
                    : underlyingPrice - pos.Strike;
                decimal timeValue = moneyness > 0 ? price - moneyness : price;

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
                catch (Exception ex)
                {
                    _logger.LogWarning("whatIfOrder failed for {Symbol} conId={ConId}: {Error}",
                        pos.Symbol, pos.ConId, ex.Message);
                }

                _context.OptionPositionsLogs.Add(new OptionPositionsLog
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
                });
                logged++;

                _logger.LogInformation("Logged {Symbol} conId={ConId}: price={Price} delta={Delta} theta={Theta} iv={Iv} margin={Margin}",
                    pos.Symbol, pos.ConId, price, delta, theta, iv, margin);

                await Task.Delay(200);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "SyncGreeks failed for {Symbol} conId={ConId}", pos.Symbol, pos.ConId);
            }
        }

        if (logged > 0)
            await _context.SaveChangesAsync();

        return logged;
    }

    private async Task<int> SyncStockPrices(IbkrConnectionManager tws)
    {
        var trades = await _context.StockPositions
            .OrderBy(t => t.Symbol).ThenBy(t => t.Date).ThenBy(t => t.Id)
            .ToListAsync();

        var tradeDtos = StockPositionComputations.ComputeRunningFields(trades);
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
                _logger.LogInformation("Requesting price for {Symbol} secType={SecType}", sym.Symbol, secType);

                var contract = new Contract
                {
                    Symbol = sym.Symbol,
                    SecType = secType,
                    Exchange = "SMART",
                    Currency = "USD",
                };

                var lastPrice = await tws.RequestLastPriceAsync(contract);
                _logger.LogInformation("{Symbol} lastPrice={LastPrice}", sym.Symbol, lastPrice);
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
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Price fetch failed for {Symbol}", sym.Symbol);
            }
        }

        if (updated > 0)
            await _context.SaveChangesAsync();

        return updated;
    }

    /// <summary>
    /// Recomputes today's Capital aggregations (Greeks totals, margin, PnL)
    /// using the latest OptionPositionsLog data. Only updates if a Capital row exists for today.
    /// </summary>
    private async Task<int> RecomputeCapitalAggregations()
    {
        var today = DateTime.UtcNow.Date;
        var capital = await _context.Capitals.FirstOrDefaultAsync(c => c.Date == today);
        if (capital == null) return 0;

        var agg = await PortfolioAggregationService.ComputeAsync(_context);

        capital.MaintenancePct = capital.NetLiquidity != 0
            ? Math.Round(capital.Maintenance * 100 / capital.NetLiquidity, 2) : 0;
        capital.TotalPnl = agg.TotalPnl;
        capital.UnrealizedPnl = agg.UnrealizedPnl;
        capital.RealizedPnl = agg.RealizedPnl;
        capital.NetDelta = agg.NetDelta;
        capital.NetTheta = agg.NetTheta;
        capital.NetVega = agg.NetVega;
        capital.NetGamma = agg.NetGamma;
        capital.AvgIv = agg.AvgIv;
        capital.TotalMargin = agg.TotalMargin;
        capital.TotalCommissions = agg.TotalCommissions;

        await _context.SaveChangesAsync();
        return 1;
    }
}
