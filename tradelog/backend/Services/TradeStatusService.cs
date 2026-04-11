using Microsoft.EntityFrameworkCore;
using tradelog.Data;
using tradelog.Models;

namespace tradelog.Services;

/// <summary>
/// Maintains the cached <see cref="Trade.Status"/> field.
/// Status logic: null = no positions linked, "Open" = any position open,
/// "Closed" = all positions closed.
/// </summary>
public class TradeStatusService
{
    private readonly DataContext _context;

    public TradeStatusService(DataContext context)
    {
        _context = context;
    }

    /// <summary>
    /// Recomputes Status for the given trade IDs.
    /// Caller is responsible for <see cref="DataContext.SaveChangesAsync"/>.
    /// </summary>
    public async Task RecomputeForAsync(IEnumerable<int> tradeIds, CancellationToken ct = default)
    {
        var ids = tradeIds.Where(id => id > 0).Distinct().ToList();
        if (ids.Count == 0) return;

        var trades = await _context.Trades
            .IgnoreQueryFilters()
            .Where(t => ids.Contains(t.Id))
            .ToListAsync(ct);

        var optionsByTrade = await _context.OptionPositions
            .IgnoreQueryFilters()
            .Where(p => p.TradeId != null && ids.Contains(p.TradeId.Value))
            .GroupBy(p => p.TradeId!.Value)
            .ToDictionaryAsync(g => g.Key, g => g.ToList(), ct);

        var stocksByTrade = await _context.StockPositions
            .IgnoreQueryFilters()
            .Where(p => p.TradeId != null && ids.Contains(p.TradeId.Value))
            .GroupBy(p => p.TradeId!.Value)
            .ToDictionaryAsync(g => g.Key, g => g.ToList(), ct);

        foreach (var trade in trades)
        {
            var opts = optionsByTrade.GetValueOrDefault(trade.Id) ?? [];
            var stks = stocksByTrade.GetValueOrDefault(trade.Id) ?? [];

            if (opts.Count == 0 && stks.Count == 0)
            {
                trade.Status = null;
                continue;
            }

            var anyOptionOpen = opts.Any(p => p.Closed == null);
            var anyStockOpen = IsAnyStockPositionOpen(stks);

            trade.Status = (anyOptionOpen || anyStockOpen) ? "Open" : "Closed";
        }
    }

    /// <summary>
    /// Backfill: recomputes Status for all trades that have linked positions
    /// but null Status. Runs once at startup, becomes a no-op on subsequent runs.
    /// </summary>
    public async Task<int> RecomputeForPendingAsync(CancellationToken ct = default)
    {
        var tradeIdsWithPositions = await _context.OptionPositions
            .IgnoreQueryFilters()
            .Where(p => p.TradeId != null)
            .Select(p => p.TradeId!.Value)
            .Union(
                _context.StockPositions
                    .IgnoreQueryFilters()
                    .Where(p => p.TradeId != null)
                    .Select(p => p.TradeId!.Value)
            )
            .Distinct()
            .ToListAsync(ct);

        var pending = await _context.Trades
            .IgnoreQueryFilters()
            .Where(t => t.Status == null && tradeIdsWithPositions.Contains(t.Id))
            .Select(t => t.Id)
            .ToListAsync(ct);

        if (pending.Count == 0) return 0;

        await RecomputeForAsync(pending, ct);
        await _context.SaveChangesAsync(ct);
        return pending.Count;
    }

    /// <summary>
    /// Determines if any stock position within a trade's linked stocks represents
    /// an open position (running total != 0).
    /// </summary>
    private static bool IsAnyStockPositionOpen(List<StockPosition> stocks)
    {
        if (stocks.Count == 0) return false;
        var dtos = StockPositionComputations.ComputeRunningFields(
            stocks.OrderBy(s => s.Symbol).ThenBy(s => s.Date).ThenBy(s => s.Id).ToList());
        // Check last entry per symbol — if TotalPos != 0, the position is open
        return dtos
            .GroupBy(d => d.Symbol)
            .Any(g => g.Last().TotalPos != 0);
    }
}
