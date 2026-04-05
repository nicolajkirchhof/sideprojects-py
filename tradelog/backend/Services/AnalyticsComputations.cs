using tradelog.Dtos;
using tradelog.Models;

namespace tradelog.Services;

public static class AnalyticsComputations
{
    /// <summary>
    /// Computes realized P/L per symbol from closed option positions and stock/futures trades.
    /// Returns a list of (symbol, closedDate, pnl, commission) tuples for downstream grouping.
    /// </summary>
    public static List<SymbolPnl> ComputeClosedPnls(
        List<OptionPosition> closedOptionPositions,
        List<StockPosition> allStockPositions)
    {
        var results = new List<SymbolPnl>();

        // Closed option positions: each position is a "trade result"
        foreach (var p in closedOptionPositions)
        {
            if (p.Closed == null || !p.ClosePrice.HasValue) continue;
            var pnl = (p.ClosePrice.Value - p.Cost) * p.Multiplier * p.Pos - p.Commission;
            results.Add(new SymbolPnl
            {
                Symbol = p.Symbol,
                Date = p.Closed.Value,
                Pnl = pnl,
                Commission = p.Commission,
            });
        }

        // Stock/futures trades: aggregate realized P/L per symbol into one logical trade
        if (allStockPositions.Count > 0)
        {
            var tradeDtos = StockPositionComputations.ComputeRunningFields(allStockPositions);
            foreach (var symbolGroup in tradeDtos.Where(d => d.Pnl != 0).GroupBy(d => d.Symbol))
            {
                results.Add(new SymbolPnl
                {
                    Symbol = symbolGroup.Key,
                    Date = symbolGroup.Max(d => d.Date),
                    Pnl = symbolGroup.Sum(d => d.Pnl),
                    Commission = symbolGroup.Sum(d => d.Commission),
                });
            }
        }

        return results;
    }

    /// <summary>
    /// Computes performance metrics from a set of P/L entries, sorted by date for drawdown accuracy.
    /// </summary>
    public static (decimal avgWin, decimal avgLoss, decimal winRate, decimal expectancy, decimal maxDrawdown)
        ComputeMetrics(List<SymbolPnl> entries)
    {
        if (entries.Count == 0)
            return (0, 0, 0, 0, 0);

        var pnls = entries.Select(e => e.Pnl).ToList();
        var wins = pnls.Where(p => p > 0).ToList();
        var losses = pnls.Where(p => p < 0).ToList();

        var avgWin = wins.Count > 0 ? Math.Round(wins.Average(), 2) : 0;
        var avgLoss = losses.Count > 0 ? Math.Round(losses.Average(), 2) : 0;
        var winRate = Math.Round((decimal)wins.Count / pnls.Count * 100, 1);

        // Expectancy = winRate/100 * avgWin + (1 - winRate/100) * avgLoss
        var wr = winRate / 100;
        var expectancy = Math.Round(wr * avgWin + (1 - wr) * avgLoss, 2);

        // Max drawdown: largest peak-to-trough in running sum, sorted chronologically
        decimal peak = 0, maxDd = 0, running = 0;
        foreach (var e in entries.OrderBy(e => e.Date))
        {
            running += e.Pnl;
            if (running > peak) peak = running;
            var dd = peak - running;
            if (dd > maxDd) maxDd = dd;
        }

        return (avgWin, avgLoss, winRate, expectancy, Math.Round(maxDd, 2));
    }

    public static List<EquityCurvePointDto> BuildEquityCurve(List<SymbolPnl> pnls)
    {
        var sorted = pnls.OrderBy(p => p.Date).ToList();
        var curve = new List<EquityCurvePointDto>();
        decimal cumulative = 0;

        foreach (var g in sorted.GroupBy(p => p.Date).OrderBy(g => g.Key))
        {
            cumulative += g.Sum(x => x.Pnl);
            curve.Add(new EquityCurvePointDto
            {
                Date = g.Key,
                CumulativePnl = Math.Round(cumulative, 2),
            });
        }

        return curve;
    }
}

public class SymbolPnl
{
    public string Symbol { get; set; } = string.Empty;
    public DateTime Date { get; set; }
    public decimal Pnl { get; set; }
    public decimal Commission { get; set; }
}
