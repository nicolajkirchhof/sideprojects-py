using tradelog.Data;
using Microsoft.EntityFrameworkCore;

namespace tradelog.Services;

/// <summary>
/// Shared computation of portfolio-level aggregations from option positions and trades.
/// Used by CapitalController (on manual create) and TwsLiveSyncService (on live sync).
/// </summary>
public static class PortfolioAggregationService
{
    public static async Task<PortfolioAggregation> ComputeAsync(DataContext context)
    {
        var positions = await context.OptionPositions.ToListAsync();
        var contractIds = positions.Select(p => p.ContractId).Distinct().ToList();

        var latestLogs = await context.OptionPositionsLogs
            .Where(l => contractIds.Contains(l.ContractId))
            .GroupBy(l => l.ContractId)
            .Select(g => g.OrderByDescending(l => l.DateTime).First())
            .ToDictionaryAsync(l => l.ContractId);

        var optionAgg = OptionComputations.ComputePortfolioAggregation(positions, latestLogs);

        var trades = await context.StockPositions
            .OrderBy(t => t.Symbol).ThenBy(t => t.Date).ThenBy(t => t.Id)
            .ToListAsync();
        var tradeDtos = StockPositionComputations.ComputeRunningFields(trades);
        var tradeRealizedPnl = tradeDtos.Sum(d => d.Pnl);
        var tradeCommissions = tradeDtos.Sum(d => d.Commission);

        return new PortfolioAggregation
        {
            TotalPnl = optionAgg.TotalPnl + tradeRealizedPnl,
            UnrealizedPnl = optionAgg.UnrealizedPnl,
            RealizedPnl = optionAgg.RealizedPnl + tradeRealizedPnl,
            NetDelta = optionAgg.NetDelta,
            NetTheta = optionAgg.NetTheta,
            NetGamma = optionAgg.NetGamma,
            NetVega = optionAgg.NetVega,
            AvgIv = optionAgg.AvgIv,
            TotalMargin = optionAgg.TotalMargin,
            TotalCommissions = optionAgg.TotalCommissions + tradeCommissions,
        };
    }
}
