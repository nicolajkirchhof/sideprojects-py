using tradelog.Dtos;
using tradelog.Models;

namespace tradelog.Services;

public static class OptionComputations
{
    /// <summary>
    /// Compute portfolio-level aggregations from option positions and their latest Greeks snapshots.
    /// Used by both InstrumentSummariesController and CapitalController.
    /// </summary>
    public static PortfolioAggregation ComputePortfolioAggregation(
        List<OptionPosition> positions,
        Dictionary<string, OptionPositionsLog> latestLogs)
    {
        var today = DateTime.UtcNow.Date;
        decimal totalPnl = 0, unrealizedPnl = 0, realizedPnl = 0;
        decimal delta = 0, theta = 0, gamma = 0, vega = 0, margin = 0, commissions = 0;
        var ivs = new List<decimal>();

        foreach (var p in positions)
        {
            var isOpen = p.Closed == null;
            var log = latestLogs.GetValueOrDefault(p.ContractId);
            commissions += p.Commission;

            if (isOpen && log != null)
            {
                var unreal = p.Pos * (log.Price - p.Cost) * p.Multiplier;
                unrealizedPnl += unreal;
                delta += log.Delta * p.Pos;
                theta += log.Theta * p.Pos;
                gamma += log.Gamma;
                vega += log.Vega;
                ivs.Add(log.Iv * 100);
            }

            if (log != null)
                margin += log.Margin;

            if (!isOpen && p.ClosePrice.HasValue)
            {
                var real = (p.ClosePrice.Value - p.Cost) * p.Multiplier * p.Pos - p.Commission;
                realizedPnl += real;
            }
        }

        totalPnl = unrealizedPnl + realizedPnl;

        return new PortfolioAggregation
        {
            TotalPnl = totalPnl,
            UnrealizedPnl = unrealizedPnl,
            RealizedPnl = realizedPnl,
            NetDelta = Math.Round(delta, 4),
            NetTheta = Math.Round(theta, 4),
            NetGamma = Math.Round(gamma, 4),
            NetVega = Math.Round(vega, 4),
            AvgIv = ivs.Count > 0 ? Math.Round(ivs.Average(), 1) : 0,
            TotalMargin = margin,
            TotalCommissions = commissions,
        };
    }
}

public class PortfolioAggregation
{
    public decimal TotalPnl { get; set; }
    public decimal UnrealizedPnl { get; set; }
    public decimal RealizedPnl { get; set; }
    public decimal NetDelta { get; set; }
    public decimal NetTheta { get; set; }
    public decimal NetGamma { get; set; }
    public decimal NetVega { get; set; }
    public decimal AvgIv { get; set; }
    public decimal TotalMargin { get; set; }
    public decimal TotalCommissions { get; set; }
}
