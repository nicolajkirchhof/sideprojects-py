using tradelog.Models;
using tradelog.Services;

namespace tradelog.Tests.Services;

public class OptionComputationsTests
{
    [Fact]
    public void SingleOpenPut_AggregatesCorrectly()
    {
        var positions = new List<OptionPosition>
        {
            new()
            {
                ContractId = "100", Symbol = "SPY", Pos = -1, Right = PositionRight.Put,
                Strike = 500m, Cost = 5m, Multiplier = 100, Commission = 1.5m
            }
        };

        var logs = new Dictionary<string, OptionPositionsLog>
        {
            ["100"] = new()
            {
                ContractId = "100", Price = 3m, Delta = -0.3m, Theta = 0.05m,
                Gamma = 0.01m, Vega = 0.15m, Iv = 0.20m, Margin = 5000m
            }
        };

        var agg = OptionComputations.ComputePortfolioAggregation(positions, logs);

        // unrealized = pos * (price - cost) * multiplier = -1 * (3 - 5) * 100 = 200
        Assert.Equal(200m, agg.UnrealizedPnl);
        Assert.Equal(0m, agg.RealizedPnl);
        Assert.Equal(200m, agg.TotalPnl);
        // delta = log.delta * pos = -0.3 * -1 = 0.3
        Assert.Equal(0.3m, agg.NetDelta);
        // theta = log.theta * pos = 0.05 * -1 = -0.05
        Assert.Equal(-0.05m, agg.NetTheta);
        Assert.Equal(5000m, agg.TotalMargin);
        Assert.Equal(1.5m, agg.TotalCommissions);
        // IV = 0.20 * 100 = 20.0
        Assert.Equal(20.0m, agg.AvgIv);
    }

    [Fact]
    public void ClosedPosition_ComputesRealizedPnl()
    {
        var positions = new List<OptionPosition>
        {
            new()
            {
                ContractId = "200", Symbol = "AAPL", Pos = -2, Right = PositionRight.Put,
                Strike = 180m, Cost = 4m, ClosePrice = 1m, Multiplier = 100,
                Commission = 3m, Closed = DateTime.UtcNow.AddDays(-1)
            }
        };

        var logs = new Dictionary<string, OptionPositionsLog>();

        var agg = OptionComputations.ComputePortfolioAggregation(positions, logs);

        // realized = (closePrice - cost) * multiplier * pos - commission = (1 - 4) * 100 * -2 - 3 = 597
        Assert.Equal(597m, agg.RealizedPnl);
        Assert.Equal(0m, agg.UnrealizedPnl);
        Assert.Equal(3m, agg.TotalCommissions);
    }

    [Fact]
    public void MixedPutCall_AggregatesGreeks()
    {
        var positions = new List<OptionPosition>
        {
            new() { ContractId = "100", Symbol = "SPY", Pos = -1, Right = PositionRight.Put, Cost = 5m, Strike = 500m, Multiplier = 100 },
            new() { ContractId = "101", Symbol = "SPY", Pos = -1, Right = PositionRight.Call, Cost = 3m, Strike = 520m, Multiplier = 100 },
        };

        var logs = new Dictionary<string, OptionPositionsLog>
        {
            ["100"] = new() { ContractId = "100", Price = 4m, Delta = -0.30m, Theta = 0.05m, Gamma = 0.01m, Vega = 0.10m, Iv = 0.20m, Margin = 3000m },
            ["101"] = new() { ContractId = "101", Price = 2m, Delta = 0.25m, Theta = 0.04m, Gamma = 0.008m, Vega = 0.08m, Iv = 0.18m, Margin = 2000m },
        };

        var agg = OptionComputations.ComputePortfolioAggregation(positions, logs);

        // net delta = (-0.30 * -1) + (0.25 * -1) = 0.30 - 0.25 = 0.05
        Assert.Equal(0.05m, agg.NetDelta);
        // net theta = (0.05 * -1) + (0.04 * -1) = -0.09
        Assert.Equal(-0.09m, agg.NetTheta);
        Assert.Equal(5000m, agg.TotalMargin);
        // avg IV = ((20 + 18) / 2) = 19.0
        Assert.Equal(19.0m, agg.AvgIv);
    }

    [Fact]
    public void EmptyPositions_ReturnsZeroAggregation()
    {
        var agg = OptionComputations.ComputePortfolioAggregation(
            new List<OptionPosition>(),
            new Dictionary<string, OptionPositionsLog>());

        Assert.Equal(0m, agg.TotalPnl);
        Assert.Equal(0m, agg.NetDelta);
        Assert.Equal(0m, agg.TotalMargin);
        Assert.Equal(0m, agg.AvgIv);
    }

    [Fact]
    public void OpenPosition_WithoutLog_SkipsGreeks()
    {
        var positions = new List<OptionPosition>
        {
            new() { ContractId = "300", Symbol = "TSLA", Pos = 1, Right = PositionRight.Call, Cost = 10m, Strike = 250m, Multiplier = 100 }
        };

        var logs = new Dictionary<string, OptionPositionsLog>(); // no log for this position

        var agg = OptionComputations.ComputePortfolioAggregation(positions, logs);

        Assert.Equal(0m, agg.UnrealizedPnl);
        Assert.Equal(0m, agg.NetDelta);
    }
}
