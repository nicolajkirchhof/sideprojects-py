using tradelog.Services;

namespace tradelog.Tests.Services;

public class AnalyticsComputationsTests
{
    [Fact]
    public void ComputeMetrics_MixedWinsAndLosses()
    {
        var entries = new List<SymbolPnl>
        {
            new() { Symbol = "A", Date = new(2025, 1, 1), Pnl = 500m },
            new() { Symbol = "B", Date = new(2025, 1, 2), Pnl = -200m },
            new() { Symbol = "C", Date = new(2025, 1, 3), Pnl = 300m },
            new() { Symbol = "D", Date = new(2025, 1, 4), Pnl = -100m },
        };

        var (avgWin, avgLoss, winRate, expectancy, maxDrawdown) =
            AnalyticsComputations.ComputeMetrics(entries);

        Assert.Equal(400m, avgWin);     // (500+300)/2
        Assert.Equal(-150m, avgLoss);   // (-200+-100)/2
        Assert.Equal(50.0m, winRate);   // 2/4
        // expectancy = 0.5 * 400 + 0.5 * -150 = 125
        Assert.Equal(125m, expectancy);
    }

    [Fact]
    public void ComputeMetrics_AllWinners()
    {
        var entries = new List<SymbolPnl>
        {
            new() { Symbol = "A", Date = new(2025, 1, 1), Pnl = 500m },
            new() { Symbol = "B", Date = new(2025, 1, 2), Pnl = 300m },
        };

        var (avgWin, avgLoss, winRate, _, _) = AnalyticsComputations.ComputeMetrics(entries);

        Assert.Equal(400m, avgWin);
        Assert.Equal(0m, avgLoss);
        Assert.Equal(100.0m, winRate);
    }

    [Fact]
    public void ComputeMetrics_AllLosers()
    {
        var entries = new List<SymbolPnl>
        {
            new() { Symbol = "A", Date = new(2025, 1, 1), Pnl = -200m },
            new() { Symbol = "B", Date = new(2025, 1, 2), Pnl = -100m },
        };

        var (avgWin, avgLoss, winRate, _, _) = AnalyticsComputations.ComputeMetrics(entries);

        Assert.Equal(0m, avgWin);
        Assert.Equal(-150m, avgLoss);
        Assert.Equal(0.0m, winRate);
    }

    [Fact]
    public void ComputeMetrics_Empty_ReturnsZeros()
    {
        var (avgWin, avgLoss, winRate, expectancy, maxDrawdown) =
            AnalyticsComputations.ComputeMetrics(new List<SymbolPnl>());

        Assert.Equal(0m, avgWin);
        Assert.Equal(0m, avgLoss);
        Assert.Equal(0m, winRate);
        Assert.Equal(0m, expectancy);
        Assert.Equal(0m, maxDrawdown);
    }

    [Fact]
    public void ComputeMetrics_MaxDrawdown_PeakToTrough()
    {
        // Running sum: 500, 300, 600, 0, 200
        // Peaks:       500, 500, 600, 600, 600
        // Drawdowns:   0,   200, 0,   600, 400
        var entries = new List<SymbolPnl>
        {
            new() { Symbol = "A", Date = new(2025, 1, 1), Pnl = 500m },
            new() { Symbol = "B", Date = new(2025, 1, 2), Pnl = -200m },
            new() { Symbol = "C", Date = new(2025, 1, 3), Pnl = 300m },
            new() { Symbol = "D", Date = new(2025, 1, 4), Pnl = -600m },
            new() { Symbol = "E", Date = new(2025, 1, 5), Pnl = 200m },
        };

        var (_, _, _, _, maxDrawdown) = AnalyticsComputations.ComputeMetrics(entries);

        Assert.Equal(600m, maxDrawdown);
    }

    [Fact]
    public void BuildEquityCurve_CumulativesByDate()
    {
        var pnls = new List<SymbolPnl>
        {
            new() { Symbol = "A", Date = new(2025, 1, 1), Pnl = 100m },
            new() { Symbol = "B", Date = new(2025, 1, 1), Pnl = 50m },  // same date
            new() { Symbol = "C", Date = new(2025, 1, 2), Pnl = -30m },
        };

        var curve = AnalyticsComputations.BuildEquityCurve(pnls);

        Assert.Equal(2, curve.Count);
        Assert.Equal(150m, curve[0].CumulativePnl);   // 100 + 50
        Assert.Equal(120m, curve[1].CumulativePnl);    // 150 - 30
    }

    [Fact]
    public void BuildEquityCurve_Empty_ReturnsEmpty()
    {
        var curve = AnalyticsComputations.BuildEquityCurve(new List<SymbolPnl>());

        Assert.Empty(curve);
    }

    [Fact]
    public void ComputeMetrics_SingleTrade()
    {
        var entries = new List<SymbolPnl>
        {
            new() { Symbol = "A", Date = new(2025, 1, 1), Pnl = 250m },
        };

        var (avgWin, avgLoss, winRate, expectancy, maxDrawdown) =
            AnalyticsComputations.ComputeMetrics(entries);

        Assert.Equal(250m, avgWin);
        Assert.Equal(0m, avgLoss);
        Assert.Equal(100.0m, winRate);
        Assert.Equal(250m, expectancy);
        Assert.Equal(0m, maxDrawdown);
    }
}
