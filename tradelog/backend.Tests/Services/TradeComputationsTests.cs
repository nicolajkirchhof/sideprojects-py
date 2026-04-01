using tradelog.Models;
using tradelog.Services;

namespace tradelog.Tests.Services;

public class TradeComputationsTests
{
    [Fact]
    public void SingleBuy_ReturnsCorrectRunningFields()
    {
        var trades = new List<Trade>
        {
            new() { Id = 1, Symbol = "AAPL", Date = new(2025, 1, 1), PosChange = 100, Price = 150m, Multiplier = 1 }
        };

        var result = TradeComputations.ComputeRunningFields(trades);

        Assert.Single(result);
        Assert.Equal(0, result[0].LastPos);
        Assert.Equal(100, result[0].TotalPos);
        Assert.Equal(150m, result[0].AvgPrice);
        Assert.Equal(0m, result[0].Pnl);
    }

    [Fact]
    public void BuyThenSell_ComputesRealizedPnl()
    {
        var trades = new List<Trade>
        {
            new() { Id = 1, Symbol = "AAPL", Date = new(2025, 1, 1), PosChange = 100, Price = 150m, Multiplier = 1 },
            new() { Id = 2, Symbol = "AAPL", Date = new(2025, 1, 2), PosChange = -100, Price = 160m, Multiplier = 1 },
        };

        var result = TradeComputations.ComputeRunningFields(trades);

        Assert.Equal(2, result.Count);
        // Sell: pnl = posChange * (avgPrice - price) * multiplier = -100 * (150 - 160) * 1 = 1000
        Assert.Equal(1000m, result[1].Pnl);
        Assert.Equal(0, result[1].TotalPos);
    }

    [Fact]
    public void BuyThenSell_WithMultiplier_ScalesPnl()
    {
        var trades = new List<Trade>
        {
            new() { Id = 1, Symbol = "ES", Date = new(2025, 1, 1), PosChange = 1, Price = 5000m, Multiplier = 50 },
            new() { Id = 2, Symbol = "ES", Date = new(2025, 1, 2), PosChange = -1, Price = 5010m, Multiplier = 50 },
        };

        var result = TradeComputations.ComputeRunningFields(trades);

        // pnl = -1 * (5000 - 5010) * 50 = 500
        Assert.Equal(500m, result[1].Pnl);
    }

    [Fact]
    public void MultipleAdds_ComputesWeightedAveragePrice()
    {
        var trades = new List<Trade>
        {
            new() { Id = 1, Symbol = "AAPL", Date = new(2025, 1, 1), PosChange = 100, Price = 150m, Multiplier = 1 },
            new() { Id = 2, Symbol = "AAPL", Date = new(2025, 1, 2), PosChange = 100, Price = 160m, Multiplier = 1 },
        };

        var result = TradeComputations.ComputeRunningFields(trades);

        Assert.Equal(200, result[1].TotalPos);
        // avgPrice = (100*150 + 100*160) / 200 = 155
        Assert.Equal(155m, result[1].AvgPrice);
    }

    [Fact]
    public void MultiSymbol_TrackedIndependently()
    {
        var trades = new List<Trade>
        {
            new() { Id = 1, Symbol = "AAPL", Date = new(2025, 1, 1), PosChange = 100, Price = 150m, Multiplier = 1 },
            new() { Id = 2, Symbol = "MSFT", Date = new(2025, 1, 1), PosChange = 50, Price = 400m, Multiplier = 1 },
        };

        var result = TradeComputations.ComputeRunningFields(trades);

        Assert.Equal(100, result[0].TotalPos);
        Assert.Equal(150m, result[0].AvgPrice);
        Assert.Equal(50, result[1].TotalPos);
        Assert.Equal(400m, result[1].AvgPrice);
    }

    [Fact]
    public void PositionFlip_LongToShort_SplitsPnl()
    {
        var trades = new List<Trade>
        {
            new() { Id = 1, Symbol = "AAPL", Date = new(2025, 1, 1), PosChange = 100, Price = 150m, Multiplier = 1 },
            new() { Id = 2, Symbol = "AAPL", Date = new(2025, 1, 2), PosChange = -150, Price = 160m, Multiplier = 1 },
        };

        var result = TradeComputations.ComputeRunningFields(trades);

        // Selling 150 when holding 100 long: realizes P&L on the 100, then opens -50 short
        // pnl = -150 * (150 - 160) * 1 = 1500
        Assert.Equal(1500m, result[1].Pnl);
        Assert.Equal(-50, result[1].TotalPos);
        // New avg price should be the flip price
        Assert.Equal(160m, result[1].AvgPrice);
    }

    [Fact]
    public void ShortSellThenCover_ComputesRealizedPnl()
    {
        var trades = new List<Trade>
        {
            new() { Id = 1, Symbol = "AAPL", Date = new(2025, 1, 1), PosChange = -100, Price = 160m, Multiplier = 1 },
            new() { Id = 2, Symbol = "AAPL", Date = new(2025, 1, 2), PosChange = 100, Price = 150m, Multiplier = 1 },
        };

        var result = TradeComputations.ComputeRunningFields(trades);

        // Cover: pnl = 100 * (160 - 150) * 1 = 1000
        Assert.Equal(1000m, result[1].Pnl);
        Assert.Equal(0, result[1].TotalPos);
    }

    [Fact]
    public void EmptyInput_ReturnsEmptyList()
    {
        var result = TradeComputations.ComputeRunningFields(new List<Trade>());

        Assert.Empty(result);
    }
}
