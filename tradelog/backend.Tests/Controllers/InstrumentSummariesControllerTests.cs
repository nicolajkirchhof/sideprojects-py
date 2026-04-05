using Microsoft.AspNetCore.Mvc;
using tradelog.Controllers;
using tradelog.Models;
using tradelog.Tests.Fixtures;

namespace tradelog.Tests.Controllers;

public class InstrumentSummariesControllerTests : IDisposable
{
    private readonly TestDbFixture _fixture;

    public InstrumentSummariesControllerTests()
    {
        _fixture = new TestDbFixture();
        SeedAccount();
    }

    private void SeedAccount()
    {
        using var ctx = _fixture.CreateContext();
        ctx.Accounts.Add(new Account { Id = _fixture.TestAccountId, IbkrAccountId = "U1234", Name = "Test" });
        ctx.SaveChanges();
    }

    [Fact]
    public async Task GetTradeOverview_AggregatesLinkedPositions()
    {
        int tradeId;
        using (var ctx = _fixture.CreateContext())
        {
            var trade = new Trade
            {
                Symbol = "SPY", Date = new(2025, 6, 1),
                TypeOfTrade = TypeOfTrade.ShortPut, Budget = Budget.Drift, Strategy = Strategy.PositiveDrift,
                AccountId = _fixture.TestAccountId
            };
            ctx.Trades.Add(trade);
            await ctx.SaveChangesAsync();
            tradeId = trade.Id;

            // Add two option legs
            ctx.OptionPositions.Add(new OptionPosition
            {
                Symbol = "SPY", ContractId = "C1", Opened = new(2025, 6, 1), Expiry = new(2025, 7, 18),
                Pos = -1, Right = PositionRight.Put, Strike = 540, Cost = 3.50m,
                TradeId = tradeId, AccountId = _fixture.TestAccountId
            });
            ctx.OptionPositions.Add(new OptionPosition
            {
                Symbol = "SPY", ContractId = "C2", Opened = new(2025, 6, 1), Expiry = new(2025, 7, 18),
                Pos = -1, Right = PositionRight.Put, Strike = 530, Cost = 2.00m,
                TradeId = tradeId, AccountId = _fixture.TestAccountId
            });

            // Add a stock leg
            ctx.StockPositions.Add(new StockPosition
            {
                Symbol = "SPY", Date = new(2025, 6, 5), PosChange = 100, Price = 545m,
                TradeId = tradeId, AccountId = _fixture.TestAccountId
            });
            await ctx.SaveChangesAsync();
        }

        using var queryCtx = _fixture.CreateContext();
        var controller = new InstrumentSummariesController(queryCtx);

        var result = await controller.GetTradeOverview(null);

        var summaries = result.Value!.ToList();
        Assert.Single(summaries);

        var summary = summaries[0];
        Assert.Equal(tradeId, summary.TradeId);
        Assert.Equal("SPY", summary.Symbol);
        Assert.Equal("ShortPut", summary.TypeOfTrade);
        Assert.Equal(2, summary.OptionLegCount);
        Assert.Equal(1, summary.StockLegCount);
        Assert.Equal("Open", summary.Status);
    }

    [Fact]
    public async Task GetTradeOverview_IncludesTradesWithNoLegs()
    {
        using (var ctx = _fixture.CreateContext())
        {
            ctx.Trades.Add(new Trade
            {
                Symbol = "QQQ", Date = new(2025, 6, 1),
                TypeOfTrade = TypeOfTrade.ShortStrangle, Budget = Budget.Drift, Strategy = Strategy.PositiveDrift,
                AccountId = _fixture.TestAccountId
            });
            await ctx.SaveChangesAsync();
        }

        using var queryCtx = _fixture.CreateContext();
        var controller = new InstrumentSummariesController(queryCtx);

        var result = await controller.GetTradeOverview(null);

        var summaries = result.Value!.ToList();
        Assert.Single(summaries);
        Assert.Equal(0, summaries[0].OptionLegCount);
        Assert.Equal(0, summaries[0].StockLegCount);
    }

    [Fact]
    public async Task GetTradeOverview_FiltersOpenOnly()
    {
        using (var ctx = _fixture.CreateContext())
        {
            // Open trade with an open option
            var openTrade = new Trade
            {
                Symbol = "SPY", Date = new(2025, 6, 1),
                TypeOfTrade = TypeOfTrade.ShortPut, Budget = Budget.Drift, Strategy = Strategy.PositiveDrift,
                AccountId = _fixture.TestAccountId
            };
            ctx.Trades.Add(openTrade);
            await ctx.SaveChangesAsync();

            ctx.OptionPositions.Add(new OptionPosition
            {
                Symbol = "SPY", ContractId = "C1", Opened = new(2025, 6, 1), Expiry = new(2025, 7, 18),
                Pos = -1, Right = PositionRight.Put, Strike = 540, Cost = 3.50m,
                TradeId = openTrade.Id, AccountId = _fixture.TestAccountId
            });

            // Closed trade with a closed option
            var closedTrade = new Trade
            {
                Symbol = "QQQ", Date = new(2025, 5, 1),
                TypeOfTrade = TypeOfTrade.LongCall, Budget = Budget.Swing, Strategy = Strategy.BreakoutMomentum,
                AccountId = _fixture.TestAccountId
            };
            ctx.Trades.Add(closedTrade);
            await ctx.SaveChangesAsync();

            ctx.OptionPositions.Add(new OptionPosition
            {
                Symbol = "QQQ", ContractId = "C2", Opened = new(2025, 5, 1), Expiry = new(2025, 6, 18),
                Pos = 1, Right = PositionRight.Call, Strike = 480, Cost = 5.00m,
                Closed = new(2025, 5, 15), ClosePrice = 8.00m,
                TradeId = closedTrade.Id, AccountId = _fixture.TestAccountId
            });
            await ctx.SaveChangesAsync();
        }

        using var queryCtx = _fixture.CreateContext();
        var controller = new InstrumentSummariesController(queryCtx);

        var result = await controller.GetTradeOverview("open");

        var summaries = result.Value!.ToList();
        Assert.Single(summaries);
        Assert.Equal("SPY", summaries[0].Symbol);
    }

    public void Dispose() => _fixture.Dispose();
}
