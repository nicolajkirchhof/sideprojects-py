using tradelog.Controllers;
using tradelog.Models;
using tradelog.Tests.Fixtures;

namespace tradelog.Tests.Controllers;

public class AnalyticsControllerTests : IDisposable
{
    private readonly TestDbFixture _fixture;

    public AnalyticsControllerTests()
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
    public async Task GetChains_ReturnsOnlyChainsWithMultipleTrades()
    {
        using (var ctx = _fixture.CreateContext())
        {
            // Chain of 2 (should appear)
            var root = new Trade
            {
                Symbol = "SPY", Date = new(2025, 5, 1),
                TypeOfTrade = TypeOfTrade.ShortPut, Budget = Budget.Drift, Strategy = Strategy.PositiveDrift,
                AccountId = _fixture.TestAccountId
            };
            ctx.Trades.Add(root);
            await ctx.SaveChangesAsync();

            ctx.Trades.Add(new Trade
            {
                Symbol = "SPY", Date = new(2025, 6, 1),
                TypeOfTrade = TypeOfTrade.ShortPut, Budget = Budget.Drift, Strategy = Strategy.PositiveDrift,
                ParentTradeId = root.Id, AccountId = _fixture.TestAccountId
            });

            // Standalone trade (should NOT appear)
            ctx.Trades.Add(new Trade
            {
                Symbol = "QQQ", Date = new(2025, 6, 1),
                TypeOfTrade = TypeOfTrade.LongCall, Budget = Budget.Swing, Strategy = Strategy.BreakoutMomentum,
                AccountId = _fixture.TestAccountId
            });
            await ctx.SaveChangesAsync();
        }

        using var queryCtx = _fixture.CreateContext();
        var controller = new AnalyticsController(queryCtx);

        var result = await controller.GetChains();

        var chains = result.Value!.ToList();
        Assert.Single(chains);
        Assert.Equal("SPY", chains[0].Symbol);
        Assert.Equal(2, chains[0].ChainLength);
    }

    [Fact]
    public async Task GetChains_AggregatesPnlAcrossChain()
    {
        using (var ctx = _fixture.CreateContext())
        {
            var root = new Trade
            {
                Symbol = "SPY", Date = new(2025, 5, 1),
                TypeOfTrade = TypeOfTrade.ShortPut, Budget = Budget.Drift, Strategy = Strategy.PositiveDrift,
                AccountId = _fixture.TestAccountId
            };
            ctx.Trades.Add(root);
            await ctx.SaveChangesAsync();

            // Closed option on root trade: realized P/L = (3.50 - 5.00) * 100 * -1 - 2 = 148
            ctx.OptionPositions.Add(new OptionPosition
            {
                Symbol = "SPY", ContractId = "C1", Opened = new(2025, 5, 1), Expiry = new(2025, 6, 20),
                Pos = -1, Right = PositionRight.Put, Strike = 540, Cost = 5.00m,
                Closed = new(2025, 5, 15), ClosePrice = 3.50m, Commission = 2m,
                TradeId = root.Id, AccountId = _fixture.TestAccountId
            });

            var child = new Trade
            {
                Symbol = "SPY", Date = new(2025, 6, 1),
                TypeOfTrade = TypeOfTrade.ShortPut, Budget = Budget.Drift, Strategy = Strategy.PositiveDrift,
                ParentTradeId = root.Id, AccountId = _fixture.TestAccountId
            };
            ctx.Trades.Add(child);
            await ctx.SaveChangesAsync();

            // Closed option on child trade: realized P/L = (2.00 - 4.00) * 100 * -1 - 1 = 199
            ctx.OptionPositions.Add(new OptionPosition
            {
                Symbol = "SPY", ContractId = "C2", Opened = new(2025, 6, 1), Expiry = new(2025, 7, 18),
                Pos = -1, Right = PositionRight.Put, Strike = 530, Cost = 4.00m,
                Closed = new(2025, 6, 15), ClosePrice = 2.00m, Commission = 1m,
                TradeId = child.Id, AccountId = _fixture.TestAccountId
            });
            await ctx.SaveChangesAsync();
        }

        using var queryCtx = _fixture.CreateContext();
        var controller = new AnalyticsController(queryCtx);

        var result = await controller.GetChains();

        var chain = result.Value!.First();
        Assert.Equal(148m + 199m, chain.TotalPnl);
        Assert.Equal(148m + 199m, chain.PremiumCollected);
        Assert.Equal(0m, chain.PremiumLost);
    }

    [Fact]
    public async Task GetChains_ShowsOpenStatusWhenAnyLegIsOpen()
    {
        using (var ctx = _fixture.CreateContext())
        {
            var root = new Trade
            {
                Symbol = "IWM", Date = new(2025, 5, 1),
                TypeOfTrade = TypeOfTrade.ShortPut, Budget = Budget.Drift, Strategy = Strategy.PositiveDrift,
                AccountId = _fixture.TestAccountId
            };
            ctx.Trades.Add(root);
            await ctx.SaveChangesAsync();

            // Closed option on root
            ctx.OptionPositions.Add(new OptionPosition
            {
                Symbol = "IWM", ContractId = "C1", Opened = new(2025, 5, 1), Expiry = new(2025, 6, 20),
                Pos = -1, Right = PositionRight.Put, Strike = 200, Cost = 3.00m,
                Closed = new(2025, 5, 20), ClosePrice = 1.00m,
                TradeId = root.Id, AccountId = _fixture.TestAccountId
            });

            var child = new Trade
            {
                Symbol = "IWM", Date = new(2025, 6, 1),
                TypeOfTrade = TypeOfTrade.ShortPut, Budget = Budget.Drift, Strategy = Strategy.PositiveDrift,
                ParentTradeId = root.Id, AccountId = _fixture.TestAccountId
            };
            ctx.Trades.Add(child);
            await ctx.SaveChangesAsync();

            // Open option on child
            ctx.OptionPositions.Add(new OptionPosition
            {
                Symbol = "IWM", ContractId = "C2", Opened = new(2025, 6, 1), Expiry = new(2025, 7, 18),
                Pos = -1, Right = PositionRight.Put, Strike = 195, Cost = 2.50m,
                TradeId = child.Id, AccountId = _fixture.TestAccountId
            });
            await ctx.SaveChangesAsync();
        }

        using var queryCtx = _fixture.CreateContext();
        var controller = new AnalyticsController(queryCtx);

        var result = await controller.GetChains();

        Assert.Equal("Open", result.Value!.First().Status);
    }

    [Fact]
    public async Task GetChains_CountsEventsAcrossChain()
    {
        using (var ctx = _fixture.CreateContext())
        {
            var root = new Trade
            {
                Symbol = "SPY", Date = new(2025, 5, 1),
                TypeOfTrade = TypeOfTrade.ShortPut, Budget = Budget.Drift, Strategy = Strategy.PositiveDrift,
                AccountId = _fixture.TestAccountId
            };
            ctx.Trades.Add(root);
            await ctx.SaveChangesAsync();

            ctx.TradeEvents.Add(new TradeEvent { TradeId = root.Id, Type = TradeEventType.ProfitTake, Date = new(2025, 5, 10), AccountId = _fixture.TestAccountId });
            ctx.TradeEvents.Add(new TradeEvent { TradeId = root.Id, Type = TradeEventType.Roll, Date = new(2025, 5, 20), AccountId = _fixture.TestAccountId });

            var child = new Trade
            {
                Symbol = "SPY", Date = new(2025, 6, 1),
                TypeOfTrade = TypeOfTrade.ShortPut, Budget = Budget.Drift, Strategy = Strategy.PositiveDrift,
                ParentTradeId = root.Id, AccountId = _fixture.TestAccountId
            };
            ctx.Trades.Add(child);
            await ctx.SaveChangesAsync();

            ctx.TradeEvents.Add(new TradeEvent { TradeId = child.Id, Type = TradeEventType.ScaleIn, Date = new(2025, 6, 5), AccountId = _fixture.TestAccountId });
            await ctx.SaveChangesAsync();
        }

        using var queryCtx = _fixture.CreateContext();
        var controller = new AnalyticsController(queryCtx);

        var result = await controller.GetChains();

        Assert.Equal(3, result.Value!.First().EventCount);
    }

    public void Dispose() => _fixture.Dispose();
}
