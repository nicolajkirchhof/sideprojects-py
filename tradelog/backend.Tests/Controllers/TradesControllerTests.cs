using Microsoft.AspNetCore.Mvc;
using tradelog.Controllers;
using tradelog.Dtos;
using tradelog.Models;
using tradelog.Tests.Fixtures;

namespace tradelog.Tests.Controllers;

public class TradesControllerTests : IDisposable
{
    private readonly TestDbFixture _fixture;

    public TradesControllerTests()
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
    public async Task GetById_ReturnsLinkedOptionAndStockPositions()
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

            ctx.OptionPositions.Add(new OptionPosition
            {
                Symbol = "SPY", ContractId = "C1", Opened = new(2025, 6, 1), Expiry = new(2025, 7, 18),
                Pos = -1, Right = PositionRight.Put, Strike = 540, Cost = 3.50m,
                TradeId = tradeId, AccountId = _fixture.TestAccountId
            });
            ctx.StockPositions.Add(new StockPosition
            {
                Symbol = "SPY", Date = new(2025, 6, 5), PosChange = 100, Price = 545m,
                TradeId = tradeId, AccountId = _fixture.TestAccountId
            });
            await ctx.SaveChangesAsync();
        }

        using var queryCtx = _fixture.CreateContext();
        var controller = new TradesController(queryCtx);

        var result = await controller.GetById(tradeId);

        var detail = result.Value!;
        Assert.Equal("SPY", detail.Symbol);
        Assert.Single(detail.OptionPositions);
        Assert.Equal("C1", detail.OptionPositions[0].ContractId);
        Assert.Single(detail.StockPositions);
        Assert.Equal(545m, detail.StockPositions[0].Price);
    }

    [Fact]
    public async Task GetById_ReturnsEmptyCollectionsWhenNoPositionsLinked()
    {
        int tradeId;
        using (var ctx = _fixture.CreateContext())
        {
            var trade = new Trade
            {
                Symbol = "QQQ", Date = new(2025, 6, 1),
                TypeOfTrade = TypeOfTrade.ShortStrangle, Budget = Budget.Drift, Strategy = Strategy.PositiveDrift,
                AccountId = _fixture.TestAccountId
            };
            ctx.Trades.Add(trade);
            await ctx.SaveChangesAsync();
            tradeId = trade.Id;
        }

        using var queryCtx = _fixture.CreateContext();
        var controller = new TradesController(queryCtx);

        var result = await controller.GetById(tradeId);

        var detail = result.Value!;
        Assert.Empty(detail.OptionPositions);
        Assert.Empty(detail.StockPositions);
    }

    [Fact]
    public async Task GetById_DoesNotReturnPositionsFromOtherAccounts()
    {
        int tradeId;
        using (var ctx = _fixture.CreateContext())
        {
            var trade = new Trade
            {
                Symbol = "IWM", Date = new(2025, 6, 1),
                TypeOfTrade = TypeOfTrade.ShortPut, Budget = Budget.Drift, Strategy = Strategy.PositiveDrift,
                AccountId = _fixture.TestAccountId
            };
            ctx.Trades.Add(trade);
            await ctx.SaveChangesAsync();
            tradeId = trade.Id;
        }

        // Add position on different account with same TradeId
        using (var ctx = _fixture.CreateContext(2))
        {
            ctx.Accounts.Add(new Account { Id = 2, IbkrAccountId = "U9999", Name = "Other" });
            ctx.OptionPositions.Add(new OptionPosition
            {
                Symbol = "IWM", ContractId = "C99", Opened = new(2025, 6, 1), Expiry = new(2025, 7, 18),
                Pos = -1, Right = PositionRight.Put, Strike = 200, Cost = 2m,
                TradeId = tradeId, AccountId = 2
            });
            await ctx.SaveChangesAsync();
        }

        using var queryCtx = _fixture.CreateContext();
        var controller = new TradesController(queryCtx);

        var result = await controller.GetById(tradeId);

        Assert.Empty(result.Value!.OptionPositions);
    }

    public void Dispose() => _fixture.Dispose();
}
