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

    // --- Follow-up chain (E2-S3) ---

    private Trade MakeTrade(string symbol, int? parentTradeId = null) => new()
    {
        Symbol = symbol, Date = new(2025, 6, 1),
        TypeOfTrade = TypeOfTrade.ShortPut, Budget = Budget.Drift, Strategy = Strategy.PositiveDrift,
        ParentTradeId = parentTradeId, AccountId = _fixture.TestAccountId
    };

    [Fact]
    public async Task GetById_ReturnsChildTradeIds()
    {
        int parentId, childId;
        using (var ctx = _fixture.CreateContext())
        {
            var parent = MakeTrade("SPY");
            ctx.Trades.Add(parent);
            await ctx.SaveChangesAsync();
            parentId = parent.Id;

            var child = MakeTrade("SPY", parentId);
            ctx.Trades.Add(child);
            await ctx.SaveChangesAsync();
            childId = child.Id;
        }

        using var queryCtx = _fixture.CreateContext();
        var controller = new TradesController(queryCtx);

        var result = await controller.GetById(parentId);

        Assert.Single(result.Value!.ChildTradeIds);
        Assert.Equal(childId, result.Value!.ChildTradeIds[0]);
    }

    [Fact]
    public async Task GetChain_ReturnsFullChainRootFirst()
    {
        int rootId, childId, grandchildId;
        using (var ctx = _fixture.CreateContext())
        {
            var root = MakeTrade("SPY");
            ctx.Trades.Add(root);
            await ctx.SaveChangesAsync();
            rootId = root.Id;

            var child = MakeTrade("SPY", rootId);
            ctx.Trades.Add(child);
            await ctx.SaveChangesAsync();
            childId = child.Id;

            var grandchild = MakeTrade("SPY", childId);
            ctx.Trades.Add(grandchild);
            await ctx.SaveChangesAsync();
            grandchildId = grandchild.Id;
        }

        using var queryCtx = _fixture.CreateContext();
        var controller = new TradesController(queryCtx);

        var result = await controller.GetChain(childId);

        var chain = result.Value!.ToList();
        Assert.Equal(3, chain.Count);
        Assert.Equal(rootId, chain[0].Id);
        Assert.Equal(childId, chain[1].Id);
        Assert.Equal(grandchildId, chain[2].Id);
    }

    [Fact]
    public async Task GetChain_FromRoot_ReturnsRootAndDescendants()
    {
        int rootId, childId;
        using (var ctx = _fixture.CreateContext())
        {
            var root = MakeTrade("SPY");
            ctx.Trades.Add(root);
            await ctx.SaveChangesAsync();
            rootId = root.Id;

            var child = MakeTrade("SPY", rootId);
            ctx.Trades.Add(child);
            await ctx.SaveChangesAsync();
            childId = child.Id;
        }

        using var queryCtx = _fixture.CreateContext();
        var controller = new TradesController(queryCtx);

        var result = await controller.GetChain(rootId);

        var chain = result.Value!.ToList();
        Assert.Equal(2, chain.Count);
        Assert.Equal(rootId, chain[0].Id);
        Assert.Equal(childId, chain[1].Id);
    }

    [Fact]
    public async Task GetChain_ReturnsNotFoundForMissingTrade()
    {
        using var ctx = _fixture.CreateContext();
        var controller = new TradesController(ctx);

        var result = await controller.GetChain(999);

        Assert.IsType<NotFoundResult>(result.Result);
    }

    [Fact]
    public async Task Delete_BlockedWhenTradeHasChildren()
    {
        int parentId;
        using (var ctx = _fixture.CreateContext())
        {
            var parent = MakeTrade("SPY");
            ctx.Trades.Add(parent);
            await ctx.SaveChangesAsync();
            parentId = parent.Id;

            ctx.Trades.Add(MakeTrade("SPY", parentId));
            await ctx.SaveChangesAsync();
        }

        using var cmdCtx = _fixture.CreateContext();
        var controller = new TradesController(cmdCtx);

        var result = await controller.Delete(parentId);

        Assert.IsType<BadRequestObjectResult>(result);
    }

    [Fact]
    public async Task Delete_AllowedWhenTradeHasNoChildren()
    {
        int tradeId;
        using (var ctx = _fixture.CreateContext())
        {
            var trade = MakeTrade("QQQ");
            ctx.Trades.Add(trade);
            await ctx.SaveChangesAsync();
            tradeId = trade.Id;
        }

        using var cmdCtx = _fixture.CreateContext();
        var controller = new TradesController(cmdCtx);

        var result = await controller.Delete(tradeId);

        Assert.IsType<NoContentResult>(result);
    }

    public void Dispose() => _fixture.Dispose();
}
