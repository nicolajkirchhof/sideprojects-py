using Microsoft.AspNetCore.Mvc;
using tradelog.Controllers;
using tradelog.Dtos;
using tradelog.Models;
using tradelog.Tests.Fixtures;

namespace tradelog.Tests.Controllers;

public class StockPositionsControllerTests : IDisposable
{
    private readonly TestDbFixture _fixture;

    public StockPositionsControllerTests()
    {
        _fixture = new TestDbFixture();
        SeedAccount();
    }

    private void SeedAccount()
    {
        using var ctx = _fixture.CreateContext();
        ctx.Accounts.Add(new Account { Id = _fixture.TestAccountId, IbkrAccountId = "U1234", Name = "Test" });
        ctx.SaveChanges();
        LookupSeeder.Seed(ctx, _fixture.TestAccountId);
    }

    [Fact]
    public async Task GetAll_ReturnsTradesWithComputedFields()
    {
        using (var ctx = _fixture.CreateContext())
        {
            ctx.StockPositions.Add(new StockPosition { Symbol = "AAPL", Date = new(2025, 1, 1), PosChange = 100, Price = 150m, Multiplier = 1, AccountId = _fixture.TestAccountId });
            ctx.StockPositions.Add(new StockPosition { Symbol = "AAPL", Date = new(2025, 1, 2), PosChange = -50, Price = 160m, Multiplier = 1, AccountId = _fixture.TestAccountId });
            await ctx.SaveChangesAsync();
        }

        using var queryCtx = _fixture.CreateContext();
        var controller = new StockPositionsController(queryCtx);

        var result = await controller.GetAll(null, null);

        var trades = result.Value!;
        Assert.Equal(2, trades.Count());
        var second = trades.Last();
        Assert.Equal(50, second.TotalPos);
        Assert.Equal(150m, second.AvgPrice);
        // pnl = -50 * (150 - 160) * 1 = 500
        Assert.Equal(500m, second.Pnl);
    }

    [Fact]
    public async Task Create_ReturnsDtoWithRunningFields()
    {
        using var ctx = _fixture.CreateContext();
        var controller = new StockPositionsController(ctx);

        var trade = new StockPosition { Symbol = "MSFT", Date = new(2025, 1, 1), PosChange = 50, Price = 400m, Multiplier = 1 };
        var result = await controller.Create(trade);

        var created = (result.Result as CreatedAtActionResult)?.Value as StockPositionDto;
        Assert.NotNull(created);
        Assert.Equal(50, created!.TotalPos);
        Assert.Equal(400m, created.AvgPrice);
    }

    [Fact]
    public async Task AccountScoping_OtherAccountTradesInvisible()
    {
        const int otherAccountId = 99;

        using (var ctx = _fixture.CreateContext())
        {
            ctx.Accounts.Add(new Account { Id = otherAccountId, IbkrAccountId = "U9999", Name = "Other" });
            await ctx.SaveChangesAsync();
        }

        // Insert trade for other account using a context scoped to that account
        using (var otherCtx = _fixture.CreateContext(otherAccountId))
        {
            otherCtx.StockPositions.Add(new StockPosition { Symbol = "SPY", Date = new(2025, 1, 1), PosChange = 10, Price = 500m, Multiplier = 1 });
            await otherCtx.SaveChangesAsync();
        }

        // Insert trade for test account
        using (var ctx = _fixture.CreateContext())
        {
            ctx.StockPositions.Add(new StockPosition { Symbol = "AAPL", Date = new(2025, 1, 1), PosChange = 10, Price = 150m, Multiplier = 1 });
            await ctx.SaveChangesAsync();
        }

        // Query with test account context (accountId=1) — should only see AAPL
        using var queryCtx = _fixture.CreateContext();
        var controller = new StockPositionsController(queryCtx);

        var result = await controller.GetAll(null, null);

        var trades = result.Value!.ToList();
        Assert.Single(trades);
        Assert.Equal("AAPL", trades[0].Symbol);
    }

    [Fact]
    public async Task GetAll_UnassignedTrue_ReturnsOnlyUnassigned()
    {
        int tradeId;
        using (var ctx = _fixture.CreateContext())
        {
            var trade = new Trade
            {
                Symbol = "AAPL", Date = new(2025, 6, 1),
                TypeOfTrade = LookupSeeder.TypeLongStock, Budget = LookupSeeder.BudgetSwing, Strategy = LookupSeeder.StrategyBreakoutMomentum,
                AccountId = _fixture.TestAccountId
            };
            ctx.Trades.Add(trade);
            await ctx.SaveChangesAsync();
            tradeId = trade.Id;

            ctx.StockPositions.Add(new StockPosition { Symbol = "AAPL", Date = new(2025, 6, 1), PosChange = 100, Price = 200m, Multiplier = 1, TradeId = tradeId, AccountId = _fixture.TestAccountId });
            ctx.StockPositions.Add(new StockPosition { Symbol = "AAPL", Date = new(2025, 6, 2), PosChange = 50, Price = 205m, Multiplier = 1, TradeId = null, AccountId = _fixture.TestAccountId });
            await ctx.SaveChangesAsync();
        }

        using var queryCtx = _fixture.CreateContext();
        var controller = new StockPositionsController(queryCtx);

        var result = await controller.GetAll(null, true);

        var positions = result.Value!.ToList();
        Assert.Single(positions);
        Assert.Equal(205m, positions[0].Price);
    }

    [Fact]
    public async Task Assign_SetsTradeId()
    {
        int posId;
        int tradeId;
        using (var ctx = _fixture.CreateContext())
        {
            var trade = new Trade
            {
                Symbol = "AAPL", Date = new(2025, 6, 1),
                TypeOfTrade = LookupSeeder.TypeLongStock, Budget = LookupSeeder.BudgetSwing, Strategy = LookupSeeder.StrategyBreakoutMomentum,
                AccountId = _fixture.TestAccountId
            };
            ctx.Trades.Add(trade);
            var pos = new StockPosition { Symbol = "AAPL", Date = new(2025, 6, 1), PosChange = 100, Price = 200m, Multiplier = 1, AccountId = _fixture.TestAccountId };
            ctx.StockPositions.Add(pos);
            await ctx.SaveChangesAsync();
            posId = pos.Id;
            tradeId = trade.Id;
        }

        using var cmdCtx = _fixture.CreateContext();
        var controller = new StockPositionsController(cmdCtx);

        var result = await controller.Assign(posId, new AssignTradeDto { TradeId = tradeId });

        Assert.IsType<NoContentResult>(result);

        using var verifyCtx = _fixture.CreateContext();
        var updated = await verifyCtx.StockPositions.FindAsync(posId);
        Assert.Equal(tradeId, updated!.TradeId);
    }

    [Fact]
    public async Task Assign_WithNull_Unassigns()
    {
        int posId;
        using (var ctx = _fixture.CreateContext())
        {
            var trade = new Trade
            {
                Symbol = "AAPL", Date = new(2025, 6, 1),
                TypeOfTrade = LookupSeeder.TypeLongStock, Budget = LookupSeeder.BudgetSwing, Strategy = LookupSeeder.StrategyBreakoutMomentum,
                AccountId = _fixture.TestAccountId
            };
            ctx.Trades.Add(trade);
            await ctx.SaveChangesAsync();

            var pos = new StockPosition { Symbol = "AAPL", Date = new(2025, 6, 1), PosChange = 100, Price = 200m, Multiplier = 1, TradeId = trade.Id, AccountId = _fixture.TestAccountId };
            ctx.StockPositions.Add(pos);
            await ctx.SaveChangesAsync();
            posId = pos.Id;
        }

        using var cmdCtx = _fixture.CreateContext();
        var controller = new StockPositionsController(cmdCtx);

        var result = await controller.Assign(posId, new AssignTradeDto { TradeId = null });

        Assert.IsType<NoContentResult>(result);

        using var verifyCtx = _fixture.CreateContext();
        var updated = await verifyCtx.StockPositions.FindAsync(posId);
        Assert.Null(updated!.TradeId);
    }

    public void Dispose() => _fixture.Dispose();
}
