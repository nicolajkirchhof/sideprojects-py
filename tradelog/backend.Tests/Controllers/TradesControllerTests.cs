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
    public async Task GetAll_ReturnsTradesWithComputedFields()
    {
        using (var ctx = _fixture.CreateContext())
        {
            ctx.Trades.Add(new Trade { Symbol = "AAPL", Date = new(2025, 1, 1), PosChange = 100, Price = 150m, Multiplier = 1, AccountId = _fixture.TestAccountId });
            ctx.Trades.Add(new Trade { Symbol = "AAPL", Date = new(2025, 1, 2), PosChange = -50, Price = 160m, Multiplier = 1, AccountId = _fixture.TestAccountId });
            await ctx.SaveChangesAsync();
        }

        using var queryCtx = _fixture.CreateContext();
        var controller = new TradesController(queryCtx);

        var result = await controller.GetAll(null);

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
        var controller = new TradesController(ctx);

        var trade = new Trade { Symbol = "MSFT", Date = new(2025, 1, 1), PosChange = 50, Price = 400m, Multiplier = 1 };
        var result = await controller.Create(trade);

        var created = (result.Result as CreatedAtActionResult)?.Value as TradeDto;
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
            otherCtx.Trades.Add(new Trade { Symbol = "SPY", Date = new(2025, 1, 1), PosChange = 10, Price = 500m, Multiplier = 1 });
            await otherCtx.SaveChangesAsync();
        }

        // Insert trade for test account
        using (var ctx = _fixture.CreateContext())
        {
            ctx.Trades.Add(new Trade { Symbol = "AAPL", Date = new(2025, 1, 1), PosChange = 10, Price = 150m, Multiplier = 1 });
            await ctx.SaveChangesAsync();
        }

        // Query with test account context (accountId=1) — should only see AAPL
        using var queryCtx = _fixture.CreateContext();
        var controller = new TradesController(queryCtx);

        var result = await controller.GetAll(null);

        var trades = result.Value!.ToList();
        Assert.Single(trades);
        Assert.Equal("AAPL", trades[0].Symbol);
    }

    public void Dispose() => _fixture.Dispose();
}
