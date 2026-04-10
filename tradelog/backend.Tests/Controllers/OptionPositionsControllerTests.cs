using Microsoft.AspNetCore.Mvc;
using tradelog.Controllers;
using tradelog.Dtos;
using tradelog.Models;
using tradelog.Services;
using tradelog.Tests.Fixtures;

namespace tradelog.Tests.Controllers;

public class OptionPositionsControllerTests : IDisposable
{
    private readonly TestDbFixture _fixture;

    public OptionPositionsControllerTests()
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

    private OptionPosition MakePosition(string symbol, string contractId, int? tradeId = null) => new()
    {
        Symbol = symbol, ContractId = contractId,
        Opened = new(2025, 6, 1), Expiry = new(2025, 7, 18),
        Pos = -1, Right = PositionRight.Put, Strike = 540, Cost = 3.50m,
        TradeId = tradeId, AccountId = _fixture.TestAccountId
    };

    [Fact]
    public async Task GetAll_UnassignedTrue_ReturnsOnlyUnassignedPositions()
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

            ctx.OptionPositions.Add(MakePosition("SPY", "C1", tradeId));
            ctx.OptionPositions.Add(MakePosition("SPY", "C2", null));
            ctx.OptionPositions.Add(MakePosition("QQQ", "C3", null));
            await ctx.SaveChangesAsync();
        }

        using var queryCtx = _fixture.CreateContext();
        var controller = new OptionPositionsController(queryCtx, new OptionPositionLogCountService(queryCtx));

        var result = await controller.GetAll(null, null, true);

        var positions = result.Value!.ToList();
        Assert.Equal(2, positions.Count);
        Assert.DoesNotContain(positions, p => p.ContractId == "C1");
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
                Symbol = "SPY", Date = new(2025, 6, 1),
                TypeOfTrade = TypeOfTrade.ShortPut, Budget = Budget.Drift, Strategy = Strategy.PositiveDrift,
                AccountId = _fixture.TestAccountId
            };
            ctx.Trades.Add(trade);
            var pos = MakePosition("SPY", "C1", null);
            ctx.OptionPositions.Add(pos);
            await ctx.SaveChangesAsync();
            posId = pos.Id;
            tradeId = trade.Id;
        }

        using var cmdCtx = _fixture.CreateContext();
        var controller = new OptionPositionsController(cmdCtx, new OptionPositionLogCountService(cmdCtx));

        var result = await controller.Assign(posId, new AssignTradeDto { TradeId = tradeId });

        Assert.IsType<NoContentResult>(result);

        using var verifyCtx = _fixture.CreateContext();
        var updated = await verifyCtx.OptionPositions.FindAsync(posId);
        Assert.Equal(tradeId, updated!.TradeId);
    }

    [Fact]
    public async Task Assign_WithNull_UnassignsPosition()
    {
        int posId;
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

            var pos = MakePosition("SPY", "C1", trade.Id);
            ctx.OptionPositions.Add(pos);
            await ctx.SaveChangesAsync();
            posId = pos.Id;
        }

        using var cmdCtx = _fixture.CreateContext();
        var controller = new OptionPositionsController(cmdCtx, new OptionPositionLogCountService(cmdCtx));

        var result = await controller.Assign(posId, new AssignTradeDto { TradeId = null });

        Assert.IsType<NoContentResult>(result);

        using var verifyCtx = _fixture.CreateContext();
        var updated = await verifyCtx.OptionPositions.FindAsync(posId);
        Assert.Null(updated!.TradeId);
    }

    [Fact]
    public async Task Assign_NotFound_Returns404()
    {
        using var ctx = _fixture.CreateContext();
        var controller = new OptionPositionsController(ctx, new OptionPositionLogCountService(ctx));

        var result = await controller.Assign(999, new AssignTradeDto { TradeId = 1 });

        Assert.IsType<NotFoundResult>(result);
    }

    [Fact]
    public async Task Update_OpenedChanged_OnOpenPosition_RecomputesLogCount()
    {
        int posId;
        using (var seed = _fixture.CreateContext())
        {
            var pos = new OptionPosition
            {
                Symbol = "SPY", ContractId = "C1",
                Opened = new(2026, 1, 1), Expiry = new(2026, 2, 1),
                Pos = -1, Right = PositionRight.Put, Strike = 540, Cost = 3.5m,
                AccountId = _fixture.TestAccountId, LogCount = 5,
            };
            seed.OptionPositions.Add(pos);
            // Two logs after the new Opened, one before.
            seed.OptionPositionsLogs.Add(new OptionPositionsLog
            { ContractId = "C1", DateTime = new(2026, 1, 5), AccountId = _fixture.TestAccountId });
            seed.OptionPositionsLogs.Add(new OptionPositionsLog
            { ContractId = "C1", DateTime = new(2026, 1, 20), AccountId = _fixture.TestAccountId });
            seed.OptionPositionsLogs.Add(new OptionPositionsLog
            { ContractId = "C1", DateTime = new(2026, 1, 25), AccountId = _fixture.TestAccountId });
            await seed.SaveChangesAsync();
            posId = pos.Id;
        }

        using var cmdCtx = _fixture.CreateContext();
        var controller = new OptionPositionsController(cmdCtx, new OptionPositionLogCountService(cmdCtx));

        var updated = new OptionPosition
        {
            Id = posId,
            Symbol = "SPY", ContractId = "C1",
            Opened = new(2026, 1, 15), Expiry = new(2026, 2, 1),
            Pos = -1, Right = PositionRight.Put, Strike = 540, Cost = 3.5m,
            AccountId = _fixture.TestAccountId,
        };

        var result = await controller.Update(posId, updated);

        Assert.IsType<NoContentResult>(result);

        using var verify = _fixture.CreateContext();
        var reloaded = await verify.OptionPositions.FindAsync(posId);
        // Only the two logs on/after 2026-01-15 remain counted.
        Assert.Equal(2, reloaded!.LogCount);
    }

    [Fact]
    public async Task Update_OpenedUnchanged_DoesNotRecomputeLogCount()
    {
        int posId;
        using (var seed = _fixture.CreateContext())
        {
            var pos = new OptionPosition
            {
                Symbol = "SPY", ContractId = "C1",
                Opened = new(2026, 1, 1), Expiry = new(2026, 2, 1),
                Pos = -1, Right = PositionRight.Put, Strike = 540, Cost = 3.5m,
                AccountId = _fixture.TestAccountId, LogCount = 42,
            };
            seed.OptionPositions.Add(pos);
            await seed.SaveChangesAsync();
            posId = pos.Id;
        }

        using var cmdCtx = _fixture.CreateContext();
        var controller = new OptionPositionsController(cmdCtx, new OptionPositionLogCountService(cmdCtx));

        var updated = new OptionPosition
        {
            Id = posId,
            Symbol = "SPY", ContractId = "C1",
            Opened = new(2026, 1, 1), Expiry = new(2026, 2, 1),
            Pos = -2, Right = PositionRight.Put, Strike = 540, Cost = 3.5m,
            AccountId = _fixture.TestAccountId,
        };

        await controller.Update(posId, updated);

        using var verify = _fixture.CreateContext();
        var reloaded = await verify.OptionPositions.FindAsync(posId);
        Assert.Equal(42, reloaded!.LogCount);
    }

    [Fact]
    public async Task Update_OpenedChanged_OnClosedPosition_DoesNotRecompute()
    {
        int posId;
        using (var seed = _fixture.CreateContext())
        {
            var pos = new OptionPosition
            {
                Symbol = "SPY", ContractId = "C1",
                Opened = new(2026, 1, 1), Expiry = new(2026, 2, 1), Closed = new(2026, 1, 20),
                Pos = -1, Right = PositionRight.Put, Strike = 540, Cost = 3.5m,
                AccountId = _fixture.TestAccountId, LogCount = 7,
            };
            seed.OptionPositions.Add(pos);
            seed.OptionPositionsLogs.Add(new OptionPositionsLog
            { ContractId = "C1", DateTime = new(2026, 1, 10), AccountId = _fixture.TestAccountId });
            seed.OptionPositionsLogs.Add(new OptionPositionsLog
            { ContractId = "C1", DateTime = new(2026, 1, 15), AccountId = _fixture.TestAccountId });
            await seed.SaveChangesAsync();
            posId = pos.Id;
        }

        using var cmdCtx = _fixture.CreateContext();
        var controller = new OptionPositionsController(cmdCtx, new OptionPositionLogCountService(cmdCtx));

        var updated = new OptionPosition
        {
            Id = posId,
            Symbol = "SPY", ContractId = "C1",
            Opened = new(2026, 1, 12), Expiry = new(2026, 2, 1), Closed = new(2026, 1, 20),
            Pos = -1, Right = PositionRight.Put, Strike = 540, Cost = 3.5m,
            AccountId = _fixture.TestAccountId,
        };

        await controller.Update(posId, updated);

        using var verify = _fixture.CreateContext();
        var reloaded = await verify.OptionPositions.FindAsync(posId);
        // Closed position is frozen — original count preserved.
        Assert.Equal(7, reloaded!.LogCount);
    }

    public void Dispose() => _fixture.Dispose();
}
