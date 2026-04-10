using tradelog.Models;
using tradelog.Services;
using tradelog.Tests.Fixtures;

namespace tradelog.Tests.Services;

public class OptionPositionLogCountServiceTests : IDisposable
{
    private readonly TestDbFixture _fixture;

    public OptionPositionLogCountServiceTests()
    {
        _fixture = new TestDbFixture();
        using var ctx = _fixture.CreateContext();
        ctx.Accounts.Add(new Account { Id = _fixture.TestAccountId, IbkrAccountId = "U1", Name = "Test" });
        ctx.SaveChanges();
    }

    private OptionPosition MakePosition(string contractId, DateTime opened, DateTime? closed = null) => new()
    {
        Symbol = "SPY",
        ContractId = contractId,
        Opened = opened,
        Expiry = opened.AddDays(30),
        Closed = closed,
        Pos = -1,
        Right = PositionRight.Put,
        Strike = 500,
        Cost = 3m,
        AccountId = _fixture.TestAccountId,
    };

    private OptionPositionsLog MakeLog(string contractId, DateTime dateTime) => new()
    {
        ContractId = contractId,
        DateTime = dateTime,
        AccountId = _fixture.TestAccountId,
    };

    [Fact]
    public async Task RecomputeForAsync_CountsLogsAtOrAfterOpened()
    {
        using (var seed = _fixture.CreateContext())
        {
            var opened = new DateTime(2026, 1, 1);
            seed.OptionPositions.Add(MakePosition("C1", opened));
            seed.OptionPositionsLogs.Add(MakeLog("C1", new DateTime(2025, 12, 31))); // before → excluded
            seed.OptionPositionsLogs.Add(MakeLog("C1", opened));                      // exact match → counted
            seed.OptionPositionsLogs.Add(MakeLog("C1", new DateTime(2026, 1, 2)));    // after → counted
            seed.OptionPositionsLogs.Add(MakeLog("C1", new DateTime(2026, 1, 3)));    // after → counted
            await seed.SaveChangesAsync();
        }

        using var ctx = _fixture.CreateContext();
        var service = new OptionPositionLogCountService(ctx);
        var position = ctx.OptionPositions.Single();

        await service.RecomputeForAsync(new[] { position });
        await ctx.SaveChangesAsync();

        Assert.Equal(3, position.LogCount);
    }

    [Fact]
    public async Task RecomputeForAsync_NoLogs_ProducesZero()
    {
        using (var seed = _fixture.CreateContext())
        {
            seed.OptionPositions.Add(MakePosition("C1", new DateTime(2026, 1, 1)));
            await seed.SaveChangesAsync();
        }

        using var ctx = _fixture.CreateContext();
        var service = new OptionPositionLogCountService(ctx);
        var position = ctx.OptionPositions.Single();

        await service.RecomputeForAsync(new[] { position });
        await ctx.SaveChangesAsync();

        Assert.Equal(0, position.LogCount);
    }

    [Fact]
    public async Task RecomputeForPendingOpenAsync_TouchesOnlyNullAndOpen()
    {
        using (var seed = _fixture.CreateContext())
        {
            // Open, null → should be recomputed
            seed.OptionPositions.Add(new OptionPosition
            {
                Symbol = "SPY", ContractId = "C-OPEN", Opened = new(2026, 1, 1), Expiry = new(2026, 2, 1),
                Pos = -1, Right = PositionRight.Put, Strike = 500, Cost = 3m,
                AccountId = _fixture.TestAccountId, LogCount = null,
            });
            // Closed, null → must stay null (legacy closed handled by migration SQL, not this method)
            seed.OptionPositions.Add(new OptionPosition
            {
                Symbol = "SPY", ContractId = "C-CLOSED-NULL", Opened = new(2026, 1, 1), Expiry = new(2026, 2, 1),
                Closed = new(2026, 1, 15), Pos = -1, Right = PositionRight.Put, Strike = 500, Cost = 3m,
                AccountId = _fixture.TestAccountId, LogCount = null,
            });
            // Open, already populated → must not be touched
            seed.OptionPositions.Add(new OptionPosition
            {
                Symbol = "SPY", ContractId = "C-OPEN-SET", Opened = new(2026, 1, 1), Expiry = new(2026, 2, 1),
                Pos = -1, Right = PositionRight.Put, Strike = 500, Cost = 3m,
                AccountId = _fixture.TestAccountId, LogCount = 99,
            });

            seed.OptionPositionsLogs.Add(MakeLog("C-OPEN", new(2026, 1, 2)));
            seed.OptionPositionsLogs.Add(MakeLog("C-CLOSED-NULL", new(2026, 1, 2)));
            seed.OptionPositionsLogs.Add(MakeLog("C-OPEN-SET", new(2026, 1, 2)));
            await seed.SaveChangesAsync();
        }

        using var ctx = _fixture.CreateContext();
        var service = new OptionPositionLogCountService(ctx);

        var touched = await service.RecomputeForPendingOpenAsync();

        Assert.Equal(1, touched);

        using var verify = _fixture.CreateContext();
        var open = verify.OptionPositions.Single(p => p.ContractId == "C-OPEN");
        var closedNull = verify.OptionPositions.Single(p => p.ContractId == "C-CLOSED-NULL");
        var openSet = verify.OptionPositions.Single(p => p.ContractId == "C-OPEN-SET");

        Assert.Equal(1, open.LogCount);
        Assert.Null(closedNull.LogCount);
        Assert.Equal(99, openSet.LogCount);
    }

    public void Dispose() => _fixture.Dispose();
}
