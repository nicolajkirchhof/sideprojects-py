using tradelog.Services;
using Microsoft.AspNetCore.Mvc;
using tradelog.Controllers;
using tradelog.Dtos;
using tradelog.Models;
using tradelog.Tests.Fixtures;

namespace tradelog.Tests.Controllers;

public class TradeEventsControllerTests : IDisposable
{
    private readonly TestDbFixture _fixture;
    private int _tradeId;

    public TradeEventsControllerTests()
    {
        _fixture = new TestDbFixture();
        SeedData();
    }

    private void SeedData()
    {
        using var ctx = _fixture.CreateContext();
        ctx.Accounts.Add(new Account { Id = _fixture.TestAccountId, IbkrAccountId = "U1234", Name = "Test" });
        ctx.SaveChanges();
        LookupSeeder.Seed(ctx, _fixture.TestAccountId);
        var trade = new Trade
        {
            Symbol = "SPY", Date = new(2025, 6, 1),
            TypeOfTrade = LookupSeeder.TypeShortPut, Budget = LookupSeeder.BudgetDrift, Strategy = LookupSeeder.StrategyPositiveDrift,
            AccountId = _fixture.TestAccountId
        };
        ctx.Trades.Add(trade);
        ctx.SaveChanges();
        _tradeId = trade.Id;
    }

    [Fact]
    public async Task GetEvents_ReturnsEventsForTrade()
    {
        using (var ctx = _fixture.CreateContext())
        {
            ctx.TradeEvents.Add(new TradeEvent
            {
                TradeId = _tradeId, Type = TradeEventType.ScaleIn,
                Date = new(2025, 6, 5), Notes = "Added more", PnlImpact = -50m,
                AccountId = _fixture.TestAccountId
            });
            ctx.TradeEvents.Add(new TradeEvent
            {
                TradeId = _tradeId, Type = TradeEventType.ProfitTake,
                Date = new(2025, 6, 10), Notes = "Took 50%", PnlImpact = 200m,
                AccountId = _fixture.TestAccountId
            });
            await ctx.SaveChangesAsync();
        }

        using var queryCtx = _fixture.CreateContext();
        var controller = new TradeEventsController(queryCtx);

        var result = await controller.GetEvents(_tradeId);

        var events = result.Value!.ToList();
        Assert.Equal(2, events.Count);
        // Chronological order
        Assert.Equal("ScaleIn", events[0].Type);
        Assert.Equal("ProfitTake", events[1].Type);
    }

    [Fact]
    public async Task CreateEvent_ReturnsCreatedEvent()
    {
        using var ctx = _fixture.CreateContext();
        var controller = new TradeEventsController(ctx);

        var dto = new TradeEventDto
        {
            Type = "Roll", Date = new(2025, 6, 15), Notes = "Rolled out", PnlImpact = -30m
        };

        var result = await controller.CreateEvent(_tradeId, dto);

        var created = (result.Result as CreatedAtActionResult)?.Value as TradeEventDto;
        Assert.NotNull(created);
        Assert.Equal("Roll", created!.Type);
        Assert.Equal(_tradeId, created.TradeId);
        Assert.Equal(-30m, created.PnlImpact);
        Assert.True(created.Id > 0);
    }

    [Fact]
    public async Task CreateEvent_ReturnsNotFoundForMissingTrade()
    {
        using var ctx = _fixture.CreateContext();
        var controller = new TradeEventsController(ctx);

        var dto = new TradeEventDto { Type = "Stop", Date = new(2025, 6, 15) };

        var result = await controller.CreateEvent(999, dto);

        Assert.IsType<NotFoundResult>(result.Result);
    }

    [Fact]
    public async Task DeleteEvent_RemovesEvent()
    {
        int eventId;
        using (var ctx = _fixture.CreateContext())
        {
            var evt = new TradeEvent
            {
                TradeId = _tradeId, Type = TradeEventType.Stop,
                Date = new(2025, 6, 20), AccountId = _fixture.TestAccountId
            };
            ctx.TradeEvents.Add(evt);
            await ctx.SaveChangesAsync();
            eventId = evt.Id;
        }

        using var cmdCtx = _fixture.CreateContext();
        var controller = new TradeEventsController(cmdCtx);

        var result = await controller.DeleteEvent(eventId);

        Assert.IsType<NoContentResult>(result);

        using var verifyCtx = _fixture.CreateContext();
        Assert.Null(await verifyCtx.TradeEvents.FindAsync(eventId));
    }

    [Fact]
    public async Task DeleteEvent_ReturnsNotFoundForMissingEvent()
    {
        using var ctx = _fixture.CreateContext();
        var controller = new TradeEventsController(ctx);

        var result = await controller.DeleteEvent(999);

        Assert.IsType<NotFoundResult>(result);
    }

    [Fact]
    public async Task GetById_IncludesEventsInTradeDetail()
    {
        using (var ctx = _fixture.CreateContext())
        {
            ctx.TradeEvents.Add(new TradeEvent
            {
                TradeId = _tradeId, Type = TradeEventType.ProfitTake,
                Date = new(2025, 6, 10), Notes = "50% off", PnlImpact = 150m,
                AccountId = _fixture.TestAccountId
            });
            await ctx.SaveChangesAsync();
        }

        using var queryCtx = _fixture.CreateContext();
        var tradesController = new TradesController(queryCtx, new TradeStatusService(queryCtx));

        var result = await tradesController.GetById(_tradeId);

        var detail = result.Value!;
        Assert.Single(detail.Events);
        Assert.Equal("ProfitTake", detail.Events[0].Type);
        Assert.Equal(150m, detail.Events[0].PnlImpact);
    }

    public void Dispose() => _fixture.Dispose();
}
