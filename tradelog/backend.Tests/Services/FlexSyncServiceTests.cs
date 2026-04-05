using Microsoft.Extensions.Logging.Abstractions;
using tradelog.Dtos;
using tradelog.Models;
using tradelog.Services;
using tradelog.Tests.Fixtures;

namespace tradelog.Tests.Services;

public class FlexSyncServiceTests : IDisposable
{
    private readonly TestDbFixture _fixture = new();
    private readonly FlexSyncService _service;

    public FlexSyncServiceTests()
    {
        _service = new FlexSyncService(
            _fixture.CreateContext(),
            NullLogger<FlexSyncService>.Instance);

        // Seed an account so the account-scoped context works
        using var ctx = _fixture.CreateContext();
        ctx.Accounts.Add(new Account { Id = 1, IbkrAccountId = "TEST", Name = "Test", IsDefault = true });
        ctx.SaveChanges();
    }

    public void Dispose() => _fixture.Dispose();

    // ── SyncOptionEventsAsync ────────────────────────────

    [Fact]
    public async Task SyncOptionEvents_Expiration_ClosesPosition()
    {
        // Arrange: create an open option position
        using (var ctx = _fixture.CreateContext())
        {
            ctx.OptionPositions.Add(new OptionPosition
            {
                Symbol = "SPY P500",
                ContractId = "900001",
                ConId = 900001,
                SecType = "OPT",
                Opened = new DateTime(2026, 1, 1),
                Expiry = new DateTime(2026, 1, 17),
                Pos = -1,
                Right = PositionRight.Put,
                Strike = 500,
                Cost = 3.50m,
                Multiplier = 100,
            });
            await ctx.SaveChangesAsync();
        }

        var events = new List<FlexOptionEventDto>
        {
            new()
            {
                ConId = 900001,
                Symbol = "SPY P500",
                AssetCategory = "OPT",
                TransactionType = "Expiration",
                Date = new DateTime(2026, 1, 17),
                Quantity = -1,
                TradePrice = 0,
                Commission = 0,
                Strike = 500,
                PutCall = "P",
                Multiplier = 100,
            }
        };

        // Act
        var processed = await _service.SyncOptionEventsAsync(events);

        // Assert
        Assert.Equal(1, processed);
        using var verifyCtx = _fixture.CreateContext();
        var pos = verifyCtx.OptionPositions.Single(p => p.ConId == 900001);
        Assert.NotNull(pos.Closed);
        Assert.Equal(new DateTime(2026, 1, 17), pos.Closed);
        Assert.Equal(0m, pos.ClosePrice);
        Assert.True(pos.CloseReasons!.Value.HasFlag(CloseReasons.TimeLimit));
    }

    [Fact]
    public async Task SyncOptionEvents_Assignment_ClosesPosition()
    {
        using (var ctx = _fixture.CreateContext())
        {
            ctx.OptionPositions.Add(new OptionPosition
            {
                Symbol = "PEP P155",
                ContractId = "723197",
                ConId = 723197,
                SecType = "OPT",
                Opened = new DateTime(2025, 12, 1),
                Expiry = new DateTime(2026, 4, 17),
                Pos = 1,
                Right = PositionRight.Put,
                Strike = 155,
                Cost = 2.00m,
                Multiplier = 100,
            });
            await ctx.SaveChangesAsync();
        }

        var events = new List<FlexOptionEventDto>
        {
            // Assignment on the option
            new()
            {
                ConId = 723197,
                Symbol = "PEP P155",
                AssetCategory = "OPT",
                TransactionType = "Assignment",
                Date = new DateTime(2026, 4, 14),
                TradePrice = 0,
                Commission = 0,
                Strike = 155,
                PutCall = "P",
                Multiplier = 100,
            },
            // Stock delivery leg — should be skipped
            new()
            {
                ConId = 11017,
                Symbol = "PEP",
                AssetCategory = "STK",
                TransactionType = "Buy",
                Date = new DateTime(2026, 4, 14),
                TradePrice = 155,
                Commission = 0,
                Multiplier = 1,
            },
        };

        var processed = await _service.SyncOptionEventsAsync(events);

        Assert.Equal(1, processed); // Only the OPT event, not the STK leg
        using var verifyCtx = _fixture.CreateContext();
        var pos = verifyCtx.OptionPositions.Single(p => p.ConId == 723197);
        Assert.NotNull(pos.Closed);
        Assert.True(pos.CloseReasons!.Value.HasFlag(CloseReasons.Other));
    }

    [Fact]
    public async Task SyncOptionEvents_AlreadyClosed_Skipped()
    {
        using (var ctx = _fixture.CreateContext())
        {
            ctx.OptionPositions.Add(new OptionPosition
            {
                Symbol = "AAPL C200",
                ContractId = "555",
                ConId = 555,
                SecType = "OPT",
                Opened = new DateTime(2026, 1, 1),
                Expiry = new DateTime(2026, 2, 20),
                Closed = new DateTime(2026, 2, 10), // Already closed by trade
                Pos = 0,
                Right = PositionRight.Call,
                Strike = 200,
                Cost = 5,
                Multiplier = 100,
            });
            await ctx.SaveChangesAsync();
        }

        var events = new List<FlexOptionEventDto>
        {
            new()
            {
                ConId = 555,
                Symbol = "AAPL C200",
                AssetCategory = "OPT",
                TransactionType = "Expiration",
                Date = new DateTime(2026, 2, 20),
                TradePrice = 0,
                Commission = 0,
                Strike = 200,
                PutCall = "C",
                Multiplier = 100,
            }
        };

        var processed = await _service.SyncOptionEventsAsync(events);

        Assert.Equal(0, processed); // Already closed — skipped
    }

    // ── SyncTradesAsync — backfill ────────────────────────────

    [Fact]
    public async Task SyncTrades_ExistingTrade_BackfillsTier1Fields()
    {
        // Arrange: insert a trade without Tier 1 fields (simulating old TWS sync)
        using (var ctx = _fixture.CreateContext())
        {
            ctx.StockPositions.Add(new StockPosition
            {
                Symbol = "AAPL",
                Date = new DateTime(2026, 1, 15),
                PosChange = 10,
                Price = 150,
                Commission = 0, // Missing from TWS
                Multiplier = 1,
                ConId = 42,
                ExecutionId = "TRADE-1",
            });
            await ctx.SaveChangesAsync();
        }

        var flexTrades = new List<FlexTradeDto>
        {
            new()
            {
                TradeId = "TRADE-1",
                ConId = 42,
                Symbol = "AAPL",
                AssetCategory = "STK",
                DateTime = new DateTime(2026, 1, 15),
                Quantity = 10,
                TradePrice = 150,
                Commission = -1.05m,
                BuySell = "BUY",
                Multiplier = 1,
                Exchange = "NYSE",
                FxRateToBase = 0.92m,
                FifoPnlRealized = 0,
                Taxes = 0.35m,
            }
        };

        // Act
        var (created, updated, _, _) = await _service.SyncTradesAsync(flexTrades);

        // Assert
        Assert.Equal(0, created);
        Assert.Equal(1, updated);

        using var verifyCtx = _fixture.CreateContext();
        var trade = verifyCtx.StockPositions.Single(t => t.ExecutionId == "TRADE-1");
        Assert.Equal(1.05m, trade.Commission);
        Assert.Equal("NYSE", trade.Exchange);
        Assert.Equal(0.92m, trade.FxRateToBase);
        Assert.Equal(0.35m, trade.Taxes);
    }
}
