using Microsoft.EntityFrameworkCore;
using tradelog.Data;
using tradelog.Models;

namespace tradelog.Tests.Fixtures;

/// <summary>
/// Seeds default lookup values for tests. Returns IDs for use in test data.
/// Must be called after the test account is created.
/// </summary>
public static class LookupSeeder
{
    // Well-known IDs (assigned by SQLite IDENTITY in order of insertion)
    public static int BudgetLongTerm { get; private set; }
    public static int BudgetDrift { get; private set; }
    public static int BudgetSwing { get; private set; }
    public static int BudgetSpeculative { get; private set; }

    public static int StrategyPositiveDrift { get; private set; }
    public static int StrategyRangeBound { get; private set; }
    public static int StrategyBreakoutMomentum { get; private set; }

    public static int TypeShortStrangle { get; private set; }
    public static int TypeShortPut { get; private set; }
    public static int TypeShortPutSpread { get; private set; }
    public static int TypeLongCall { get; private set; }
    public static int TypeLongStock { get; private set; }

    public static int DirectionalBullish { get; private set; }
    public static int DirectionalNeutral { get; private set; }

    public static int TimeframeOneDay { get; private set; }

    public static int MgmtAsPlanned { get; private set; }

    public static void Seed(DataContext ctx, int accountId)
    {
        var values = new List<LookupValue>
        {
            new() { AccountId = accountId, Category = LookupCategory.Budget, Name = "Long-Term", SortOrder = 0 },
            new() { AccountId = accountId, Category = LookupCategory.Budget, Name = "Drift", SortOrder = 1 },
            new() { AccountId = accountId, Category = LookupCategory.Budget, Name = "Swing", SortOrder = 2 },
            new() { AccountId = accountId, Category = LookupCategory.Budget, Name = "Speculative", SortOrder = 3 },

            new() { AccountId = accountId, Category = LookupCategory.Strategy, Name = "Positive Drift", SortOrder = 0 },
            new() { AccountId = accountId, Category = LookupCategory.Strategy, Name = "Range Bound", SortOrder = 1 },
            new() { AccountId = accountId, Category = LookupCategory.Strategy, Name = "Breakout Momentum", SortOrder = 2 },

            new() { AccountId = accountId, Category = LookupCategory.TypeOfTrade, Name = "Short Strangle", SortOrder = 0 },
            new() { AccountId = accountId, Category = LookupCategory.TypeOfTrade, Name = "Short Put", SortOrder = 1 },
            new() { AccountId = accountId, Category = LookupCategory.TypeOfTrade, Name = "Short Put Spread", SortOrder = 2 },
            new() { AccountId = accountId, Category = LookupCategory.TypeOfTrade, Name = "Long Call", SortOrder = 3 },
            new() { AccountId = accountId, Category = LookupCategory.TypeOfTrade, Name = "Long Stock", SortOrder = 4 },

            new() { AccountId = accountId, Category = LookupCategory.Directional, Name = "Bullish", SortOrder = 0 },
            new() { AccountId = accountId, Category = LookupCategory.Directional, Name = "Neutral", SortOrder = 1 },

            new() { AccountId = accountId, Category = LookupCategory.Timeframe, Name = "1 Day", SortOrder = 0 },

            new() { AccountId = accountId, Category = LookupCategory.ManagementRating, Name = "As Planned", SortOrder = 0 },
        };

        ctx.LookupValues.AddRange(values);
        ctx.SaveChanges();

        // Read back assigned IDs
        var all = ctx.LookupValues.IgnoreQueryFilters()
            .Where(lv => lv.AccountId == accountId)
            .ToList();

        LookupValue Get(string cat, string name) => all.First(lv => lv.Category == cat && lv.Name == name);

        BudgetLongTerm = Get(LookupCategory.Budget, "Long-Term").Id;
        BudgetDrift = Get(LookupCategory.Budget, "Drift").Id;
        BudgetSwing = Get(LookupCategory.Budget, "Swing").Id;
        BudgetSpeculative = Get(LookupCategory.Budget, "Speculative").Id;

        StrategyPositiveDrift = Get(LookupCategory.Strategy, "Positive Drift").Id;
        StrategyRangeBound = Get(LookupCategory.Strategy, "Range Bound").Id;
        StrategyBreakoutMomentum = Get(LookupCategory.Strategy, "Breakout Momentum").Id;

        TypeShortStrangle = Get(LookupCategory.TypeOfTrade, "Short Strangle").Id;
        TypeShortPut = Get(LookupCategory.TypeOfTrade, "Short Put").Id;
        TypeShortPutSpread = Get(LookupCategory.TypeOfTrade, "Short Put Spread").Id;
        TypeLongCall = Get(LookupCategory.TypeOfTrade, "Long Call").Id;
        TypeLongStock = Get(LookupCategory.TypeOfTrade, "Long Stock").Id;

        DirectionalBullish = Get(LookupCategory.Directional, "Bullish").Id;
        DirectionalNeutral = Get(LookupCategory.Directional, "Neutral").Id;

        TimeframeOneDay = Get(LookupCategory.Timeframe, "1 Day").Id;

        MgmtAsPlanned = Get(LookupCategory.ManagementRating, "As Planned").Id;
    }
}
