using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace tradelog.Migrations
{
    /// <inheritdoc />
    public partial class ConfigurableEnums : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "LookupValues",
                columns: table => new
                {
                    Id = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    AccountId = table.Column<int>(type: "int", nullable: false),
                    Category = table.Column<string>(type: "nvarchar(30)", maxLength: 30, nullable: false),
                    Name = table.Column<string>(type: "nvarchar(100)", maxLength: 100, nullable: false),
                    SortOrder = table.Column<int>(type: "int", nullable: false),
                    IsActive = table.Column<bool>(type: "bit", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_LookupValues", x => x.Id);
                });

            migrationBuilder.CreateIndex(
                name: "IX_LookupValues_AccountId_Category_Name",
                table: "LookupValues",
                columns: new[] { "AccountId", "Category", "Name" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_LookupValues_AccountId_Category_SortOrder",
                table: "LookupValues",
                columns: new[] { "AccountId", "Category", "SortOrder" });

            // ── Seed lookup values for every existing account ─────────
            // Uses old enum ordinals as SortOrder so the FK remap below
            // can JOIN on (AccountId, Category, SortOrder = old int value).

            var seedSql = @"
-- Budget: LongTerm=0, Drift=1, Swing=2, Speculative=3
INSERT INTO LookupValues (AccountId, Category, Name, SortOrder, IsActive)
SELECT Id, 'Budget', v.Name, v.SortOrder, 1
FROM Accounts CROSS APPLY (VALUES
  (0,'Long-Term'),(1,'Drift'),(2,'Swing'),(3,'Speculative')
) AS v(SortOrder, Name);

-- Strategy: 0..9
INSERT INTO LookupValues (AccountId, Category, Name, SortOrder, IsActive)
SELECT Id, 'Strategy', v.Name, v.SortOrder, 1
FROM Accounts CROSS APPLY (VALUES
  (0,'Positive Drift'),(1,'Range Bound'),(2,'PEAD'),(3,'Breakout Momentum'),
  (4,'IV Mean Reversion'),(5,'Sector Strength'),(6,'Sector Weakness'),
  (7,'Green Line Breakout'),(8,'Slingshot'),(9,'Pre-Earnings')
) AS v(SortOrder, Name);

-- TypeOfTrade: 0..19
INSERT INTO LookupValues (AccountId, Category, Name, SortOrder, IsActive)
SELECT Id, 'TypeOfTrade', v.Name, v.SortOrder, 1
FROM Accounts CROSS APPLY (VALUES
  (0,'Short Strangle'),(1,'Short Put Spread'),(2,'Short Call Spread'),
  (3,'Long Call'),(4,'Long Put'),(5,'Long Call Vertical'),
  (6,'Long Put Vertical'),(7,'Synthetic Long'),(8,'Covered Strangle'),
  (9,'Butterfly'),(10,'Ratio Diagonal Spread'),(11,'Long Strangle'),
  (12,'Short Put'),(13,'Short Call'),(14,'Long Stock'),(15,'Short Stock'),
  (16,'Iron Condor'),(17,'XYZ'),(18,'PMCC'),(19,'Calendar Spread')
) AS v(SortOrder, Name);

-- DirectionalBias: 0..2
INSERT INTO LookupValues (AccountId, Category, Name, SortOrder, IsActive)
SELECT Id, 'Directional', v.Name, v.SortOrder, 1
FROM Accounts CROSS APPLY (VALUES
  (0,'Bullish'),(1,'Neutral'),(2,'Bearish')
) AS v(SortOrder, Name);

-- Timeframe: 0..2
INSERT INTO LookupValues (AccountId, Category, Name, SortOrder, IsActive)
SELECT Id, 'Timeframe', v.Name, v.SortOrder, 1
FROM Accounts CROSS APPLY (VALUES
  (0,'1 Day'),(1,'1 Week'),(2,'Delta Band')
) AS v(SortOrder, Name);

-- ManagementRating: 0..2
INSERT INTO LookupValues (AccountId, Category, Name, SortOrder, IsActive)
SELECT Id, 'ManagementRating', v.Name, v.SortOrder, 1
FROM Accounts CROSS APPLY (VALUES
  (0,'As Planned'),(1,'Minor Adjustments'),(2,'Rogue')
) AS v(SortOrder, Name);
";
            migrationBuilder.Sql(seedSql);

            // ── Remap FK columns: old enum ordinal → LookupValues.Id ──
            // Trades table has: TypeOfTrade, Directional, Timeframe, Budget, Strategy, ManagementRating
            var remapSql = @"
-- Trades.Budget
UPDATE t SET t.Budget = lv.Id
FROM Trades t
INNER JOIN LookupValues lv ON lv.AccountId = t.AccountId
  AND lv.Category = 'Budget' AND lv.SortOrder = t.Budget;

-- Trades.Strategy
UPDATE t SET t.Strategy = lv.Id
FROM Trades t
INNER JOIN LookupValues lv ON lv.AccountId = t.AccountId
  AND lv.Category = 'Strategy' AND lv.SortOrder = t.Strategy;

-- Trades.TypeOfTrade
UPDATE t SET t.TypeOfTrade = lv.Id
FROM Trades t
INNER JOIN LookupValues lv ON lv.AccountId = t.AccountId
  AND lv.Category = 'TypeOfTrade' AND lv.SortOrder = t.TypeOfTrade;

-- Trades.Directional (nullable)
UPDATE t SET t.Directional = lv.Id
FROM Trades t
INNER JOIN LookupValues lv ON lv.AccountId = t.AccountId
  AND lv.Category = 'Directional' AND lv.SortOrder = t.Directional
WHERE t.Directional IS NOT NULL;

-- Trades.Timeframe (nullable)
UPDATE t SET t.Timeframe = lv.Id
FROM Trades t
INNER JOIN LookupValues lv ON lv.AccountId = t.AccountId
  AND lv.Category = 'Timeframe' AND lv.SortOrder = t.Timeframe
WHERE t.Timeframe IS NOT NULL;

-- Trades.ManagementRating (nullable)
UPDATE t SET t.ManagementRating = lv.Id
FROM Trades t
INNER JOIN LookupValues lv ON lv.AccountId = t.AccountId
  AND lv.Category = 'ManagementRating' AND lv.SortOrder = t.ManagementRating
WHERE t.ManagementRating IS NOT NULL;

-- Portfolios.Budget
UPDATE p SET p.Budget = lv.Id
FROM Portfolios p
INNER JOIN LookupValues lv ON lv.AccountId = p.AccountId
  AND lv.Category = 'Budget' AND lv.SortOrder = p.Budget;

-- Portfolios.Strategy
UPDATE p SET p.Strategy = lv.Id
FROM Portfolios p
INNER JOIN LookupValues lv ON lv.AccountId = p.AccountId
  AND lv.Category = 'Strategy' AND lv.SortOrder = p.Strategy;
";
            migrationBuilder.Sql(remapSql);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "LookupValues");
        }
    }
}
