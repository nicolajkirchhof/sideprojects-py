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
SELECT a.Id, 'Budget', v.Name, v.SortOrder, 1
FROM Accounts a, (
  SELECT 0 AS SortOrder, 'Long-Term' AS Name UNION ALL
  SELECT 1, 'Drift' UNION ALL SELECT 2, 'Swing' UNION ALL SELECT 3, 'Speculative'
) v;

-- Strategy: 0..9
INSERT INTO LookupValues (AccountId, Category, Name, SortOrder, IsActive)
SELECT a.Id, 'Strategy', v.Name, v.SortOrder, 1
FROM Accounts a, (
  SELECT 0 AS SortOrder, 'Positive Drift' AS Name UNION ALL
  SELECT 1, 'Range Bound' UNION ALL SELECT 2, 'PEAD' UNION ALL
  SELECT 3, 'Breakout Momentum' UNION ALL SELECT 4, 'IV Mean Reversion' UNION ALL
  SELECT 5, 'Sector Strength' UNION ALL SELECT 6, 'Sector Weakness' UNION ALL
  SELECT 7, 'Green Line Breakout' UNION ALL SELECT 8, 'Slingshot' UNION ALL
  SELECT 9, 'Pre-Earnings'
) v;

-- TypeOfTrade: 0..19
INSERT INTO LookupValues (AccountId, Category, Name, SortOrder, IsActive)
SELECT a.Id, 'TypeOfTrade', v.Name, v.SortOrder, 1
FROM Accounts a, (
  SELECT 0 AS SortOrder, 'Short Strangle' AS Name UNION ALL
  SELECT 1, 'Short Put Spread' UNION ALL SELECT 2, 'Short Call Spread' UNION ALL
  SELECT 3, 'Long Call' UNION ALL SELECT 4, 'Long Put' UNION ALL
  SELECT 5, 'Long Call Vertical' UNION ALL SELECT 6, 'Long Put Vertical' UNION ALL
  SELECT 7, 'Synthetic Long' UNION ALL SELECT 8, 'Covered Strangle' UNION ALL
  SELECT 9, 'Butterfly' UNION ALL SELECT 10, 'Ratio Diagonal Spread' UNION ALL
  SELECT 11, 'Long Strangle' UNION ALL SELECT 12, 'Short Put' UNION ALL
  SELECT 13, 'Short Call' UNION ALL SELECT 14, 'Long Stock' UNION ALL
  SELECT 15, 'Short Stock' UNION ALL SELECT 16, 'Iron Condor' UNION ALL
  SELECT 17, 'XYZ' UNION ALL SELECT 18, 'PMCC' UNION ALL SELECT 19, 'Calendar Spread'
) v;

-- DirectionalBias: 0..2
INSERT INTO LookupValues (AccountId, Category, Name, SortOrder, IsActive)
SELECT a.Id, 'Directional', v.Name, v.SortOrder, 1
FROM Accounts a, (
  SELECT 0 AS SortOrder, 'Bullish' AS Name UNION ALL
  SELECT 1, 'Neutral' UNION ALL SELECT 2, 'Bearish'
) v;

-- Timeframe: 0..2
INSERT INTO LookupValues (AccountId, Category, Name, SortOrder, IsActive)
SELECT a.Id, 'Timeframe', v.Name, v.SortOrder, 1
FROM Accounts a, (
  SELECT 0 AS SortOrder, '1 Day' AS Name UNION ALL
  SELECT 1, '1 Week' UNION ALL SELECT 2, 'Delta Band'
) v;

-- ManagementRating: 0..2
INSERT INTO LookupValues (AccountId, Category, Name, SortOrder, IsActive)
SELECT a.Id, 'ManagementRating', v.Name, v.SortOrder, 1
FROM Accounts a, (
  SELECT 0 AS SortOrder, 'As Planned' AS Name UNION ALL
  SELECT 1, 'Minor Adjustments' UNION ALL SELECT 2, 'Rogue'
) v;
";
            migrationBuilder.Sql(seedSql);

            // ── Remap FK columns: old enum ordinal → LookupValues.Id ──
            // Trades table has: TypeOfTrade, Directional, Timeframe, Budget, Strategy, ManagementRating
            var remapSql = @"
-- Trades.Budget
UPDATE Trades SET Budget = (
  SELECT lv.Id FROM LookupValues lv
  WHERE lv.AccountId = Trades.AccountId AND lv.Category = 'Budget' AND lv.SortOrder = Trades.Budget
);

-- Trades.Strategy
UPDATE Trades SET Strategy = (
  SELECT lv.Id FROM LookupValues lv
  WHERE lv.AccountId = Trades.AccountId AND lv.Category = 'Strategy' AND lv.SortOrder = Trades.Strategy
);

-- Trades.TypeOfTrade
UPDATE Trades SET TypeOfTrade = (
  SELECT lv.Id FROM LookupValues lv
  WHERE lv.AccountId = Trades.AccountId AND lv.Category = 'TypeOfTrade' AND lv.SortOrder = Trades.TypeOfTrade
);

-- Trades.Directional (nullable)
UPDATE Trades SET Directional = (
  SELECT lv.Id FROM LookupValues lv
  WHERE lv.AccountId = Trades.AccountId AND lv.Category = 'Directional' AND lv.SortOrder = Trades.Directional
) WHERE Directional IS NOT NULL;

-- Trades.Timeframe (nullable)
UPDATE Trades SET Timeframe = (
  SELECT lv.Id FROM LookupValues lv
  WHERE lv.AccountId = Trades.AccountId AND lv.Category = 'Timeframe' AND lv.SortOrder = Trades.Timeframe
) WHERE Timeframe IS NOT NULL;

-- Trades.ManagementRating (nullable)
UPDATE Trades SET ManagementRating = (
  SELECT lv.Id FROM LookupValues lv
  WHERE lv.AccountId = Trades.AccountId AND lv.Category = 'ManagementRating' AND lv.SortOrder = Trades.ManagementRating
) WHERE ManagementRating IS NOT NULL;

-- Portfolios.Budget
UPDATE Portfolios SET Budget = (
  SELECT lv.Id FROM LookupValues lv
  WHERE lv.AccountId = Portfolios.AccountId AND lv.Category = 'Budget' AND lv.SortOrder = Portfolios.Budget
);

-- Portfolios.Strategy
UPDATE Portfolios SET Strategy = (
  SELECT lv.Id FROM LookupValues lv
  WHERE lv.AccountId = Portfolios.AccountId AND lv.Category = 'Strategy' AND lv.SortOrder = Portfolios.Strategy
);
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
