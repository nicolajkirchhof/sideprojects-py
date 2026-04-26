using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace tradelog.Migrations
{
    /// <inheritdoc />
    public partial class WidenSymbolColumns : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            // AlterColumn operations omitted: SQLite ignores string length constraints,
            // so nvarchar(20) → nvarchar(60) is a no-op at the storage level.

            // Reset staging data for clean Flex sync baseline
            migrationBuilder.Sql("DELETE FROM [OptionPositionsLogs];");
            migrationBuilder.Sql("DELETE FROM [OptionPositions];");
            migrationBuilder.Sql("DELETE FROM [Trades];");
            migrationBuilder.Sql("DELETE FROM [Capitals];");
            migrationBuilder.Sql("DELETE FROM [TradeEntries];");
            migrationBuilder.Sql("DELETE FROM [WeeklyPreps];");
            migrationBuilder.Sql("DELETE FROM [Portfolios];");
            migrationBuilder.Sql("DELETE FROM [StockPriceCaches];");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            // AlterColumn operations omitted: no-op on SQLite.
        }
    }
}
