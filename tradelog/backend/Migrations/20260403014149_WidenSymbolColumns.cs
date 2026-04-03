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
            migrationBuilder.AlterColumn<string>(
                name: "Symbol",
                table: "Trades",
                type: "nvarchar(60)",
                maxLength: 60,
                nullable: false,
                oldClrType: typeof(string),
                oldType: "nvarchar(20)",
                oldMaxLength: 20);

            migrationBuilder.AlterColumn<string>(
                name: "Symbol",
                table: "OptionPositions",
                type: "nvarchar(60)",
                maxLength: 60,
                nullable: false,
                oldClrType: typeof(string),
                oldType: "nvarchar(20)",
                oldMaxLength: 20);

            migrationBuilder.AlterColumn<string>(
                name: "ContractId",
                table: "OptionPositions",
                type: "nvarchar(40)",
                maxLength: 40,
                nullable: false,
                oldClrType: typeof(string),
                oldType: "nvarchar(20)",
                oldMaxLength: 20);

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
            migrationBuilder.AlterColumn<string>(
                name: "Symbol",
                table: "Trades",
                type: "nvarchar(20)",
                maxLength: 20,
                nullable: false,
                oldClrType: typeof(string),
                oldType: "nvarchar(60)",
                oldMaxLength: 60);

            migrationBuilder.AlterColumn<string>(
                name: "Symbol",
                table: "OptionPositions",
                type: "nvarchar(20)",
                maxLength: 20,
                nullable: false,
                oldClrType: typeof(string),
                oldType: "nvarchar(60)",
                oldMaxLength: 60);

            migrationBuilder.AlterColumn<string>(
                name: "ContractId",
                table: "OptionPositions",
                type: "nvarchar(20)",
                maxLength: 20,
                nullable: false,
                oldClrType: typeof(string),
                oldType: "nvarchar(40)",
                oldMaxLength: 40);
        }
    }
}
