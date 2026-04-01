using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace tradelog.Migrations
{
    /// <inheritdoc />
    public partial class ExtendTradeAndOptionFields : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "Exchange",
                table: "Trades",
                type: "nvarchar(20)",
                maxLength: 20,
                nullable: true);

            migrationBuilder.AddColumn<decimal>(
                name: "FifoPnlRealized",
                table: "Trades",
                type: "decimal(18,6)",
                precision: 18,
                scale: 6,
                nullable: false,
                defaultValue: 0m);

            migrationBuilder.AddColumn<decimal>(
                name: "FxRateToBase",
                table: "Trades",
                type: "decimal(18,6)",
                precision: 18,
                scale: 6,
                nullable: false,
                defaultValue: 0m);

            migrationBuilder.AddColumn<decimal>(
                name: "Taxes",
                table: "Trades",
                type: "decimal(18,6)",
                precision: 18,
                scale: 6,
                nullable: false,
                defaultValue: 0m);

            migrationBuilder.AddColumn<int>(
                name: "UnderlyingConid",
                table: "OptionPositions",
                type: "int",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "UnderlyingSymbol",
                table: "OptionPositions",
                type: "nvarchar(20)",
                maxLength: 20,
                nullable: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "Exchange",
                table: "Trades");

            migrationBuilder.DropColumn(
                name: "FifoPnlRealized",
                table: "Trades");

            migrationBuilder.DropColumn(
                name: "FxRateToBase",
                table: "Trades");

            migrationBuilder.DropColumn(
                name: "Taxes",
                table: "Trades");

            migrationBuilder.DropColumn(
                name: "UnderlyingConid",
                table: "OptionPositions");

            migrationBuilder.DropColumn(
                name: "UnderlyingSymbol",
                table: "OptionPositions");
        }
    }
}
