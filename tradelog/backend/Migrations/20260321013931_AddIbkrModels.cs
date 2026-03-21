using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace tradelog.Migrations
{
    /// <inheritdoc />
    public partial class AddIbkrModels : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<int>(
                name: "ConId",
                table: "Trades",
                type: "int",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "ExecutionId",
                table: "Trades",
                type: "nvarchar(50)",
                maxLength: 50,
                nullable: true);

            migrationBuilder.AddColumn<int>(
                name: "ConId",
                table: "OptionPositions",
                type: "int",
                nullable: true);

            migrationBuilder.CreateTable(
                name: "IbkrConfigs",
                columns: table => new
                {
                    Id = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    Host = table.Column<string>(type: "nvarchar(100)", maxLength: 100, nullable: false),
                    Port = table.Column<int>(type: "int", nullable: false),
                    ClientId = table.Column<int>(type: "int", nullable: false),
                    LastSyncAt = table.Column<DateTime>(type: "datetime2", nullable: true),
                    LastSyncResult = table.Column<string>(type: "nvarchar(500)", maxLength: 500, nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_IbkrConfigs", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "StockPriceCaches",
                columns: table => new
                {
                    Id = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    Symbol = table.Column<string>(type: "nvarchar(20)", maxLength: 20, nullable: false),
                    LastPrice = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false),
                    UpdatedAt = table.Column<DateTime>(type: "datetime2", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_StockPriceCaches", x => x.Id);
                });

            migrationBuilder.CreateIndex(
                name: "IX_Trades_ExecutionId",
                table: "Trades",
                column: "ExecutionId");

            migrationBuilder.CreateIndex(
                name: "IX_StockPriceCaches_Symbol",
                table: "StockPriceCaches",
                column: "Symbol",
                unique: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "IbkrConfigs");

            migrationBuilder.DropTable(
                name: "StockPriceCaches");

            migrationBuilder.DropIndex(
                name: "IX_Trades_ExecutionId",
                table: "Trades");

            migrationBuilder.DropColumn(
                name: "ConId",
                table: "Trades");

            migrationBuilder.DropColumn(
                name: "ExecutionId",
                table: "Trades");

            migrationBuilder.DropColumn(
                name: "ConId",
                table: "OptionPositions");
        }
    }
}
