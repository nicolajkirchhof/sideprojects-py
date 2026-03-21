using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace tradelog.Migrations
{
    /// <inheritdoc />
    public partial class AddBestExitFields : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropIndex(
                name: "IX_OptionPositionsLogs_ContractId_DateTime",
                table: "OptionPositionsLogs");

            migrationBuilder.AddColumn<DateTime>(
                name: "BestExitDate",
                table: "Trades",
                type: "datetime2",
                nullable: true);

            migrationBuilder.AddColumn<decimal>(
                name: "BestExitPrice",
                table: "Trades",
                type: "decimal(18,6)",
                precision: 18,
                scale: 6,
                nullable: true);

            migrationBuilder.AddColumn<DateTime>(
                name: "BestExitDate",
                table: "OptionPositions",
                type: "datetime2",
                nullable: true);

            migrationBuilder.AddColumn<decimal>(
                name: "BestExitPrice",
                table: "OptionPositions",
                type: "decimal(18,6)",
                precision: 18,
                scale: 6,
                nullable: true);

            migrationBuilder.CreateIndex(
                name: "IX_OptionPositionsLogs_AccountId_ContractId_DateTime",
                table: "OptionPositionsLogs",
                columns: new[] { "AccountId", "ContractId", "DateTime" },
                unique: true,
                descending: new[] { false, false, true });
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropIndex(
                name: "IX_OptionPositionsLogs_AccountId_ContractId_DateTime",
                table: "OptionPositionsLogs");

            migrationBuilder.DropColumn(
                name: "BestExitDate",
                table: "Trades");

            migrationBuilder.DropColumn(
                name: "BestExitPrice",
                table: "Trades");

            migrationBuilder.DropColumn(
                name: "BestExitDate",
                table: "OptionPositions");

            migrationBuilder.DropColumn(
                name: "BestExitPrice",
                table: "OptionPositions");

            migrationBuilder.CreateIndex(
                name: "IX_OptionPositionsLogs_ContractId_DateTime",
                table: "OptionPositionsLogs",
                columns: new[] { "ContractId", "DateTime" },
                unique: true,
                descending: new[] { false, true });
        }
    }
}
