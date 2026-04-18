using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace tradelog.Migrations
{
    /// <inheritdoc />
    public partial class AddDailyPrepAndTradeAnalysis : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "DailyPreps",
                columns: table => new
                {
                    Id = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    AccountId = table.Column<int>(type: "int", nullable: false),
                    Date = table.Column<DateTime>(type: "datetime2", nullable: false),
                    MarketSummary = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    Watchlist = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    EmailCount = table.Column<int>(type: "int", nullable: false),
                    CandidateCount = table.Column<int>(type: "int", nullable: false),
                    CreatedAt = table.Column<DateTime>(type: "datetime2", nullable: false),
                    UpdatedAt = table.Column<DateTime>(type: "datetime2", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_DailyPreps", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "TradeAnalyses",
                columns: table => new
                {
                    Id = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    AccountId = table.Column<int>(type: "int", nullable: false),
                    TradeId = table.Column<int>(type: "int", nullable: false),
                    AnalysisDate = table.Column<DateTime>(type: "datetime2", nullable: false),
                    Score = table.Column<int>(type: "int", nullable: false),
                    Analysis = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    Model = table.Column<string>(type: "nvarchar(50)", maxLength: 50, nullable: true),
                    CreatedAt = table.Column<DateTime>(type: "datetime2", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_TradeAnalyses", x => x.Id);
                    table.ForeignKey(
                        name: "FK_TradeAnalyses_Trades_TradeId",
                        column: x => x.TradeId,
                        principalTable: "Trades",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateIndex(
                name: "IX_DailyPreps_AccountId_Date",
                table: "DailyPreps",
                columns: new[] { "AccountId", "Date" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_TradeAnalyses_TradeId_AnalysisDate",
                table: "TradeAnalyses",
                columns: new[] { "TradeId", "AnalysisDate" },
                unique: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "DailyPreps");

            migrationBuilder.DropTable(
                name: "TradeAnalyses");
        }
    }
}
