using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace tradelog.Migrations
{
    /// <inheritdoc />
    public partial class InitialCreate : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "Capitals",
                columns: table => new
                {
                    Id = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    Date = table.Column<DateTime>(type: "datetime2", nullable: false),
                    NetLiquidity = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false),
                    Maintenance = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false),
                    ExcessLiquidity = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false),
                    Bpr = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false),
                    MaintenancePct = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false),
                    TotalPnl = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false),
                    UnrealizedPnl = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false),
                    RealizedPnl = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false),
                    NetDelta = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false),
                    NetTheta = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false),
                    NetVega = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false),
                    NetGamma = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false),
                    AvgIv = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false),
                    TotalMargin = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false),
                    TotalCommissions = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Capitals", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "OptionPositions",
                columns: table => new
                {
                    Id = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    Symbol = table.Column<string>(type: "nvarchar(20)", maxLength: 20, nullable: false),
                    ContractId = table.Column<string>(type: "nvarchar(20)", maxLength: 20, nullable: false),
                    Opened = table.Column<DateTime>(type: "datetime2", nullable: false),
                    Expiry = table.Column<DateTime>(type: "datetime2", nullable: false),
                    Closed = table.Column<DateTime>(type: "datetime2", nullable: true),
                    Pos = table.Column<int>(type: "int", nullable: false),
                    Right = table.Column<int>(type: "int", nullable: false),
                    Strike = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false),
                    Cost = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false),
                    ClosePrice = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: true),
                    Commission = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false),
                    Multiplier = table.Column<int>(type: "int", nullable: false),
                    CloseReasons = table.Column<int>(type: "int", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_OptionPositions", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "OptionPositionsLogs",
                columns: table => new
                {
                    Id = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    DateTime = table.Column<DateTime>(type: "datetime2", nullable: false),
                    ContractId = table.Column<string>(type: "nvarchar(20)", maxLength: 20, nullable: false),
                    Underlying = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false),
                    Iv = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false),
                    Price = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false),
                    TimeValue = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false),
                    Delta = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false),
                    Theta = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false),
                    Gamma = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false),
                    Vega = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false),
                    Margin = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_OptionPositionsLogs", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "Portfolios",
                columns: table => new
                {
                    Id = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    Budget = table.Column<int>(type: "int", nullable: false),
                    Strategy = table.Column<int>(type: "int", nullable: false),
                    MinAllocation = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false),
                    MaxAllocation = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Portfolios", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "TradeEntries",
                columns: table => new
                {
                    Id = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    Symbol = table.Column<string>(type: "nvarchar(20)", maxLength: 20, nullable: false),
                    Date = table.Column<DateTime>(type: "datetime2", nullable: false),
                    TypeOfTrade = table.Column<int>(type: "int", nullable: false),
                    Notes = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    Directional = table.Column<int>(type: "int", nullable: true),
                    Timeframe = table.Column<int>(type: "int", nullable: true),
                    Budget = table.Column<int>(type: "int", nullable: false),
                    Strategy = table.Column<int>(type: "int", nullable: false),
                    NewsCatalyst = table.Column<bool>(type: "bit", nullable: false),
                    RecentEarnings = table.Column<bool>(type: "bit", nullable: false),
                    SectorSupport = table.Column<bool>(type: "bit", nullable: false),
                    Ath = table.Column<bool>(type: "bit", nullable: false),
                    Rvol = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: true),
                    InstitutionalSupport = table.Column<string>(type: "nvarchar(100)", maxLength: 100, nullable: true),
                    GapPct = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: true),
                    XAtrMove = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: true),
                    TaFaNotes = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    IntendedManagement = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    ActualManagement = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    ManagementRating = table.Column<int>(type: "int", nullable: true),
                    Learnings = table.Column<string>(type: "nvarchar(max)", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_TradeEntries", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "Trades",
                columns: table => new
                {
                    Id = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    Symbol = table.Column<string>(type: "nvarchar(20)", maxLength: 20, nullable: false),
                    Date = table.Column<DateTime>(type: "datetime2", nullable: false),
                    PosChange = table.Column<int>(type: "int", nullable: false),
                    Price = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false),
                    Commission = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: false),
                    Multiplier = table.Column<int>(type: "int", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Trades", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "WeeklyPreps",
                columns: table => new
                {
                    Id = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    Date = table.Column<DateTime>(type: "datetime2", nullable: false),
                    IndexBias = table.Column<string>(type: "nvarchar(20)", maxLength: 20, nullable: true),
                    Breadth = table.Column<string>(type: "nvarchar(20)", maxLength: 20, nullable: true),
                    NotableSectors = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    VolatilityNotes = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    OpenPositionsRequiringManagement = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    CurrentPortfolioRisk = table.Column<string>(type: "nvarchar(20)", maxLength: 20, nullable: true),
                    PortfolioNotes = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    ScanningFor = table.Column<string>(type: "nvarchar(50)", maxLength: 50, nullable: true),
                    IndexSectorPreference = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    Watchlist = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    Learnings = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    FocusForImprovement = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    ExternalComments = table.Column<string>(type: "nvarchar(max)", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_WeeklyPreps", x => x.Id);
                });

            migrationBuilder.CreateIndex(
                name: "IX_OptionPositions_ContractId",
                table: "OptionPositions",
                column: "ContractId");

            migrationBuilder.CreateIndex(
                name: "IX_OptionPositions_Symbol",
                table: "OptionPositions",
                column: "Symbol");

            migrationBuilder.CreateIndex(
                name: "IX_OptionPositionsLogs_ContractId_DateTime",
                table: "OptionPositionsLogs",
                columns: new[] { "ContractId", "DateTime" },
                unique: true,
                descending: new[] { false, true });

            migrationBuilder.CreateIndex(
                name: "IX_TradeEntries_Symbol",
                table: "TradeEntries",
                column: "Symbol");

            migrationBuilder.CreateIndex(
                name: "IX_Trades_Symbol_Date",
                table: "Trades",
                columns: new[] { "Symbol", "Date" });
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "Capitals");

            migrationBuilder.DropTable(
                name: "OptionPositions");

            migrationBuilder.DropTable(
                name: "OptionPositionsLogs");

            migrationBuilder.DropTable(
                name: "Portfolios");

            migrationBuilder.DropTable(
                name: "TradeEntries");

            migrationBuilder.DropTable(
                name: "Trades");

            migrationBuilder.DropTable(
                name: "WeeklyPreps");
        }
    }
}
