using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace tradelog.Migrations
{
    /// <inheritdoc />
    public partial class TradecentricRename : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            // ── Step 1: Rename Trades → StockPositions (the IBKR fills table) ──
            migrationBuilder.RenameTable(name: "Trades", newName: "StockPositions");

            // Rename existing indexes to match new table name
            migrationBuilder.RenameIndex(
                name: "IX_Trades_Symbol_Date", table: "StockPositions", newName: "IX_StockPositions_Symbol_Date");
            migrationBuilder.RenameIndex(
                name: "IX_Trades_ExecutionId", table: "StockPositions", newName: "IX_StockPositions_ExecutionId");

            // Add TradeId FK column to StockPositions
            migrationBuilder.AddColumn<int>(
                name: "TradeId", table: "StockPositions", type: "int", nullable: true);
            migrationBuilder.CreateIndex(
                name: "IX_StockPositions_TradeId", table: "StockPositions", column: "TradeId");

            // ── Step 2: Rename TradeEntries → Trades (the parent trade/journal table) ──
            migrationBuilder.RenameTable(name: "TradeEntries", newName: "Trades");

            // Rename existing index
            migrationBuilder.RenameIndex(
                name: "IX_TradeEntries_Symbol", table: "Trades", newName: "IX_Trades_Symbol");

            // Add ParentTradeId self-FK
            migrationBuilder.AddColumn<int>(
                name: "ParentTradeId", table: "Trades", type: "int", nullable: true);
            migrationBuilder.CreateIndex(
                name: "IX_Trades_ParentTradeId", table: "Trades", column: "ParentTradeId");

            // ── Step 3: Add TradeId FK to OptionPositions ──
            migrationBuilder.AddColumn<int>(
                name: "TradeId", table: "OptionPositions", type: "int", nullable: true);
            migrationBuilder.CreateIndex(
                name: "IX_OptionPositions_TradeId", table: "OptionPositions", column: "TradeId");

            // ── Step 4: Create TradeEvents table ──
            migrationBuilder.CreateTable(
                name: "TradeEvents",
                columns: table => new
                {
                    Id = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    AccountId = table.Column<int>(type: "int", nullable: false),
                    TradeId = table.Column<int>(type: "int", nullable: false),
                    Type = table.Column<string>(type: "TEXT", nullable: false),
                    Date = table.Column<DateTime>(type: "datetime2", nullable: false),
                    Notes = table.Column<string>(type: "TEXT", nullable: true),
                    PnlImpact = table.Column<decimal>(type: "decimal(18,6)", precision: 18, scale: 6, nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_TradeEvents", x => x.Id);
                    table.ForeignKey(
                        name: "FK_TradeEvents_Trades_TradeId",
                        column: x => x.TradeId,
                        principalTable: "Trades",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });
            migrationBuilder.CreateIndex(
                name: "IX_TradeEvents_TradeId", table: "TradeEvents", column: "TradeId");

            // ── Step 5: Add FK constraints ──
            migrationBuilder.AddForeignKey(
                name: "FK_OptionPositions_Trades_TradeId",
                table: "OptionPositions", column: "TradeId",
                principalTable: "Trades", principalColumn: "Id",
                onDelete: ReferentialAction.SetNull);

            migrationBuilder.AddForeignKey(
                name: "FK_StockPositions_Trades_TradeId",
                table: "StockPositions", column: "TradeId",
                principalTable: "Trades", principalColumn: "Id",
                onDelete: ReferentialAction.SetNull);

            migrationBuilder.AddForeignKey(
                name: "FK_Trades_Trades_ParentTradeId",
                table: "Trades", column: "ParentTradeId",
                principalTable: "Trades", principalColumn: "Id",
                onDelete: ReferentialAction.Restrict);

            // ── Step 6: Migrate enum values ──
            // Budget: Core(0) → Drift(1), Speculative(1) → Speculative(3)
            // New order: LongTerm=0, Drift=1, Swing=2, Speculative=3
            migrationBuilder.Sql("UPDATE [Trades] SET [Budget] = 3 WHERE [Budget] = 1;"); // Speculative → 3
            migrationBuilder.Sql("UPDATE [Trades] SET [Budget] = 1 WHERE [Budget] = 0;"); // Core → Drift(1)

            // Strategy: reorder to match new enum
            // Old: PositiveDrift=0, RangeBoundCommodities=1, PEADS=2, Momentum=3, IVMeanReversion=4,
            //      SectorStrength=5, SectorWeakness=6, Breakout=7, GreenLineBreakout=8, Slingshot=9, PreEarnings=10
            // New: PositiveDrift=0, RangeBound=1, PEAD=2, BreakoutMomentum=3, IVMeanReversion=4,
            //      SectorStrength=5, SectorWeakness=6, GreenLineBreakout=7, Slingshot=8, PreEarnings=9
            // Mappings that change: Breakout(7) → BreakoutMomentum(3), GreenLineBreakout(8)→7, Slingshot(9)→8, PreEarnings(10)→9
            // Must process in correct order to avoid collisions
            migrationBuilder.Sql("UPDATE [Trades] SET [Strategy] = 3 WHERE [Strategy] = 7;");  // Breakout → BreakoutMomentum
            migrationBuilder.Sql("UPDATE [Trades] SET [Strategy] = 7 WHERE [Strategy] = 8;");  // GreenLineBreakout 8→7
            migrationBuilder.Sql("UPDATE [Trades] SET [Strategy] = 8 WHERE [Strategy] = 9;");  // Slingshot 9→8
            migrationBuilder.Sql("UPDATE [Trades] SET [Strategy] = 9 WHERE [Strategy] = 10;"); // PreEarnings 10→9
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            // Reverse enum migrations
            migrationBuilder.Sql("UPDATE [Trades] SET [Strategy] = 10 WHERE [Strategy] = 9;");
            migrationBuilder.Sql("UPDATE [Trades] SET [Strategy] = 9 WHERE [Strategy] = 8;");
            migrationBuilder.Sql("UPDATE [Trades] SET [Strategy] = 8 WHERE [Strategy] = 7;");
            migrationBuilder.Sql("UPDATE [Trades] SET [Strategy] = 7 WHERE [Strategy] = 3 AND [Strategy] != 3;");
            migrationBuilder.Sql("UPDATE [Trades] SET [Budget] = 0 WHERE [Budget] = 1;");
            migrationBuilder.Sql("UPDATE [Trades] SET [Budget] = 1 WHERE [Budget] = 3;");

            // Drop FKs
            migrationBuilder.DropForeignKey(name: "FK_Trades_Trades_ParentTradeId", table: "Trades");
            migrationBuilder.DropForeignKey(name: "FK_StockPositions_Trades_TradeId", table: "StockPositions");
            migrationBuilder.DropForeignKey(name: "FK_OptionPositions_Trades_TradeId", table: "OptionPositions");

            // Drop new tables
            migrationBuilder.DropTable(name: "TradeEvents");

            // Remove added columns
            migrationBuilder.DropIndex(name: "IX_OptionPositions_TradeId", table: "OptionPositions");
            migrationBuilder.DropColumn(name: "TradeId", table: "OptionPositions");

            migrationBuilder.DropIndex(name: "IX_Trades_ParentTradeId", table: "Trades");
            migrationBuilder.DropColumn(name: "ParentTradeId", table: "Trades");

            migrationBuilder.DropIndex(name: "IX_StockPositions_TradeId", table: "StockPositions");
            migrationBuilder.DropColumn(name: "TradeId", table: "StockPositions");

            // Rename back: Trades → TradeEntries
            migrationBuilder.RenameIndex(
                name: "IX_Trades_Symbol", table: "Trades", newName: "IX_TradeEntries_Symbol");
            migrationBuilder.RenameTable(name: "Trades", newName: "TradeEntries");

            // Rename back: StockPositions → Trades
            migrationBuilder.RenameIndex(
                name: "IX_StockPositions_Symbol_Date", table: "StockPositions", newName: "IX_Trades_Symbol_Date");
            migrationBuilder.RenameIndex(
                name: "IX_StockPositions_ExecutionId", table: "StockPositions", newName: "IX_Trades_ExecutionId");
            migrationBuilder.RenameTable(name: "StockPositions", newName: "Trades");
        }
    }
}
