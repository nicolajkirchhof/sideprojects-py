using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace tradelog.Migrations
{
    /// <inheritdoc />
    public partial class AddOptionPositionLogCount : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<int>(
                name: "LogCount",
                table: "OptionPositions",
                type: "int",
                nullable: true);

            // Legacy closed positions freeze at 0 — we intentionally discard any
            // historical sample count since LogCount is a forward-looking convenience
            // field. Legacy open positions stay NULL and are populated by the
            // application's startup backfill.
            migrationBuilder.Sql(
                "UPDATE OptionPositions SET LogCount = 0 WHERE Closed IS NOT NULL");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "LogCount",
                table: "OptionPositions");
        }
    }
}
