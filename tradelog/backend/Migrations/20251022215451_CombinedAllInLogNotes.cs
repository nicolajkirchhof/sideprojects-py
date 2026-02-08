using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace backend.net.Migrations
{
    /// <inheritdoc />
    public partial class CombinedAllInLogNotes : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "ExpectedOutcome",
                table: "Logs");

            migrationBuilder.DropColumn(
                name: "FA",
                table: "Logs");

            migrationBuilder.DropColumn(
                name: "Lernings",
                table: "Logs");

            migrationBuilder.DropColumn(
                name: "TA",
                table: "Logs");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "ExpectedOutcome",
                table: "Logs",
                type: "nvarchar(max)",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "FA",
                table: "Logs",
                type: "nvarchar(max)",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "Lernings",
                table: "Logs",
                type: "nvarchar(max)",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "TA",
                table: "Logs",
                type: "nvarchar(max)",
                nullable: true);
        }
    }
}
