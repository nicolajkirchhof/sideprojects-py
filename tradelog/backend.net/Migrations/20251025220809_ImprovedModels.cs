using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace backend.net.Migrations
{
    /// <inheritdoc />
    public partial class ImprovedModels : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "CloseReason",
                table: "Positions");

            migrationBuilder.DropColumn(
                name: "Multiplier",
                table: "Positions");

            migrationBuilder.DropColumn(
                name: "Strategy",
                table: "Logs");

            migrationBuilder.AddColumn<int>(
                name: "CloseReasons",
                table: "Positions",
                type: "int",
                nullable: true);

            migrationBuilder.AddColumn<int>(
                name: "ProfitMechanism",
                table: "Logs",
                type: "int",
                nullable: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "CloseReasons",
                table: "Positions");

            migrationBuilder.DropColumn(
                name: "ProfitMechanism",
                table: "Logs");

            migrationBuilder.AddColumn<string>(
                name: "CloseReason",
                table: "Positions",
                type: "nvarchar(max)",
                nullable: true);

            migrationBuilder.AddColumn<int>(
                name: "Multiplier",
                table: "Positions",
                type: "int",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<string>(
                name: "Strategy",
                table: "Logs",
                type: "nvarchar(max)",
                nullable: true);
        }
    }
}
