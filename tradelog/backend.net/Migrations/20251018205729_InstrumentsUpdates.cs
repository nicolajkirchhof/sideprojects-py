using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace backend.net.Migrations
{
    /// <inheritdoc />
    public partial class InstrumentsUpdates : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "ContractId",
                table: "Instruments");

            migrationBuilder.AddColumn<string>(
                name: "InstrumentSpecifics",
                table: "Positions",
                type: "nvarchar(20)",
                maxLength: 20,
                nullable: true);

            migrationBuilder.AlterColumn<int>(
                name: "SecType",
                table: "Instruments",
                type: "int",
                nullable: false,
                oldClrType: typeof(string),
                oldType: "nvarchar(50)",
                oldMaxLength: 50);

            migrationBuilder.CreateIndex(
                name: "IX_Positions_InstrumentId",
                table: "Positions",
                column: "InstrumentId");

            migrationBuilder.AddForeignKey(
                name: "FK_Positions_Instruments_InstrumentId",
                table: "Positions",
                column: "InstrumentId",
                principalTable: "Instruments",
                principalColumn: "Id",
                onDelete: ReferentialAction.Cascade);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_Positions_Instruments_InstrumentId",
                table: "Positions");

            migrationBuilder.DropIndex(
                name: "IX_Positions_InstrumentId",
                table: "Positions");

            migrationBuilder.DropColumn(
                name: "InstrumentSpecifics",
                table: "Positions");

            migrationBuilder.AlterColumn<string>(
                name: "SecType",
                table: "Instruments",
                type: "nvarchar(50)",
                maxLength: 50,
                nullable: false,
                oldClrType: typeof(int),
                oldType: "int");

            migrationBuilder.AddColumn<int>(
                name: "ContractId",
                table: "Instruments",
                type: "int",
                nullable: false,
                defaultValue: 0);
        }
    }
}
