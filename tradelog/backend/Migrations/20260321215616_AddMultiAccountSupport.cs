using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace tradelog.Migrations
{
    /// <inheritdoc />
    public partial class AddMultiAccountSupport : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<int>(
                name: "AccountId",
                table: "WeeklyPreps",
                type: "int",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<int>(
                name: "AccountId",
                table: "Trades",
                type: "int",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<int>(
                name: "AccountId",
                table: "TradeEntries",
                type: "int",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<int>(
                name: "AccountId",
                table: "Portfolios",
                type: "int",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<int>(
                name: "AccountId",
                table: "OptionPositionsLogs",
                type: "int",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<int>(
                name: "AccountId",
                table: "OptionPositions",
                type: "int",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<int>(
                name: "AccountId",
                table: "Capitals",
                type: "int",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.CreateTable(
                name: "Accounts",
                columns: table => new
                {
                    Id = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    IbkrAccountId = table.Column<string>(type: "nvarchar(20)", maxLength: 20, nullable: false),
                    Name = table.Column<string>(type: "nvarchar(50)", maxLength: 50, nullable: false),
                    Host = table.Column<string>(type: "nvarchar(100)", maxLength: 100, nullable: false),
                    Port = table.Column<int>(type: "int", nullable: false),
                    ClientId = table.Column<int>(type: "int", nullable: false),
                    IsDefault = table.Column<bool>(type: "bit", nullable: false),
                    LastSyncAt = table.Column<DateTime>(type: "datetime2", nullable: true),
                    LastSyncResult = table.Column<string>(type: "nvarchar(500)", maxLength: 500, nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Accounts", x => x.Id);
                });

            migrationBuilder.CreateIndex(
                name: "IX_Accounts_IbkrAccountId",
                table: "Accounts",
                column: "IbkrAccountId",
                unique: true);

            // Backfill: create a default account and assign all existing records to it
            migrationBuilder.Sql(@"
                -- Migrate IbkrConfig settings to the default Account (if IbkrConfigs had data)
                IF EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'IbkrConfigs')
                BEGIN
                    INSERT INTO Accounts (IbkrAccountId, Name, Host, Port, ClientId, IsDefault, LastSyncAt, LastSyncResult)
                    SELECT TOP 1 'UNKNOWN', 'Default', Host, Port, ClientId, 1, LastSyncAt, LastSyncResult
                    FROM IbkrConfigs
                END

                -- If no IbkrConfig existed but there is data, create a placeholder account
                IF NOT EXISTS (SELECT 1 FROM Accounts)
                   AND (EXISTS (SELECT 1 FROM Trades) OR EXISTS (SELECT 1 FROM OptionPositions))
                BEGIN
                    INSERT INTO Accounts (IbkrAccountId, Name, Host, Port, ClientId, IsDefault)
                    VALUES ('UNKNOWN', 'Default', '127.0.0.1', 7497, 1, 1)
                END

                -- Assign all existing records to the default account
                IF EXISTS (SELECT 1 FROM Accounts)
                BEGIN
                    DECLARE @defaultAccountId INT = (SELECT TOP 1 Id FROM Accounts WHERE IsDefault = 1)
                    IF @defaultAccountId IS NOT NULL
                    BEGIN
                        UPDATE Trades SET AccountId = @defaultAccountId WHERE AccountId = 0
                        UPDATE TradeEntries SET AccountId = @defaultAccountId WHERE AccountId = 0
                        UPDATE OptionPositions SET AccountId = @defaultAccountId WHERE AccountId = 0
                        UPDATE OptionPositionsLogs SET AccountId = @defaultAccountId WHERE AccountId = 0
                        UPDATE Capitals SET AccountId = @defaultAccountId WHERE AccountId = 0
                        UPDATE WeeklyPreps SET AccountId = @defaultAccountId WHERE AccountId = 0
                        UPDATE Portfolios SET AccountId = @defaultAccountId WHERE AccountId = 0
                    END
                END
            ");

            migrationBuilder.DropTable(
                name: "IbkrConfigs");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "Accounts");

            migrationBuilder.DropColumn(
                name: "AccountId",
                table: "WeeklyPreps");

            migrationBuilder.DropColumn(
                name: "AccountId",
                table: "Trades");

            migrationBuilder.DropColumn(
                name: "AccountId",
                table: "TradeEntries");

            migrationBuilder.DropColumn(
                name: "AccountId",
                table: "Portfolios");

            migrationBuilder.DropColumn(
                name: "AccountId",
                table: "OptionPositionsLogs");

            migrationBuilder.DropColumn(
                name: "AccountId",
                table: "OptionPositions");

            migrationBuilder.DropColumn(
                name: "AccountId",
                table: "Capitals");

            migrationBuilder.CreateTable(
                name: "IbkrConfigs",
                columns: table => new
                {
                    Id = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    ClientId = table.Column<int>(type: "int", nullable: false),
                    Host = table.Column<string>(type: "nvarchar(100)", maxLength: 100, nullable: false),
                    LastSyncAt = table.Column<DateTime>(type: "datetime2", nullable: true),
                    LastSyncResult = table.Column<string>(type: "nvarchar(500)", maxLength: 500, nullable: true),
                    Port = table.Column<int>(type: "int", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_IbkrConfigs", x => x.Id);
                });
        }
    }
}
