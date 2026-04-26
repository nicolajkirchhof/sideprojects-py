using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace tradelog.Migrations
{
    /// <inheritdoc />
    public partial class PendingModelChanges : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            // All operations in this migration are column type renames between SQL Server
            // (nvarchar/int) and SQLite (TEXT/INTEGER) type names. SQLite uses type affinity
            // and ignores the specific type name, so these are all no-ops at the storage level.
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
        }
    }
}
