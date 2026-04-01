using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;
using tradelog.Data;
using tradelog.Services;

namespace tradelog.Tests.Fixtures;

/// <summary>
/// Creates a SQLite in-memory DataContext for integration tests.
/// Each call to CreateContext() shares the same in-memory database (same connection).
/// Dispose the fixture to close the connection.
/// </summary>
public class TestDbFixture : IDisposable
{
    private readonly SqliteConnection _connection;
    private readonly DbContextOptions<DataContext> _options;

    public int TestAccountId { get; }

    public TestDbFixture(int testAccountId = 1)
    {
        TestAccountId = testAccountId;

        // SQLite in-memory DB lives as long as the connection is open
        _connection = new SqliteConnection("DataSource=:memory:");
        _connection.Open();

        _options = new DbContextOptionsBuilder<DataContext>()
            .UseSqlite(_connection)
            .Options;

        // Create schema
        using var ctx = CreateContext();
        ctx.Database.EnsureCreated();
    }

    public DataContext CreateContext()
    {
        return new DataContext(_options, new StubAccountContext(TestAccountId));
    }

    /// <summary>
    /// Creates a DataContext scoped to a different account ID (for testing account isolation).
    /// </summary>
    public DataContext CreateContext(int accountId)
    {
        return new DataContext(_options, new StubAccountContext(accountId));
    }

    public void Dispose()
    {
        _connection.Dispose();
    }
}

/// <summary>
/// Stub IAccountContext that returns a fixed account ID for testing.
/// </summary>
public class StubAccountContext : IAccountContext
{
    public int CurrentAccountId { get; }

    public StubAccountContext(int accountId)
    {
        CurrentAccountId = accountId;
    }
}
