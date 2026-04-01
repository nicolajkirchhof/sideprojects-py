using tradelog.Models;
using tradelog.Services;
using Microsoft.EntityFrameworkCore;

namespace tradelog.Data;

public class DataContext : DbContext
{
    private readonly IAccountContext? _accountContext;

    private int CurrentAccountId => _accountContext?.CurrentAccountId ?? 0;

    public DataContext(DbContextOptions<DataContext> options, IAccountContext? accountContext = null)
        : base(options)
    {
        _accountContext = accountContext;
    }

    public DbSet<Account> Accounts { get; set; }
    public DbSet<TradeEntry> TradeEntries { get; set; }
    public DbSet<OptionPosition> OptionPositions { get; set; }
    public DbSet<OptionPositionsLog> OptionPositionsLogs { get; set; }
    public DbSet<Trade> Trades { get; set; }
    public DbSet<Capital> Capitals { get; set; }
    public DbSet<WeeklyPrep> WeeklyPreps { get; set; }
    public DbSet<Portfolio> Portfolios { get; set; }
    public DbSet<StockPriceCache> StockPriceCaches { get; set; }

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        base.OnModelCreating(modelBuilder);

        // Default precision for all decimal properties: 18 digits, 6 decimal places
        foreach (var entity in modelBuilder.Model.GetEntityTypes())
        {
            foreach (var property in entity.GetProperties()
                         .Where(p => p.ClrType == typeof(decimal) || p.ClrType == typeof(decimal?)))
            {
                property.SetPrecision(18);
                property.SetScale(6);
            }
        }

        // ────────────────────────────────────────────
        // Global query filters — scope all account-owned entities
        // Reference CurrentAccountId property (not a local variable) so EF Core
        // parameterizes the filter and re-evaluates it per query execution.
        // ────────────────────────────────────────────

        modelBuilder.Entity<TradeEntry>().HasQueryFilter(e => e.AccountId == CurrentAccountId);
        modelBuilder.Entity<OptionPosition>().HasQueryFilter(e => e.AccountId == CurrentAccountId);
        modelBuilder.Entity<OptionPositionsLog>().HasQueryFilter(e => e.AccountId == CurrentAccountId);
        modelBuilder.Entity<Trade>().HasQueryFilter(e => e.AccountId == CurrentAccountId);
        modelBuilder.Entity<Capital>().HasQueryFilter(e => e.AccountId == CurrentAccountId);
        modelBuilder.Entity<WeeklyPrep>().HasQueryFilter(e => e.AccountId == CurrentAccountId);
        modelBuilder.Entity<Portfolio>().HasQueryFilter(e => e.AccountId == CurrentAccountId);

        // ────────────────────────────────────────────
        // Indexes
        // ────────────────────────────────────────────

        // OptionPositionsLog: composite index for "latest snapshot" queries (account-scoped)
        modelBuilder.Entity<OptionPositionsLog>()
            .HasIndex(e => new { e.AccountId, e.ContractId, e.DateTime })
            .IsDescending(false, false, true)
            .IsUnique();

        modelBuilder.Entity<OptionPosition>().HasIndex(e => e.Symbol);
        modelBuilder.Entity<OptionPosition>().HasIndex(e => e.ContractId);
        modelBuilder.Entity<Trade>().HasIndex(e => new { e.Symbol, e.Date });
        modelBuilder.Entity<TradeEntry>().HasIndex(e => e.Symbol);
        modelBuilder.Entity<StockPriceCache>().HasIndex(e => e.Symbol).IsUnique();
        modelBuilder.Entity<Trade>().HasIndex(e => e.ExecutionId);
        modelBuilder.Entity<Account>().HasIndex(e => e.IbkrAccountId).IsUnique();

        // Store CloseReasons bitmask as int
        modelBuilder.Entity<OptionPosition>()
            .Property(e => e.CloseReasons)
            .HasConversion<int?>();
    }

    public override int SaveChanges()
    {
        SetAccountIdOnNewEntities();
        return base.SaveChanges();
    }

    public override Task<int> SaveChangesAsync(CancellationToken cancellationToken = default)
    {
        SetAccountIdOnNewEntities();
        return base.SaveChangesAsync(cancellationToken);
    }

    private void SetAccountIdOnNewEntities()
    {
        var accountId = _accountContext?.CurrentAccountId ?? 0;
        if (accountId == 0) return;

        foreach (var entry in ChangeTracker.Entries()
                     .Where(e => e.State == EntityState.Added && e.Entity is IAccountScoped))
        {
            ((IAccountScoped)entry.Entity).AccountId = accountId;
        }
    }
}

/// <summary>Marker interface for entities that are scoped to an account.</summary>
public interface IAccountScoped
{
    int AccountId { get; set; }
}
