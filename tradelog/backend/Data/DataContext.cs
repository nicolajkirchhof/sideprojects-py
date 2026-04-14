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
    public DbSet<Trade> Trades { get; set; }
    public DbSet<OptionPosition> OptionPositions { get; set; }
    public DbSet<OptionPositionsLog> OptionPositionsLogs { get; set; }
    public DbSet<StockPosition> StockPositions { get; set; }
    public DbSet<Capital> Capitals { get; set; }
    public DbSet<WeeklyPrep> WeeklyPreps { get; set; }
    public DbSet<StockPriceCache> StockPriceCaches { get; set; }
    public DbSet<LookupValue> LookupValues { get; set; }
    public DbSet<Document> Documents { get; set; }
    public DbSet<DocumentStrategyLink> DocumentStrategyLinks { get; set; }

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
        // Global query filters
        // ────────────────────────────────────────────

        modelBuilder.Entity<Trade>().HasQueryFilter(e => e.AccountId == CurrentAccountId);
        modelBuilder.Entity<OptionPosition>().HasQueryFilter(e => e.AccountId == CurrentAccountId);
        modelBuilder.Entity<OptionPositionsLog>().HasQueryFilter(e => e.AccountId == CurrentAccountId);
        modelBuilder.Entity<StockPosition>().HasQueryFilter(e => e.AccountId == CurrentAccountId);
        modelBuilder.Entity<Capital>().HasQueryFilter(e => e.AccountId == CurrentAccountId);
        modelBuilder.Entity<WeeklyPrep>().HasQueryFilter(e => e.AccountId == CurrentAccountId);
        modelBuilder.Entity<LookupValue>().HasQueryFilter(e => e.AccountId == CurrentAccountId);
        modelBuilder.Entity<Document>().HasQueryFilter(e => e.AccountId == CurrentAccountId);

        // ────────────────────────────────────────────
        // Lookup values — unique name per category per account
        // ────────────────────────────────────────────

        modelBuilder.Entity<LookupValue>()
            .HasIndex(e => new { e.AccountId, e.Category, e.Name })
            .IsUnique();

        modelBuilder.Entity<LookupValue>()
            .HasIndex(e => new { e.AccountId, e.Category, e.SortOrder });

        // ────────────────────────────────────────────
        // Document ↔ Strategy many-to-many junction
        // ────────────────────────────────────────────

        modelBuilder.Entity<DocumentStrategyLink>()
            .HasKey(dsl => new { dsl.DocumentId, dsl.LookupValueId });

        modelBuilder.Entity<DocumentStrategyLink>()
            .HasOne(dsl => dsl.Document)
            .WithMany(d => d.StrategyLinks)
            .HasForeignKey(dsl => dsl.DocumentId)
            .OnDelete(DeleteBehavior.Cascade);

        modelBuilder.Entity<DocumentStrategyLink>()
            .HasOne(dsl => dsl.LookupValue)
            .WithMany()
            .HasForeignKey(dsl => dsl.LookupValueId)
            .OnDelete(DeleteBehavior.Cascade);

        // ────────────────────────────────────────────
        // Indexes
        // ────────────────────────────────────────────

        modelBuilder.Entity<OptionPositionsLog>()
            .HasIndex(e => new { e.AccountId, e.ContractId, e.DateTime })
            .IsDescending(false, false, true)
            .IsUnique();

        modelBuilder.Entity<OptionPosition>().HasIndex(e => e.Symbol);
        modelBuilder.Entity<OptionPosition>().HasIndex(e => e.ContractId);
        modelBuilder.Entity<StockPosition>().HasIndex(e => new { e.Symbol, e.Date });
        modelBuilder.Entity<Trade>().HasIndex(e => e.Symbol);
        modelBuilder.Entity<StockPriceCache>().HasIndex(e => e.Symbol).IsUnique();
        modelBuilder.Entity<StockPosition>().HasIndex(e => e.ExecutionId);
        modelBuilder.Entity<Account>().HasIndex(e => e.IbkrAccountId).IsUnique();

        // Store CloseReasons bitmask as int
        modelBuilder.Entity<OptionPosition>()
            .Property(e => e.CloseReasons)
            .HasConversion<int?>();

        // ────────────────────────────────────────────
        // Foreign key configurations
        // ────────────────────────────────────────────

        modelBuilder.Entity<Trade>()
            .HasOne<Trade>()
            .WithMany()
            .HasForeignKey(t => t.ParentTradeId)
            .OnDelete(DeleteBehavior.Restrict);

        modelBuilder.Entity<OptionPosition>()
            .HasOne<Trade>()
            .WithMany(t => t.OptionPositions)
            .HasForeignKey(p => p.TradeId)
            .OnDelete(DeleteBehavior.SetNull);

        modelBuilder.Entity<StockPosition>()
            .HasOne<Trade>()
            .WithMany(t => t.StockPositions)
            .HasForeignKey(p => p.TradeId)
            .OnDelete(DeleteBehavior.SetNull);
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
