using tradelog.Models;
using Microsoft.EntityFrameworkCore;

namespace tradelog.Data;

public class DataContext : DbContext
{
    public DataContext(DbContextOptions<DataContext> options) : base(options)
    {
    }

    public DbSet<TradeEntry> TradeEntries { get; set; }
    public DbSet<OptionPosition> OptionPositions { get; set; }
    public DbSet<OptionPositionsLog> OptionPositionsLogs { get; set; }
    public DbSet<Trade> Trades { get; set; }
    public DbSet<Capital> Capitals { get; set; }
    public DbSet<WeeklyPrep> WeeklyPreps { get; set; }
    public DbSet<Portfolio> Portfolios { get; set; }

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        base.OnModelCreating(modelBuilder);

        // Default precision for all decimal properties: 18 digits, 6 decimal places
        // Covers prices, Greeks, percentages, and monetary values
        foreach (var entity in modelBuilder.Model.GetEntityTypes())
        {
            foreach (var property in entity.GetProperties()
                         .Where(p => p.ClrType == typeof(decimal) || p.ClrType == typeof(decimal?)))
            {
                property.SetPrecision(18);
                property.SetScale(6);
            }
        }

        // OptionPositionsLog: composite index for "latest snapshot" queries
        modelBuilder.Entity<OptionPositionsLog>()
            .HasIndex(e => new { e.ContractId, e.DateTime })
            .IsDescending(false, true)
            .IsUnique();

        // OptionPosition: index on Symbol for grouping queries
        modelBuilder.Entity<OptionPosition>()
            .HasIndex(e => e.Symbol);

        // OptionPosition: index on ContractId for log lookups
        modelBuilder.Entity<OptionPosition>()
            .HasIndex(e => e.ContractId);

        // Trade: composite index for running position queries
        modelBuilder.Entity<Trade>()
            .HasIndex(e => new { e.Symbol, e.Date });

        // TradeEntry: index on Symbol for instrument summary lookups
        modelBuilder.Entity<TradeEntry>()
            .HasIndex(e => e.Symbol);

        // Store CloseReasons bitmask as int
        modelBuilder.Entity<OptionPosition>()
            .Property(e => e.CloseReasons)
            .HasConversion<int?>();
    }
}
