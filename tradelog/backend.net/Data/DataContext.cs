using backend.net.Models;
using Microsoft.EntityFrameworkCore;

namespace backend.net.Data;

public class DataContext : DbContext
{
    public DataContext(DbContextOptions<DataContext> options) : base(options)
    {
    }

    public DbSet<Position> Positions { get; set; }
    public DbSet<Trade> Trades { get; set; }
    public DbSet<TradeIdea> TradeIdeas { get; set; }
    public DbSet<TradeLog> TradeLogs { get; set; }
}
