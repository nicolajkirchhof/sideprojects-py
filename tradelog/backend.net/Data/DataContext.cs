using backend.net.Models;
using Microsoft.EntityFrameworkCore;

namespace backend.net.Data;

public class DataContext : DbContext
{
    public DataContext(DbContextOptions<DataContext> options) : base(options)
    {
    }

    public DbSet<Position> Positions { get; set; }
    public DbSet<Log> Logs { get; set; }
    public DbSet<Capital> Capitals { get; set; }
    public DbSet<Tracking> Trackings { get; set; }
    public DbSet<Instrument> Instruments { get; set; }
}
