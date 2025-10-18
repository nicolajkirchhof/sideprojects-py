using System.ComponentModel.DataAnnotations;

namespace backend.net.Models;

public enum SecTypes
{
    Stock,
    Future,
    Forex
}

public class Instrument
{
    public int Id { get; set; }
    public SecTypes SecType { get; set; }
    [StringLength(50)]
    public string Symbol { get; set; }
    public int Multiplier { get; set; }
    [StringLength(50)]
    public string Sector { get; set; }
    [StringLength(50)]
    public string? Subsector { get; set; }
}
