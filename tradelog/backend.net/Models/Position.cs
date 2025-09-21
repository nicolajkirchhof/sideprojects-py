using System.ComponentModel.DataAnnotations.Schema;

namespace backend.net.Models;

public class Position
{
    public int Id { get; set; }
    public int TradeId { get; set; }
    public string ContractId { get; set; }
    public string Type { get; set; }
    public DateTime Opened { get; set; }
    public DateTime Expiry { get; set; }
    public DateTime? Closed { get; set; }
    public int PositionValue { get; set; }
    public string? Right { get; set; }
    public double Strike { get; set; }
    public double Cost { get; set; }
    public double? Close { get; set; }
    public int Multiplier { get; set; }

    [ForeignKey("TradeId")]
    public Trade Trade { get; set; }
}
