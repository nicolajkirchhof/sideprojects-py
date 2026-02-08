using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace backend.net.Models;

public enum PositionTypes
{
    Call,
    Put,
    Underlying,
}

[Flags]
public enum CloseReasons
{
   TakeLoss = 1,
    TakeProfit = 2,
    Roll = 4,
    AssumptionInvalidated = 8,
    TimeLimit = 16,
    Other = 32,
}

public class Position
{
    public int Id { get; set; }
    public int InstrumentId { get; set; }
    public Instrument Instrument { get; set; }

    [StringLength(20)]
    public string? InstrumentSpecifics { get; set; } // E.G. Future Contract

    [StringLength(20)]
    public string ContractId { get; set; }
    public PositionTypes Type { get; set; }
    public DateTime Opened { get; set; }
    public DateTime Expiry { get; set; }
    public DateTime? Closed { get; set; }
    public int Size{ get; set; }
    public double Strike { get; set; }
    public double Cost { get; set; }
    public double? Close { get; set; }
    public double? Comission { get; set; }
    public int? CloseReasons { get; set; }
}
