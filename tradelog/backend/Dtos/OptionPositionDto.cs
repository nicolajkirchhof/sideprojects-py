using tradelog.Models;

namespace tradelog.Dtos;

public class OptionPositionDto
{
    // Stored fields
    public int Id { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public string ContractId { get; set; } = string.Empty;
    public DateTime Opened { get; set; }
    public DateTime Expiry { get; set; }
    public DateTime? Closed { get; set; }
    public int Pos { get; set; }
    public PositionRight Right { get; set; }
    public decimal Strike { get; set; }
    public decimal Cost { get; set; }
    public decimal? ClosePrice { get; set; }
    public decimal Commission { get; set; }
    public int Multiplier { get; set; }
    public CloseReasons? CloseReasons { get; set; }

    // From latest OptionPositionsLog
    public decimal? LastPrice { get; set; }
    public decimal? LastValue { get; set; }
    public decimal? TimeValue { get; set; }
    public decimal? Delta { get; set; }
    public decimal? Theta { get; set; }
    public decimal? Gamma { get; set; }
    public decimal? Vega { get; set; }
    public decimal? Iv { get; set; }
    public decimal? Margin { get; set; }

    // Computed P/L
    public decimal? UnrealizedPnl { get; set; }
    public decimal? UnrealizedPnlPct { get; set; }
    public decimal? RealizedPnl { get; set; }
    public decimal? RealizedPnlPct { get; set; }
    public decimal? DurationPct { get; set; }
    public decimal? Roic { get; set; }
}
