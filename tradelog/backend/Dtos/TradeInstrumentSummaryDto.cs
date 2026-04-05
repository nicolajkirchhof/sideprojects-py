namespace tradelog.Dtos;

public class TradeInstrumentSummaryDto
{
    public string Symbol { get; set; } = string.Empty;
    public string Status { get; set; } = "Closed";

    // From latest Trade
    public string? Budget { get; set; }
    public string? PositionType { get; set; }
    public string? IntendedManagement { get; set; }

    // Current position state
    public int TotalPos { get; set; }
    public decimal AvgPrice { get; set; }
    public int Multiplier { get; set; }

    // P/L
    public decimal Pnl { get; set; }
    public decimal UnrealizedPnl { get; set; }
    public decimal? UnrealizedPnlPct { get; set; }
    public decimal RealizedPnl { get; set; }
    public decimal Commissions { get; set; }
}
