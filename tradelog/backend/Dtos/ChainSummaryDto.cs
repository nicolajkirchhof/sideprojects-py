namespace tradelog.Dtos;

public class ChainSummaryDto
{
    public int RootTradeId { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public string Strategy { get; set; } = string.Empty;
    public string Budget { get; set; } = string.Empty;
    public int ChainLength { get; set; }
    public string Status { get; set; } = "Closed";
    public DateTime StartDate { get; set; }

    // Aggregated across all trades in chain
    public decimal TotalPnl { get; set; }
    public decimal PremiumCollected { get; set; }
    public decimal PremiumLost { get; set; }
    public int EventCount { get; set; }
}
