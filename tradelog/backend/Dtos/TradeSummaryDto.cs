namespace tradelog.Dtos;

public class TradeSummaryDto
{
    public int TradeId { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public string TypeOfTrade { get; set; } = string.Empty;
    public string Strategy { get; set; } = string.Empty;
    public string Budget { get; set; } = string.Empty;
    public string Status { get; set; } = "Open";
    public DateTime Date { get; set; }

    // Leg counts
    public int OptionLegCount { get; set; }
    public int StockLegCount { get; set; }

    // Aggregated P/L
    public decimal Pnl { get; set; }
    public decimal UnrealizedPnl { get; set; }
    public decimal RealizedPnl { get; set; }
    public decimal Commissions { get; set; }

    // Aggregated Greeks (from option legs only)
    public decimal Delta { get; set; }
    public decimal Theta { get; set; }
    public decimal Gamma { get; set; }
    public decimal Vega { get; set; }
    public decimal Margin { get; set; }
}
