namespace tradelog.Dtos;

public class StockPositionDto
{
    // Stored fields
    public int Id { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public DateTime Date { get; set; }
    public int PosChange { get; set; }
    public decimal Price { get; set; }
    public decimal Commission { get; set; }
    public int Multiplier { get; set; }

    public int? TradeId { get; set; }
    public string? Notes { get; set; }

    // Computed (running position tracking)
    public int LastPos { get; set; }
    public int TotalPos { get; set; }
    public decimal AvgPrice { get; set; }
    public decimal Pnl { get; set; }
}
