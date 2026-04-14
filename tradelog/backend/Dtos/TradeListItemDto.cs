namespace tradelog.Dtos;

/// <summary>
/// Lightweight DTO for the trades list (GET /api/trades).
/// Includes computed P&L from linked positions.
/// </summary>
public class TradeListItemDto
{
    public int Id { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public DateTime Date { get; set; }
    public int TypeOfTrade { get; set; }
    public int? Directional { get; set; }
    public int Budget { get; set; }
    public int Strategy { get; set; }
    public int? ManagementRating { get; set; }
    public string? Status { get; set; }
    public int? ParentTradeId { get; set; }

    /// <summary>Aggregated P&L from all linked option + stock positions.</summary>
    public decimal? Pnl { get; set; }
}
