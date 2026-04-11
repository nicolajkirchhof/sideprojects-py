namespace tradelog.Dtos;

/// <summary>
/// Request body for POST /api/trades — creates a trade and optionally
/// links unassigned positions in a single transaction.
/// </summary>
public class TradeCreateDto
{
    public string Symbol { get; set; } = string.Empty;
    public DateTime Date { get; set; }
    public int TypeOfTrade { get; set; }
    public string? Notes { get; set; }
    public int? Directional { get; set; }
    public int? Timeframe { get; set; }
    public int Budget { get; set; }
    public int Strategy { get; set; }
    public bool NewsCatalyst { get; set; }
    public bool RecentEarnings { get; set; }
    public bool SectorSupport { get; set; }
    public bool Ath { get; set; }
    public decimal? Rvol { get; set; }
    public string? InstitutionalSupport { get; set; }
    public decimal? GapPct { get; set; }
    public decimal? XAtrMove { get; set; }
    public string? TaFaNotes { get; set; }
    public string? IntendedManagement { get; set; }
    public string? ActualManagement { get; set; }
    public int? ManagementRating { get; set; }
    public string? Learnings { get; set; }
    public int? ParentTradeId { get; set; }

    /// <summary>Optional: link these unassigned option positions to the new trade.</summary>
    public List<int>? OptionPositionIds { get; set; }

    /// <summary>Optional: link these unassigned stock positions to the new trade.</summary>
    public List<int>? StockPositionIds { get; set; }
}
